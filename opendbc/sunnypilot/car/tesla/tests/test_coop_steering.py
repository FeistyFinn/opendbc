"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import pytest

from opendbc.car import structs
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.tesla.carcontroller import get_safety_CP
from opendbc.sunnypilot.car.tesla.values import TeslaFlagsSP
from opendbc.sunnypilot.car.tesla.coop_steering import (
  CoopSteeringCarController,
  CoopSteeringCarControllerParams,
  apply_bounds,
  apply_deadzone,
  get_steer_from_lat_accel,
  get_override_torque_to_angle,
  DT_LAT_CTRL,
  STEER_OVERRIDE_MIN_TORQUE,
  STEER_OVERRIDE_MAX_LAT_ACCEL,
  STEER_OVERRIDE_TARGET_ANGLE_MAX,
  STEER_RESUME_RATE_LIMIT_RAMP_RATE,
)

# Tesla uses the Model Y vehicle model for lateral limiting (matches safety)
VM = VehicleModel(get_safety_CP())
STEER_ANGLE_MAX = CoopSteeringCarControllerParams.ANGLE_LIMITS.STEER_ANGLE_MAX


def _cs(steering_torque=0.0, v_ego=5.0, steering_angle=0.0, steering_rate_deg=0.0):
  out = structs.CarState()
  out.vEgo = v_ego
  out.vEgoRaw = v_ego
  out.steeringAngleDeg = steering_angle
  out.steeringTorque = steering_torque
  out.steeringRateDeg = steering_rate_deg
  return SimpleNamespace(out=out)


def _cp_sp(coop=True):
  cp_sp = structs.CarParamsSP()
  flags = 0
  if coop:
    flags |= TeslaFlagsSP.COOP_STEERING.value
  cp_sp.flags = flags
  return cp_sp


def _settle(controller, cp_sp, cs, apply_angle=0.0, frames=100):
  out = None
  for _ in range(frames):
    out = controller.update(apply_angle, True, cp_sp, cs, VM)
  return out


# --- pure helpers ---

def test_apply_bounds():
  assert apply_bounds(10.0, 5.0) == 5.0
  assert apply_bounds(-10.0, 5.0) == -5.0
  assert apply_bounds(2.0, 5.0) == 2.0


def test_apply_deadzone():
  # inside the deadzone -> zero
  assert apply_deadzone(0.4, STEER_OVERRIDE_MIN_TORQUE) == 0.0
  assert apply_deadzone(-0.4, STEER_OVERRIDE_MIN_TORQUE) == 0.0
  # outside the deadzone -> reduced by the deadzone, sign preserved
  assert apply_deadzone(1.5, STEER_OVERRIDE_MIN_TORQUE) == 1.5 - STEER_OVERRIDE_MIN_TORQUE
  assert apply_deadzone(-1.5, STEER_OVERRIDE_MIN_TORQUE) == -(1.5 - STEER_OVERRIDE_MIN_TORQUE)


def test_get_steer_from_lat_accel_direction():
  # higher lateral accel demand -> larger steering angle, same sign
  small = get_steer_from_lat_accel(1.0, 10.0, VM)
  large = get_steer_from_lat_accel(2.0, 10.0, VM)
  assert large > small > 0


# --- passthrough: feature off / not engaged ---

def test_passthrough_when_coop_disabled():
  c = CoopSteeringCarController()
  out = _settle(c, _cp_sp(coop=False), _cs(steering_torque=3.0))
  assert out.steeringAngleDeg == 0.0  # input apply_angle unchanged
  assert c.angle_override == 0


def test_passthrough_when_not_lat_active():
  c = CoopSteeringCarController()
  out = c.update(10.0, False, _cp_sp(coop=True), _cs(steering_torque=3.0, steering_angle=10.0), VM)
  assert out.steeringAngleDeg == 10.0
  assert out.lat_active is False
  assert c.angle_override == 0


# --- core behavior: torque -> angle offset ---

def test_small_torque_stays_in_deadzone():
  c = CoopSteeringCarController()
  _settle(c, _cp_sp(coop=True), _cs(steering_torque=0.3), frames=50)  # below MIN_TORQUE
  assert abs(c.angle_override) < 1e-6


def test_torque_produces_offset_in_torque_direction():
  pos = CoopSteeringCarController()
  out_pos = _settle(pos, _cp_sp(coop=True), _cs(steering_torque=2.0))
  assert pos.angle_override > 0.0
  assert out_pos.steeringAngleDeg > 0.0

  neg = CoopSteeringCarController()
  _settle(neg, _cp_sp(coop=True), _cs(steering_torque=-2.0))
  assert neg.angle_override < 0.0


def test_override_ramps_not_jumps():
  c = CoopSteeringCarController()
  cp_sp, cs = _cp_sp(coop=True), _cs(steering_torque=4.0)
  c.update(0.0, True, cp_sp, cs, VM)
  one_frame = abs(c.angle_override)
  _settle(c, cp_sp, cs)
  settled = abs(c.angle_override)
  # rate-limited: a single step is a fraction of the settled offset, not an instant jump
  assert 0.0 < one_frame < settled


def test_override_bounded_by_steer_angle_max():
  c = CoopSteeringCarController()
  _settle(c, _cp_sp(coop=True), _cs(steering_torque=50.0), frames=500)  # unrealistically large torque
  assert abs(c.angle_override) <= STEER_ANGLE_MAX + 1e-6


# --- release: no residual offset / snap-back ---

def test_release_resets_override_to_input_angle():
  c = CoopSteeringCarController()
  cp_sp = _cp_sp(coop=True)
  _settle(c, cp_sp, _cs(steering_torque=3.0))
  assert c.angle_override != 0.0

  # driver lets go and lateral disengages
  out = c.update(12.0, False, cp_sp, _cs(steering_torque=0.0, steering_angle=12.0), VM)
  assert c.angle_override == 0
  assert out.steeringAngleDeg == 12.0  # returns commanded angle, no leftover override


# ============================================================================
# Baseline-algorithm coverage added for upstreaming (cards #4 / #5 / #6 / #7 / #10).
# ============================================================================


def _run_moving_planner(cp_sp, torque, v_ego, planner_step, frames):
  """Step apply_angle by planner_step each frame (a moving planner), lat_active True."""
  c = CoopSteeringCarController()
  a = 0.0
  for _ in range(frames):
    a += planner_step
    c.update(a, True, cp_sp, _cs(steering_torque=torque, v_ego=v_ego), VM)
  return c


# --- #10: away/center rate gains collapsed to one symmetric limit ---

def test_symmetric_ramp_no_directional_asymmetry():
  # + and - torque of equal magnitude must produce exact mirror-image override trajectories, frame by
  # frame -- proving the ramp has no directional (away-vs-center) asymmetry after the #10 collapse.
  cp_sp = _cp_sp(coop=True)
  pos, neg = CoopSteeringCarController(), CoopSteeringCarController()
  for _ in range(60):
    p = pos.update(0.0, True, cp_sp, _cs(steering_torque=1.5, v_ego=10.0), VM).steeringAngleDeg
    n = neg.update(0.0, True, cp_sp, _cs(steering_torque=-1.5, v_ego=10.0), VM).steeringAngleDeg
    assert abs(p + n) < 1e-6


# --- #5: high-speed behavior (all legacy tests use v_ego = 5) ---

@pytest.mark.parametrize("v_ego", [5.0, 15.0, 30.0])
def test_settles_to_vm_predicted_offset_at_speed(v_ego):
  # The settled override tracks the vehicle-model torque->angle oracle (NOT literal 1/v^2 -- the slip
  # term makes it deviate ~26% by 30 m/s). gain = get_override_torque_to_angle(v, VM, MAX_LAT_ACCEL).
  cp_sp = _cp_sp(coop=True)
  torque = 1.5
  out = _settle(CoopSteeringCarController(), cp_sp, _cs(steering_torque=torque, v_ego=v_ego), frames=500)
  gain = get_override_torque_to_angle(v_ego, VM, STEER_OVERRIDE_MAX_LAT_ACCEL)
  expected = apply_deadzone(torque, STEER_OVERRIDE_MIN_TORQUE) * gain
  assert out.steeringAngleDeg == pytest.approx(expected, rel=0.03, abs=0.5)


@pytest.mark.parametrize("v_ego", [5.0, 15.0, 30.0])
def test_release_returns_to_zero_at_speed(v_ego):
  # Torque -> 0 with lat_active still True: the override unwinds back to ~0 at every speed.
  cp_sp = _cp_sp(coop=True)
  c = CoopSteeringCarController()
  _settle(c, cp_sp, _cs(steering_torque=2.0, v_ego=v_ego), frames=300)
  assert abs(c.angle_override) > 0.0
  out = _settle(c, cp_sp, _cs(steering_torque=0.0, v_ego=v_ego), frames=500)
  assert abs(out.steeringAngleDeg) < 0.5


# --- #7: speed-dependent max-offset envelope ---

def test_offset_envelope_monotone_nonincreasing():
  # At max torque, the settled offset is largest at the 1 m/s floor and monotone NON-increasing with
  # speed (the 360-deg clip pins the gain flat below ~3.3 m/s -> assert non-strict), never exceeding the
  # target-angle cap.
  cp_sp = _cp_sp(coop=True)
  speeds = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0, 30.0, 40.0]
  offsets = [_settle(CoopSteeringCarController(), cp_sp, _cs(steering_torque=3.0, v_ego=v), frames=600).steeringAngleDeg
             for v in speeds]
  assert offsets[0] == max(offsets)                       # largest at the floor
  for a, b in zip(offsets, offsets[1:], strict=False):
    assert b <= a + 1e-6                                  # non-increasing (flat region below ~3.3 m/s)
  assert all(o <= STEER_OVERRIDE_TARGET_ANGLE_MAX + 1e-6 for o in offsets)


# --- #4: interaction with a MOVING planner (the double-count branch, spec'd by #11) ---

def test_moving_planner_same_dir_subtracts_opposite_keeps_authority():
  # Planner moving the SAME way as the override -> its delta is subtracted (no double-count) so the
  # accumulated override is smaller than with a still planner. Planner moving OPPOSITE -> no subtraction,
  # full authority (identical to a still planner, which never triggers the branch). Never sign-flips.
  cp_sp = _cp_sp(coop=True)
  still = CoopSteeringCarController()
  _settle(still, cp_sp, _cs(steering_torque=1.5, v_ego=8.0), frames=60)
  same = _run_moving_planner(cp_sp, 1.5, 8.0, +0.1, 60)
  opp = _run_moving_planner(cp_sp, 1.5, 8.0, -0.1, 60)
  assert same.angle_override >= 0.0                                      # never sign-flipped
  assert same.angle_override < still.angle_override                     # same-dir planner suppresses it
  assert opp.angle_override == pytest.approx(still.angle_override, rel=0.01)  # opposite -> full authority


def test_moving_planner_reversal_keeps_override_continuous():
  # Across a planner direction reversal the double-count subtraction toggles on/off; because it is
  # bounded by abs(override_delta) the override never steps (stays continuous).
  cp_sp = _cp_sp(coop=True)
  c = CoopSteeringCarController()
  a, prev, max_jump = 0.0, 0.0, 0.0
  step = +0.2
  for i in range(60):
    if i == 30:
      step = -0.2
    a += step
    c.update(a, True, cp_sp, _cs(steering_torque=1.2, v_ego=10.0), VM)
    max_jump = max(max_jump, abs(c.angle_override - prev))
    prev = c.angle_override
  assert max_jump <= CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE + 1e-6


# --- #6: engage/disengage transitions (the resume rate limiter) ---

def test_resume_ramp_acceleration_builds():
  # On re-engage with a jumping planner angle and no driver torque, the resume rate limiter caps the
  # first-frame angle change to RAMP_RATE * DT^2 (~0.12 deg) and the allowed step accelerates each frame.
  cp_sp = _cp_sp(coop=True)
  c = CoopSteeringCarController()
  c.update(0.0, False, cp_sp, _cs(steering_torque=0.0), VM)  # inactive: baseline the limiter at 0
  steps, prev = [], 0.0
  for _ in range(5):
    out = c.update(30.0, True, cp_sp, _cs(steering_torque=0.0), VM)
    steps.append(out.steeringAngleDeg - prev)
    prev = out.steeringAngleDeg
  first = STEER_RESUME_RATE_LIMIT_RAMP_RATE * DT_LAT_CTRL ** 2
  assert steps[0] == pytest.approx(first, abs=1e-6)         # ~0.12 deg, not 30
  for a, b in zip(steps, steps[1:], strict=False):
    assert b > a - 1e-9                                     # accelerating (step grows each frame)


def test_engage_with_residual_torque_ramps_via_override_limiter():
  # Engaging with above-deadzone torque (apply_angle held) ramps the override -- the first-frame step is
  # a fraction of the settled target, bounded by the override limiter, not a jump.
  cp_sp = _cp_sp(coop=True)
  c = CoopSteeringCarController()
  c.update(0.0, False, cp_sp, _cs(steering_torque=1.5), VM)  # inactive
  c.update(0.0, True, cp_sp, _cs(steering_torque=1.5), VM)   # engage
  one_frame = abs(c.angle_override)
  settled = _settle(CoopSteeringCarController(), cp_sp, _cs(steering_torque=1.5), frames=300).steeringAngleDeg
  assert 0.0 < one_frame < abs(settled)


def test_disengage_resets_override_same_call():
  # lat_active dropping mid-override zeroes the override IN THE SAME update() call (not next frame) and
  # returns the raw passthrough angle.
  cp_sp = _cp_sp(coop=True)
  c = CoopSteeringCarController()
  _settle(c, cp_sp, _cs(steering_torque=3.0))
  assert c.angle_override != 0.0
  out = c.update(8.0, False, cp_sp, _cs(steering_torque=3.0, steering_angle=8.0), VM)
  assert out.steeringAngleDeg == 8.0
  assert c.angle_override == 0


def test_resume_ramp_bypassed_while_inactive():
  # While inactive the resume limiter is bypassed: a passthrough angle is unramped (a full jump is fine).
  cp_sp = _cp_sp(coop=True)
  c = CoopSteeringCarController()
  out = c.update(45.0, False, cp_sp, _cs(steering_torque=0.0, steering_angle=45.0), VM)
  assert out.steeringAngleDeg == 45.0
