"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

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
  STEER_INERTIA_J,
  STEER_INERTIA_J_MAX,
  STEER_INERTIA_TORQUE_LIMIT,
  STEER_OVERRIDE_MIN_TORQUE,
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


def _cp_sp(coop=True, inertia_comp=True, inertia_j=0.0):
  cp_sp = structs.CarParamsSP()
  flags = 0
  if coop:
    flags |= TeslaFlagsSP.COOP_STEERING.value
    if inertia_comp:
      flags |= TeslaFlagsSP.COOP_STEERING_INERTIA_COMP.value
  cp_sp.flags = flags
  cp_sp.teslaCoopSteeringInertiaJ = inertia_j  # 0.0 -> module default
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


# --- inertia compensation ---

def test_inertia_no_false_override_from_rotation_alone():
  # Wheel is rotating (large alpha from a rate ramp) but driver is not engaging:
  # the compensation must not manufacture a phantom intent torque that grows the override.
  # Guard: with |driver_torque| <= deadzone, tau_inertia is forced to 0.
  c = CoopSteeringCarController()
  cp_sp = _cp_sp(coop=True, inertia_comp=True)
  rate = 0.0
  for _ in range(50):
    c.update(0.0, True, cp_sp, _cs(steering_torque=0.0, steering_rate_deg=rate), VM)
    rate += 2.0  # 100 deg/s^2 constant acceleration -> non-trivial raw alpha
  assert abs(c.tau_inertia_last) < 1e-9
  assert abs(c.angle_override) < 1e-9


def test_inertia_noop_at_steady_state():
  # Above-deadzone driver torque with zero wheel rotation: alpha settles to 0, tau_inertia -> 0,
  # behavior matches the pre-compensation baseline (sub-toggle off).
  c_on = CoopSteeringCarController()
  c_off = CoopSteeringCarController()
  _settle(c_on, _cp_sp(coop=True, inertia_comp=True), _cs(steering_torque=2.0))
  _settle(c_off, _cp_sp(coop=True, inertia_comp=False), _cs(steering_torque=2.0))
  assert abs(c_on.tau_inertia_last) < 1e-9
  assert abs(c_on.angle_override - c_off.angle_override) < 1e-6


def test_inertia_attenuates_response_during_torque_step_with_rotation():
  # Driver torque above deadzone with wheel accelerating in the same direction (alpha > 0):
  # tau_inertia > 0 -> driver_torque_intent = tau - J*alpha < tau -> smaller target ->
  # angle_override grows more slowly than with compensation off.
  def run(cp_sp):
    c = CoopSteeringCarController()
    rate = 0.0
    for _ in range(20):
      c.update(0.0, True, cp_sp, _cs(steering_torque=2.0, steering_rate_deg=rate), VM)
      rate += 5.0  # +250 deg/s^2 (~4.4 rad/s^2) sustained alpha
    return c.angle_override

  with_comp = run(_cp_sp(coop=True, inertia_comp=True))
  without_comp = run(_cp_sp(coop=True, inertia_comp=False))
  assert with_comp > 0.0
  assert without_comp > 0.0
  assert with_comp < without_comp


def test_inertia_engagement_no_spike():
  # Pre-engage the wheel is already rotating fast. After engaging, the differentiator must not
  # see a spurious (current_rate - 0) / dt spike on the first frame -- reset_override_state seeds
  # prev_steering_rate_deg with the current rate and re-arms the filter to snap on first sample.
  c = CoopSteeringCarController()
  cp_sp = _cp_sp(coop=True, inertia_comp=True)
  c.update(0.0, False, cp_sp, _cs(steering_torque=2.0, steering_rate_deg=20.0), VM)
  c.update(0.0, True, cp_sp, _cs(steering_torque=2.0, steering_rate_deg=20.0), VM)
  # No differentiator spike: tau_inertia is well below the clamp on the engagement frame.
  assert abs(c.tau_inertia_last) < 0.1


def test_inertia_clamp_protects_against_runaway_alpha():
  # Extreme alpha (stale-then-jump pattern) must clamp tau_inertia at STEER_INERTIA_TORQUE_LIMIT
  # so a sensor glitch can't drive the override past sane bounds.
  c = CoopSteeringCarController()
  cp_sp = _cp_sp(coop=True, inertia_comp=True)
  # Drive the filter into a steady state first so it is initialized
  for _ in range(5):
    c.update(0.0, True, cp_sp, _cs(steering_torque=2.0, steering_rate_deg=0.0), VM)
  # Jump the rate: huge alpha
  c.update(0.0, True, cp_sp, _cs(steering_torque=2.0, steering_rate_deg=500.0), VM)
  assert abs(c.tau_inertia_last) <= STEER_INERTIA_TORQUE_LIMIT + 1e-9


def test_inertia_sub_toggle_off_matches_baseline():
  # Master COOP_STEERING on, COOP_STEERING_INERTIA_COMP off: even with nonzero rate signal,
  # no compensation runs (tau_inertia stays 0). Override behavior matches a run where the
  # rate signal is zero with the sub-flag on.
  c_off = CoopSteeringCarController()
  c_ref = CoopSteeringCarController()
  rate = 0.0
  for _ in range(80):
    c_off.update(0.0, True, _cp_sp(coop=True, inertia_comp=False),
                 _cs(steering_torque=2.0, steering_rate_deg=rate), VM)
    c_ref.update(0.0, True, _cp_sp(coop=True, inertia_comp=True),
                 _cs(steering_torque=2.0, steering_rate_deg=0.0), VM)
    rate += 5.0
  assert c_off.tau_inertia_last == 0.0
  assert abs(c_off.angle_override - c_ref.angle_override) < 1e-6


# --- field-tunable inertia J (TeslaCoopSteeringInertiaJ param) ---

def test_inertia_j_param_default_when_unset():
  # J param unset (0.0) -> the module default STEER_INERTIA_J is used.
  c = CoopSteeringCarController()
  c.update(0.0, True, _cp_sp(coop=True, inertia_comp=True, inertia_j=0.0),
           _cs(steering_torque=2.0, steering_rate_deg=10.0), VM)
  assert abs(c.inertia_j_used - STEER_INERTIA_J) < 1e-9


def test_inertia_j_param_overrides_default():
  # Larger J subtracts more inertial torque -> smaller override than a smaller J.
  def run(j):
    c = CoopSteeringCarController()
    rate = 0.0
    for _ in range(20):
      c.update(0.0, True, _cp_sp(coop=True, inertia_comp=True, inertia_j=j),
               _cs(steering_torque=2.0, steering_rate_deg=rate), VM)
      rate += 5.0
    return c.angle_override, c.inertia_j_used
  small_ovr, small_j = run(0.02)
  large_ovr, large_j = run(0.15)
  assert abs(small_j - 0.02) < 1e-9 and abs(large_j - 0.15) < 1e-9
  assert 0.0 < large_ovr < small_ovr


def test_inertia_j_param_clamped_to_safe_range():
  # Out-of-range param J is hard-clamped so a bad value can never blow up the FF.
  c_hi = CoopSteeringCarController()
  c_hi.update(0.0, True, _cp_sp(coop=True, inertia_comp=True, inertia_j=10.0),
              _cs(steering_torque=2.0, steering_rate_deg=10.0), VM)
  assert abs(c_hi.inertia_j_used - STEER_INERTIA_J_MAX) < 1e-9
  c_neg = CoopSteeringCarController()
  c_neg.update(0.0, True, _cp_sp(coop=True, inertia_comp=True, inertia_j=-1.0),
               _cs(steering_torque=2.0, steering_rate_deg=10.0), VM)
  assert c_neg.inertia_j_used == 0.0
