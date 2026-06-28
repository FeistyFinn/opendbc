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
  DitherCalibrator,
  apply_bounds,
  apply_deadzone,
  get_steer_from_lat_accel,
  STEER_INERTIA_J,
  STEER_INERTIA_J_MAX,
  STEER_INERTIA_TORQUE_LIMIT,
  STEER_OVERRIDE_MIN_TORQUE,
  DITHER_AMP_DEG,
  DITHER_RAMP_FRAMES,
  DITHER_ABORT_FRAMES,
  DITHER_MAX_FRAMES,
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


def _cp_sp(coop=True, inertia_j=0.0, dither=False):
  # The inertia FF is shadow-only: it is always computed + logged while coop steering is active but
  # never applied to steering (the live-apply toggle was removed). inertia_j scales the logged FF.
  # dither arms the standstill inertia-J calibration excitation (ALPHA).
  cp_sp = structs.CarParamsSP()
  flags = 0
  if coop:
    flags |= TeslaFlagsSP.COOP_STEERING.value
  if dither:
    flags |= TeslaFlagsSP.COOP_STEERING_DITHER_CALIB_ALPHA.value
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


# --- inertia compensation (SHADOW-ONLY: FF is always computed + logged, never applied) ---

def test_inertia_no_false_override_from_rotation_alone():
  # Wheel is rotating (large alpha from a rate ramp) but driver is not engaging:
  # the compensation must not manufacture a phantom intent torque that grows the override.
  # Guard: with |driver_torque| <= deadzone, tau_inertia is forced to 0.
  c = CoopSteeringCarController()
  cp_sp = _cp_sp(coop=True)
  rate = 0.0
  for _ in range(50):
    c.update(0.0, True, cp_sp, _cs(steering_torque=0.0, steering_rate_deg=rate), VM)
    rate += 2.0  # 100 deg/s^2 constant acceleration -> non-trivial raw alpha
  assert abs(c.tau_inertia_last) < 1e-9
  assert abs(c.angle_override) < 1e-9


def test_inertia_noop_at_steady_state():
  # Above-deadzone driver torque with zero wheel rotation: alpha settles to 0 so tau_inertia -> 0,
  # and the override is the plain torque->angle baseline.
  c = CoopSteeringCarController()
  _settle(c, _cp_sp(coop=True), _cs(steering_torque=2.0))
  assert abs(c.tau_inertia_last) < 1e-9
  assert c.angle_override > 0.0


def test_ff_logged_but_not_applied():
  # Shadow-only: with the wheel accelerating under above-deadzone driver torque, the FF is computed
  # and logged (tau_inertia / alpha nonzero), but it does NOT change the applied override -- the
  # override matches an otherwise-identical run with no wheel rotation (no FF influence on steering).
  c_rot = CoopSteeringCarController()
  c_still = CoopSteeringCarController()
  rate = 0.0
  for _ in range(20):
    c_rot.update(0.0, True, _cp_sp(coop=True), _cs(steering_torque=2.0, steering_rate_deg=rate), VM)
    c_still.update(0.0, True, _cp_sp(coop=True), _cs(steering_torque=2.0, steering_rate_deg=0.0), VM)
    rate += 5.0  # +250 deg/s^2 (~4.4 rad/s^2) sustained alpha so the FF is non-trivial
  assert abs(c_rot.tau_inertia_last) > 0.0   # FF computed + logged
  assert abs(c_rot.alpha_filt_last) > 0.0
  assert c_rot.angle_override > 0.0
  assert abs(c_rot.angle_override - c_still.angle_override) < 1e-6  # FF not applied to steering


def test_inertia_engagement_no_spike():
  # Pre-engage the wheel is already rotating fast. After engaging, the differentiator must not
  # see a spurious (current_rate - 0) / dt spike on the first frame -- reset_override_state seeds
  # prev_steering_rate_deg with the current rate and re-arms the filter to snap on first sample.
  c = CoopSteeringCarController()
  cp_sp = _cp_sp(coop=True)
  c.update(0.0, False, cp_sp, _cs(steering_torque=2.0, steering_rate_deg=20.0), VM)
  c.update(0.0, True, cp_sp, _cs(steering_torque=2.0, steering_rate_deg=20.0), VM)
  # No differentiator spike: tau_inertia is well below the clamp on the engagement frame.
  assert abs(c.tau_inertia_last) < 0.1


def test_inertia_clamp_protects_against_runaway_alpha():
  # Extreme alpha (stale-then-jump pattern) must clamp the logged tau_inertia at
  # STEER_INERTIA_TORQUE_LIMIT so a sensor glitch can't blow up the FF telemetry.
  c = CoopSteeringCarController()
  cp_sp = _cp_sp(coop=True)
  # Drive the filter into a steady state first so it is initialized
  for _ in range(5):
    c.update(0.0, True, cp_sp, _cs(steering_torque=2.0, steering_rate_deg=0.0), VM)
  # Jump the rate: huge alpha
  c.update(0.0, True, cp_sp, _cs(steering_torque=2.0, steering_rate_deg=500.0), VM)
  assert abs(c.tau_inertia_last) <= STEER_INERTIA_TORQUE_LIMIT + 1e-9


# --- field-tunable inertia J (TeslaCoopSteeringInertiaJ param) -- scales the logged FF only ---

def test_inertia_j_param_default_when_unset():
  # J param unset (0.0) -> the module default STEER_INERTIA_J is used.
  c = CoopSteeringCarController()
  c.update(0.0, True, _cp_sp(coop=True, inertia_j=0.0),
           _cs(steering_torque=2.0, steering_rate_deg=10.0), VM)
  assert abs(c.inertia_j_used - STEER_INERTIA_J) < 1e-9


def test_j_param_scales_logged_ff_only():
  # Larger J -> larger logged inertial torque (|tau_inertia|) and matching inertia_j_used, but
  # shadow-only means the applied override is identical regardless of J.
  def run(j):
    c = CoopSteeringCarController()
    rate = 0.0
    for _ in range(20):
      c.update(0.0, True, _cp_sp(coop=True, inertia_j=j),
               _cs(steering_torque=2.0, steering_rate_deg=rate), VM)
      rate += 5.0
    return c.angle_override, c.inertia_j_used, abs(c.tau_inertia_last)
  small_ovr, small_j, small_ff = run(0.02)
  large_ovr, large_j, large_ff = run(0.15)
  assert abs(small_j - 0.02) < 1e-9 and abs(large_j - 0.15) < 1e-9
  assert large_ff > small_ff > 0.0          # larger J -> larger logged FF
  assert abs(small_ovr - large_ovr) < 1e-6  # but the applied override is unaffected (shadow-only)
  assert small_ovr > 0.0


def test_inertia_j_param_clamped_to_safe_range():
  # Out-of-range param J is hard-clamped so a bad value can never blow up the FF.
  c_hi = CoopSteeringCarController()
  c_hi.update(0.0, True, _cp_sp(coop=True, inertia_j=10.0),
              _cs(steering_torque=2.0, steering_rate_deg=10.0), VM)
  assert abs(c_hi.inertia_j_used - STEER_INERTIA_J_MAX) < 1e-9
  c_neg = CoopSteeringCarController()
  c_neg.update(0.0, True, _cp_sp(coop=True, inertia_j=-1.0),
               _cs(steering_torque=2.0, steering_rate_deg=10.0), VM)
  assert c_neg.inertia_j_used == 0.0


# --- standstill inertia-J calibration dither (DitherCalibrator, ALPHA) ---

def _drive_dither(cal, n, armed=True, lat_active=True, standstill=True, hands_off=True):
  acts, cmds = [], []
  for _ in range(n):
    a, c = cal.update(armed, lat_active, standstill, hands_off)
    acts.append(a)
    cmds.append(c)
  return acts, cmds


def test_dither_inert_unless_all_preconditions():
  # off by default and fully inert if ANY single precondition is missing
  for kw in ({"armed": False}, {"lat_active": False}, {"standstill": False}, {"hands_off": False}):
    cal = DitherCalibrator()
    acts, cmds = _drive_dither(cal, 100, **kw)
    assert not any(acts) and all(c == 0.0 for c in cmds)


def test_dither_bounded_amplitude():
  # |command| never exceeds the configured sum-of-tones peak, across a full-length run
  cal = DitherCalibrator()
  _, cmds = _drive_dither(cal, DITHER_MAX_FRAMES)
  assert max(abs(c) for c in cmds) <= DITHER_AMP_DEG + 1e-9
  assert max(abs(c) for c in cmds) > 0.05   # actually excites (well above noise, below the bound)


def test_dither_ramps_in_no_step():
  # the envelope ramps linearly from 0; the first commanded frame is ~0, not a full-amplitude jump
  cal = DitherCalibrator()
  acts, cmds = _drive_dither(cal, DITHER_RAMP_FRAMES + 5)
  assert acts[0] and abs(cmds[0]) < 1e-6
  for i, c in enumerate(cmds[:DITHER_RAMP_FRAMES]):
    assert abs(c) <= (i + 1) / DITHER_RAMP_FRAMES * DITHER_AMP_DEG + 1e-9  # under the growing envelope
  assert abs(cal.env - 1.0) < 1e-9            # fully ramped in after DITHER_RAMP_FRAMES


def test_dither_aborts_fast_on_lost_precondition():
  # run to full envelope, then lose a precondition -> envelope collapses within the abort window
  cal = DitherCalibrator()
  _drive_dither(cal, DITHER_RAMP_FRAMES + 10)
  assert cal.active and cal.env > 0.99
  _drive_dither(cal, DITHER_ABORT_FRAMES + 1, hands_off=False)
  assert not cal.active and cal.env == 0.0
  assert not cal.finished                     # an aborted (cut-short) run does NOT latch -> can retry


def test_dither_one_run_per_arming_latches_then_rearms():
  # a run that reaches the duration cap latches and will not re-fire while still armed; dropping
  # armed (the offroad->onroad re-key) clears the latch and lets it run once more
  cal = DitherCalibrator()
  _drive_dither(cal, DITHER_MAX_FRAMES + DITHER_RAMP_FRAMES + 5)
  assert cal.finished and not cal.active
  acts, cmds = _drive_dither(cal, 50)
  assert not any(acts) and all(c == 0.0 for c in cmds)   # still armed -> stays inert (one run)
  cal.update(False, True, True, True)                    # disarm one frame
  assert not cal.finished
  acts2, _ = _drive_dither(cal, DITHER_RAMP_FRAMES + 5)
  assert any(acts2)                                       # re-armed -> runs again


# --- dither injection through the carcontroller (safety invariant: clamped by the panda limiter) ---

MAX_ANGLE_RATE = CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE


def test_dither_not_injected_unless_armed():
  # coop on but the dither flag is OFF: a standstill hands-off output is the plain baseline (no dither)
  c = CoopSteeringCarController()
  cp_sp, cs = _cp_sp(coop=True, dither=False), _cs(steering_torque=0.0, v_ego=0.0)
  outs = [c.update(0.0, True, cp_sp, cs, VM).steeringAngleDeg for _ in range(100)]
  assert all(abs(o) < 1e-6 for o in outs)
  assert not c.dither_active_last


def test_dither_injected_when_armed_and_clamped():
  # coop + dither armed, standstill + hands-off: a bounded oscillation is injected, and every frame
  # stays within the configured amplitude AND the panda per-frame angle-rate limit (clamp invariant)
  c = CoopSteeringCarController()
  cp_sp, cs = _cp_sp(coop=True, dither=True), _cs(steering_torque=0.0, v_ego=0.0)
  outs = [c.update(0.0, True, cp_sp, cs, VM).steeringAngleDeg for _ in range(120)]
  assert max(abs(o) for o in outs) > 0.05                      # the dither actually moved the angle
  assert max(abs(o) for o in outs) <= DITHER_AMP_DEG + 1e-6    # within the configured amplitude
  deltas = [abs(b - a) for a, b in zip(outs[:-1], outs[1:], strict=True)]
  assert max(deltas) <= MAX_ANGLE_RATE + 1e-6                  # within the panda per-frame angle-rate limit


def test_dither_inert_when_moving():
  # armed but above the standstill ceiling: no dither (a wiggle at speed would deviate the path)
  c = CoopSteeringCarController()
  cp_sp, cs = _cp_sp(coop=True, dither=True), _cs(steering_torque=0.0, v_ego=5.0)
  for _ in range(60):
    c.update(0.0, True, cp_sp, cs, VM)
  assert not c.dither_active_last


def test_dither_aborts_on_driver_touch():
  # injecting at standstill, then the driver grabs the wheel -> the dither aborts within the window
  c = CoopSteeringCarController()
  cp_sp = _cp_sp(coop=True, dither=True)
  for _ in range(DITHER_RAMP_FRAMES + 10):
    c.update(0.0, True, cp_sp, _cs(steering_torque=0.0, v_ego=0.0), VM)
  assert c.dither_active_last
  for _ in range(DITHER_ABORT_FRAMES + 2):
    c.update(0.0, True, cp_sp, _cs(steering_torque=3.0, v_ego=0.0), VM)  # hands ON
  assert not c.dither_active_last
