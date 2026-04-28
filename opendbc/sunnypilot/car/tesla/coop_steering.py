"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import numpy as np
from collections import namedtuple
from dataclasses import replace

from opendbc.car import structs, rate_limit, DT_CTRL
from opendbc.car.common.filter_simple import FirstOrderFilter
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.lateral import apply_steer_angle_limits_vm
from opendbc.car.tesla.values import CarControllerParams
from opendbc.sunnypilot.car.tesla.values import TeslaFlagsSP


DT_LAT_CTRL = DT_CTRL * CarControllerParams.STEER_STEP

# limit steering acceleration when engaging
STEER_RESUME_RATE_LIMIT_RAMP_RATE = 300 # deg/s^2

class CoopSteeringCarControllerParams(CarControllerParams):
  ANGLE_LIMITS = replace(CarControllerParams.ANGLE_LIMITS, MAX_ANGLE_RATE=5)

# angle override
# Inertia compensation: tau_intent = tau_measured - J * alpha_wheel
# Tesla Model 3/Y steering wheel + column rotational inertia is not published;
# literature pegs comparable column assemblies in the 0.05-0.15 kg*m^2 range.
# Start at the conservative low end so v1 under-compensates rather than over-compensates.
# Sign convention from carstate.py: steeringTorque = -EPAS3S_torsionBarTorque and
# steeringRateDeg = -SCCM_steeringAngleSpeed, so they share the same sign convention
# (positive = wheel turning right / driver applying right torque), and the inertia term subtracts.
STEER_INERTIA_J = 0.08 # kg*m^2 (Nm per rad/s^2) - default; per-vehicle override via the TeslaCoopSteeringInertiaJ param
STEER_INERTIA_J_MAX = 0.15 # kg*m^2 - hard safety clamp on the field-tunable J (literature upper bound)
STEER_ALPHA_FILTER_RC = 0.04 # s
STEER_OVERRIDE_MIN_TORQUE = 0.5 # Nm - based on typical steering bias + noise - used for the deadzone
STEER_OVERRIDE_MAX_TORQUE = 2.5 # Nm - typical torque before EPS disengages due to hands_on_level=3
STEER_INERTIA_TORQUE_LIMIT = STEER_OVERRIDE_MAX_TORQUE # safety clamp on the FF term
STEER_OVERRIDE_TORQUE_RANGE = STEER_OVERRIDE_MAX_TORQUE - STEER_OVERRIDE_MIN_TORQUE

STEER_OVERRIDE_STANDSTILL_VEGO = 0.1 # m/s - below this speed the holding-torque estimate collapses (angle/torque_to_angle blows up)
STEER_OVERRIDE_MAX_LAT_ACCEL = 2.0 # m/s^2 - determines angle rate - speed dependent - similar to Tesla comfort steering mode
STEER_OVERRIDE_TARGET_ANGLE_MAX = CarControllerParams.ANGLE_LIMITS.STEER_ANGLE_MAX  # deg

# override angle ramp control.
# 125 == MAX_ANGLE_RATE / DT_LAT_CTRL / STEER_OVERRIDE_TORQUE_RANGE (5 / 0.02 / 2.0), i.e. exactly the
# internal per-Nm ceiling that calc_override_angle_delta_limit enforces, so this gain sits at that cap.
STEER_OVERRIDE_DELTA_GAIN_LIMIT = 125 # deg/s/Nm


CoopSteeringDataSP = namedtuple("CoopSteeringDataSP",
                                ["steeringAngleDeg", "lat_active"])

def get_steer_from_lat_accel(lat_accel, vEgo: float, VM: VehicleModel):
  """Calculate the maximum steering angle based on lateral acceleration."""
  curvature = lat_accel / (max(1, vEgo) ** 2)  # 1/m
  return math.degrees(VM.get_steer_from_curvature(curvature, vEgo, 0))  # deg


def apply_bounds(signal: float, limit: float) -> float:
  """Limit input to a range."""
  return float(np.clip(signal, -limit, limit))


def apply_deadzone(signal: float, deadzone: float) -> float:
  """Apply deadzone to input."""
  return signal - apply_bounds(signal, deadzone)


def get_override_torque_to_angle(vEgo: float, VM: VehicleModel, lat_accel: float) -> float:
  """
  Convert effective override torque to steering angle gain.
  """

  # lateral accel is linear in respect to angle so it's fine to interpolate it with torque
  steer_from_lat_accel = apply_bounds(get_steer_from_lat_accel(lat_accel, vEgo, VM), STEER_OVERRIDE_TARGET_ANGLE_MAX)
  return steer_from_lat_accel / STEER_OVERRIDE_TORQUE_RANGE


def calc_override_angle_delta_limit(torque: float, gain_limit: float) -> float:
  """
  Convert torque magnitude to a per-step steering angle delta limit.
  """
  delta_gain_limit_max = CoopSteeringCarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE / DT_LAT_CTRL / STEER_OVERRIDE_TORQUE_RANGE
  return torque * min(gain_limit, delta_gain_limit_max) * DT_LAT_CTRL


class SteerRateLimiter:
  """Handles rate limiting of steering angle changes with a configurable rate."""
  def __init__(self):
    self._last = 0.0

  def reset(self, angle: float) -> None:
    """Reset the rate limiter state with the given angle."""
    self._last = angle

  def update(self, angle: float, angle_delta_lim: float) -> float:
    angle_lim = rate_limit(angle, self._last, -angle_delta_lim, angle_delta_lim)
    self._last = angle_lim
    return angle_lim


class CoopSteeringCarController:
  def __init__(self):
    self.apply_angle_last = 0
    self.coop_apply_angle_sat_last = 0
    self.angle_override = 0
    self.resume_rate_limiter_delta = SteerRateLimiter()
    self.resume_rate_limiter = SteerRateLimiter()
    self.prev_steering_rate_deg = 0.0
    self.alpha_filter = FirstOrderFilter(0.0, STEER_ALPHA_FILTER_RC, DT_LAT_CTRL, initialized=False)
    self.tau_inertia_last = 0.0
    self.tau_intent_last = 0.0
    self.alpha_filt_last = 0.0
    self.inertia_j_used = 0.0

  def reset_override_state(self, apply_angle: float, current_steering_rate_deg: float = 0.0) -> None:
    self.apply_angle_last = apply_angle
    self.angle_override = 0
    self.coop_apply_angle_sat_last = apply_angle
    self.prev_steering_rate_deg = current_steering_rate_deg
    self.alpha_filter.x = 0.0
    self.alpha_filter.initialized = False
    self.tau_inertia_last = 0.0
    self.tau_intent_last = 0.0
    self.alpha_filt_last = 0.0
    self.inertia_j_used = 0.0

  def update_shadow_inertia_ff(self, driver_torque: float, steering_rate_deg: float, inertia_j: float) -> None:
    """
    Compute + log the inertia feed-forward (FF) term J * alpha_wheel. SHADOW-ONLY: always computed and
    logged while coop steering is active (so wheel-acceleration ghost-torque is measured on every drive),
    but NEVER applied to steering -- the override always runs off the raw measured driver torque. The
    live-apply path was removed pending a workable J; the offline fit
    (tools/sunnypilot/vtb/fit_steer_inertia.py) consumes this logged telemetry.
    """
    alpha_raw_deg_per_s2 = (steering_rate_deg - self.prev_steering_rate_deg) / DT_LAT_CTRL
    alpha_filt_rad_per_s2 = math.radians(self.alpha_filter.update(alpha_raw_deg_per_s2))
    tau_inertia = apply_bounds(inertia_j * alpha_filt_rad_per_s2, STEER_INERTIA_TORQUE_LIMIT)
    # Only count the FF when the driver is actually engaging the wheel. Without this guard,
    # openpilot-driven wheel rotation (no driver torque, but nonzero alpha) would manufacture
    # a phantom negative intent torque outside the deadzone and grow a spurious override.
    if abs(driver_torque) <= STEER_OVERRIDE_MIN_TORQUE:
      tau_inertia = 0.0
    self.alpha_filt_last = alpha_filt_rad_per_s2
    self.inertia_j_used = inertia_j
    self.prev_steering_rate_deg = steering_rate_deg
    self.tau_inertia_last = tau_inertia
    self.tau_intent_last = driver_torque - tau_inertia  # logged for analysis (the would-be live intent)

  def compute_override_targets(self, vEgo: float, steering_torque: float, VM: VehicleModel) -> tuple[float, float]:
    """Returns (angle_override_target, override_torque): the driver's target angle and net torque above neutral."""
    torque_to_angle = get_override_torque_to_angle(vEgo, VM, STEER_OVERRIDE_MAX_LAT_ACCEL)
    driver_torque_with_deadzone = apply_deadzone(steering_torque, STEER_OVERRIDE_MIN_TORQUE)
    neutral_torque = 0.0 if abs(vEgo) <= STEER_OVERRIDE_STANDSTILL_VEGO else self.angle_override / torque_to_angle
    return driver_torque_with_deadzone * torque_to_angle, driver_torque_with_deadzone - neutral_torque

  def override_slew_step(self, angle_override_target: float, override_torque: float) -> float:
    """Per-frame slew toward the driver torque target; the slew rate scales with the excess torque.
    Symmetric: deflect and center are bounded equally (the away/center gains collapse to one)."""
    target_error = angle_override_target - self.angle_override
    slew_rate = calc_override_angle_delta_limit(abs(override_torque), STEER_OVERRIDE_DELTA_GAIN_LIMIT)
    return float(np.clip(target_error, -slew_rate, slew_rate))

  @staticmethod
  def adjust_slew_for_planner(slew_step: float, apply_angle_step: float, override_torque: float) -> float:
    """
    Same-direction: subtract the planner overlap to avoid double-counting.
    Opposing: grow the slew opposite to the planner, scaled by driver effort.
    """
    direction = slew_step * apply_angle_step
    if direction > 0:
      return slew_step - apply_bounds(apply_angle_step, abs(slew_step))
    if direction < 0:
      driver_effort = abs(override_torque) / STEER_OVERRIDE_TORQUE_RANGE
      return slew_step - driver_effort * apply_angle_step
    return slew_step

  @staticmethod
  def unwind_on_saturation(angle_override: float, sat_error: float) -> float:
    """Anti-windup: if the override drove past the angle limit, pull it back by the overshoot."""
    if angle_override * sat_error <= 0:
      return angle_override
    return angle_override - apply_bounds(sat_error, abs(angle_override))

  def resume_steer_desired_rate_limit(self, lat_active: bool, apply_angle: float) -> float:
    """Limits steering wheel acceleration when resuming steering"""
    if not lat_active:
      # reset and bypass
      self.resume_rate_limiter_delta.reset(0)
      self.resume_rate_limiter.reset(apply_angle)
      return apply_angle

    angle_rate_delta_lim = self.resume_rate_limiter_delta.update(CarControllerParams.ANGLE_LIMITS.MAX_ANGLE_RATE,
                                                         STEER_RESUME_RATE_LIMIT_RAMP_RATE * DT_LAT_CTRL**2)
    apply_angle_lim = self.resume_rate_limiter.update(apply_angle, angle_rate_delta_lim)
    return apply_angle_lim

  def update(self, apply_angle, lat_active, CP_SP: structs.CarParamsSP, CS: structs.CarState, VM: VehicleModel) -> CoopSteeringDataSP:
    angle_coop_enabled = CP_SP.flags & TeslaFlagsSP.COOP_STEERING.value
    # per-vehicle tunable J (param), hard-clamped so a bad value can never blow up the FF; 0 -> default.
    # Shadow-only: J scales the logged FF only; it does not affect the applied steering angle.
    inertia_j = float(np.clip(CP_SP.teslaCoopSteeringInertiaJ or STEER_INERTIA_J, 0.0, STEER_INERTIA_J_MAX))

    # avoid sudden rotation on engagement
    apply_angle = self.resume_steer_desired_rate_limit(lat_active, apply_angle)

    if not lat_active or not angle_coop_enabled:
      self.reset_override_state(apply_angle, CS.out.steeringRateDeg)
      return CoopSteeringDataSP(apply_angle, lat_active)

    apply_angle_step = apply_angle - self.apply_angle_last
    self.apply_angle_last = apply_angle

    # Shadow-only inertia FF: computed + logged every active frame, never applied to the steering angle.
    self.update_shadow_inertia_ff(CS.out.steeringTorque, CS.out.steeringRateDeg, inertia_j)

    angle_override_target, override_torque = self.compute_override_targets(CS.out.vEgo, CS.out.steeringTorque, VM)
    slew_step = self.override_slew_step(angle_override_target, override_torque)
    slew_step = self.adjust_slew_for_planner(slew_step, apply_angle_step, override_torque)
    self.angle_override += slew_step
    apply_angle += self.angle_override

    # final rate limit - matching panda safety
    self.coop_apply_angle_sat_last = apply_steer_angle_limits_vm(apply_angle, self.coop_apply_angle_sat_last, CS.out.vEgoRaw,
                                                    CS.out.steeringAngleDeg, lat_active, CoopSteeringCarControllerParams, VM)
    sat_error = apply_angle - self.coop_apply_angle_sat_last
    self.angle_override = self.unwind_on_saturation(self.angle_override, sat_error)

    return CoopSteeringDataSP(self.coop_apply_angle_sat_last, lat_active)
