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
  STEER_OVERRIDE_MIN_TORQUE,
)

# Tesla uses the Model Y vehicle model for lateral limiting (matches safety)
VM = VehicleModel(get_safety_CP())
STEER_ANGLE_MAX = CoopSteeringCarControllerParams.ANGLE_LIMITS.STEER_ANGLE_MAX


def _cs(steering_torque=0.0, v_ego=5.0, steering_angle=0.0):
  out = structs.CarState()
  out.vEgo = v_ego
  out.vEgoRaw = v_ego
  out.steeringAngleDeg = steering_angle
  out.steeringTorque = steering_torque
  return SimpleNamespace(out=out)


def _cp_sp(coop=True):
  cp_sp = structs.CarParamsSP()
  cp_sp.flags = TeslaFlagsSP.COOP_STEERING.value if coop else 0
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
