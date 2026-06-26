"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntFlag


class TeslaFlagsSP(IntFlag):
  HAS_VEHICLE_BUS = 1  # Multi-finger infotainment press signal is present on the VEHICLE bus with the deprecated Tesla harness installed
  COOP_STEERING = 2  # Coop steering
  MADS_SCREEN_BUTTON_3_FINGER = 4
  MADS_SCREEN_BUTTON_4_FINGER = 8
  MADS_SCREEN_BUTTON_5_FINGER = 16
  # Bits 32/64 briefly held this fork's COOP_STEERING_INERTIA_COMP / _SHADOW. The inertia FF is now
  # SHADOW-ONLY -- always computed + logged off CP_SP.teslaCoopSteeringInertiaJ, never applied to
  # steering -- until a workable J ships, so no flag gates it and both bits are free again. When the
  # live FF returns for in-car alpha testing, allocate a FRESH bit and name the param/flag with "alpha"
  # in it (e.g. TeslaCoopSteeringInertiaCompAlpha / COOP_STEERING_INERTIA_COMP_ALPHA).
  # On bit 4: an earlier fork-local revision retired it (it held COOP_STEERING_INERTIA_COMP) and said
  # "do not reuse". That is superseded -- the port DELIBERATELY reclaims bit 4 for upstream's 3-finger
  # flag above, converging on upstream's 4/8/16 layout. Safe: TeslaFlagsSP is recomputed from params at
  # car init and never persisted, so the only cost is reading pre-2026-07-26 logs.


class MadsScreenButtonType:
  OFF = 0
  THREE_FINGER = 1
  FOUR_FINGER = 2
  FIVE_FINGER = 3


class TeslaSafetyFlagsSP:
  HAS_VEHICLE_BUS = 1
  # Threaded to the panda so its MADS touch-point grant uses the SAME count as carstate_ext (>= N),
  # not a hardcoded 3. No bit set == MadsScreenButtonType.OFF: the panda leaves the button UNAVAILABLE.
  MADS_SCREEN_BUTTON_3_FINGER = 2
  MADS_SCREEN_BUTTON_4_FINGER = 4
  MADS_SCREEN_BUTTON_5_FINGER = 8
