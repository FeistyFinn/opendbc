"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntFlag


class TeslaFlagsSP(IntFlag):
  HAS_VEHICLE_BUS = 1  # Infotainment multi-touch signal is present on the VEHICLE bus with the deprecated Tesla harness installed
  COOP_STEERING = 2  # Coop steering (master)
  # bit 4 retired (was COOP_STEERING_INERTIA_COMP): the inertia FF is now SHADOW-ONLY -- always computed +
  # logged, never applied to steering -- until a workable J ships. Do not reuse bit 4. When the live FF
  # returns for in-car alpha testing, allocate a FRESH bit and name the param/flag with "alpha" in it
  # (e.g. TeslaCoopSteeringInertiaCompAlpha / COOP_STEERING_INERTIA_COMP_ALPHA) to mark it as alpha.
  MADS_TOGGLE_FINGERS_4 = 8  # Two-bit encoding of MADS-toggle finger count (with _5):
  MADS_TOGGLE_FINGERS_5 = 16  #   00 -> 3 fingers (legacy), 01 -> 4, 10 -> 5
  # bit 32 retired: an earlier inertia-mode bit; do not reuse.
  # FRESH bit for the standstill inertia-J calibration dither (the "live FF returns for in-car alpha"
  # case the bit-4 retirement comment anticipated): active excitation, ALPHA, gated to COOP_STEERING.
  COOP_STEERING_DITHER_CALIB_ALPHA = 64


class TeslaSafetyFlagsSP:
  HAS_VEHICLE_BUS = 1
  # Two-bit MADS-toggle finger count (mirrors TeslaFlagsSP, with _5): 00 -> 3 (legacy), 01 -> 4, 10 -> 5.
  # Threaded to the panda so its MADS touch-point grant uses the SAME count as carstate_ext (>= N), not a hardcoded 3.
  MADS_TOGGLE_FINGERS_4 = 2
  MADS_TOGGLE_FINGERS_5 = 4
