"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntFlag


class TeslaFlagsSP(IntFlag):
  HAS_VEHICLE_BUS = 1  # Infotainment multi-touch signal is present on the VEHICLE bus with the deprecated Tesla harness installed
  COOP_STEERING = 2  # Coop steering
  # MADS-toggle finger count: a 2-bit field at bit values 8/16 (bit value 4 is reserved).
  MADS_TOGGLE_FINGERS_4 = 8  # with _5: 00 -> 3 fingers (legacy), 01 -> 4, 10 -> 5
  MADS_TOGGLE_FINGERS_5 = 16


class TeslaSafetyFlagsSP:
  HAS_VEHICLE_BUS = 1
  # Two-bit MADS-toggle finger count (mirrors TeslaFlagsSP, with _5): 00 -> 3 (legacy), 01 -> 4, 10 -> 5.
  # Threaded to the panda so its MADS touch-point grant uses the SAME count as carstate_ext (>= N), not a hardcoded 3.
  MADS_TOGGLE_FINGERS_4 = 2
  MADS_TOGGLE_FINGERS_5 = 4
