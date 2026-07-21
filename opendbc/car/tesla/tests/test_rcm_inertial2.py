"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

from opendbc.can import CANDefine, CANPacker, CANParser

DBC = "tesla_model3_party"
RCM_INERTIAL2_ADDR = 273


def test_rcm_inertial2_val_enums():
  # The RCM_inertial2 VAL_ tables parse into CANDefine: each accel axis has an SNA sentinel and a QF
  # fault flag. Guards the enum layout of the additive signal (there is no runtime consumer yet).
  dv = CANDefine(DBC).dv[RCM_INERTIAL2_ADDR]
  for axis in ("RCM_lateralAccel", "RCM_longitudinalAccel", "RCM_verticalAccel"):
    assert dv[axis] == {32768: "SNA"}
    assert dv[axis + "QF"] == {0: "FAULTED", 1: "NOT_FAULTED"}


def test_rcm_inertial2_round_trip():
  # Bit layout, endianness (@1- little-endian signed) and the 0.00125 m/s^2 scale round-trip: packing
  # each accel axis and parsing it back recovers the value within one LSB.
  packer = CANPacker(DBC)
  parser = CANParser(DBC, [("RCM_inertial2", 0)], 0)
  vals = {"RCM_lateralAccel": 1.0, "RCM_longitudinalAccel": -2.0, "RCM_verticalAccel": 9.81}
  msg = packer.make_can_msg("RCM_inertial2", 0, vals)
  parser.update([0, [msg]])
  vl = parser.vl["RCM_inertial2"]
  for k, v in vals.items():
    assert math.isclose(vl[k], v, abs_tol=0.00125), (k, vl[k], v)
