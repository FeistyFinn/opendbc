"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import StrEnum

from opendbc.car import Bus, create_button_events, structs
from opendbc.can.parser import CANParser
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.tesla.values import DBC, CANBUS
from opendbc.sunnypilot.car.tesla.values import TeslaFlagsSP

ButtonType = structs.CarState.ButtonEvent.Type


class CarStateExt:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP

    self.mads_gesture_armed = True
    self.coop_steering_debug = None  # carcontroller stashes its coop_steer here each frame for telemetry
    if CP_SP.flags & TeslaFlagsSP.MADS_TOGGLE_FINGERS_5:
      self.mads_toggle_fingers = 5
    elif CP_SP.flags & TeslaFlagsSP.MADS_TOGGLE_FINGERS_4:
      self.mads_toggle_fingers = 4
    else:
      self.mads_toggle_fingers = 3

    self.gas_combo_prev = False  # rising-edge tracking for the gas+scroll gap-adjust combo

  def mads_gesture_button_events(self, touch_points: int) -> list[structs.CarState.ButtonEvent]:
    """One lkas toggle per deliberate N-finger screen gesture, with hysteresis to reject touch-count
    jitter. Fires a single lkas press(+release) when the active touch-point count first reaches >= N
    (while armed), then disarms until all fingers lift (count returns to 0).

    The old create_button_events(raw_count, {N: lkas}) only matched the count == N exactly and emitted
    a spurious unknown ButtonEvent for every other count, so a real N-finger tap (whose count bounces
    e.g. 4<->5 while held) re-fired lkas on every bounce -> rapid MADS toggle -> the MADS/panda state
    diverged and controlsMismatchLateral fired (TAKE CONTROL + siren)."""
    if touch_points == 0:
      self.mads_gesture_armed = True
    elif touch_points >= self.mads_toggle_fingers and self.mads_gesture_armed:
      self.mads_gesture_armed = False
      return [structs.CarState.ButtonEvent(pressed=True, type=ButtonType.lkas),
              structs.CarState.ButtonEvent(pressed=False, type=ButtonType.lkas)]
    return []

  def update(self, ret: structs.CarState, ret_sp: structs.CarStateSP, can_parsers: dict[StrEnum, CANParser]) -> None:
    button_events = []
    if self.CP_SP.flags & TeslaFlagsSP.HAS_VEHICLE_BUS:
      cp_adas = can_parsers[Bus.adas]
      button_events += self.mads_gesture_button_events(int(cp_adas.vl["UI_status2"]["UI_activeTouchPoints"]))

    cp_party = can_parsers[Bus.party]
    cp_ap_party = can_parsers[Bus.ap_party]

    speed_units = self.can_define.dv["DI_state"]["DI_speedUnits"].get(int(cp_party.vl["DI_state"]["DI_speedUnits"]), None)
    speed_limit = cp_ap_party.vl["DAS_status"]["DAS_fusedSpeedLimit"]
    if self.can_define.dv["DAS_status"]["DAS_fusedSpeedLimit"].get(int(speed_limit), None) in ["NONE", "UNKNOWN_SNA"]:
      ret_sp.speedLimit = 0
    else:
      if speed_units == "KPH":
        ret_sp.speedLimit = speed_limit * CV.KPH_TO_MS
      elif speed_units == "MPH":
        ret_sp.speedLimit = speed_limit * CV.MPH_TO_MS

    ret.genericToggle = cp_party.vl["UI_warning"]["scrollWheelPressed"] != 0

    # gas + scroll-wheel press combo -> gap/personality adjust (implements the DAS "Gap adjust button").
    # Scroll is on the party bus, so this is independent of the deprecated vehicle-bus infotainment path.
    gas_combo = ret.gasPressed and ret.genericToggle and ret.cruiseState.enabled
    button_events += create_button_events(int(gas_combo), int(self.gas_combo_prev), {1: ButtonType.gapAdjustCruise})
    self.gas_combo_prev = gas_combo

    ret.buttonEvents = button_events

  def update_coop_steering_sp(self, ret_sp: structs.CarStateSP) -> None:
    """Log the cooperative-steering inertia-FF internals to CarStateSP for telemetry. The FF is
    shadow-only: always computed + logged while coop steering is active, never applied to steering.
    inertiaCompActive is therefore always False; shadowActive == coopActive. (Both fields are kept
    for log-schema stability; the live-apply path was removed pending a workable J.) Sourced from the
    carcontroller's coop_steer instance, which stashes itself on this CarState each frame (1-frame
    lag). Defensive: telemetry must never break carstate."""
    coop_steer = self.coop_steering_debug
    if coop_steer is None:
      return
    try:
      sp = ret_sp.coopSteering
      coop_active = bool(self.CP_SP.flags & TeslaFlagsSP.COOP_STEERING)
      sp.coopActive = coop_active
      sp.inertiaCompActive = False        # shadow-only: FF is never applied live
      sp.shadowActive = coop_active       # FF computed + logged whenever coop steering is on
      sp.alphaFilt = float(coop_steer.alpha_filt_last)
      sp.tauInertia = float(coop_steer.tau_inertia_last)
      sp.tauIntent = float(coop_steer.tau_intent_last)
      sp.inertiaJUsed = float(coop_steer.inertia_j_used)
      sp.angleOverride = float(coop_steer.angle_override)
      sp.blendedAngleDeg = float(coop_steer.coop_apply_angle_sat_last)  # delivered angle (post-saturation)
    except Exception:  # telemetry must never break carstate
      pass

  @staticmethod
  def get_parser(CP: structs.CarParams, CP_SP: structs.CarParamsSP) -> dict[StrEnum, CANParser]:
    messages = {}

    if CP_SP.flags & TeslaFlagsSP.HAS_VEHICLE_BUS:
      messages[Bus.adas] = CANParser(DBC[CP.carFingerprint][Bus.adas], [], CANBUS.vehicle)

    return messages
