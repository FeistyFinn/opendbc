"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

from opendbc.car import Bus, structs
from opendbc.can import CANDefine
from opendbc.sunnypilot.car.tesla.carstate_ext import CarStateExt
from opendbc.sunnypilot.car.tesla.values import TeslaFlagsSP

ButtonType = structs.CarState.ButtonEvent.Type


def _ext(fingers: int = 5) -> CarStateExt:
  cp_sp = structs.CarParamsSP()
  flags = TeslaFlagsSP.HAS_VEHICLE_BUS.value
  if fingers == 5:
    flags |= TeslaFlagsSP.MADS_TOGGLE_FINGERS_5.value
  elif fingers == 4:
    flags |= TeslaFlagsSP.MADS_TOGGLE_FINGERS_4.value
  cp_sp.flags = flags
  return CarStateExt(structs.CarParams(), cp_sp)


def _feed(ext: CarStateExt, counts: list[int]) -> list[tuple[bool, object]]:
  """Feed a sequence of active-touch-point counts; return the flattened (pressed, type) events."""
  out = []
  for c in counts:
    out.extend((be.pressed, be.type) for be in ext.mads_gesture_button_events(c))
  return out


def _lkas_presses(events) -> int:
  return sum(1 for pressed, t in events if pressed and t == ButtonType.lkas)


def test_finger_count_from_flags():
  assert _ext(5).mads_toggle_fingers == 5
  assert _ext(4).mads_toggle_fingers == 4
  assert _ext(3).mads_toggle_fingers == 3  # no bits set -> legacy 3-finger


def test_clean_gesture_is_one_toggle():
  # fingers land 0->5 then lift 5->0: exactly one lkas toggle, no unknown events
  evs = _feed(_ext(5), [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
  assert _lkas_presses(evs) == 1
  assert all(t == ButtonType.lkas for _, t in evs)


def test_jitter_while_held_does_not_retoggle():
  # reach 5, then the count bounces 4<->5 while the user holds 5 fingers: still ONE toggle
  evs = _feed(_ext(5), [0, 5, 4, 5, 4, 5, 4, 5, 0])
  assert _lkas_presses(evs) == 1


def test_rearm_requires_return_to_zero():
  # never lifting all fingers (count stays >= 1) must not re-fire, even dipping below/above N
  evs = _feed(_ext(5), [0, 5, 6, 4, 6, 5, 1, 5])
  assert _lkas_presses(evs) == 1


def test_two_separate_gestures_two_toggles():
  # gesture, full release to 0, gesture again -> two toggles
  evs = _feed(_ext(5), [0, 5, 0, 5, 0])
  assert _lkas_presses(evs) == 2


def test_below_threshold_never_toggles():
  # 3-finger map zoom / 4-finger brush with N=5 must never engage
  evs = _feed(_ext(5), [0, 1, 2, 3, 2, 0, 3, 4, 3, 0])
  assert evs == []


def test_no_unknown_events_ever():
  # exhaustive ramp up and down emits only lkas events (never the old spurious unknown)
  evs = _feed(_ext(5), list(range(8)) + list(range(7, -1, -1)))
  assert ButtonType.unknown not in [t for _, t in evs]


def test_lower_threshold_still_single_toggle():
  # with N=4 a clean 4-finger gesture is one toggle and 3 fingers never fires
  evs = _feed(_ext(4), [0, 3, 0, 4, 5, 4, 0])
  assert _lkas_presses(evs) == 1


# --- coop-steering FF telemetry (CarStateSP.coopSteering) ---

def test_coop_steering_telemetry_populates_carstatesp():
  # carstate_ext logs the carcontroller's coop_steer internals to CarStateSP.coopSteering for
  # data-gathering; must be best-effort (never raise). Shadow-only: the FF is always computed +
  # logged while coop steering is on but never applied, so inertiaCompActive is always False and
  # shadowActive == coopActive.
  ext = _ext(5)
  ext.CP_SP.flags |= TeslaFlagsSP.COOP_STEERING.value

  # before the carcontroller stashes anything: no-op, no crash
  ret_sp = structs.CarStateSP()
  ext.update_coop_steering_sp(ret_sp)
  assert ret_sp.coopSteering.alphaFilt == 0.0

  # carcontroller stashes its coop_steer (faked)
  ext.coop_steering_debug = SimpleNamespace(alpha_filt_last=3.2, tau_inertia_last=0.25,
                                            tau_intent_last=1.75, inertia_j_used=0.08, angle_override=4.1,
                                            coop_apply_angle_sat_last=6.3)
  ret_sp = structs.CarStateSP()
  ext.update_coop_steering_sp(ret_sp)
  c = ret_sp.coopSteering
  # coop on -> FF logged in shadow; never applied live
  assert c.coopActive and c.shadowActive and not c.inertiaCompActive
  assert abs(c.alphaFilt - 3.2) < 1e-6 and abs(c.inertiaJUsed - 0.08) < 1e-6 and abs(c.angleOverride - 4.1) < 1e-6
  # delivered (post-saturation) angle -- distinct from the raw offset
  assert abs(c.blendedAngleDeg - 6.3) < 1e-6 and c.blendedAngleDeg != c.angleOverride

  # coop off -> not active, not shadow
  ext.CP_SP.flags &= ~TeslaFlagsSP.COOP_STEERING.value
  ret_sp = structs.CarStateSP()
  ext.update_coop_steering_sp(ret_sp)
  c = ret_sp.coopSteering
  assert not c.coopActive and not c.inertiaCompActive and not c.shadowActive
  assert abs(c.alphaFilt - 3.2) < 1e-6  # FF telemetry still logged regardless

  # a malformed debug object must never raise (telemetry is best-effort)
  ext.coop_steering_debug = SimpleNamespace()  # missing attrs
  ext.update_coop_steering_sp(structs.CarStateSP())


# --- scroll-wheel genericToggle + gas+scroll gap-adjust combo (drives the full update() path) ---

def _ext_no_vehicle_bus() -> CarStateExt:
  # CarStateExt without the vehicle bus, so update() skips the infotainment N-finger path and only the
  # party-bus scroll-wheel logic runs. Inject the real party CANDefine the way the mixed CarState does.
  cp_sp = structs.CarParamsSP()
  cp_sp.flags = 0
  ext = CarStateExt(structs.CarParams(), cp_sp)
  ext.can_define = CANDefine("tesla_model3_party")
  return ext


def _parsers(scroll: int):
  # Minimal fake CAN parsers with the .vl entries update() reads (speed-limit + scroll wheel).
  return {
    Bus.party: SimpleNamespace(vl={"DI_state": {"DI_speedUnits": 0},
                                   "UI_warning": {"scrollWheelPressed": scroll}}),
    Bus.ap_party: SimpleNamespace(vl={"DAS_status": {"DAS_fusedSpeedLimit": 0}}),
  }


def test_generic_toggle_tracks_scroll_wheel():
  ext = _ext_no_vehicle_bus()
  ret, ret_sp = structs.CarState(), structs.CarStateSP()
  ext.update(ret, ret_sp, _parsers(scroll=0))
  assert not ret.genericToggle
  ext.update(ret, ret_sp, _parsers(scroll=1))
  assert ret.genericToggle


def _gap_presses(ret) -> int:
  return sum(1 for be in ret.buttonEvents if be.pressed and be.type == ButtonType.gapAdjustCruise)


def test_gas_scroll_combo_emits_one_gap_adjust():
  # gas + scroll pressed while cruise is enabled -> exactly one gapAdjustCruise press on the rising
  # edge, and none while the combo is held (rising-edge only, no re-fire).
  ext = _ext_no_vehicle_bus()
  ret_sp = structs.CarStateSP()
  # each frame is a FRESH CarState (the base rebuilds it every cycle); only ext state persists.
  ret = structs.CarState()
  ret.gasPressed = True
  ret.cruiseState.enabled = True
  ext.update(ret, ret_sp, _parsers(scroll=1))
  assert _gap_presses(ret) == 1
  ret = structs.CarState()  # next frame, combo still held
  ret.gasPressed = True
  ret.cruiseState.enabled = True
  ext.update(ret, ret_sp, _parsers(scroll=1))
  assert _gap_presses(ret) == 0


def test_gas_scroll_combo_requires_cruise_and_gas():
  # No gap-adjust when cruise is disabled (or gas released), even with the scroll wheel pressed.
  ret, ret_sp = structs.CarState(), structs.CarStateSP()
  ret.gasPressed = True
  ret.cruiseState.enabled = False
  ext = _ext_no_vehicle_bus()
  ext.update(ret, ret_sp, _parsers(scroll=1))
  assert _gap_presses(ret) == 0

  ret2 = structs.CarState()
  ret2.gasPressed = False
  ret2.cruiseState.enabled = True
  ext2 = _ext_no_vehicle_bus()
  ext2.update(ret2, ret_sp, _parsers(scroll=1))
  assert _gap_presses(ret2) == 0


def test_update_preserves_base_button_events():
  # ext.update() runs AFTER base carstate.py has already populated ret.buttonEvents (e.g. the ACC
  # cancel emitted on DAS_accState->13). It must APPEND its scroll/gap events, not replace the list,
  # or the stock cancel/disengage ButtonEvent is dropped before it reaches controls.
  ext = _ext_no_vehicle_bus()
  ret, ret_sp = structs.CarState(), structs.CarStateSP()
  ret.buttonEvents = [structs.CarState.ButtonEvent(pressed=True, type=ButtonType.cancel)]
  ret.gasPressed = True
  ret.cruiseState.enabled = True
  ext.update(ret, ret_sp, _parsers(scroll=1))
  types = [be.type for be in ret.buttonEvents]
  assert ButtonType.cancel in types            # base ACC-cancel preserved (regression guard)
  assert ButtonType.gapAdjustCruise in types   # ext gap-adjust appended alongside
