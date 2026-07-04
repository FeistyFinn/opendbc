"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

from opendbc.car import structs
from opendbc.sunnypilot.car.tesla.carstate_ext import CarStateExt
from opendbc.sunnypilot.car.tesla.values import TeslaFlagsSP

ButtonType = structs.CarState.ButtonEvent.Type


def _ext(fingers: int = 5) -> CarStateExt:
  """fingers == 0 means MadsScreenButtonType.OFF: no finger flag is set at all."""
  cp_sp = structs.CarParamsSP()
  flags = TeslaFlagsSP.HAS_VEHICLE_BUS.value
  if fingers == 5:
    flags |= TeslaFlagsSP.MADS_SCREEN_BUTTON_5_FINGER.value
  elif fingers == 4:
    flags |= TeslaFlagsSP.MADS_SCREEN_BUTTON_4_FINGER.value
  elif fingers == 3:
    flags |= TeslaFlagsSP.MADS_SCREEN_BUTTON_3_FINGER.value
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
  assert _ext(5).mads_screen_button_fingers == 5
  assert _ext(4).mads_screen_button_fingers == 4
  assert _ext(3).mads_screen_button_fingers == 3
  assert _ext(0).mads_screen_button_fingers == 0  # no finger bit set -> screen button OFF


def test_off_emits_no_gesture_events():
  # MadsScreenButtonType.OFF sets no finger flag: the screen gesture is fully disabled, so no touch
  # count -- however high -- may emit an lkas ButtonEvent. Mirrors panda's `fingers != 0U` gate.
  assert _feed(_ext(0), [0, 3, 4, 5, 6, 5, 0, 5, 0]) == []


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
