"""#1451 회귀 테스트: _merge_note 견고화 — 생성 시퀀스의 조용한 노트 손실 방지.

원본 _merge_note는 세 경로에서 노트를 소리 없이 버렸다:
  1) orphan note_off (활성 note_on 없음)  — 정상 (버릴 게 맞음)
  2) 같은 pitch 재타건 (off 없이 note_on 재발생) — 이전 노트 손실
  3) 시퀀스 끝까지 안 닫힌 note_on         — 노트 손실
정규(인코더 출력) 데이터엔 2/3이 없으므로 출력은 불변이어야 하고,
문법이 불완전한 생성 데이터에선 2/3 노트가 보존돼야 한다.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "music_transformer", "third_party"))
from midi_processor import processor as P


def _events(pairs):
    return [P.Event(event_type=t, value=v) for t, v in pairs]


def test_rearticulation_preserved():
    # C4 on, shift, C4 on(재타건), shift, off, shift, off
    ev = _events([('note_on', 60), ('time_shift', 49), ('note_on', 60),
                  ('time_shift', 49), ('note_off', 60),
                  ('time_shift', 49), ('note_off', 60)])
    sn = P._event_seq2snote_seq(ev)
    notes = P._merge_note(sn)
    # 재타건이므로 노트 2개가 나와야 한다 (원본은 1개만)
    assert len(notes) == 2, f"expected 2 notes, got {len(notes)}"


def test_unclosed_note_emitted():
    # note_on 뒤 note_off가 영영 안 옴 -> 폴백 duration으로 살아야 함
    ev = _events([('note_on', 72), ('time_shift', 49)])
    sn = P._event_seq2snote_seq(ev)
    notes = P._merge_note(sn, unclosed_note_dur=0.5)
    assert len(notes) == 1, f"expected 1 fallback note, got {len(notes)}"
    assert abs(notes[0].end - notes[0].start - 0.5) < 1e-6


def test_orphan_note_off_discarded():
    # 활성 note_on 없는 note_off -> 버려야 정상 (노트 생성 X)
    ev = _events([('note_off', 65), ('time_shift', 49),
                  ('note_on', 65), ('time_shift', 49), ('note_off', 65)])
    sn = P._event_seq2snote_seq(ev)
    notes = P._merge_note(sn)
    assert len(notes) == 1


def test_legacy_dur_zero_matches_old_behavior():
    # unclosed_note_dur=0 이면 미닫힘 note_on은 옛날처럼 버려짐 (재현 호환)
    ev = _events([('note_on', 72), ('time_shift', 49)])
    sn = P._event_seq2snote_seq(ev)
    assert len(P._merge_note(sn, unclosed_note_dur=0)) == 0


if __name__ == "__main__":
    test_rearticulation_preserved()
    test_unclosed_note_emitted()
    test_orphan_note_off_discarded()
    test_legacy_dur_zero_matches_old_behavior()
    print("all decode-note-loss tests passed")
