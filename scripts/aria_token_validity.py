"""Validity of generated Aria tokens after a prompt (docs/experiments/WJAZZD_PILOT.md).

Replaces aria_cond_loss_arms.grammar_check for diagnosis. That checker took the largest onset of
the whole prompt as the starting point even when a <T> had reset the segment, and counted the
tokenizer's <D> ("diminish", placed between notes near the end of long training sequences) as an
error. Here:
- between notes: a piano note start, <T> (segment reset), <D> or <E> (stop) are valid
- a note is (piano, p, v) -> (onset, x) -> (dur, d); onsets may repeat but not go back inside a segment
- the first internal error is classified: token_order, onset_reversal, non_piano_instrument,
  special_token; later errors are counted only as a cascade (resync at the next piano note start)
- generation cut by the token cap in the middle of a note is an incomplete tail, not an error
Raw tokens are never changed.
"""
from __future__ import annotations

NOTE_START = "piano"


def _kind(t):
    return t[0] if isinstance(t, tuple) else t


def segment_state(prefix):
    """Last onset inside the current 5 s segment at the end of the prompt, and the expected token."""
    last, expect = 0, "note"
    for t in prefix:
        k = _kind(t)
        if k == "<T>":
            last = 0
        elif k == "onset":
            last = t[1]
            expect = "dur"
        elif k == NOTE_START:
            expect = "onset"
        elif k == "dur":
            expect = "note"
    return last, expect


def check(prefix, new) -> dict:
    last, expect = segment_state(prefix)
    first, cascade, notes_before_first, notes = None, 0, 0, 0
    ended_by_eos = False
    for i, t in enumerate(new):
        k = _kind(t)
        err = None
        if expect == "note":
            if k == "<E>":
                ended_by_eos = True
                break
            if k == "<T>":
                last = 0
            elif k == "<D>":
                pass
            elif k == NOTE_START:
                expect = "onset"
            elif isinstance(t, tuple) and k not in ("onset", "dur", "prefix"):
                err = "non_piano_instrument"
            elif isinstance(t, tuple):
                err = "token_order"
            else:
                err = "special_token"
        elif expect == "onset":
            if k != "onset":
                err = "token_order"
            elif t[1] < last:
                err = "onset_reversal"
            else:
                last = t[1]
                expect = "dur"
        else:
            if k != "dur":
                err = "token_order"
            else:
                expect = "note"
                notes += 1
        if err:
            if first is None:
                first = {"index": i, "type": err, "token": str(t)}
                notes_before_first = notes
            else:
                cascade += 1
            expect = "onset" if k == NOTE_START else "note"     # resync
    tail = None if ended_by_eos or expect == "note" else "incomplete_tail"
    return {"valid": first is None, "first_error": first, "cascade_errors": cascade,
            "complete_notes": notes, "notes_before_first_error": notes_before_first if first else notes,
            "ended_by_eos": ended_by_eos, "tail": tail}
