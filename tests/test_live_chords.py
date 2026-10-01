"""Live chord input: recognition, tracking and runtime flags (docs/experiments/LIVE_CHORDS.md)."""
from __future__ import annotations

import tempfile
import unittest

from mido import Message

from inference.app.fallback import parse_chord
from inference.control.live_chords import LiveChordTracker, held_notes, recognize
from inference.realtime.continuous import TimedInputMessage
from scripts.run_continuous_jazz import main
from scripts.run_live_chord_probe import PLAN, follow_metrics, recognition_matches, voicing


def on(t, p):
    return TimedInputMessage(received_ns=t, message=Message("note_on", note=p, velocity=80))


def off(t, p):
    return TimedInputMessage(received_ns=t, message=Message("note_off", note=p, velocity=0))


def chord(t, *pitches):
    return [on(t + i, p) for i, p in enumerate(pitches)]


class RecognizeTest(unittest.TestCase):
    def test_four_note_chords_in_root_position_and_inversions(self) -> None:
        self.assertEqual(recognize([50, 53, 57, 60]), "Dm7")
        self.assertEqual(recognize([43, 47, 50, 53]), "G7")
        self.assertEqual(recognize([48, 52, 55, 59]), "Cmaj7")
        self.assertEqual(recognize([52, 55, 59, 60]), "Cmaj7")          # first inversion
        self.assertEqual(recognize([47, 50, 53, 57]), "Bm7b5")
        self.assertEqual(recognize([47, 50, 53, 56]), "Bdim")
        self.assertEqual(recognize([42, 46, 49, 53]), "Gbmaj7")

    def test_triads_and_shells_map_to_the_guide_vocabulary(self) -> None:
        self.assertEqual(recognize([48, 52, 55]), "Cmaj7")
        self.assertEqual(recognize([45, 48, 52]), "Am7")
        self.assertEqual(recognize([43, 47, 53]), "G7")                 # root, 3rd, 7th
        self.assertEqual(recognize([47, 50, 53]), "Bm7b5")

    def test_names_round_trip_through_parse_chord(self) -> None:
        for pitches in ([50, 53, 57, 60], [43, 47, 50, 53], [48, 52, 55, 59], [47, 50, 53, 57], [47, 50, 53, 56]):
            name = recognize(pitches)
            root, iv = parse_chord(name)
            self.assertEqual({(root + i) % 12 for i in iv}, {p % 12 for p in pitches}, name)

    def test_too_few_or_foreign_notes_give_none(self) -> None:
        self.assertIsNone(recognize([48, 55]))
        self.assertIsNone(recognize([48, 49, 50]))                    # cluster
        self.assertIsNone(recognize([48, 52, 55, 58, 62]))            # C9: D is not a chord tone here
        self.assertEqual(recognize([52, 55, 59]), "Em7")              # minor triad
        self.assertEqual(recognize([50, 53, 57, 59]), "Bm7b5")        # D-F-A-B: Bm7b5 over D

    def test_held_notes_follow_note_off_and_split(self) -> None:
        events = chord(0, 48, 52, 55, 72) + [off(10, 52)]
        self.assertEqual(sorted(held_notes(events, below=60)), [48, 55])


class TrackerTest(unittest.TestCase):
    def test_follow_uses_the_held_chord_then_keeps_it_after_release(self) -> None:
        t = LiveChordTracker(split=60, follow=True)
        self.assertEqual(t.update([], 0, "Cmaj7"), "Cmaj7")
        held = chord(1000, 50, 51, 52)                  # cluster: not a chord
        self.assertEqual(t.update(held, 1, "Cmaj7"), "Cmaj7")
        dm7 = chord(2000, 50, 53, 57)
        self.assertEqual(t.update(dm7, 2, "Cmaj7"), "Dm7")
        released = dm7 + [off(3000, p) for p in (50, 53, 57)]
        self.assertEqual(t.update(released, 3, "Cmaj7"), "Dm7")
        self.assertEqual([b["source"] for b in t.report(lambda b: b * 1000, 0)["blocks"]],
                         ["static", "static", "live", "held"])

    def test_observe_records_but_keeps_the_launch_progression(self) -> None:
        t = LiveChordTracker(split=60, follow=False)
        self.assertEqual(t.update(chord(2000, 43, 47, 50, 53), 2, "Cmaj7"), "Cmaj7")
        self.assertEqual([c["chord"] for c in t.changes], ["G7"])
        self.assertEqual(t.chord_for(2, "x"), "Cmaj7")

    def test_report_latency_from_chord_onset_to_first_block_using_it(self) -> None:
        t = LiveChordTracker(split=60, follow=True)
        t.update(chord(1_500_000_000, 43, 47, 50, 53), 2, "Cmaj7")    # onset = last note at 1.5 s + 3 ns
        rep = t.report(lambda b: b * 937_500_000, 0)
        c = rep["changes"][0]
        self.assertEqual((c["chord"], c["first_block"], c["seen_in_block"]), ("G7", 2, 2))
        self.assertAlmostEqual(c["latency_ms"], 375.0, places=0)
        self.assertEqual(c["latency_ms"], c["seen_latency_ms"])
        obs = LiveChordTracker(split=60, follow=False)
        obs.update(chord(1_500_000_000, 43, 47, 50, 53), 2, "Cmaj7")
        o = obs.report(lambda b: b * 937_500_000, 0)["changes"][0]
        self.assertIsNone(o["first_block"])
        self.assertAlmostEqual(o["seen_latency_ms"], 375.0, places=0)

    def test_the_same_chord_again_is_not_a_change(self) -> None:
        t = LiveChordTracker(split=60, follow=True)
        t.update(chord(0, 48, 52, 55), 0, "x")
        t.update(chord(0, 48, 52, 55, 59), 1, "x")
        self.assertEqual(len(t.changes), 1)


class ProbeTest(unittest.TestCase):
    def test_plan_voicings_stay_below_the_split_and_are_recognised(self) -> None:
        for name in PLAN.split(","):
            v = voicing(name)
            self.assertTrue(max(v) < 60, name)
            self.assertTrue(recognition_matches([{"chord": recognize(v)}], [name]), name)

    def test_recognition_compares_content_not_spelling(self) -> None:
        self.assertTrue(recognition_matches([{"chord": "Gbmaj7"}], ["F#maj7"]))
        self.assertFalse(recognition_matches([{"chord": "Gb7"}], ["F#maj7"]))
        self.assertFalse(recognition_matches([{"chord": "Dm7"}], ["Dm7", "G7"]))

    def test_follow_metrics_use_the_chord_held_at_block_start_and_adopted_blocks(self) -> None:
        # 120 BPM: half-bar block = 1.0 s. G7 arrives at 0.5 s, so it is the target from block 1.
        report = {"bpm": 120, "blocks": 4, "adopted_blocks": [0, 1, 2],
                  "played_bars": [{"notes": [[60, 0.1, 0.2], [67, 1.1, 1.2], [61, 1.5, 1.6]]},
                                  {"notes": [[71, 0.2, 0.3], [62, 1.2, 1.3]]}],
                  "live_chords": {"changes": [{"chord": "G7", "onset_ms": 500.0, "latency_ms": 500.0,
                                               "seen_latency_ms": 500.0}]}}
        m = follow_metrics(report)
        self.assertEqual(m["blocks"], 2)                    # block 0 precedes the chord, block 3 not adopted
        self.assertEqual((m["notes"], m["chord_tone_ratio"]), (3, 2 / 3))
        self.assertEqual(m["latencies_ms"], [500.0])


class FlagTest(unittest.TestCase):
    def _exit(self, *extra):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            main(["--output-dir", d, "--checkpoint", "x.pt", "--conditioning-midi", "p.mid",
                  "--chord-primer", "--chord-blocks-per-bar", "2", *extra])

    def test_live_chords_needs_input_and_half_bar_blocks(self) -> None:
        self._exit("--half-bar-blocks", "--live-chords", "follow")
        self._exit("--input-port", "x", "--live-chords", "observe")
        self._exit("--half-bar-blocks", "--input-port", "x", "--live-chords", "follow", "--chord-split", "0")
        self._exit("--half-bar-blocks", "--input-port", "x", "--live-chords", "follow", "--chord-split", "129")

    def test_split_128_counts_every_held_note(self) -> None:
        self.assertEqual(sorted(held_notes(chord(0, 60, 64, 67, 127), below=128)), [60, 64, 67, 127])


if __name__ == "__main__":
    unittest.main()
