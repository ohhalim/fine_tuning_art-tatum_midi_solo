from __future__ import annotations

import unittest

import pretty_midi

from inference.control.chord_primer import (
    build_chord_primer,
    chord_guide_notes,
    chord_tone_pitch_classes,
)

BPM = 128
# The checkpoint's training sets top out at token id 388 and contain zero
# control tokens; anything above that was never given a gradient.
MAX_TRAINED_TOKEN_ID = 388


class VocabularyCompatibilityTests(unittest.TestCase):
    """The whole point of this route is that it uses only trained tokens."""

    def test_primer_stays_inside_the_trained_token_range(self):
        tokens, used = build_chord_primer(["Dm7"], bpm=BPM, bars=1)

        self.assertTrue(used)
        self.assertTrue(tokens)
        self.assertLessEqual(max(tokens), MAX_TRAINED_TOKEN_ID)

    def test_primer_contains_no_control_tokens(self):
        from utilities.constants import TOKEN_COND_SEP, TOKEN_STAGE_B_CHORD_QUALITY_END

        tokens, _ = build_chord_primer(["Cm7", "F7"], bpm=BPM, bars=2)
        control = [t for t in tokens if TOKEN_COND_SEP <= t <= TOKEN_STAGE_B_CHORD_QUALITY_END]

        self.assertEqual([], control)


class GuideTests(unittest.TestCase):
    def test_different_chords_produce_different_pitches(self):
        d_minor = {n.pitch % 12 for n in chord_guide_notes(["Dm7"], bpm=BPM, bars=1)}
        g_seven = {n.pitch % 12 for n in chord_guide_notes(["G7"], bpm=BPM, bars=1)}

        self.assertNotEqual(d_minor, g_seven)

    def test_guide_sits_below_the_solo_register(self):
        notes = chord_guide_notes(["Cmaj7"], bpm=BPM, bars=1)

        self.assertTrue(notes)
        self.assertLessEqual(max(n.pitch for n in notes), 60)

    def test_guide_pitches_belong_to_the_chord(self):
        for chord in ("Dm7", "G7", "Cmaj7", "C7"):
            with self.subTest(chord=chord):
                tones = chord_tone_pitch_classes(chord)
                notes = chord_guide_notes([chord], bpm=BPM, bars=1)
                self.assertTrue(all(n.pitch % 12 in tones for n in notes))


class MelodicContextTests(unittest.TestCase):
    def test_recent_playing_is_merged_onto_the_same_timeline(self):
        played = [pretty_midi.Note(velocity=90, pitch=72, start=0.1, end=0.3)]

        tokens, used = build_chord_primer(["Dm7"], bpm=BPM, bars=1, melodic_notes=played)

        self.assertTrue(used)
        self.assertIn(72, tokens)  # note_on for the played pitch survives

    def test_empty_progression_reports_no_chord(self):
        tokens, used = build_chord_primer([], bpm=BPM, bars=1)

        self.assertFalse(used)
        self.assertEqual([], tokens)


class ScoringTests(unittest.TestCase):
    """Scoring must never be able to change a note."""

    def test_chord_tone_classes_are_read_only(self):
        self.assertEqual({2, 5, 9, 0}, chord_tone_pitch_classes("Dm7"))
        self.assertEqual({7, 11, 2, 5}, chord_tone_pitch_classes("G7"))

    def test_score_against_counts_without_filtering(self):
        from scripts.run_chord_primer_ab import score_against

        notes = [pretty_midi.Note(90, 62, 0.0, 0.2),   # D, in Dm7
                 pretty_midi.Note(90, 61, 0.2, 0.4)]   # C#, not in Dm7
        before = [(n.pitch, n.start) for n in notes]

        ratio = score_against(notes, ["Dm7"], bars=1, bar_seconds=1.875)

        self.assertAlmostEqual(0.5, ratio)
        self.assertEqual(before, [(n.pitch, n.start) for n in notes])

    def test_empty_notes_score_as_none_not_zero(self):
        from scripts.run_chord_primer_ab import score_against

        self.assertIsNone(score_against([], ["Dm7"], bars=1, bar_seconds=1.875))
