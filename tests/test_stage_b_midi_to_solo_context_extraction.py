from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.extract_stage_b_midi_to_solo_context import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloContextExtractionError,
    build_context_report,
    build_fixture_midi,
    extract_context_from_midi,
    validate_context_report,
)
from scripts.stage_b_tokens import POSITIONS_PER_BAR


class StageBMidiToSoloContextExtractionTest(unittest.TestCase):
    def test_extracts_required_context_from_fixture_midi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            midi_path = build_fixture_midi(output_dir / "fixture.mid")
            report = build_context_report(
                midi_path=midi_path,
                output_dir=output_dir,
                target_context_bars=8,
                issue_number=483,
            )
            summary = validate_context_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                min_context_bars=4,
                require_no_final_claim=True,
            )

            self.assertEqual(summary["context_bars"], 8)
            self.assertEqual(summary["context_event_count"], 8 * POSITIONS_PER_BAR)
            self.assertGreaterEqual(summary["inferred_chord_bar_count"], 4)
            self.assertGreaterEqual(summary["carry_forward_chord_bar_count"], 4)
            self.assertEqual(summary["unknown_chord_bar_count"], 0)
            self.assertGreater(summary["bass_note_bar_count"], 0)
            self.assertFalse(summary["midi_to_solo_mvp_claimed"])
            self.assertFalse(summary["harmony_analysis_quality_claimed"])
            self.assertFalse(summary["critical_user_input_required"])

    def test_explicit_chord_event_overrides_pitch_class_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            midi_path = build_fixture_midi(Path(tmp) / "fixture_with_text.mid")
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            pm.text_events.append(pretty_midi.Text("Dm7", 0.0))
            pm.write(str(midi_path))

            context = extract_context_from_midi(midi_path, target_context_bars=4)
            first_bar = context["bar_contexts"][0]

            self.assertEqual(first_bar["chord_root"], "D")
            self.assertEqual(first_bar["chord_quality"], "min7")
            self.assertEqual(first_bar["chord_source"], "explicit_text_event")
            self.assertEqual(first_bar["chord_confidence"], 1.0)

    def test_rejects_missing_input_midi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StageBMidiToSoloContextExtractionError):
                extract_context_from_midi(Path(tmp) / "missing.mid")

    def test_rejects_too_few_context_bars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            midi_path = build_fixture_midi(output_dir / "fixture.mid")
            report = build_context_report(
                midi_path=midi_path,
                output_dir=output_dir,
                target_context_bars=4,
                issue_number=483,
            )

            with self.assertRaises(StageBMidiToSoloContextExtractionError):
                validate_context_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    min_context_bars=8,
                    require_no_final_claim=True,
                )

    def test_rejects_final_quality_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            midi_path = build_fixture_midi(output_dir / "fixture.mid")
            report = build_context_report(
                midi_path=midi_path,
                output_dir=output_dir,
                target_context_bars=8,
                issue_number=483,
            )
            report["readiness"]["harmony_analysis_quality_claimed"] = True

            with self.assertRaises(StageBMidiToSoloContextExtractionError):
                validate_context_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    min_context_bars=4,
                    require_no_final_claim=True,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_context_extraction_mvp")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_training_resource_probe")


if __name__ == "__main__":
    unittest.main()
