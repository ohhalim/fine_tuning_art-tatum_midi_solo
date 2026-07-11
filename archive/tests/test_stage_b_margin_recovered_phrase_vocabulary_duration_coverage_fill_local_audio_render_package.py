from __future__ import annotations

import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.build_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package import (
    DurationCoverageFillLocalAudioRenderPackageError,
    build_local_audio_render_package,
    validate_local_audio_render_package,
)


def write_midi(path: Path, pitch: int) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="solo")
    piano.notes.append(pretty_midi.Note(velocity=76, pitch=pitch, start=0.0, end=0.2))
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def external_boundary() -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary_v1",
        "external_review_boundary": {
            "boundary": "external_human_audio_review_required_for_human_preference_claim",
            "status": "pending_external_review_input",
            "human_audio_preference_claimed": False,
        },
    }


def audio_review_package(root: Path) -> dict:
    source_path = root / "source.mid"
    fill_path = root / "fill.mid"
    context_path = root / "fill_context.mid"
    write_midi(source_path, 60)
    write_midi(fill_path, 64)
    write_midi(context_path, 67)
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package_v1",
        "candidate_id": "duration_fill_candidate",
        "review_items": [
            {
                "role": "source_constrained_partial",
                "candidate_id": "source_candidate",
                "midi_file": {"path": str(source_path), "required": True},
                "context_midi_file": {"path": "", "required": False},
                "metric_summary": {"note_count": 1},
            },
            {
                "role": "duration_coverage_fill_keep",
                "candidate_id": "duration_fill_candidate",
                "midi_file": {"path": str(fill_path), "required": True},
                "context_midi_file": {"path": str(context_path), "required": True},
                "metric_summary": {"note_count": 1},
            },
        ],
        "package_boundary": {
            "status": "ready_for_external_review_input",
            "preference_claimed": False,
        },
    }


def fake_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class StageBMarginRecoveredPhraseVocabularyDurationCoverageFillLocalAudioRenderPackageTest(unittest.TestCase):
    def test_records_renderer_unavailable_without_audio_claim(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_local_audio_render_package(
                external_boundary(),
                audio_review_package(root),
                output_dir=root / "render_package",
                renderer_paths={"fluidsynth": "", "timidity": ""},
            )
            summary = validate_local_audio_render_package(
                report,
                expected_candidate_id="duration_fill_candidate",
                expected_status="renderer_unavailable",
                require_required_midi_exists=True,
                require_no_audio_claim=True,
            )

            self.assertEqual(summary["render_status"], "renderer_unavailable")
            self.assertEqual(summary["planned_audio_output_count"], 2)
            self.assertFalse(summary["render_attempted"])
            self.assertFalse(summary["audio_rendered_quality_claimed"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertEqual(report["planned_audio_outputs"][0]["render_command"], [])

    def test_records_ready_command_when_renderer_and_soundfont_exist(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            fake_executable(renderer)
            soundfont.write_bytes(b"sf2")
            report = build_local_audio_render_package(
                external_boundary(),
                audio_review_package(root),
                output_dir=root / "render_package",
                requested_renderer="fluidsynth",
                soundfont_path=str(soundfont),
                renderer_paths={"fluidsynth": str(renderer), "timidity": ""},
            )
            summary = validate_local_audio_render_package(
                report,
                expected_candidate_id="duration_fill_candidate",
                expected_status="ready_for_local_render",
                require_required_midi_exists=True,
                require_no_audio_claim=True,
            )

            self.assertEqual(summary["selected_renderer_name"], "fluidsynth")
            self.assertTrue(report["planned_audio_outputs"][0]["render_command"])
            self.assertIn(str(soundfont), report["planned_audio_outputs"][0]["render_command"])

    def test_rejects_external_human_audio_preference_claim(self) -> None:
        with TemporaryDirectory() as temp_dir:
            boundary = external_boundary()
            boundary["external_review_boundary"]["human_audio_preference_claimed"] = True
            with self.assertRaises(DurationCoverageFillLocalAudioRenderPackageError):
                build_local_audio_render_package(
                    boundary,
                    audio_review_package(Path(temp_dir)),
                    output_dir=Path(temp_dir) / "render_package",
                )


if __name__ == "__main__":
    unittest.main()
