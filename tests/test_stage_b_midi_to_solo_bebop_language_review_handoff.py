import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_bebop_language_review_handoff import (
    BebopLanguageReviewHandoffError,
    build_handoff,
    validate_best_of_package,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"artifact")
    return str(path)


def candidate(tmp: Path, rank: int, *, changed: bool, repair_steps: int) -> dict:
    case = "dominant_cycle" if rank == 1 else "minor_backdoor"
    return {
        "rank": rank,
        "case_label": case,
        "variant_index": rank * 10,
        "chords": ["Dm7", "G7", "Cmaj7", "A7"],
        "score": 0.1 + rank * 0.01,
        "gate_penalty": 0.0,
        "midi_path": touch(tmp / f"candidate_{rank:02d}.mid"),
        "context_midi_path": touch(tmp / f"candidate_{rank:02d}_context.mid"),
        "solo_audio": {
            "wav_file": {
                "path": touch(tmp / f"candidate_{rank:02d}.wav"),
                "exists": True,
                "size_bytes": 8,
            }
        },
        "context_audio": {
            "wav_file": {
                "path": touch(tmp / f"candidate_{rank:02d}_context.wav"),
                "exists": True,
                "size_bytes": 8,
            }
        },
        "motion_balance_repair": {
            "changed": changed,
            "step_count": repair_steps,
        },
        "rhythm_articulation_repair": {
            "accepted": True,
        },
        "objective_metrics": {
            "step_motion_ratio": 0.40 + rank * 0.01,
            "chromatic_step_ratio": 0.22 + rank * 0.01,
            "third_fourth_motion_ratio": 0.54,
            "large_leap_ratio": 0.05,
            "chord_tone_ratio": 0.80,
            "strong_beat_chord_tone_ratio": 1.0,
            "offbeat_non_chord_ratio": 0.40,
            "offbeat_non_chord_resolution_ratio": 1.0,
            "offbeat_unresolved_non_chord_ratio": 0.0,
            "dominant_altered_offbeat_ratio": 0.12,
            "enclosure_proxy_ratio": 0.31,
            "adjacent_repeat_ratio": 0.0,
            "interval_trigram_repeat_ratio": 0.01,
            "max_bar_pitch_class_jaccard": 0.62,
            "unique_pitch_count": 14,
        },
    }


def package(tmp: Path, *, quality_claimed: bool = False) -> dict:
    first = candidate(tmp, 1, changed=True, repair_steps=3)
    second = candidate(tmp, 2, changed=False, repair_steps=0)
    return {
        "schema_version": "stage_b_midi_to_solo_bebop_language_best_of_package_v1",
        "boundary": "stage_b_midi_to_solo_bebop_language_best_of_package",
        "aggregate": {
            "generated_candidate_count": 40,
            "selected_candidate_count": 2,
            "avg_step_motion_ratio": 0.42,
            "avg_chromatic_step_ratio": 0.24,
            "avg_large_leap_ratio": 0.04,
            "avg_enclosure_proxy_ratio": 0.32,
            "avg_offbeat_non_chord_ratio": 0.40625,
            "avg_max_bar_pitch_class_jaccard": 0.63,
            "max_gate_penalty": 0.0,
            "avg_adjacent_repeat_ratio": 0.0,
            "avg_offbeat_non_chord_resolution_ratio": 1.0,
            "avg_offbeat_unresolved_non_chord_ratio": 0.0,
        },
        "listen_first": {
            "files": [
                {
                    "rank": 1,
                    "solo_wav": first["solo_audio"]["wav_file"]["path"],
                    "context_wav": first["context_audio"]["wav_file"]["path"],
                }
            ]
        },
        "selected_candidates": [first, second],
        "quality_claimed": quality_claimed,
        "model_direct_claimed": False,
    }


class BebopLanguageReviewHandoffTest(unittest.TestCase):
    def test_builds_review_handoff_with_baseline_delta_and_artifact_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            current_path = tmp / "current" / "bebop_language_best_of_package.json"
            baseline_path = tmp / "baseline" / "bebop_language_best_of_package.json"
            note_review_path = tmp / "note_review" / "bebop_language_note_review.json"
            current_package = package(tmp / "artifacts")
            baseline_package = package(tmp / "baseline_artifacts")
            baseline_package["aggregate"]["avg_step_motion_ratio"] = 0.37
            baseline_package["aggregate"]["avg_chromatic_step_ratio"] = 0.20
            baseline_package["aggregate"]["avg_large_leap_ratio"] = 0.06
            write_json(current_path, current_package)
            write_json(baseline_path, baseline_package)
            write_json(
                note_review_path,
                {
                    "schema_version": "stage_b_midi_to_solo_bebop_language_note_review_v1",
                    "source_package": str(current_path),
                    "candidate_count": 2,
                    "candidate_scope": "all_selected",
                    "max_notes_per_candidate": 32,
                    "quality_claimed": False,
                    "model_direct_claimed": False,
                    "candidates": [
                        {"rank": 1, "first_notes": [{"pitch": 60}]},
                        {"rank": 2, "first_notes": [{"pitch": 62}]},
                    ],
                },
            )

            report = build_handoff(
                package_path=current_path,
                baseline_package_path=baseline_path,
                note_review_path=note_review_path,
                expected_candidate_count=2,
                output_dir=tmp / "out",
            )

            self.assertTrue(report["review_readiness"]["review_ready"])
            self.assertEqual(report["repair_summary"]["motion_balance_changed_candidates"], 1)
            self.assertEqual(report["repair_summary"]["motion_balance_pitch_repair_steps"], 3)
            self.assertEqual(report["note_review"]["candidate_count"], 2)
            status_by_metric = {
                row["metric"]: row["status"]
                for row in report["baseline_comparison"]
            }
            self.assertEqual(status_by_metric["avg_step_motion_ratio"], "improved")
            self.assertEqual(status_by_metric["avg_large_leap_ratio"], "improved")
            self.assertEqual(status_by_metric["avg_offbeat_non_chord_ratio"], "same")
            self.assertTrue((tmp / "out" / "bebop_language_review_handoff.json").exists())
            self.assertTrue((tmp / "out" / "bebop_language_review_handoff.md").exists())

    def test_rejects_quality_claimed_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            current = package(tmp / "artifacts", quality_claimed=True)

            with self.assertRaises(BebopLanguageReviewHandoffError):
                validate_best_of_package(
                    current,
                    package_path=tmp / "current" / "bebop_language_best_of_package.json",
                    expected_candidate_count=2,
                )

    def test_rejects_missing_review_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            current = package(tmp / "artifacts")
            current["selected_candidates"][0]["solo_audio"]["wav_file"]["path"] = str(tmp / "missing.wav")

            with self.assertRaises(BebopLanguageReviewHandoffError):
                validate_best_of_package(
                    current,
                    package_path=tmp / "current" / "bebop_language_best_of_package.json",
                    expected_candidate_count=2,
                )


if __name__ == "__main__":
    unittest.main()
