from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_jazz_training_manifests import (
    build_manifest_payload,
    split_grouped_rows,
    write_outputs,
)


def row(
    path: str,
    *,
    artist: str,
    album: str,
    sha1: str,
    brad: bool = False,
    recommendation: str = "candidate",
) -> dict:
    return {
        "path": path,
        "sha1": sha1,
        "source": "studio",
        "artist": artist,
        "album": album,
        "is_brad_mehldau": brad,
        "duration_sec": 120.0,
        "non_drum_note_count": 100,
        "piano_program_note_ratio": 1.0,
        "max_note_duration_ratio": 0.02,
        "recommendation": recommendation,
    }


class JazzTrainingManifestsTest(unittest.TestCase):
    def test_manifest_isolates_generic_and_brad_candidates(self) -> None:
        audit_payload = {
            "summary": {"candidate_file_count": 6},
            "files": [
                row("a.mid", artist="A", album="One", sha1="a1"),
                row("b.mid", artist="B", album="Two", sha1="b1"),
                row("brad1.mid", artist="Brad Mehldau", album="Brad One", sha1="c1", brad=True),
                row("brad2.mid", artist="Brad Mehldau", album="Brad Two", sha1="d1", brad=True),
                row("review.mid", artist="C", album="Three", sha1="e1", recommendation="review_too_long"),
                row("reject.mid", artist="D", album="Four", sha1="f1", recommendation="reject_too_few_notes"),
            ],
        }

        payload = build_manifest_payload(
            audit_payload,
            audit_json=Path("audit.json"),
            seed=7,
            generic_train_ratio=0.5,
            generic_val_ratio=0.5,
            brad_train_ratio=0.5,
            brad_val_ratio=0.25,
            brad_holdout_ratio=0.25,
            group_fields=["artist", "album"],
        )

        generic_paths = {
            item["path"]
            for split in ("generic_jazz_train", "generic_jazz_val")
            for item in payload["splits"][split]
        }
        brad_paths = {
            item["path"]
            for split in ("brad_adaptation_train", "brad_adaptation_val", "brad_test_holdout")
            for item in payload["splits"][split]
        }

        self.assertEqual(generic_paths, {"a.mid", "b.mid"})
        self.assertEqual(brad_paths, {"brad1.mid", "brad2.mid"})
        self.assertEqual([item["path"] for item in payload["diagnostics"]["review"]], ["review.mid"])
        self.assertEqual([item["path"] for item in payload["diagnostics"]["rejected"]], ["reject.mid"])

    def test_grouped_split_keeps_sha_and_album_boundaries(self) -> None:
        rows = [
            row("same_sha_a.mid", artist="A", album="One", sha1="dup"),
            row("same_sha_b.mid", artist="B", album="Two", sha1="dup"),
            row("same_album_a.mid", artist="C", album="Group", sha1="c1"),
            row("same_album_b.mid", artist="C", album="Group", sha1="c2"),
            row("solo.mid", artist="D", album="Solo", sha1="d1"),
        ]

        splits = split_grouped_rows(
            rows,
            {"train": 0.5, "val": 0.5},
            seed=3,
            group_fields=["artist", "album"],
        )

        owners = {}
        for split_name, split_rows in splits.items():
            for item in split_rows:
                owners[item["path"]] = split_name

        self.assertEqual(owners["same_sha_a.mid"], owners["same_sha_b.mid"])
        self.assertEqual(owners["same_album_a.mid"], owners["same_album_b.mid"])

    def test_write_outputs_creates_json_and_text_manifests(self) -> None:
        audit_payload = {
            "summary": {"candidate_file_count": 2},
            "files": [
                row("a.mid", artist="A", album="One", sha1="a1"),
                row("brad.mid", artist="Brad Mehldau", album="Brad", sha1="b1", brad=True),
            ],
        }
        payload = build_manifest_payload(
            audit_payload,
            audit_json=Path("audit.json"),
            seed=1,
            generic_train_ratio=0.9,
            generic_val_ratio=0.1,
            brad_train_ratio=0.7,
            brad_val_ratio=0.15,
            brad_holdout_ratio=0.15,
            group_fields=["artist", "album"],
        )

        with self.subTest("write output files"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                out = Path(tmp_dir)
                write_outputs(out, payload)
                self.assertTrue((out / "jazz_training_manifests.json").exists())
                self.assertTrue((out / "generic_jazz_train.txt").exists())
                self.assertTrue((out / "brad_adaptation_train.txt").exists())
                self.assertTrue((out / "jazz_training_manifests.md").exists())


if __name__ == "__main__":
    unittest.main()
