from __future__ import annotations

import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.audit_chord_progression_coverage import (
    audit_midi_text_events,
    audit_role_meta_files,
    audit_sidecar_files,
    build_decision,
    count_chord_symbols,
    find_chord_fields,
)


class ChordProgressionCoverageAuditTest(unittest.TestCase):
    def test_count_chord_symbols_detects_quality_suffixes(self) -> None:
        self.assertGreaterEqual(count_chord_symbols("Cm7 F7 Bbmaj7 Ebmaj7"), 4)
        self.assertEqual(count_chord_symbols("This sentence has capital letters but no changes"), 0)

    def test_find_chord_fields_detects_explicit_chord_progression(self) -> None:
        matches = find_chord_fields({"meta": {"chord_progression": ["Cm7", "F7"]}})

        self.assertTrue(any(match["key"] == "chord_progression" for match in matches))

    def test_audit_role_meta_files_counts_chord_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with_chords = root / "with" / "meta.json"
            without_chords = root / "without" / "meta.json"
            with_chords.parent.mkdir(parents=True)
            without_chords.parent.mkdir(parents=True)
            with_chords.write_text(
                json.dumps({"sample_id": "1", "source_midi": "a.mid", "chord_progression": ["Cm7", "F7"]}),
                encoding="utf-8",
            )
            without_chords.write_text(json.dumps({"sample_id": "2", "source_midi": "b.mid"}), encoding="utf-8")

            report = audit_role_meta_files([with_chords, without_chords])

            self.assertEqual(report["scanned_file_count"], 2)
            self.assertEqual(report["with_chord_field_count"], 1)

    def test_audit_sidecar_files_marks_chord_like_text_candidate(self) -> None:
        with TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "changes.txt"
            sidecar.write_text("Cm7 F7 Bbmaj7 Ebmaj7\n", encoding="utf-8")

            report = audit_sidecar_files([sidecar])

            self.assertEqual(report["candidate_file_count"], 1)

    def test_audit_sidecar_files_reads_mxl_container(self) -> None:
        with TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "changes.mxl"
            with zipfile.ZipFile(sidecar, mode="w") as archive:
                archive.writestr("META-INF/container.xml", "<container />")
                archive.writestr("score.musicxml", "<credit-words>Cm7 F7 Bbmaj7 Ebmaj7</credit-words>")

            report = audit_sidecar_files([sidecar])

            self.assertEqual(report["candidate_file_count"], 1)

    def test_audit_midi_text_events_marks_chord_like_lyric_candidate(self) -> None:
        with TemporaryDirectory() as tmp:
            midi_path = Path(tmp) / "text_chords.mid"
            midi = pretty_midi.PrettyMIDI(initial_tempo=120)
            instrument = pretty_midi.Instrument(program=0)
            instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
            midi.instruments.append(instrument)
            for index, text in enumerate(["Cm7", "F7", "Bbmaj7", "Ebmaj7"]):
                midi.lyrics.append(pretty_midi.Lyric(text=text, time=float(index)))
            midi.write(str(midi_path))

            report = audit_midi_text_events([midi_path])

            self.assertEqual(report["candidate_file_count"], 1)
            self.assertEqual(report["with_text_event_count"], 1)

    def test_build_decision_routes_to_chord_inference_when_no_hits(self) -> None:
        decision = build_decision(
            {
                "role_meta": {"with_chord_field_count": 0},
                "sidecars": {"candidate_file_count": 0},
                "midi_text_events": {"candidate_file_count": 0},
            }
        )

        self.assertEqual(decision["next_step"], "create_chord_inference_or_lead_sheet_alignment_issue")


if __name__ == "__main__":
    unittest.main()
