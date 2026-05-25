from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.run_stage_b_data_motif_generation_compare import (
    build_compare_summary,
    build_review_export,
    chord_tone_pitch_classes,
    analyze_contour_landing_profile,
    contour_landing_summary,
    data_motif_contour_landing_repair_tokens,
    data_motif_guide_tones_tokens,
    data_motif_phrase_recovery_tokens,
    data_motif_rhythm_phrase_variation_tokens,
    data_motif_tokens,
    duration_tokens_from_steps,
    fit_duration_tokens_to_positions,
    guide_tone_pitch_classes,
    nearest_allowed_pitch_token,
    normalize_position_deltas,
    overlap_free_solo_notes,
    parse_baseline_modes,
    phrase_cadence_tokens,
    phrase_recovery_tokens,
    straight_guide_tones_tokens,
    straight_grid_tokens,
    varied_grid_position_duration_steps,
    varied_grid_tokens,
    varied_guide_tones_tokens,
)
from scripts.run_stage_b_generation_probe import (
    analyze_stage_b_note_grammar,
    analyze_stage_b_rhythm_profile,
    build_stage_b_primer,
    extract_stage_b_note_groups,
)
from scripts.stage_b_tokens import (
    TOKEN_NOTE_PITCH_END,
    TOKEN_NOTE_PITCH_START,
    duration_steps_from_token,
    pitch_from_token,
)


def template_report() -> dict:
    return {
        "summary": {
            "top_rhythm_templates": [
                {
                    "count": 3,
                    "key": {
                        "position_deltas": [0, 2, 5, 7],
                        "duration_steps": [2, 1, 3, 2],
                    },
                },
                {
                    "count": 1,
                    "key": {
                        "position_deltas": [0, 3, 4, 9],
                        "duration_steps": [1, 4, 2, 1],
                    },
                },
            ],
            "top_contour_templates": [
                {
                    "count": 2,
                    "key": {
                        "pitch_intervals": [0, 2, 5, 3],
                    },
                },
                {
                    "count": 1,
                    "key": {
                        "pitch_intervals": [0, -2, 1, 0],
                    },
                },
            ],
        }
    }


class StageBDataMotifGenerationCompareTest(unittest.TestCase):
    def test_parse_baseline_modes_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            parse_baseline_modes("hand_written_swing,random")

    def test_parse_baseline_modes_accepts_straight_guide_tones(self) -> None:
        self.assertEqual(
            parse_baseline_modes("straight_grid,straight_guide_tones,data_motif_guide_tones"),
            ["straight_grid", "straight_guide_tones", "data_motif_guide_tones"],
        )

    def test_parse_baseline_modes_accepts_phrase_cadence(self) -> None:
        self.assertEqual(parse_baseline_modes("phrase_cadence"), ["phrase_cadence"])

    def test_parse_baseline_modes_accepts_phrase_recovery(self) -> None:
        self.assertEqual(parse_baseline_modes("phrase_recovery"), ["phrase_recovery"])

    def test_parse_baseline_modes_accepts_data_motif_phrase_recovery(self) -> None:
        self.assertEqual(parse_baseline_modes("data_motif_phrase_recovery"), ["data_motif_phrase_recovery"])

    def test_parse_baseline_modes_accepts_contour_landing_repair(self) -> None:
        self.assertEqual(
            parse_baseline_modes("data_motif_contour_landing_repair"),
            ["data_motif_contour_landing_repair"],
        )

    def test_parse_baseline_modes_accepts_rhythm_phrase_variation(self) -> None:
        self.assertEqual(
            parse_baseline_modes("data_motif_rhythm_phrase_variation"),
            ["data_motif_rhythm_phrase_variation"],
        )

    def test_build_compare_summary_allows_selected_modes_without_hand_written_reference(self) -> None:
        def valid_summary() -> dict:
            return {
                "sample_count": 3,
                "valid_sample_count": 3,
                "strict_valid_sample_count": 3,
                "avg_syncopated_onset_ratio": 0.5,
                "avg_unique_bar_position_pattern_ratio": 0.5,
                "avg_duration_diversity_ratio": 0.25,
                "avg_most_common_duration_ratio": 0.5,
                "avg_ioi_diversity_ratio": 0.25,
                "avg_most_common_ioi_ratio": 0.5,
                "avg_tension_ratio": 0.2,
                "avg_root_tone_ratio": 0.0,
                "passed_strict_review_gate": True,
            }

        summary = build_compare_summary(
            {
                "phrase_cadence": valid_summary(),
                "varied_guide_tones": valid_summary(),
            },
            min_strict_valid_samples=1,
        )

        self.assertFalse(summary["comparison_ready"])
        self.assertTrue(summary["passed_selected_modes_gate"])
        self.assertTrue(summary["passed_compare_gate"])

    def test_normalize_position_deltas_scales_into_slot(self) -> None:
        positions = normalize_position_deltas([0, 3, 9, 12], slot_start=8, slot_size=8)

        self.assertEqual(positions[0], 8)
        self.assertLessEqual(positions[-1], 15)
        self.assertEqual(positions, sorted(set(positions)))

    def test_duration_tokens_from_steps_clamps_count(self) -> None:
        tokens = duration_tokens_from_steps([0, 2, 99], target_count=4)
        steps = [duration_steps_from_token(token) for token in tokens]

        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[0], 1)
        self.assertGreaterEqual(steps[-1], 1)

    def test_fit_duration_tokens_to_positions_prevents_overlap(self) -> None:
        tokens = fit_duration_tokens_to_positions([0, 1, 3, 7], [8, 8, 8, 8])
        steps = [duration_steps_from_token(token) for token in tokens]

        self.assertEqual(steps, [1, 2, 4, 4])

    def test_nearest_allowed_pitch_token_avoids_recent_pitch_when_possible(self) -> None:
        allowed = list(range(TOKEN_NOTE_PITCH_START, TOKEN_NOTE_PITCH_END + 1))
        recent_pitch = pitch_from_token(allowed[0])

        token = nearest_allowed_pitch_token(recent_pitch, allowed[:3], recent_pitches=[recent_pitch])

        self.assertNotEqual(pitch_from_token(token), recent_pitch)

    def test_data_motif_tokens_builds_strictly_increasing_solo_line_groups(self) -> None:
        primer = build_stage_b_primer(["Cm7", "Fm7"], 124)
        tokens = data_motif_tokens(
            primer_tokens=primer,
            chords=["Cm7", "Fm7"],
            bars=2,
            note_groups_per_bar=8,
            template_report=template_report(),
            seed=17,
        )

        grammar = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        positions_by_bar: dict[int, list[int]] = {}
        for group in groups:
            positions_by_bar.setdefault(int(group["bar"]), []).append(int(group["position"]))

        self.assertTrue(grammar["grammar_valid"])
        self.assertEqual(len(groups), 16)
        for positions in positions_by_bar.values():
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(len(positions), len(set(positions)))

    def test_data_motif_guide_tones_preserves_data_motif_rhythm_shape(self) -> None:
        primer = build_stage_b_primer(["Cm7", "F7"], 124)
        tokens = data_motif_guide_tones_tokens(
            primer_tokens=primer,
            chords=["Cm7", "F7"],
            bars=2,
            note_groups_per_bar=8,
            template_report=template_report(),
            seed=17,
        )

        grammar = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        positions_by_bar: dict[int, list[int]] = {}
        for group in groups:
            positions_by_bar.setdefault(int(group["bar"]), []).append(int(group["position"]))

        self.assertTrue(grammar["grammar_valid"])
        self.assertEqual(len(groups), 16)
        self.assertGreater(len({group["position"] for group in groups}), 4)
        for positions in positions_by_bar.values():
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(len(positions), len(set(positions)))

    def test_data_motif_guide_tones_uses_guide_tones_on_strong_beats(self) -> None:
        chords = ["Cm7", "F7"]
        primer = build_stage_b_primer(chords, 124)
        tokens = data_motif_guide_tones_tokens(
            primer_tokens=primer,
            chords=chords,
            bars=2,
            note_groups_per_bar=8,
            template_report=template_report(),
            seed=17,
        )
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))

        strong_groups = [group for group in groups if int(group["position"]) in {0, 4, 8, 12}]
        self.assertGreaterEqual(len(strong_groups), 2)
        for group in strong_groups:
            chord = chords[int(group["bar"]) % len(chords)]
            pitch_class = int(group["pitch"]) % 12
            self.assertIn(pitch_class, set(guide_tone_pitch_classes(chord)))

    def test_data_motif_guide_tones_avoids_chromatic_scale_runs(self) -> None:
        chords = ["Cm7", "F7"]
        primer = build_stage_b_primer(chords, 124)
        tokens = data_motif_guide_tones_tokens(
            primer_tokens=primer,
            chords=chords,
            bars=2,
            note_groups_per_bar=8,
            template_report=template_report(),
            seed=17,
        )
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        non_chord_run = 0

        for group in groups:
            chord = chords[int(group["bar"]) % len(chords)]
            if int(group["pitch"]) % 12 in chord_tone_pitch_classes(chord):
                non_chord_run = 0
            else:
                non_chord_run += 1
            self.assertLessEqual(non_chord_run, 1)

    def test_straight_grid_tokens_stay_on_even_subdivision_grid(self) -> None:
        primer = build_stage_b_primer(["Cm7", "Fm7"], 124)
        tokens = straight_grid_tokens(
            primer_tokens=primer,
            chords=["Cm7", "Fm7"],
            bars=2,
            note_groups_per_bar=8,
            seed=17,
        )

        grammar = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))

        self.assertTrue(grammar["grammar_valid"])
        self.assertEqual(len(groups), 16)
        self.assertEqual({group["position"] % 2 for group in groups}, {0})

    def test_straight_guide_tones_tokens_use_guide_tones_on_strong_beats(self) -> None:
        chords = ["Cm7", "F7"]
        primer = build_stage_b_primer(chords, 124)
        tokens = straight_guide_tones_tokens(
            primer_tokens=primer,
            chords=chords,
            bars=2,
            note_groups_per_bar=8,
            seed=17,
        )

        grammar = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))

        self.assertTrue(grammar["grammar_valid"])
        self.assertEqual(len(groups), 16)
        self.assertEqual({group["position"] % 2 for group in groups}, {0})
        for group in groups:
            if int(group["position"]) in {0, 4, 8, 12}:
                chord = chords[int(group["bar"]) % len(chords)]
                pitch_class = int(group["pitch"]) % 12
                self.assertIn(pitch_class, set(guide_tone_pitch_classes(chord)))

    def test_straight_guide_tones_tokens_avoid_chromatic_scale_runs(self) -> None:
        chords = ["Cm7", "F7"]
        primer = build_stage_b_primer(chords, 124)
        tokens = straight_guide_tones_tokens(
            primer_tokens=primer,
            chords=chords,
            bars=2,
            note_groups_per_bar=8,
            seed=17,
        )
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        non_chord_run = 0

        for group in groups:
            chord = chords[int(group["bar"]) % len(chords)]
            if int(group["pitch"]) % 12 in chord_tone_pitch_classes(chord):
                non_chord_run = 0
            else:
                non_chord_run += 1
            self.assertLessEqual(non_chord_run, 1)

    def test_varied_grid_position_duration_steps_do_not_collapse_duration(self) -> None:
        positions, durations = varied_grid_position_duration_steps(8)

        self.assertEqual(positions, [0, 1, 4, 6, 8, 9, 12, 14])
        self.assertEqual(len(set(durations)), 2)
        self.assertLess(max(durations.count(duration) for duration in set(durations)) / len(durations), 0.75)
        for index, position in enumerate(positions[:-1]):
            self.assertLessEqual(position + durations[index], positions[index + 1])

    def test_varied_grid_tokens_use_multiple_durations(self) -> None:
        chords = ["Cm7", "F7"]
        primer = build_stage_b_primer(chords, 124)
        tokens = varied_grid_tokens(
            primer_tokens=primer,
            chords=chords,
            bars=2,
            note_groups_per_bar=8,
            seed=17,
        )
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        durations = [int(group["duration_steps"]) for group in groups]

        self.assertGreaterEqual(len(set(durations)), 2)

    def test_varied_guide_tones_tokens_use_guide_tone_pitches_and_multiple_durations(self) -> None:
        chords = ["Cm7", "F7"]
        primer = build_stage_b_primer(chords, 124)
        tokens = varied_guide_tones_tokens(
            primer_tokens=primer,
            chords=chords,
            bars=2,
            note_groups_per_bar=8,
            seed=17,
        )
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        durations = [int(group["duration_steps"]) for group in groups]
        non_guide_run = 0

        self.assertGreaterEqual(len(set(durations)), 2)
        for group in groups:
            chord = chords[int(group["bar"]) % len(chords)]
            if int(group["pitch"]) % 12 in set(guide_tone_pitch_classes(chord)):
                non_guide_run = 0
            else:
                non_guide_run += 1
            self.assertLessEqual(non_guide_run, 1)

    def test_phrase_cadence_tokens_reduce_scalar_and_chromatic_motion(self) -> None:
        chords = ["Cm7", "F7", "Bbmaj7", "Ebmaj7"]
        primer = build_stage_b_primer(chords, 124)
        tokens = phrase_cadence_tokens(
            primer_tokens=primer,
            chords=chords,
            bars=4,
            note_groups_per_bar=8,
            seed=17,
        )
        grammar = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        pitches = [int(group["pitch"]) for group in groups]
        intervals = [abs(right - left) for left, right in zip(pitches, pitches[1:])]
        stepwise_ratio = sum(1 for interval in intervals if interval in {1, 2}) / len(intervals)
        chromatic_ratio = sum(1 for interval in intervals if interval == 1) / len(intervals)
        durations = [int(group["duration_steps"]) for group in groups]

        self.assertTrue(grammar["grammar_valid"])
        self.assertEqual(len(groups), 32)
        self.assertGreaterEqual(len(set(durations)), 2)
        self.assertLess(stepwise_ratio, 0.70)
        self.assertLess(chromatic_ratio, 0.35)

    def test_phrase_recovery_tokens_resolve_large_leaps(self) -> None:
        chords = ["Cm7", "F7", "Bbmaj7", "Ebmaj7"]
        primer = build_stage_b_primer(chords, 124)
        tokens = phrase_recovery_tokens(
            primer_tokens=primer,
            chords=chords,
            bars=4,
            note_groups_per_bar=8,
            seed=17,
        )
        grammar = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        pitches = [int(group["pitch"]) for group in groups]
        intervals = [right - left for left, right in zip(pitches, pitches[1:])]
        large_leap_indexes = [index for index, interval in enumerate(intervals) if abs(interval) >= 7]
        unresolved = 0
        for index in large_leap_indexes:
            if index + 1 >= len(intervals):
                unresolved += 1
                continue
            direction = 1 if intervals[index] > 0 else -1
            next_direction = 1 if intervals[index + 1] > 0 else -1 if intervals[index + 1] < 0 else 0
            if next_direction != -direction or not 1 <= abs(intervals[index + 1]) <= 5:
                unresolved += 1
        unresolved_ratio = unresolved / len(large_leap_indexes)

        self.assertTrue(grammar["grammar_valid"])
        self.assertEqual(len(groups), 32)
        self.assertLess(unresolved_ratio, 0.45)

    def test_data_motif_phrase_recovery_preserves_data_rhythm_and_resolves_leaps(self) -> None:
        chords = ["Cm7", "F7", "Bbmaj7", "Ebmaj7"]
        primer = build_stage_b_primer(chords, 124)
        tokens = data_motif_phrase_recovery_tokens(
            primer_tokens=primer,
            chords=chords,
            bars=4,
            note_groups_per_bar=8,
            template_report=template_report(),
            seed=17,
        )
        grammar = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        pitches = [int(group["pitch"]) for group in groups]
        intervals = [right - left for left, right in zip(pitches, pitches[1:])]
        large_leap_indexes = [index for index, interval in enumerate(intervals) if abs(interval) >= 7]
        unresolved = 0
        for index in large_leap_indexes:
            if index + 1 >= len(intervals):
                unresolved += 1
                continue
            direction = 1 if intervals[index] > 0 else -1
            next_direction = 1 if intervals[index + 1] > 0 else -1 if intervals[index + 1] < 0 else 0
            if next_direction != -direction or not 1 <= abs(intervals[index + 1]) <= 5:
                unresolved += 1
        unresolved_ratio = unresolved / len(large_leap_indexes)

        self.assertTrue(grammar["grammar_valid"])
        self.assertEqual(len(groups), 32)
        self.assertGreater(len({group["position"] for group in groups}), 4)
        self.assertLess(unresolved_ratio, 0.45)

    def test_data_motif_contour_landing_repair_resolves_final_landing_and_smooths_register(self) -> None:
        chords = ["Cm7", "F7", "Bbmaj7", "Ebmaj7"]
        primer = build_stage_b_primer(chords, 124)
        tokens = data_motif_contour_landing_repair_tokens(
            primer_tokens=primer,
            chords=chords,
            bars=4,
            note_groups_per_bar=8,
            template_report=template_report(),
            seed=17,
        )
        grammar = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        profile = analyze_contour_landing_profile(tokens, chords=chords, primer_size=len(primer))
        pitches = [int(group["pitch"]) for group in groups]
        repeated_ratio = sum(1 for left, right in zip(pitches, pitches[1:]) if left == right) / (
            len(pitches) - 1
        )

        self.assertTrue(grammar["grammar_valid"])
        self.assertEqual(len(groups), 32)
        self.assertGreater(len({group["position"] for group in groups}), 4)
        self.assertTrue(profile["final_landing_resolved"])
        self.assertIn(profile["final_landing_role"], {"guide", "chord_tone"})
        self.assertLess(repeated_ratio, 0.20)
        self.assertLessEqual(profile["max_abs_interval"], 12)
        self.assertEqual(profile["abrupt_register_reset_count"], 0)

    def test_data_motif_rhythm_phrase_variation_preserves_landing_and_varies_rhythm(self) -> None:
        chords = ["Cm7", "F7", "Bbmaj7", "Ebmaj7"]
        primer = build_stage_b_primer(chords, 124)
        tokens = data_motif_rhythm_phrase_variation_tokens(
            primer_tokens=primer,
            chords=chords,
            bars=4,
            note_groups_per_bar=8,
            template_report=template_report(),
            seed=17,
        )
        grammar = analyze_stage_b_note_grammar(tokens, primer_size=len(primer))
        groups = extract_stage_b_note_groups(tokens, primer_size=len(primer))
        profile = analyze_contour_landing_profile(tokens, chords=chords, primer_size=len(primer))
        rhythm = analyze_stage_b_rhythm_profile(tokens, primer_size=len(primer))
        pitches = [int(group["pitch"]) for group in groups]

        self.assertTrue(grammar["grammar_valid"])
        self.assertEqual(len(groups), 32)
        self.assertGreaterEqual(min(pitches), 48)
        self.assertTrue(profile["final_landing_resolved"])
        self.assertIn(profile["final_landing_role"], {"guide", "chord_tone"})
        self.assertLessEqual(profile["max_abs_interval"], 6)
        self.assertGreater(rhythm["ioi_diversity_ratio"], 0.09)
        self.assertLess(rhythm["most_common_ioi_ratio"], 0.40)

    def test_contour_landing_summary_counts_resolved_landings(self) -> None:
        summary = contour_landing_summary(
            [
                {
                    "contour_landing_profile": {
                        "final_landing_resolved": True,
                        "final_landing_role": "guide",
                        "max_abs_interval": 7,
                        "abrupt_register_reset_count": 0,
                    }
                },
                {
                    "contour_landing_profile": {
                        "final_landing_resolved": False,
                        "final_landing_role": "outside",
                        "max_abs_interval": 14,
                        "abrupt_register_reset_count": 1,
                    }
                },
            ]
        )

        self.assertEqual(summary["final_landing_resolved_count"], 1)
        self.assertEqual(summary["final_landing_role_counts"]["guide"], 1)
        self.assertEqual(summary["final_landing_role_counts"]["outside"], 1)
        self.assertEqual(summary["total_abrupt_register_reset_count"], 1)
        self.assertEqual(summary["max_abs_interval"], 14)

    def test_build_review_export_copies_named_mode_files(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_midi = tmp_path / "source.mid"
            midi = pretty_midi.PrettyMIDI(initial_tempo=124)
            instrument = pretty_midi.Instrument(program=0, name="Solo")
            instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
            midi.instruments.append(instrument)
            midi.write(str(source_midi))
            samples = {
                "data_motif": [
                    {
                        "sample_index": 1,
                        "sample_seed": 17,
                        "valid": True,
                        "strict_valid": True,
                        "midi_path": str(source_midi),
                        "metrics": {
                            "note_count": 8,
                            "unique_pitch_count": 4,
                            "dead_air_ratio": 0.1,
                        },
                        "rhythm_profile": {
                            "syncopated_onset_ratio": 0.6,
                            "unique_bar_position_pattern_ratio": 1.0,
                            "duration_diversity_ratio": 0.2,
                            "most_common_duration_ratio": 0.4,
                            "ioi_diversity_ratio": 0.2,
                            "most_common_ioi_ratio": 0.4,
                        },
                        "pitch_roles": {
                            "tension_ratio": 0.2,
                            "root_tone_ratio": 0.0,
                        },
                    }
                ]
            }

            manifest = build_review_export(
                samples,
                output_dir=tmp_path / "review",
                top_n=1,
                copy_midi=True,
                chords=["Cm7", "Fm7"],
                bpm=124,
                bars=2,
            )

            copied = Path(manifest["candidates"][0]["review_midi_path"])
            context = Path(manifest["candidates"][0]["context_midi_path"])
            self.assertTrue(copied.exists())
            self.assertTrue(context.exists())
            self.assertIn("data_motif", copied.name)
            self.assertTrue(Path(manifest["chord_guide_midi_path"]).exists())
            self.assertTrue((tmp_path / "review" / "review_manifest.json").exists())
            self.assertTrue((tmp_path / "review" / "review_candidates.md").exists())

    def test_overlap_free_solo_notes_trims_to_next_onset(self) -> None:
        notes = [
            pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0),
            pretty_midi.Note(velocity=82, pitch=62, start=0.5, end=1.5),
            pretty_midi.Note(velocity=70, pitch=64, start=1.0, end=1.25),
        ]

        solo_notes, report = overlap_free_solo_notes(notes)

        self.assertEqual(len(solo_notes), 3)
        self.assertLessEqual(solo_notes[0].end, solo_notes[1].start)
        self.assertLessEqual(solo_notes[1].end, solo_notes[2].start)
        self.assertEqual(report["trimmed_note_count"], 2)
        self.assertEqual(report["after_max_simultaneous_notes"], 1)

    def test_build_review_export_can_write_overlap_free_variant(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_midi = tmp_path / "source.mid"
            midi = pretty_midi.PrettyMIDI(initial_tempo=124)
            instrument = pretty_midi.Instrument(program=0, name="Solo")
            instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
            instrument.notes.append(pretty_midi.Note(velocity=80, pitch=62, start=0.5, end=1.5))
            midi.instruments.append(instrument)
            midi.write(str(source_midi))
            samples = {
                "data_motif": [
                    {
                        "sample_index": 1,
                        "sample_seed": 17,
                        "valid": True,
                        "strict_valid": True,
                        "midi_path": str(source_midi),
                        "metrics": {"note_count": 2, "unique_pitch_count": 2, "dead_air_ratio": 0.0},
                        "rhythm_profile": {
                            "syncopated_onset_ratio": 0.0,
                            "unique_bar_position_pattern_ratio": 1.0,
                            "duration_diversity_ratio": 0.5,
                            "most_common_duration_ratio": 0.5,
                            "ioi_diversity_ratio": 0.5,
                            "most_common_ioi_ratio": 0.5,
                        },
                        "pitch_roles": {"tension_ratio": 0.0, "root_tone_ratio": 0.0},
                    }
                ]
            }

            manifest = build_review_export(
                samples,
                output_dir=tmp_path / "review",
                top_n=1,
                copy_midi=True,
                chords=["Cm7"],
                bpm=124,
                bars=1,
                overlap_free_review_midi=True,
            )

            candidate = manifest["candidates"][0]
            exported = pretty_midi.PrettyMIDI(candidate["review_midi_path"])
            exported_notes = exported.instruments[0].notes

            self.assertEqual(candidate["review_variant"], "overlap_free_solo_line")
            self.assertTrue(candidate["review_midi_path"].endswith("_overlap_free.mid"))
            self.assertEqual(candidate["review_postprocess_report"]["after_max_simultaneous_notes"], 1)
            self.assertLessEqual(exported_notes[0].end, exported_notes[1].start)


if __name__ == "__main__":
    unittest.main()
