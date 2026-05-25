# Stage B Contour Repair MIDI-Note Proxy Review

작성일: 2026-05-25

## 목적

Issue #115의 `data_motif_contour_landing_repair` 후보 3개와 `data_motif_phrase_recovery` baseline 후보 3개를
같은 listening review notes schema로 채우고, 다음 generation rule 방향을 정리한다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note timing, pitch contour, objective MIDI metrics, context chord guide track을 읽은 proxy review다.
- 따라서 `keep` 후보를 만들지 않는다.
- 이 문서는 broad training 또는 Brad style adaptation 성공 근거가 아니다.

## 입력

Review notes template:

```text
outputs/stage_b_listening_review_notes/harness_stage_b_contour_landing_repair/review_notes_template.json
```

Proxy-filled notes:

```text
outputs/stage_b_listening_review_notes/harness_stage_b_contour_landing_repair_codex_proxy/contour_repair_listening_review_notes_codex_midi_proxy.json
```

Aggregate:

```text
outputs/stage_b_listening_review_aggregate/harness_stage_b_contour_landing_repair_codex_proxy/listening_review_aggregate.md
```

대상 후보:

- `data_motif_contour_landing_repair_rank_1_sample_1`
- `data_motif_contour_landing_repair_rank_2_sample_2`
- `data_motif_contour_landing_repair_rank_3_sample_3`
- `data_motif_phrase_recovery_rank_1_sample_1`
- `data_motif_phrase_recovery_rank_2_sample_2`
- `data_motif_phrase_recovery_rank_3_sample_3`

## Objective Context

모든 후보는 objective MIDI review 기준으로 clean이다.

| mode | candidates | objective flags | max interval | final landing |
|---|---:|---|---:|---|
| `data_motif_contour_landing_repair` | 3 | `{}` | 7 | `guide`, `guide`, `guide` |
| `data_motif_phrase_recovery` | 3 | `{}` | 12-13 | `guide`, `tension`, `tension` |

공통 조건:

- note count: `63`
- max active solo notes: `1`
- off-grid count: `0`
- repeated pitch interval ratio: `0.000`
- objective bucket: `clean`

이것은 성공 판정이 아니다. 의미는 "기본 MIDI artifact 없이 review할 수 있다"는 정도다.

## Proxy Review Result

Aggregate:

- candidate count: `6`
- reviewed count: `6`
- pending count: `0`
- decisions:
  - `needs_followup`: `5`
  - `reject`: `1`
  - `keep`: `0`
- phrase quality:
  - `phrase`: `1`
  - `fragment`: `4`
  - `exercise`: `1`
- timing:
  - `too_stiff`: `6`
- chord fit:
  - `fits`: `4`
  - `unclear`: `2`
- issue counts:
  - `bad_timing`: `6`
  - `too_mechanical`: `6`
  - `too_repetitive`: `6`
  - `weak_phrase`: `5`
  - `other`: `2`

Candidate decisions:

| candidate | phrase | timing | chord_fit | decision | proxy note |
|---|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` | objective landing works, but the line falls into a very low C1-A#1 register |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `phrase` | `too_stiff` | `fits` | `needs_followup` | strongest repaired candidate; guide landing and max interval 7, but rhythm remains rigid |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | guide landing works, but one unresolved large leap remains and contour is weaker than rank 2 |
| `data_motif_phrase_recovery_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` | previous baseline; guide landing but many octave-like jumps and max interval 13 |
| `data_motif_phrase_recovery_rank_2_sample_2` | `fragment` | `too_stiff` | `unclear` | `needs_followup` | reaches F6 and ends on tension; useful negative comparison for landing repair |
| `data_motif_phrase_recovery_rank_3_sample_3` | `exercise` | `too_stiff` | `unclear` | `reject` | extreme F3-G#6 range, high-register spike, final tension landing |

## Interpretation

`data_motif_contour_landing_repair` improved the specific contour/landing target.

- rank 2 is the strongest current candidate by MIDI-note proxy review.
- all repaired candidates end on guide tones.
- repaired candidates keep max interval at `7`, while the baseline reaches `12-13`.
- repaired candidates avoid the worst high-register spikes of `data_motif_phrase_recovery`.

But the review still does not produce a `keep` candidate.

Remaining blockers:

- all candidates are `too_stiff`
- all candidates are `too_mechanical`
- all candidates are `too_repetitive`
- duration/IOI template is still nearly fixed
- repaired rank 1 drops into an unnaturally low register
- repaired rank 3 still has one unresolved large-leap objective signal

Aggregate recommended follow-ups:

| code | count | interpretation |
|---|---:|---|
| `improve_phrase_vocabulary` | 16 | fragments/exercise/mechanical phrase readings dominate |
| `fix_timing_grid` | 12 | timing remains stiff; do not add loose swing before fixing template rigidity |
| `increase_motif_variation` | 6 | repeated rhythm/motif template remains a consistent issue |

## Decision

The next task should not be another landing repair.

Recommended next issue:

```text
Stage B rhythm/phrase vocabulary variation probe
```

The next probe should test:

- more varied data-derived position/duration template selection
- phrase-level motif memory that avoids repeating the same onset/duration skeleton
- register bounds that prevent repaired phrases from falling into C1/A#1 solo register
- whether the strongest repaired candidate can stay as a comparison baseline

Still do not:

- move to broad training from these candidates
- claim a reliable jazz improviser
- pivot to audio diffusion or backend work
- commit generated MIDI artifacts under `outputs/`

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py \
  --run_id harness_stage_b_contour_landing_repair_codex_proxy \
  --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_contour_landing_repair_codex_proxy/contour_repair_listening_review_notes_codex_midi_proxy.json

.venv/bin/python scripts/summarize_listening_review_notes.py \
  --run_id harness_stage_b_contour_landing_repair_codex_proxy \
  --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_contour_landing_repair_codex_proxy/contour_repair_listening_review_notes_codex_midi_proxy.json

bash scripts/agent_harness.sh quick
```

Validation result:

```json
{
  "candidate_count": 6,
  "reviewed_count": 6,
  "pending_count": 0,
  "decision_counts": {
    "needs_followup": 5,
    "reject": 1,
    "keep": 0
  }
}
```
