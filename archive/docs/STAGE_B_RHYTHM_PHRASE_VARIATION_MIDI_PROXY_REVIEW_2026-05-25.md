# Stage B Rhythm/Phrase Variation MIDI Proxy Review

작성일: 2026-05-25

## 목적

Issue #120은 Issue #118 rhythm/phrase variation 후보를 같은 listening review notes schema로 채운 proxy review다.

중요한 경계:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note timing, pitch contour, objective MIDI metrics, context chord guide track, exact note-sequence comparison을 기준으로 한 proxy review다.
- `keep` 후보를 만들기 위한 단계가 아니라, 다음 generation rule 병목을 정리하는 단계다.

## 입력

Review 대상:

- `data_motif_contour_landing_repair` 후보 3개
- `data_motif_rhythm_phrase_variation` 후보 3개

사용한 생성 artifact:

- review notes template:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_rhythm_phrase_variation/review_notes_template.json`
- review candidates:
  - `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_candidates.md`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_rhythm_phrase_variation/objective_midi_note_review.json`
- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_rhythm_phrase_variation_codex_proxy/rhythm_phrase_variation_review_notes_codex_midi_proxy.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_rhythm_phrase_variation_codex_proxy/listening_review_aggregate.md`

`outputs/`는 생성 artifact이므로 커밋하지 않는다.

## MIDI Note Inspection

`data_motif_rhythm_phrase_variation` 후보의 개선점:

- pitch range: `51-80`
- large leap ratio: `0.000`
- unresolved large leap ratio: `0.000`
- objective bucket: `clean`
- objective flags: `{}`
- unique duration count: `5`
- most common duration ratio: `0.467`
- bar position count: `16`

Contour repair baseline 대비 개선:

- low-register floor artifact가 사라졌다.
- max interval이 `7`에서 `6`으로 줄었다.
- duration/IOI diversity가 소폭 증가했다.

하지만 가장 중요한 발견은 sample diversity failure다.

`data_motif_rhythm_phrase_variation_rank_1_sample_1`, `rank_2_sample_2`, `rank_3_sample_3`는 MIDI note/start/duration sequence가 완전히 동일했다.

따라서 variation mode는 objective profile은 개선했지만, ranked review package가 독립 후보 3개를 제공하지 못했다.

## Review Result

Decision counts:

| decision | count |
|---|---:|
| `keep` | 0 |
| `needs_followup` | 4 |
| `reject` | 2 |

Candidate decisions:

| candidate | phrase | timing | chord fit | decision | reason |
|---|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` | objective-clean, but drops into C1-A1 register artifact |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `phrase` | `too_stiff` | `fits` | `needs_followup` | strongest contour-repair baseline, still rigid |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` | one unresolved large leap and weaker contour |
| `data_motif_rhythm_phrase_variation_rank_1_sample_1` | `phrase` | `too_stiff` | `fits` | `needs_followup` | best representative of variation mode |
| `data_motif_rhythm_phrase_variation_rank_2_sample_2` | `phrase` | `too_stiff` | `fits` | `reject` | exact duplicate of rank 1 |
| `data_motif_rhythm_phrase_variation_rank_3_sample_3` | `phrase` | `too_stiff` | `fits` | `reject` | exact duplicate of rank 1 |

Aggregate follow-up signals:

| code | count | interpretation |
|---|---:|---|
| `fix_timing_grid` | 12 | candidates still read as grid-stiff |
| `improve_phrase_vocabulary` | 10 | phrase/exercise/mechanical issues remain |
| `increase_motif_variation` | 6 | duplicate/repetitive candidate evidence remains |

## 해석

Issue #118의 rhythm/phrase variation은 실패가 아니다.

좋아진 점:

- register floor가 훨씬 안전해졌다.
- large leap / unresolved large leap 문제가 사라졌다.
- duration/IOI objective diversity가 baseline보다 좋아졌다.
- chord/tension profile은 objective-clean 범위에 남았다.

하지만 musical `keep` 후보는 아직 없다.

남은 병목:

- timing은 여전히 sixteenth-grid deterministic pattern으로 들을 위험이 높다.
- phrase vocabulary는 아직 mechanical하다.
- candidate rank 1-3이 모두 같은 note sequence라 review package의 sample diversity가 깨졌다.

따라서 다음 단계는 broad training이나 backend/UI가 아니라 variation mode의 sample-level diversity를 고치는 것이다.

## Decision

Issue #120 결론:

- `data_motif_rhythm_phrase_variation_rank_1_sample_1`은 다음 probe의 representative candidate로 남긴다.
- `rank_2`와 `rank_3`는 objective-clean이지만 exact duplicate이므로 review evidence로는 reject한다.
- 다음 generation issue는 template seed/candidate selection diversity를 고쳐야 한다.

Recommended next issue:

```text
Stage B rhythm/phrase variation sample diversity repair
```

목표:

- variation candidate rank 1-3이 exact duplicate이 되지 않게 한다.
- sample seed가 rhythm slot boundary, motif template selection, pitch contour, cadence target 중 최소 하나에 반영되도록 한다.
- duplicate sequence를 review export/ranking에서 감지한다.
- objective-clean gate를 유지하면서 independent review candidates를 만든다.

## 검증

실행한 검증:

```bash
.venv/bin/python scripts/build_listening_review_notes.py --run_id harness_stage_b_rhythm_phrase_variation_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_rhythm_phrase_variation_codex_proxy/rhythm_phrase_variation_review_notes_codex_midi_proxy.json
.venv/bin/python scripts/summarize_listening_review_notes.py --run_id harness_stage_b_rhythm_phrase_variation_codex_proxy --review_notes outputs/stage_b_listening_review_notes/harness_stage_b_rhythm_phrase_variation_codex_proxy/rhythm_phrase_variation_review_notes_codex_midi_proxy.json
```

Commit 전 필수 quick harness도 실행한다.
