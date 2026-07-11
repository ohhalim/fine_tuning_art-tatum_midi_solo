# Stage B Phrase Vocabulary/Motif Proxy Keep Focused Package

작성일: 2026-05-27

## 목적

Issue #176은 Issue #174 proxy keep 후보를 focused context review package로 분리한 작업이다.

중요한 경계:

- 이 패키지는 focused context review를 위한 입력이다.
- proxy `keep`은 final musical quality가 아니다.
- copied MIDI files는 `outputs/` artifact로만 두고 커밋하지 않는다.

## 입력

Source artifacts:

- filled review notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_phrase_vocab_motif_variation_proxy_review/phrase_vocab_motif_variation_repaired_review_notes.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_phrase_vocab_motif_variation_repair/objective_midi_note_review.json`

Generated package:

- package JSON:
  - `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/focused_review_package.json`
- package markdown:
  - `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/focused_review_package.md`
- copied solo MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free.mid`
- copied context MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_02_sample_02_overlap_free_with_context.mid`

## Result

Focused package summary:

- candidate count: `1`
- decision filter: `keep`
- copied MIDI files: `2`
- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- mode: `data_motif_rhythm_phrase_variation`
- sample seed: `18`
- valid: `true`
- strict valid: `true`
- review variant: `overlap_free_solo_line`

Candidate metrics:

| metric | value |
|---|---:|
| note count | `64` |
| unique pitch count | `18` |
| source syncopated onset ratio | `0.719` |
| source duration diversity ratio | `0.094` |
| source most-common duration ratio | `0.406` |
| source IOI diversity ratio | `0.095` |
| source most-common IOI ratio | `0.397` |
| source tension ratio | `0.344` |
| objective chord-tone ratio | `0.531` |
| objective tension ratio | `0.469` |
| objective stepwise interval ratio | `0.460` |
| objective unresolved large leap ratio | `0.000` |

Objective status:

- objective bucket: `clean`
- objective flags: `[]`
- objective penalty: `0`
- max active notes: `1`
- polyphonic tick ratio: `0.000`
- off-sixteenth-grid count: `0`

## 판단

Issue #176은 proxy keep 후보를 focused context review 가능한 단일 package로 격리했다.

이 단계에서 유지되는 장점:

- objective-clean solo-line
- safe register
- final guide landing
- copied solo/context MIDI pair
- objective first-note summary 포함

남은 위험:

- listening notes에 `too_mechanical` issue가 남아 있다.
- source tension ratio는 높지 않다.
- duration/grid trace가 아직 강하다.

## 다음 작업

`Stage B phrase vocabulary motif focused context decision`

목표:

- focused package의 solo/context MIDI를 기준으로 proxy keep을 유지할지 결정한다.
- focused context에서도 유지되면 next step은 focused listening review notes template이다.
- focused context에서 탈락하면 다음 repair는 timing grid와 phrase vocabulary를 다시 좁힌다.

## 검증

실행한 검증:

```bash
SOURCE_RUN_ID=harness_stage_b_phrase_vocab_motif_variation_proxy_review OBJECTIVE_RUN_ID=harness_stage_b_phrase_vocab_motif_variation_repair REVIEW_NOTES_FILE=phrase_vocab_motif_variation_repaired_review_notes.json RUN_ID=harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package bash scripts/agent_harness.sh stage-b-proxy-keep-focused-package
```
