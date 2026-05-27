# Stage B Focused Timing Vocabulary Proxy Keep Focused Package

작성일: 2026-05-27

## 목적

Issue #188은 Issue #186 proxy keep 후보를 focused context review package로 분리한 작업이다.

중요한 경계:

- 이 패키지는 focused context review를 위한 입력이다.
- proxy `keep`은 final musical quality가 아니다.
- copied MIDI files는 `outputs/` artifact로만 두고 커밋하지 않는다.

## 입력

Source artifacts:

- filled review notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_focused_timing_vocab_proxy_review/focused_timing_vocab_repaired_review_notes.json`
- objective MIDI review:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_focused_timing_vocab_followup_repair/objective_midi_note_review.json`

Generated package:

- package JSON:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_proxy_keep_focused_package/focused_review_package.json`
- package markdown:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_proxy_keep_focused_package/focused_review_package.md`
- copied solo MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_proxy_keep_focused_package/midi/02_data_motif_rhythm_phrase_variation_rank_03_sample_03_overlap_free.mid`
- copied context MIDI:
  - `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_proxy_keep_focused_package/context_midi/02_data_motif_rhythm_phrase_variation_rank_03_sample_03_overlap_free_with_context.mid`

## Result

Focused package summary:

- candidate count: `1`
- decision filter: `keep`
- copied MIDI files: `2`
- candidate: `data_motif_rhythm_phrase_variation_rank_3_sample_3`
- mode: `data_motif_rhythm_phrase_variation`
- sample seed: `19`
- valid: `true`
- strict valid: `true`
- review variant: `overlap_free_solo_line`

Candidate metrics:

| metric | value |
|---|---:|
| note count | `64` |
| unique pitch count | `20` |
| source syncopated onset ratio | `0.703` |
| source duration diversity ratio | `0.078` |
| source most-common duration ratio | `0.406` |
| source IOI diversity ratio | `0.095` |
| source most-common IOI ratio | `0.397` |
| source tension ratio | `0.297` |
| objective chord-tone ratio | `0.547` |
| objective tension ratio | `0.453` |
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

Issue #188은 proxy keep 후보를 focused context review 가능한 단일 package로 격리했다.

이 단계에서 유지되는 장점:

- objective-clean solo-line
- max interval `4`
- final guide landing
- copied solo/context MIDI pair
- objective first-note summary 포함
- wider pitch vocabulary than the previous proxy keep path

남은 위험:

- listening notes에 `too_mechanical` issue가 남아 있다.
- source tension ratio는 `0.297`로 낮다.
- duration/grid trace가 아직 강하다.
- focused context에서도 phrase continuation과 landing을 다시 확인해야 한다.

## 다음 작업

`Stage B focused timing vocabulary focused context decision`

목표:

- focused package의 solo/context MIDI를 기준으로 proxy keep을 유지할지 결정한다.
- focused context에서도 유지되면 next step은 focused listening review notes template이다.
- focused context에서 탈락하면 다음 repair는 timing grid와 phrase vocabulary를 다시 좁힌다.

## 검증

실행한 검증:

```bash
SOURCE_RUN_ID=harness_stage_b_focused_timing_vocab_proxy_review OBJECTIVE_RUN_ID=harness_stage_b_focused_timing_vocab_followup_repair REVIEW_NOTES_FILE=focused_timing_vocab_repaired_review_notes.json RUN_ID=harness_stage_b_focused_timing_vocab_proxy_keep_focused_package bash scripts/agent_harness.sh stage-b-proxy-keep-focused-package
```
