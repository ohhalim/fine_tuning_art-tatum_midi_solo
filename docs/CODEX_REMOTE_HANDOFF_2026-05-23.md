# Codex Remote Handoff

작성일: 2026-05-23

이 문서는 새 Codex 세션이나 Codex 앱의 원격 작업 환경이 이 저장소를 바로 이해하고 이어서 작업할 수 있도록 만든 handoff 문서다.

먼저 이 파일을 읽고, 더 자세한 배경이 필요하면 `docs/CORE_PLAN.md`와 `docs/CURRENT_STATUS_AND_PLAN.md`를 보면 된다.

## 0. One-Line Summary

이 저장소는 **symbolic MIDI 기반 jazz piano solo generation pipeline**을 만드는 중이다.

지금 목표는 제품/백엔드/라이브 플러그인이 아니라, **코드 진행 위에서 review 가능한 jazz piano solo-line MIDI를 안정적으로 만드는 model-core MVP**다.

## 1. Current Repository State

현재 기준 브랜치:

```text
main
```

최신 handoff 문서 커밋:

```text
9fa7470 docs: Codex 원격 handoff 문서 추가
```

최신 기능 merge:

```text
685ce5b Merge pull request #112 from ohhalim/issue-111-stage-b-clean-context-diagnostics
```

최근 완료된 PR:

- PR #110: `Stage B objective clean 리뷰 패키지 추가`
  - Issue #109 closed
  - merge commit: `4168397a9f26756d85e6824823acfb518806b238`
- PR #112: `Stage B clean context 진단 리포트 추가`
  - Issue #111 closed
  - merge commit: `685ce5b2826788e979c3a529fb75e74b56fed889`

새 환경에서는 먼저 최신 main을 기준으로 시작한다.

```bash
git switch main
git pull --ff-only origin main
```

## 2. Project Goal

최종 목표:

```text
house/techno/dance groove 위에서 쓸 수 있는 jazz piano solo MIDI generator
```

장기 입력:

- BPM
- chord progression
- section
- energy
- density
- optional recent MIDI context

장기 출력:

- 1-2 bar 또는 4-8 bar jazz piano solo MIDI
- FL Studio, Ableton, piano VST, future live controller에서 사용 가능해야 함

하지만 지금은 최종 제품 단계가 아니다.

현재 MVP:

```text
valid하고 review 가능한 symbolic MIDI jazz solo-line 후보를 생성/검증하는 pipeline
```

## 3. Important Strategic Decision

MIDI를 계속 사용한다.

최근 Live Music Diffusion Models, LMDM 논문을 장기 참고문헌으로 검토했지만, 현재 작업을 audio diffusion으로 pivot하지 않는다.

LMDM에서 참고할 개념:

- block-wise generation
- sliding context
- live input/output scheduling
- long-horizon drift control

하지만 지금 이 저장소의 핵심은:

- symbolic MIDI
- chord-aware phrase generation
- note-level diagnostics
- reviewable MIDI
- jazz phrase vocabulary

현재 하지 말 것:

- raw audio diffusion 구현
- LMDM repo 환경 세팅
- JUCE/ONNX/live plugin
- Spring Boot/API/backend MVP
- SaaS/UI
- broad training without reviewable MIDI

## 4. Why Stage A Failed

Stage A `control_v1`은 pipeline은 돌았지만 musical output이 실패했다.

문제:

- note count가 너무 적음
- one-note/two-note collapse
- 긴 sustain block
- chord block처럼 보이는 출력
- solo-line으로 볼 수 없는 구조

결론:

```text
Stage A를 더 세게 postprocess하지 않는다.
Stage B symbolic representation으로 넘어간다.
```

## 5. Why Stage B Exists

Stage B는 REMI/Jazz Transformer 계열 판단을 따른다.

핵심 representation:

- `BAR`
- `POSITION`
- `CHORD_ROOT`
- `CHORD_QUALITY`
- `NOTE_PITCH`
- `NOTE_DURATION`
- `VELOCITY`
- tempo/role control

핵심 판단:

```text
현재 문제는 Transformer architecture 자체보다 representation, phrase window, chord/context control, review gate 문제다.
```

## 6. What Has Been Built

대략적인 완료 흐름:

1. dataset audit
2. Brad Mehldau subset audit
3. Stage A training/generation probe
4. Stage A failure review
5. Stage B tokenization spec
6. Stage B phrase/window dataset
7. Stage B vocab/model training path
8. Stage B generation/decode probe
9. grammar-constrained generation
10. overlap/dedup gate
11. multi-sample review gate
12. collapse diagnostics
13. coverage-aware generation
14. candidate ranking
15. chord-aware pitch control
16. longer 4-bar and 8-bar phrase probes
17. swing/motif phrase grammar
18. real phrase reference statistics
19. data-derived motif extraction
20. data-motif generation baseline
21. context MIDI export
22. chord-labeled evaluation bridge
23. objective MIDI note review
24. objective flags review flow
25. overlap-free review MIDI
26. duration variation
27. phrase/cadence baseline
28. phrase naturalness metrics
29. phrase recovery baseline
30. data motif phrase recovery baseline
31. objective clean review package
32. clean context diagnostics
33. clean listening review notes template
34. clean MIDI-note proxy review
35. contour/cadence landing repair probe
36. contour repair MIDI-note proxy review
37. rhythm/phrase vocabulary variation probe
38. rhythm/phrase variation MIDI-note proxy review
39. rhythm/phrase variation sample diversity repair
40. sample-diverse rhythm variation MIDI-note proxy review

자세한 전체 기록은 `docs/CORE_PLAN.md`에 있다.

## 7. Latest Meaningful Result

최신 의미 있는 결과는 Stage B sample-diverse rhythm variation MIDI-note proxy review다.

Issue #124는 Issue #122에서 sample diversity를 고친 rhythm variation 후보를 다시 proxy review로 채웠다.

중요한 경계:

```text
이것은 실제 오디오 청취 리뷰가 아니다.
MIDI-note / piano-roll proxy review이며, 최종 subjective jazz quality proof가 아니다.
```

결과:

- candidate count: `6`
- reviewed count: `6`
- pending count: `0`
- decisions:
  - `needs_followup`: `6`
  - `reject`: `0`
  - `keep`: `0`
- timing:
  - `too_stiff`: `6`
- duplicate note sequences: `0`
- aggregate follow-ups:
  - `improve_phrase_vocabulary`: `14`
  - `fix_timing_grid`: `12`
  - `increase_motif_variation`: `6`

핵심 발견:

- sample diversity repair는 유효하다.
- exact duplicate 문제는 더 이상 현재 병목이 아니다.
- 새 variation 후보들도 no-keep이고 timing-stiff/template-mechanical 문제가 남았다.
- 다음 작업은 timing-grid repetition repair다.

Docs:

```text
docs/STAGE_B_SAMPLE_DIVERSE_RHYTHM_PROXY_REVIEW_2026-05-25.md
```

The previous probe was Stage B rhythm/phrase variation sample diversity repair.

Issue #122는 Issue #120 proxy review에서 확인된 exact duplicate rank candidate 문제를 고쳤다.

결과:

- candidate count: `6`
- unique note sequences: `6`
- duplicate note sequences: `0`
- objective MIDI flag counts: `{}`
- `data_motif_rhythm_phrase_variation`:
  - strict: `3/3`
  - final landing resolved: `3/3`
  - max interval: `6`
  - avg syncopation: `0.697`
  - avg duration diversity: `0.106`
  - avg IOI diversity: `0.108`
  - avg most-common IOI ratio: `0.497`

Implemented:

- sample seed now affects rhythm template row, contour template row, slot boundary, duration variation, pitch-cell selection, and approach target.
- review manifest records `note_sequence_signature`, `is_duplicate_note_sequence`, and `duplicate_of_candidate_id`.
- review markdown shows duplicate status.

Docs:

```text
docs/STAGE_B_RHYTHM_VARIATION_SAMPLE_DIVERSITY_2026-05-25.md
```

The previous review was Stage B rhythm/phrase variation MIDI-note proxy review.

Issue #120은 Issue #118 variation 후보와 contour repair baseline 후보를 같은 listening review notes schema로 채웠다.

중요한 경계:

```text
이것은 실제 오디오 청취 리뷰가 아니다.
MIDI-note / piano-roll proxy review이며, 최종 subjective jazz quality proof가 아니다.
```

결과:

- candidate count: `6`
- reviewed count: `6`
- pending count: `0`
- decisions:
  - `needs_followup`: `4`
  - `reject`: `2`
  - `keep`: `0`
- timing:
  - `too_stiff`: `6`
- aggregate follow-ups:
  - `fix_timing_grid`: `12`
  - `improve_phrase_vocabulary`: `10`
  - `increase_motif_variation`: `6`

핵심 발견:

- `data_motif_rhythm_phrase_variation_rank_1_sample_1`은 register floor, max interval, large-leap metrics가 개선된 representative follow-up candidate다.
- `data_motif_rhythm_phrase_variation_rank_2_sample_2`와 `rank_3_sample_3`는 rank 1과 MIDI note/start/duration sequence가 완전히 동일해서 duplicate review evidence로 reject한다.
- 다음 작업은 broad training이 아니라 variation mode의 sample-level diversity repair다.

Docs:

```text
docs/STAGE_B_RHYTHM_PHRASE_VARIATION_MIDI_PROXY_REVIEW_2026-05-25.md
```

The previous probe was Stage B rhythm/phrase vocabulary variation.

Issue #118은 Issue #116 contour repair MIDI-note proxy review에서 나온 `too_stiff=6`, `too_mechanical=6`, `too_repetitive=6` 문제를 좁혀서 검증했다.

결과:

- compared modes:
  - `data_motif_contour_landing_repair`
  - `data_motif_rhythm_phrase_variation`
- candidate count: `6`
- `data_motif_rhythm_phrase_variation`:
  - strict: `3/3`
  - final landing resolved: `3/3`
  - max interval: `6`
  - objective MIDI flags: `{}`
  - unresolved large leap ratio: `0.000`
  - repeated pitch interval ratio: `0.000`
  - pitch range floor: `>=51`
  - syncopation: `0.694`
  - duration diversity: `0.097`
  - IOI diversity: `0.115`
- comparison `data_motif_contour_landing_repair`:
  - max interval: `7`
  - syncopation: `0.625`
  - duration diversity: `0.062`
  - IOI diversity: `0.079`

Docs:

```text
docs/STAGE_B_RHYTHM_PHRASE_VARIATION_2026-05-25.md
```

The previous review was Stage B contour repair MIDI-note proxy review.

Issue #116은 Issue #115 이후 repair-vs-baseline 후보 6개를 같은 listening review notes schema로 채웠다.

중요한 경계:

```text
이것은 실제 오디오 청취 리뷰가 아니다.
MIDI-note / piano-roll proxy review이며, 최종 subjective jazz quality proof가 아니다.
```

결과:

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
- issue counts:
  - `bad_timing`: `6`
  - `too_mechanical`: `6`
  - `too_repetitive`: `6`
  - `weak_phrase`: `5`

Docs:

```text
docs/STAGE_B_CONTOUR_REPAIR_MIDI_PROXY_REVIEW_2026-05-25.md
```

The previous probe was Stage B data-derived contour/cadence landing repair.

이 probe는 2026-05-24 clean MIDI-note proxy review에서 드러난 contour/landing 문제를 좁혀서 검증했다.

결과:

- compared modes:
  - `data_motif_contour_landing_repair`
  - `data_motif_phrase_recovery`
- candidate count: `6`
- `data_motif_contour_landing_repair`:
  - strict: `3/3`
  - final landing resolved: `3/3`
  - max interval: `7`
  - abrupt register resets: `0`
  - objective MIDI flags: `{}`
- `data_motif_phrase_recovery` comparison:
  - strict: `3/3`
  - final landing resolved: `1/3`
  - max interval: `13`
- contour repair MIDI-note proxy review later filled those notes with `reviewed=6`, `keep=0`.

Docs:

```text
docs/STAGE_B_CONTOUR_LANDING_REPAIR_2026-05-25.md
```

이전 근거는 Issue #113 이후의 clean MIDI-note proxy review다.

Issue #109에서 objective-clean 후보 3개를 골랐다.

후보:

- `data_motif_phrase_recovery_rank_1_sample_1`
- `data_motif_phrase_recovery_rank_2_sample_2`
- `data_motif_phrase_recovery_rank_3_sample_3`

Issue #111에서 이 후보들을 MIDI note-level로 다시 진단했고, Issue #113에서 같은 schema로 review할 수 있는 clean listening review notes template을 만들었다.

2026-05-24 로컬 follow-up에서는 Codex가 MIDI note timing, pitch contour, context chord guide track을 읽어 proxy review를 작성했다.

중요한 경계:

```text
이것은 실제 오디오 청취 리뷰가 아니다.
MIDI-note / piano-roll proxy review이며, 최종 subjective jazz quality proof가 아니다.
```

결과:

- candidate count: `3`
- reviewed count: `3`
- pending count: `0`
- decision counts:
  - `needs_followup`: `2`
  - `reject`: `1`
  - `keep`: `0`
- all candidates:
  - note count: `63`
  - bar coverage: `8/8`
  - off-grid ratio: `0.000`
  - max duration: `1.000` beat
  - max simultaneous solo notes: `1`
  - context MIDI exists
  - chord guide exists
  - bass root guide exists

Candidate decisions:

| candidate | timing | chord_fit | phrase | landing | vocabulary | decision |
|---|---|---|---|---|---|---|
| `data_motif_phrase_recovery_rank_1_sample_1` | `stiff` | `acceptable` | `acceptable` | `acceptable` | `thin` | `needs_followup` |
| `data_motif_phrase_recovery_rank_2_sample_2` | `stiff` | `acceptable` | `weak` | `unresolved` | `thin` | `needs_followup` |
| `data_motif_phrase_recovery_rank_3_sample_3` | `stiff` | `acceptable` | `broken` | `unresolved` | `exercise_like` | `reject` |

해석:

```text
객관 진단상 clean이지만, proxy review 기준으로는 keep 후보가 없다.
```

중요:

```text
다음 병목은 basic MIDI validity가 아니라 rhythm stiffness, duration/IOI diversity 부족, thin vocabulary를 청취로 확인하는 일이다.
```

## 8. Generated Outputs Are Not Committed

`outputs/`는 생성 artifact이며 커밋하지 않는다.

원격 Codex 환경에서는 `outputs/`가 비어 있을 수 있다. 최신 결과를 다시 만들려면 아래를 실행한다.

```bash
bash scripts/agent_harness.sh stage-b-clean-review-package
bash scripts/agent_harness.sh stage-b-clean-context-diagnostics
bash scripts/agent_harness.sh stage-b-clean-listening-review-notes
```

생성될 주요 파일:

```text
outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package/clean_review_package.md
outputs/stage_b_clean_context_diagnostics/harness_stage_b_clean_context_diagnostics/clean_context_diagnostics.md
outputs/stage_b_clean_listening_review_notes/harness_stage_b_clean_listening_review_notes/clean_listening_review_notes_template.json
```

로컬에서 마지막으로 사용한 context MIDI 파일:

```text
outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package/context_midi/02_data_motif_phrase_recovery_rank_01_sample_01_overlap_free_with_context.mid
outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package/context_midi/02_data_motif_phrase_recovery_rank_02_sample_02_overlap_free_with_context.mid
outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package/context_midi/02_data_motif_phrase_recovery_rank_03_sample_03_overlap_free_with_context.mid
```

## 9. Key Files To Read First

Read these in order:

```text
docs/CODEX_REMOTE_HANDOFF_2026-05-23.md
docs/CORE_PLAN.md
docs/CURRENT_STATUS_AND_PLAN.md
AGENTS.md
```

Latest implementation files:

```text
scripts/run_stage_b_data_motif_generation_compare.py
scripts/build_clean_review_package.py
scripts/build_clean_context_diagnostics.py
scripts/build_clean_listening_review_notes.py
scripts/agent_harness.sh
tests/test_stage_b_data_motif_generation_compare.py
tests/test_clean_review_package.py
tests/test_clean_context_diagnostics.py
tests/test_clean_listening_review_notes.py
```

Latest result docs:

```text
docs/STAGE_B_CLEAN_REVIEW_PACKAGE_2026-05-23.md
docs/STAGE_B_CLEAN_CONTEXT_DIAGNOSTICS_2026-05-23.md
docs/STAGE_B_CLEAN_LISTENING_REVIEW_NOTES_2026-05-23.md
docs/STAGE_B_CLEAN_MIDI_PROXY_REVIEW_2026-05-24.md
docs/STAGE_B_CONTOUR_LANDING_REPAIR_2026-05-25.md
docs/STAGE_B_CONTOUR_REPAIR_MIDI_PROXY_REVIEW_2026-05-25.md
docs/STAGE_B_RHYTHM_PHRASE_VARIATION_MIDI_PROXY_REVIEW_2026-05-25.md
docs/STAGE_B_RHYTHM_VARIATION_SAMPLE_DIVERSITY_2026-05-25.md
docs/STAGE_B_SAMPLE_DIVERSE_RHYTHM_PROXY_REVIEW_2026-05-25.md
docs/STAGE_B_RHYTHM_PHRASE_VARIATION_2026-05-25.md
```

Core historical docs:

```text
docs/STAGE_B_DATA_MOTIF_PHRASE_RECOVERY_2026-05-22.md
docs/STAGE_B_PHRASE_RECOVERY_REVIEW_2026-05-22.md
docs/STAGE_B_PHRASE_NATURALNESS_OBJECTIVES_2026-05-22.md
docs/STAGE_B_REVIEW_CONTEXT_GRID_2026-05-22.md
docs/STAGE_B_DATA_MOTIF_GENERATION_2026-05-21.md
docs/REFERENCES.md
```

## 10. Required Commands

Always run quick before committing:

```bash
bash scripts/agent_harness.sh quick
```

For the latest clean review package:

```bash
bash scripts/agent_harness.sh stage-b-clean-review-package
```

For the latest clean context diagnostics:

```bash
bash scripts/agent_harness.sh stage-b-clean-context-diagnostics
```

For the latest clean listening review notes template:

```bash
bash scripts/agent_harness.sh stage-b-clean-listening-review-notes
```

For the latest contour/cadence landing repair probe:

```bash
bash scripts/agent_harness.sh stage-b-contour-landing-repair
```

Expected quick result:

```text
Ran 222 tests
OK
```

CUDA warning is expected on local CPU/Mac environments:

```text
WARNING: CUDA devices not detected
```

This warning is not a failure for these harnesses.

## 11. Quality Gate

Never say a MIDI output succeeded just because a `.mid` file exists.

Minimum checks:

- note count is not tiny
- enough unique pitches
- requested bars are covered
- no long sustain block
- no chord-block/polyphonic review artifact
- no one-note/two-note collapse
- no extreme dead-air
- grid/timing is explainable
- context MIDI exists when listening review needs chord context

Current latest contour/landing repair candidates pass the objective MIDI gate, but MIDI-note proxy review still produced no `keep` candidate.
Current latest rhythm/phrase variation candidates pass the objective MIDI gate and are no longer exact duplicate note sequences, but sample-diverse MIDI-note proxy review still produced no `keep` candidate.

## 12. What To Do Next

The next correct task is **not** broad training, audio diffusion, or backend work.

Next step after the sample-diverse rhythm variation MIDI-note proxy review:

```text
Stage B rhythm variation timing-grid repetition repair
```

The next probe should target:

- reducing most-common IOI ratio while preserving objective-clean gate
- avoiding long deterministic rest/onset template cells
- keeping duplicate note sequence count at `0`
- preserving final guide landing and max interval bound

Recommended next issue title:

```text
Stage B rhythm variation timing-grid repetition repair
```

## 13. Do Not Do Next

Do not do these unless the user explicitly changes scope:

- do not start Spring Boot
- do not build API/backend
- do not build UI/SaaS
- do not implement LMDM/audio diffusion
- do not install heavy CUDA/audio diffusion dependencies
- do not upload datasets/checkpoints/generated outputs
- do not commit `outputs/`
- do not claim Brad Mehldau style model works
- do not claim this is a reliable jazz improviser
- do not use exact artist clone wording in public docs

Use safer wording:

```text
symbolic MIDI jazz solo generation probe
style-conditioned generation
personalized improvisation research direction
```

## 14. How To Explain Current Status

Short explanation:

```text
We are building a symbolic MIDI jazz piano solo generation pipeline.
Stage A failed musically, so the project moved to Stage B with explicit bar/position/chord/duration tokens.
The latest work added a rhythm/phrase variation probe after contour repair proxy review.
Objective note-level diagnostics show no flags, and rhythm variation improves syncopation, duration diversity, and IOI diversity while preserving resolved landings.
The next step is MIDI-note proxy review of those variation candidates, not broad training.
```

Very short explanation:

```text
The pipeline can now improve contour/landing and some rhythm metrics, but the new variation candidates still need MIDI-note proxy review before more generation changes.
```

## 15. Remote Codex Working Rules

When continuing work:

1. Start from latest `main`.
2. Create one issue per focused task.
3. Create one branch per issue.
4. Make small commits with Korean commit messages.
5. Run the relevant harness before commits.
6. Keep generated artifacts out of git.
7. Update docs honestly.
8. Push/PR/merge only according to the active `AGENTS.md` and user permission rules in the current environment.

Recommended branch naming:

```text
issue-<number>-stage-b-<short-scope>
```

Recommended commit style:

```text
feat: Stage B contour/cadence landing repair 추가
docs: Stage B MIDI-note proxy review 결과 기록
```

## 16. Critical Handoff Warning

The project has reached a review boundary.

More objective scripts can be written, but they should not pretend to answer the musical question:

```text
Does this sound like a jazz piano phrase?
```

That question requires listening review with chord context.

The code can help organize the review and measure obvious failures. It should not falsely claim subjective jazz quality.
