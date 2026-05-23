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

최신 main merge:

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

자세한 전체 기록은 `docs/CORE_PLAN.md`에 있다.

## 7. Latest Meaningful Result

최신 의미 있는 결과는 Issue #111이다.

Issue #109에서 objective-clean 후보 3개를 골랐다.

후보:

- `data_motif_phrase_recovery_rank_1_sample_1`
- `data_motif_phrase_recovery_rank_2_sample_2`
- `data_motif_phrase_recovery_rank_3_sample_3`

Issue #111에서 이 후보들을 MIDI note-level로 다시 진단했다.

결과:

- candidate count: `3`
- diagnostic flags: none
- decision hint:
  - `listen_with_context`: `3`
- all candidates:
  - note count: `63`
  - bar coverage: `8/8`
  - off-grid ratio: `0.000`
  - max duration: `1.000` beat
  - context MIDI exists
  - chord guide exists
  - bass root guide exists

해석:

```text
객관 진단상 지금 후보 3개는 더 자동으로 거르기보다 들어볼 단계다.
```

중요:

```text
이것은 jazz quality 성공이 아니다.
이제 병목은 subjective listening review다.
```

## 8. Generated Outputs Are Not Committed

`outputs/`는 생성 artifact이며 커밋하지 않는다.

원격 Codex 환경에서는 `outputs/`가 비어 있을 수 있다. 최신 결과를 다시 만들려면 아래를 실행한다.

```bash
bash scripts/agent_harness.sh stage-b-clean-review-package
bash scripts/agent_harness.sh stage-b-clean-context-diagnostics
```

생성될 주요 파일:

```text
outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package/clean_review_package.md
outputs/stage_b_clean_context_diagnostics/harness_stage_b_clean_context_diagnostics/clean_context_diagnostics.md
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
scripts/build_clean_review_package.py
scripts/build_clean_context_diagnostics.py
scripts/agent_harness.sh
tests/test_clean_review_package.py
tests/test_clean_context_diagnostics.py
```

Latest result docs:

```text
docs/STAGE_B_CLEAN_REVIEW_PACKAGE_2026-05-23.md
docs/STAGE_B_CLEAN_CONTEXT_DIAGNOSTICS_2026-05-23.md
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

Expected quick result:

```text
Ran 213 tests
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

Current latest candidates pass the objective gate, but still need listening review.

## 12. What To Do Next

The next correct task is **not** to add another generation rule immediately.

Next step:

```text
Create or fill a clean listening review notes package for the 3 context MIDI candidates.
```

The review should classify each candidate:

- timing:
  - `good`
  - `too_loose`
  - `too_straight`
  - `unclear`
- chord fit:
  - `fits`
  - `too_safe`
  - `too_outside`
  - `unclear`
- phrase continuation:
  - `phrase_like`
  - `fragmented`
  - `exercise_like`
  - `unclear`
- landing:
  - `clear`
  - `weak`
  - `missing`
  - `unclear`
- jazz vocabulary:
  - `present`
  - `weak`
  - `absent`
  - `unclear`

If the 3 candidates sound acceptable:

```text
Move toward model-side phrase continuation / generic jazz base probe.
```

If they still sound beginner-like:

```text
Do not pivot to audio.
Do not add backend.
Strengthen data-derived phrase/cadence vocabulary.
```

Recommended next issue title:

```text
Stage B clean listening review notes 추가
```

Alternative next issue if listening feedback says "still beginner-like":

```text
Stage B data-derived cadence vocabulary 추가
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
The latest work narrowed generated candidates down to 3 objective-clean, 8-bar, context-aware MIDI candidates.
Objective note-level diagnostics show no current blocker, so the next step is listening review for phrase quality.
```

Very short explanation:

```text
The pipeline can now produce objective-clean context MIDI candidates, but it has not yet proven real jazz phrase quality.
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
feat: Stage B clean listening review notes 추가
docs: Stage B listening review 결과 기록
```

## 16. Critical Handoff Warning

The project has reached a review boundary.

More objective scripts can be written, but they should not pretend to answer the musical question:

```text
Does this sound like a jazz piano phrase?
```

That question requires listening review with chord context.

The code can help organize the review and measure obvious failures. It should not falsely claim subjective jazz quality.
