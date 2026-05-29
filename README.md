# Jazz Piano MIDI 생성 검증 파이프라인

## 개요

Symbolic MIDI 기반 jazz piano solo-line 생성 모델 개발을 위한 model-core 검증 프로젝트.

현재 범위:

- MIDI dataset audit
- Stage B duration-explicit tokenization
- tiny training / generation probe
- generated token MIDI decode
- objective MIDI review gate
- focused context / listening review package
- repeatability sweep 및 repair boundary

현재 상태:

- pipeline MVP: 완료
- model-core 검증: 진행 중
- broad trained-model quality: 미검증
- Brad style adaptation: 미진행
- realtime DAW/plugin, backend/API, SaaS/UI: 범위 밖

최신 판단:

- outside-soloing repair objective path 완료
- final boundary: `outside_soloing_repair_objective_path_complete`
- next boundary: `stage_b_model_core_evidence_readme_refresh`
- human/audio preference claim: `false`
- broad model quality claim: `false`

## 구현 범위

| 영역 | 구현 내용 |
|---|---|
| Dataset audit | MIDI corpus readable file, candidate file, Brad subset, duplicate hash 점검 |
| Stage B representation | `BAR`, `POSITION`, `CHORD_ROOT`, `CHORD_QUALITY`, `NOTE_PITCH`, `NOTE_DURATION`, `VELOCITY` 기반 token 구조 |
| Training / generation probe | tiny-overfit training path, raw generation probe, constrained generation probe |
| Decode / postprocess | generated token sequence MIDI 복원, overlap-free solo-line 후보 생성 |
| Objective MIDI review | note count, unique pitch, phrase coverage, dead-air, max active notes, repeated cell, max interval, chord/tension/outside ratio, final landing 검증 |
| Focused review | solo MIDI와 chord/bass context MIDI 분리, listening review notes schema 생성 |
| Audio review boundary | MIDI/WAV review package, review input guard, preference claim boundary 분리 |
| Repeatability / repair | seed/sample sweep, dead-air repair, phrase/vocabulary repair, duration/coverage fill, outside-soloing repair |
| Harness | unit test, compile check, whitespace check, Stage B probe 실행 모드 관리 |
| Docs | issue 단위 실험 결과, 관측 지표, 제외된 claim, 다음 boundary 기록 |

## 문제 / 해결 / 결과

| 문제 | 관측값 | 해결 | 결과 |
|---|---|---|---|
| `.mid` 파일 존재만으로 성공 판단 위험 | one-note collapse, long sustain block, chord block 출력 | `.mid exists` 성공 조건 제외, objective MIDI review 추가 | note-level gate 기반 실패 분리 |
| Stage A representation 한계 | `NOTE_ON/OFF` 중심 구조에서 duration/phrase 제어 어려움 | Stage B duration-explicit tokenization 전환 | `POSITION`, `NOTE_DURATION`, chord context 기반 probe 가능 |
| raw generation 불안정 | raw generated sample gate 실패, note count `3 < 6` | constrained generation과 review gate 분리 | `stage-b-overlap-gate` valid/strict/grammar `1/1` |
| seed-level margin 부족 | 6-file seed `17` strict `1/3` | candidate count margin recovery, 5 samples per seed | 6-file 5-sample strict `12/15`, warning seed 없음 |
| dead-air outlier | seed `31` sample `1`, dead-air `0.857` | dead-air diagnostics, candidate selection gate, duration/coverage fill | selected fill dead-air `0.5714 -> 0.2941` |
| pitch vocabulary 부족 | focused unique pitch `5`, keep 승격 실패 | seed/top-k sweep 및 phrase/vocabulary repair | qualified `2/96`, selected unique pitch `8`, max interval `7` |
| repeated cell / adjacent repeat | adjacent repeat, duplicated pitch-class chunk 발생 | pitch reuse 제한, repeated-cell metric, repair sweep | adjacent repeats `0`, duplicated 3-note chunks `0` |
| focused context 미검증 | objective gate 통과만으로 final landing 판단 불가 | solo/context package, chord/bass guide 검증 | max active `1`, final landing chord/tension 기록 |
| 단일 후보 과장 위험 | focused fill `keep`이 broad quality로 오해 가능 | claim boundary 분리 | `single_postprocess_candidate_keep_support`, broad quality `false` |
| duplicate output 위험 | selected/peer source는 다르지만 sample seed 동일 | note signature, metric fingerprint, sample seed audit | output diversity `absent`, distinct sample-seed repair로 이동 |
| distinct 후보 phrase/vocabulary 부족 | distinct candidate timing은 acceptable, phrase/vocabulary는 weak/thin | remaining blocker summary, constrained adjacent repair, duration/coverage fill | target-qualified `0/48`, duration fill qualified `2/4` |
| MIDI evidence와 청감 preference 혼동 | MIDI metric 우세가 human preference로 오해 가능 | MIDI evidence consolidation, external human/audio boundary, review input guard | MIDI evidence preference support, human/audio preference `false` |
| 사용자 청취 이후 repeatability 후보 난해함 | repeatability WAV 2개 모두 outside-soloing-like follow-up | pitch-role repair sweep, objective evidence consolidation | repaired source `2/2`, qualified variants `6/6` |
| outside-soloing repair 반복성 미확인 | 단일 repair 후보만으로 policy support 부족 | `chord_tone_snap`, `guide_tone_landing`, `contour_resolution` policy sweep | policy support `3/3`, variants qualified `6/6` |
| final claim boundary 필요 | objective support와 pending review 상태 분리 필요 | repeatability consolidation, final decision | `outside_soloing_repair_objective_path_complete`, preference claim `false` |

## 주요 검증 결과

| 항목 | 결과 |
|---|---|
| dataset readable files | `2777` |
| candidate files | `2775` |
| Brad candidate files | `72` |
| exact duplicate hash groups | `0` |
| Stage B window samples | `70` |
| train / val split | `63 / 7` |
| vocab size | `547` |
| raw generation repeatability | 2 files / 3 seeds / strict `8/9` |
| broader source gate | 3 files / strict `7/9`, dead-air outlier rate `0.222` |
| candidate count recovery | 6 files / 5 samples per seed / strict `12/15` |
| phrase/vocabulary repair | qualified `2/96`, selected/peer keep `2` |
| duration/coverage fill | qualified `2/4`, fill additions `6`, dead-air `0.5714 -> 0.2941` |
| duration/coverage MIDI evidence | fill score delta `+79.731`, human/audio preference `false` |
| repeatability audio review | 2 WAV candidates, user review `outside_or_unclear`, keep claim `false` |
| outside-soloing repair sweep | repaired source `2/2`, qualified variants `6/6` |
| outside-soloing repair objective evidence | objective support source `2/2`, chord-tone pass `2/2`, non-chord run pass `2/2`, interval pass `2/2` |
| outside-soloing repair policy repeatability | policy support `3/3`, variants qualified `6/6`, chord-tone min `1.000`, non-chord max `0` |
| outside-soloing repair final boundary | `outside_soloing_repair_objective_path_complete` |

## 증명한 것 / 증명하지 않은 것

| 구분 | 상태 |
|---|---|
| Dataset -> tokenization -> training -> generation -> decode -> review 연결 | 검증 |
| `.mid` 파일 존재 기반 성공 판정 제거 | 검증 |
| one-note / long sustain / chord block 실패 감지 | 검증 |
| objective-clean solo-line 후보 선별 | 검증 |
| duration/coverage fill 기반 dead-air repair | 검증 |
| outside-soloing repair objective support | 검증 |
| outside-soloing repair policy repeatability | 검증 |
| human/audio preference | 미검증 또는 단일 사용자 범위 |
| broad unconstrained trained-model quality | 미검증 |
| Brad style adaptation | 미진행 |
| generic jazz pianist base 완성 | 미검증 |
| production-ready improviser | 미검증 |

## 주요 실행

환경 설치:

```bash
pip install -r requirements.txt
```

기본 검증:

```bash
bash scripts/agent_harness.sh quick
```

Stage B generation probe:

```bash
bash scripts/agent_harness.sh stage-b-generation-probe
```

outside-soloing repair final decision:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-final-decision
```

## 참고 문서

- `docs/CURRENT_STATUS_AND_PLAN.md`
- `docs/CORE_PLAN.md`
- `docs/STAGE_B_MODEL_CORE_MVP_COMPLETION_AUDIT_2026-05-28.md`
- `docs/STAGE_B_DURATION_COVERAGE_FILL_OUTSIDE_SOLOING_REPAIR_FINAL_DECISION_2026-05-29.md`
