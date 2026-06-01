# Jazz Piano MIDI 모델 검증 파이프라인

## 개요

Symbolic MIDI 기반 jazz piano solo-line 생성 모델의 학습, 생성, 디코딩, 검증 흐름을 작은 실험 단위로 검증한 model-core 프로젝트.

현재 범위는 완성형 재즈 연주 모델이 아니라, MIDI dataset에서 window를 만들고 checkpoint를 학습한 뒤 생성 결과를 objective MIDI gate로 검증하는 pipeline이다.

## 현재 상태

| 항목 | 상태 |
|---|---|
| pipeline MVP | 완료 |
| generic model-core path | 검증 진행 |
| full generic window preparation | train `154136` / val `21845` tokenized records |
| scale checkpoint training smoke | train `128` / val `32`, best validation loss `5.9031`, checkpoint `1` |
| raw checkpoint generation | valid / strict / grammar `0 / 0 / 0` |
| constrained objective repair | valid / strict / grammar `3 / 3 / 3` |
| objective repeatability sweep | seeds `44/52/60`, valid / strict / grammar `9 / 9 / 9` |
| human/audio preference | 미검증 |
| broad trained-model quality | 미검증 |
| Brad style adaptation | 미진행 |
| realtime DAW/plugin, backend/API, SaaS/UI | 범위 밖 |

최신 판단:

- evidence boundary: `stage_b_generic_base_scale_checkpoint_repeatability_consolidation`
- documentation status: `stage_b_model_core_evidence_readme_refresh`
- next documentation boundary: `stage_b_model_core_portfolio_bullet_refresh`
- objective MIDI gate repeatability claim: `true`
- musical quality claim: `false`
- human/audio preference claim: `false`
- broad trained-model quality claim: `false`
- Brad style adaptation claim: `false`

## 구현 범위

| 영역 | 구현 내용 |
|---|---|
| Dataset audit | MIDI corpus readable file, candidate file, Brad subset, duplicate hash 점검 |
| Manifest split | generic train/val, Brad holdout split, style leakage guard |
| Stage B representation | `BAR`, `POSITION`, `CHORD_ROOT`, `CHORD_QUALITY`, `NOTE_PITCH`, `NOTE_DURATION`, `VELOCITY` token 구조 |
| Window preparation | full generic manifest 기반 2-bar duration-explicit window 생성 |
| Training smoke | generic base scale checkpoint 학습 smoke, validation loss와 checkpoint artifact 검증 |
| Generation probe | checkpoint 기반 raw generation, constrained generation, coverage-aware position, duration token 적용 |
| Decode / postprocess | generated token sequence MIDI 복원, overlap-free solo-line 후보 생성 |
| Objective MIDI review | note count, unique pitch, phrase coverage, dead-air, max active notes, long-note ratio, repeated cell, max interval 검증 |
| Repair boundary | density/coverage, duration/long-note, sustained coverage/dead-air target 분리 |
| Repeatability sweep | seed 범위 확장, aggregate pass-rate, failure reason, claim boundary 기록 |
| Harness | unit test, compile check, whitespace check, Stage B 전용 probe 실행 모드 관리 |
| Docs | issue 단위 관측값, 제외된 claim, 다음 boundary 기록 |

## 문제 / 해결 / 결과

| 문제 | 관측값 | 해결 | 결과 |
|---|---|---|---|
| `.mid` 파일 존재만으로 성공 판단 위험 | one-note collapse, long sustain block, chord block 출력 | `.mid exists` 성공 조건 제외, objective MIDI review gate 추가 | note-level gate 기반 실패 분리 |
| Stage A representation 한계 | `NOTE_ON/OFF` 중심 구조에서 duration/phrase 제어 어려움 | Stage B duration-explicit tokenization 전환 | `POSITION`, `NOTE_DURATION`, chord context 기반 generation probe 가능 |
| generic base 준비 기준 부재 | Brad style adaptation 이전 generic corpus 검증 필요 | generic/Brad manifest split, leakage guard, full window preparation | generic train/val `2433 / 270`, Brad split `47 / 11 / 14` |
| full window vocab overflow 위험 | tokenized record 생성 시 vocab boundary 검증 필요 | max token id와 vocab size guard 추가 | train/val tokenized records `154136 / 21845`, max token id/vocab `544 / 547` |
| training path 과장 위험 | full training 전 scale smoke만 실행된 상태 | scale checkpoint training smoke로 범위 제한 | selected train/val `128 / 32`, best validation loss `5.9031`, checkpoint `1` |
| raw checkpoint generation 실패 | sample `3`, valid/strict/grammar `0/0/0`, note count `2-4` | grammar/representation decision, density/coverage repair target 분리 | raw generation quality claim 제외 |
| density 부족 | note-count failure `3/3` | constrained note-group density와 coverage-aware position 적용 | density/coverage repair valid/strict/grammar `1/1/3` |
| long-note ratio failure | long-note failures `2` | jazz duration token과 duration/long-note repair 적용 | duration repair valid/strict/grammar `2/2/3`, long-note failure delta `2` |
| dead-air 잔여 병목 | dead-air failure `1`, sustained coverage regression 관측 | sustained coverage/dead-air repair, constrained note groups per bar `8` | repair valid/strict/grammar `3/3/3`, dead-air/long-note `0/0` |
| 단일 seed 과장 위험 | objective gate support가 single seed set에 한정 | objective gate repeatability sweep 추가 | seeds `44/52/60`, valid/strict/grammar `9/9/9`, failure reasons none |
| 음악 품질 claim 과장 위험 | objective MIDI gate와 청감 품질의 분리 필요 | claim boundary 문서화 | musical quality, human/audio preference, broad quality, Brad style adaptation claim `false` |

## 주요 검증 결과

| 항목 | 결과 |
|---|---|
| dataset readable files | `2777` |
| candidate files | `2775` |
| Brad candidate files | `72` |
| exact duplicate hash groups | `0` |
| generic train / val manifest files | `2433 / 270` |
| Brad split files | `47 / 11 / 14` |
| full generic tokenized train / val records | `154136 / 21845` |
| max token id / vocab size | `544 / 547` |
| scale smoke selected train / val records | `128 / 32` |
| scale smoke best validation loss | `5.9031` |
| scale checkpoint count | `1` |
| raw generation probe | sample `3`, valid/strict/grammar `0/0/0` |
| density/coverage repair | valid/strict/grammar `1/1/3`, note-count failure delta `3` |
| duration/long-note repair | valid/strict/grammar `2/2/3`, long-note failure delta `2` |
| sustained coverage/dead-air repair | valid/strict/grammar `3/3/3`, dead-air/long-note `0/0` |
| objective gate repeatability sweep | seeds `44/52/60`, sample `9`, valid/strict/grammar `9/9/9` |
| avg onset / sustained coverage | `0.4236111111111111 / 0.6805555555555556` |
| max longest sustained empty run steps | `4` |

## 증명한 것 / 증명하지 않은 것

| 구분 | 상태 |
|---|---|
| Dataset -> window -> training smoke -> checkpoint -> generation -> decode -> review 연결 | 검증 |
| full generic manifest window preparation | 검증 |
| scale checkpoint training smoke | 검증 |
| raw checkpoint generation 실패 감지 | 검증 |
| constrained objective repair path | 검증 |
| objective MIDI gate repeatability | configured seed sweep 범위 검증 |
| `.mid` 파일 존재 기반 성공 판정 제거 | 검증 |
| one-note / long sustain / chord block 실패 감지 | 검증 |
| human/audio preference | 미검증 |
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

generic base scale checkpoint repeatability consolidation:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-scale-checkpoint-repeatability-consolidation
```
