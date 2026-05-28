# Jazz Piano MIDI 생성 검증 파이프라인

> Symbolic MIDI 생성 모델의 출력 실패를 note-level metric으로 분석하고, reviewable solo-line 후보까지 좁히는 검증 파이프라인

## 프로젝트 한 줄 요약

재즈 피아노 솔로 MIDI 생성 실험에서 `.mid` 파일 생성만으로 성공을 판단하지 않도록, **tokenization, generation, decoding, objective review, focused review** 흐름을 구현한 프로젝트입니다.

완성된 음악 생성 모델이 아니라, MIDI 생성 모델 개발을 위한 **실패 분석 및 검증 기반**이 핵심입니다.

## 구현한 것

| 구현 영역 | 구현 내용 |
|---|---|
| Dataset audit | jazz piano MIDI corpus 읽기 가능 여부, 후보 파일, Brad subset, 중복 여부 점검 |
| Stage B tokenization | `BAR`, `POSITION`, `CHORD_ROOT`, `CHORD_QUALITY`, `NOTE_PITCH`, `NOTE_DURATION`, `VELOCITY` 기반 duration-explicit token 구조 |
| Generation probe | grammar-constrained generation, coverage-aware generation, chord-aware pitch constraint, data-derived motif rhythm generation |
| MIDI decode / postprocess | generated token sequence를 MIDI로 복원하고 overlap-free solo-line variant 생성 |
| Objective MIDI review | note count, unique pitch, polyphony, phrase coverage, repeated cell, interval, chord/tension/outside ratio, final landing 검증 |
| Focused review package | proxy keep 후보의 solo MIDI와 context MIDI를 분리해 focused review artifact 생성 |
| Listening review notes | timing, chord fit, phrase continuation, landing, jazz vocabulary, decision을 structured field로 기록 |
| Validation harness | unit test, compile check, whitespace check, Stage B probe 실행을 harness mode로 관리 |
| Documentation | issue 단위 실험 결과, 실패 원인, repair target, remaining risk 문서화 |

## 문제와 해결

| 문제 | 원인 / 관찰 | 해결 | 결과 |
|---|---|---|---|
| `.mid` 파일은 생성되지만 solo-line으로 보기 어려움 | one-note collapse, long sustain block, chord block 출력 | `.mid exists`를 성공 조건에서 제외하고 objective MIDI review 추가 | 생성 결과를 note-level metric으로 재검증 |
| Stage A 출력 품질 실패 | `NOTE_ON/OFF` 중심 representation에서 duration과 phrase 구조 제어 어려움 | Stage B duration-explicit tokenization으로 전환 | `POSITION`, `NOTE_DURATION`, chord context 기반 생성 probe 가능 |
| 동시 발음 / chord block 위험 | 같은 onset의 note가 겹치며 solo-line 검증 불가 | overlap-free postprocess 및 max active notes 검증 | focused 후보 max active notes `1` 유지 |
| 반복 pitch-cell 문제 | adjacent repeat, duplicated pitch-class chunk 발생 | pitch reuse 제한, fallback 후보 조정, repeated-cell metric 추가 | focused 후보 adjacent pitch repeats `0`, duplicated 3/4/8 chunks `0 / 0 / 0` |
| final landing 검증 부족 | 마지막 음이 chord context와 맞는지 판단 어려움 | context MIDI, chord guide, bass root guide와 함께 focused context review 구성 | focused 후보 final landing `D5` over `Ebmaj7` 확인 |
| 주관적 리뷰 기록 불일치 | "좋다/나쁘다" 식의 loose comment로 다음 repair target 불명확 | listening review notes schema 추가 | timing, chord fit, phrase continuation, landing, vocabulary, decision 분리 |
| 실험 결과 과장 위험 | 단일 후보 keep을 모델 완성으로 오해 가능 | proven / not proven / remaining risk 문서화 | current best focused candidate와 broad quality claim 분리 |
| margin-recovered 후보의 focused keep 실패 | 후보 3개 모두 pitch vocabulary 부족, rank 2는 dead-air도 높음 | 기존 seed31 checkpoint에서 top_k4 12-sample repair 후보 재선별 | sample 8에서 dead-air `0.444 -> 0.294`, focused unique pitch `4 -> 5`, remaining flag `low_pitch_variety` |

## 파이프라인 구조

```mermaid
flowchart LR
    A["Dataset audit"] --> B["Stage B tokenization"]
    B --> C["Generation probe"]
    C --> D["MIDI decode"]
    D --> E["Overlap-free postprocess"]
    E --> F["Objective MIDI review"]
    F --> G["Proxy review"]
    G --> H["Focused context package"]
    H --> I["Focused listening notes"]
    I --> J["Keep / follow-up decision"]
```

## 핵심 결과

Issue #254 기준 model-core MVP:

| 항목 | 결과 |
|---|---|
| core 여부 | dataset, tokenization, training, generation, decode, review gate가 연결된 model-core 작업 |
| pipeline MVP | 완료 |
| raw generation gate | `stage-b-generation-probe` 통과 |
| raw generation mode | `unconstrained` token sampling |
| repair 조건 | 50 epoch tiny-overfit, top_k `4`, overlap postprocess |
| repeatability sweep | 2 source files / 3 seeds / 9 samples |
| repeatability result | strict `8/9`, grammar `9/9`, dead-air outlier `1` |
| dead-air diagnostics | seed `31` sample `1`, dead-air `0.857`, collapse warning false |
| candidate selection gate | selected best seed `17` sample `3`, dead-air `0.333` |
| broader source gate | 3 source files / strict `7/9`, dead-air outlier rate `0.222`, selected best dead-air `0.222` |
| larger source boundary | 4/5/6 source files hard gate 통과, 6-file seed `17` strict margin `1/3` |
| seed strict margin diagnostics | 6-file seed `17`: sample `1` dead-air, sample `2` unique pitch, sample `3` strict-valid |
| seed margin warning gate | hard gate 유지, warning min strict per seed `2`, warning seed `17` 기록 |
| candidate count recovery | 6 source files / 5 samples per seed / strict `12/15`, warning seed 없음 |
| margin-recovered review export | seed별 best 후보 3개 objective metric table 추출, selected best seed `23` sample `1` |
| listening review notes | margin-recovered 후보 3개 pending review template 생성, selected best count `1` |
| MIDI proxy review fill | rank `2` seed `31` sample `5` proxy keep, rank `1` dead-air best는 needs_followup |
| proxy keep consolidation | dead-air 단일 기준 selected best와 phrase-rich proxy keep 후보의 claim boundary 문서화 |
| margin-recovered focused package | rank `2` 후보만 solo/context review package로 격리, focused solo-line max active `1` |
| margin-recovered focused context decision | rank `2` proxy keep을 `needs_followup`으로 하향, low pitch variety / dead-air blocker 기록 |
| margin-recovered fallback comparison | rank `1/2/3` 전체 focused context 비교, focused keep `0/3`, 공통 blocker low pitch variety |
| margin-recovered pitch/dead-air repair | top_k4 12-sample 재선별로 sample `8` 선택, dead-air `0.294`, focused unique pitch `5`, remaining flag `low_pitch_variety` |
| constrained review gate | `stage-b-overlap-gate` 통과 |
| focused candidate path | `stage-b-rhythm-phrase-variation` 통과 |

MVP 근거:

- Stage B window/token dataset preparation 정상 동작
- tiny training path 정상 실행, best validation loss `1.6905`
- raw generated samples valid/strict/grammar `5/5`
- complete note groups `21-22`, invalid token count `0`
- postprocess 후 note count `13-18`, unique pitch count `4-6`
- 2-file/3-seed repeatability sweep에서 strict pass-rate `0.889`
- dead-air outlier가 collapse/postprocess 문제가 아니라 낮은 onset/sustained coverage 문제임을 분리
- dead-air outlier rate `0.111`을 기록하고 strict-valid 후보 중 best candidate를 선택
- 3-file repeatability에서 strict `7/9`, dead-air outlier rate `0.222 <= 0.250` 확인
- 4/5/6-file repeatability hard gate 통과, 6-file seed `17`에서 strict `1/3` 및 unique pitch failure 확인
- 6-file seed `17`의 dead-air failure와 unique-pitch failure가 서로 다른 후보에서 발생함을 sample 단위로 분리
- per-seed strict margin warning을 repeatability summary에 추가해 aggregate pass-rate로 가려지는 후보 안정성 리스크 기록
- samples per seed를 `3`에서 `5`로 늘려 6-file seed `17` strict margin을 `1/3`에서 `3/5`로 회복
- 5-sample run의 seed별 best 후보를 review rank로 정리하고, dead-air 기준 selected best와 coverage가 높은 대안 후보를 분리
- margin-recovered 후보 3개를 listening review notes template으로 묶고 실제 청감 판단 전 pending 상태로 보존
- MIDI metric proxy review에서 dead-air 최저 후보보다 phrase/onset/sustained coverage가 높은 rank `2` 후보를 keep으로 분리
- rank `2` seed `31` sample `5`는 MIDI metric proxy keep이며, human listening preference나 broad model quality claim과 분리
- rank `2` 후보를 focused package로 격리하고 source note count `19` -> focused solo-line note count `14`, max simultaneous notes `2` -> `1` 변환 기록
- focused context decision에서 unique pitch `4`, dead-air `0.444`로 `needs_followup` 판정해 proxy keep을 최종 후보로 과장하지 않음
- margin-recovered 후보 3개 전체를 같은 focused context 기준으로 비교해 fallback 후보 없음, low pitch variety `3/3` 확인
- 기존 seed `31` checkpoint의 top_k4 12-sample repair에서 dead-air를 `0.444 -> 0.294`로 낮추고 focused unique pitch를 `4 -> 5`로 올린 partial repair 확인
- repair sample `8`도 focused unique pitch gate `6`에는 미달하므로 focused keep이나 broad quality로 승격하지 않음
- constrained/postprocessed generation의 strict review gate 통과
- objective-clean focused candidates `6/6`
- listening review pending `6`

## 구현 범위 요약

| 구분 | 내용 |
|---|---|
| 만든 것 | symbolic MIDI 생성 모델의 dataset, tokenization, training, generation, decode, objective review, proxy review pipeline |
| 겪은 문제 | `.mid` 파일 존재만으로 성공 판단 불가, one-note collapse, long sustain block, chord block, dead-air outlier, seed-level margin 부족 |
| 해결 방식 | duration-explicit token 구조, grammar/coverage/chord-aware probe, overlap-free postprocess, repeatability sweep, dead-air diagnostics, proxy review scoring, repair candidate selection |
| 검증 결과 | raw generation local gate 통과, 6-file 5-sample recovery strict `12/15`, margin-recovered fallback focused keep `0/3`, repair sample dead-air `0.294` |
| 주장 경계 | reviewable MIDI 후보 생성 검증 파이프라인까지 가능, human listening preference / Brad style adaptation / broad production quality는 미검증 |

Issue #210 기준 current best focused review candidate:

| 항목 | 결과 |
|---|---|
| candidate | `data_motif_rhythm_phrase_variation_rank_2_sample_2` |
| decision | current best focused review candidate |
| note count | `64` |
| unique pitch count | `19` |
| range | `G3-G5` |
| phrase span | `32.0` beats |
| max active notes | `1` |
| max interval | `4` |
| objective flags | `[]` |
| adjacent pitch repeats | `0` |
| duplicated 3/4/8-note pitch-class chunks | `0 / 0 / 0` |
| final landing | `D5` over `Ebmaj7` |
| focused timing | `acceptable` |
| focused chord fit | `strong` |
| focused landing | `strong` |
| focused jazz vocabulary | `acceptable` |

결과 해석:

- reviewable MIDI outcome 확보
- objective-clean focused candidate 확보
- repeated-cell blocker 제거
- proxy review -> focused context decision -> focused listening fill 경로 검증
- 단일 후보 기준 current best candidate 확보

## 아직 증명하지 않은 것

| 항목 | 상태 |
|---|---|
| broad unconstrained trained-model generation quality | 미검증 |
| broad multi-seed model quality | 부분 검증 / 6-file 3-seed 5-sample local sweep hard gate 통과, seed-level margin warning 해소 |
| dead-air outlier control | 부분 검증 / candidate selection gate와 sample-level repair 추가, pitch vocabulary gate는 미완료 |
| human/audio listening preference | 미검증 |
| Brad Mehldau style adaptation | 미검증 |
| generic jazz pianist base 완성 | 미검증 |
| realtime DAW/plugin readiness | 범위 밖 |
| backend/API/product MVP | 범위 밖 |

## 주요 검증 기준

Objective MIDI review 기준:

- non-zero note count
- unique pitch count
- max simultaneous notes
- polyphonic tick ratio
- phrase coverage
- dead-air ratio
- max note duration ratio
- repeated pitch/cell ratio
- max interval
- unresolved large leap ratio
- chord-tone/tension/outside/root ratio
- final guide/chord landing
- IOI/duration diversity

성공 조건에서 제외한 항목:

- `.mid` 파일 존재만으로 성공 처리
- one-note / two-note output
- long sustain block
- chord block output
- repeated-cell collapse
- final landing 미검증 결과

## Dataset audit 결과

| 항목 | 값 |
|---|---:|
| active dataset tree | `midi_dataset/midi` |
| readable files | `2777` |
| candidate files | `2775` |
| candidate non-Brad files | `2703` |
| candidate Brad files | `72` |
| exact duplicate hash groups | `0` |

Dataset 판단:

- Brad subset 직접 scratch training 제외
- generic jazz base 이후 adaptation / holdout 후보 분리
- generation 확장 전 dataset audit 선행

## 실행 방법

환경 설치:

```bash
pip install -r requirements.txt
```

빠른 검증:

```bash
bash scripts/agent_harness.sh quick
```

Stage B rhythm/phrase variation probe:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

Focused listening review notes:

```bash
bash scripts/agent_harness.sh stage-b-focused-listening-review-notes
```
