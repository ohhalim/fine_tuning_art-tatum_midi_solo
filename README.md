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
| pitch vocabulary gate 미달 | Issue #254 후보가 dead-air는 낮지만 focused unique pitch `5`에 머무름 | seed/top-k sweep으로 48개 후보 재평가 | seed17 top_k5 sample 4에서 focused unique pitch `6`, dead-air `0.400`, qualified `1/48` |
| context review 전 후보 과장 위험 | pitch vocabulary gate 통과만으로 listening 후보라고 보기 어려움 | selected candidate를 solo/context package로 격리해 focused context decision 실행 | decision `keep_for_focused_listening`, flags `{}`, max active `1`, final `G#4` over `Fm7` chord tone |
| 청감 리뷰 기록 누락 위험 | context keep 후보라도 실제 timing/phrase/vocabulary 판단은 별도 기록 필요 | focused listening notes template 생성 | candidate `1`, pending `1`, risks `dead_air_ratio_at_gate`, `adjacent_pitch_repeats` |
| focused listening fill 후 최종 keep 실패 | dead-air가 gate 상한이고 adjacent repeat가 남음 | MIDI/context evidence 기준 listening fields 채움 | timing `stiff`, phrase `weak`, vocabulary `thin`, decision `needs_followup` |
| timing/repetition repair 필요 | Issue #262 후보의 chord fit과 landing은 strong이지만 timing과 phrase continuation이 약함 | top_k7, temperature `0.86`, seed `37/41` sweep으로 dead-air와 adjacent repeat 동시 개선 후보 선택 | selected sample `39`, dead-air `0.400 -> 0.353`, adjacent repeats `3 -> 2`, unique pitch `6 -> 7` |
| repair 후보 context 검증 필요 | objective repair만으로 final landing과 context guide 적합성 판단 불가 | solo/context package를 만들고 focused context decision 재실행 | decision `keep_for_focused_listening`, flags `{}`, final `A#4` over `Fm7` tension |
| context keep 후보 청감 판단 보류 | context keep만으로 timing/phrase/vocabulary 최종 판단 불가 | focused listening notes template 생성 | candidate `1`, pending `1`, risks `dead_air_ratio_remaining`, `adjacent_pitch_repeats`, `wide_interval_review` |
| focused listening fill 후 후속 개선 필요 | timing은 개선됐지만 adjacent repeat와 wide interval이 남음 | MIDI/context evidence 기준 listening fields 채움 | timing `acceptable`, phrase `weak`, vocabulary `thin`, decision `needs_followup` |
| phrase/vocabulary blocker | adjacent repeat `2`, max interval `16`이 phrase/vocabulary risk로 남음 | top_k7, temperature `0.82`, seed `43/61` sweep으로 후보 재선택 | sample `43`, adjacent repeats `0`, max interval `7`, dead-air `0.333`, unique pitch `8` |
| phrase/vocabulary repair 후보 context 검증 필요 | objective gate 통과만으로 final landing과 context guide 적합성 판단 불가 | solo/context package를 만들고 focused context decision 재실행 | decision `keep_for_focused_listening`, flags `{}`, final `C5` over `Fm7` chord tone |
| context keep 후보 청감 판단 보류 | focused context keep만으로 timing/phrase/vocabulary 최종 판단 불가 | focused listening notes template 생성 | candidate `1`, pending `1`, risk `sustained_coverage_review` |
| focused listening fill 후 keep 판정 경계 | sustained coverage가 threshold 근처라 최종 음악 품질로 과장할 위험 | MIDI/context evidence 기준으로 listening fields 채움 | timing `acceptable`, phrase `acceptable`, vocabulary `acceptable`, decision `keep`, human/audio proof는 미검증 |
| keep 후보 과장 위험 | evidence fill의 `keep`을 broad model quality로 오해할 수 있음 | proven / not proven / next boundary로 consolidation | current margin-recovered evidence keep candidate와 human/audio 미검증 범위 분리 |
| keep 후보 안정성 미확인 | selected keep이 단일 sample일 수 있음 | 96개 sweep 후보에서 qualified peer 분포 비교 | qualified `2/96`, seed43/61 각각 1개, narrow two-source support |
| qualified peer fallback 미검증 | peer 후보가 objective metric만 통과했을 수 있음 | peer 후보를 별도 solo/context package로 격리해 focused context decision 실행 | decision `keep_for_focused_listening`, flags `{}`, final `C5` over `Fm7` chord tone |
| peer 청감 판단 보류 | peer context keep만으로 selected keep과 같은 fallback인지 판단 불가 | peer focused listening notes template 생성 | candidate `1`, pending `1`, risk `sustained_coverage_review` |
| peer fallback keep 여부 미확정 | peer notes가 pending 상태라 selected keep과 비교 불가 | peer notes를 MIDI/context evidence 기준으로 fill | timing `acceptable`, phrase `acceptable`, vocabulary `acceptable`, decision `keep` |

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

Issue #288 기준 model-core MVP:

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
| margin-recovered pitch vocabulary sweep | seed17/31 top_k5 48개 후보 중 `1`개 qualified, selected focused unique pitch `6`, dead-air `0.400` |
| margin-recovered pitch vocabulary focused context | selected qualified 후보를 context package로 격리, decision `keep_for_focused_listening`, flags `{}` |
| margin-recovered pitch vocabulary focused listening notes | focused listening template 생성, candidate `1`, pending `1`, prior decision `keep_for_focused_listening` |
| margin-recovered pitch vocabulary focused listening fill | reviewed `1`, pending `0`, decision `needs_followup`, timing `stiff`, vocabulary `thin` |
| margin-recovered timing/repetition repair | seed37/41 top_k7 temp0.86 96개 후보 중 `2`개 qualified, sample `39` 선택, dead-air `0.353`, adjacent repeats `2` |
| margin-recovered timing/repetition focused context | selected repair 후보를 solo/context package로 격리, decision `keep_for_focused_listening`, flags `{}` |
| margin-recovered timing/repetition focused listening notes | focused listening template 생성, candidate `1`, pending `1`, review risks `3` |
| margin-recovered timing/repetition focused listening fill | reviewed `1`, pending `0`, timing `acceptable`, phrase `weak`, vocabulary `thin`, decision `needs_followup` |
| margin-recovered phrase vocabulary repair | seed43/61 top_k7 temp0.82 96개 후보 중 `2`개 qualified, sample `43` 선택, adjacent repeats `0`, max interval `7` |
| margin-recovered phrase vocabulary focused context | selected repair 후보를 solo/context package로 격리, decision `keep_for_focused_listening`, flags `{}` |
| margin-recovered phrase vocabulary focused listening notes | focused listening template 생성, candidate `1`, pending `1`, review risk `sustained_coverage_review` |
| margin-recovered phrase vocabulary focused listening fill | reviewed `1`, pending `0`, timing `acceptable`, phrase `acceptable`, vocabulary `acceptable`, decision `keep` |
| margin-recovered phrase vocabulary keep consolidation | current evidence keep candidate 정리, human/audio proof와 broad quality claim boundary 분리 |
| margin-recovered phrase vocabulary keep stability | qualified `2/96`, qualified source `2`, selected keep과 peer 후보 metric 동일 수준 |
| margin-recovered phrase vocabulary peer focused context | qualified peer 후보 context decision `keep_for_focused_listening`, flags `{}` |
| margin-recovered phrase vocabulary peer focused listening notes | peer focused listening template 생성, candidate `1`, pending `1`, review risk `sustained_coverage_review` |
| margin-recovered phrase vocabulary peer focused listening fill | peer reviewed `1`, decision `keep`, timing `acceptable`, phrase `acceptable`, vocabulary `acceptable` |
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
- seed/top-k sweep 48개 후보 중 focused unique pitch `6`, dead-air `0.400`, note count `13`, duplicated 3-note chunk `0`인 qualified 후보 `1`개 확인
- Issue #256 후보는 Issue #254 대비 dead-air가 `+0.106`, adjacent repeat이 `+2`라서 focused context review 전 최종 후보로 승격하지 않음
- selected pitch-vocab 후보를 focused context package로 격리해 context guide 존재, max active `1`, final landing chord tone, decision `keep_for_focused_listening` 확인
- focused context keep은 listening review 진입 조건이며, dead-air `0.400`과 adjacent repeats `3`은 다음 review risk로 유지
- focused listening notes template에 prior decision `keep_for_focused_listening`, pending fields, review risks `dead_air_ratio_at_gate` / `adjacent_pitch_repeats` 기록
- focused listening fill에서 chord fit과 landing은 `strong`이지만 timing `stiff`, phrase continuation `weak`, jazz vocabulary `thin`으로 `needs_followup` 판정
- timing/repetition repair sweep에서 focused unique pitch `7`, note count `14`, max active `1`, duplicated 3-note chunk `0`, dead-air `0.353`, adjacent repeats `2`인 qualified 후보 선택
- Issue #264 후보는 Issue #262 대비 objective timing/repetition metric은 개선됐지만, focused context/listening 재검증 전 최종 keep으로 보지 않음
- timing/repetition repair 후보를 solo/context package로 격리해 range `C#4-G5`, phrase span `6.5` beats, max active `1`, final `A#4` over `Fm7` tension, context decision `keep_for_focused_listening` 확인
- context keep 후보를 focused listening notes template으로 넘기고 timing, phrase continuation, landing, vocabulary, final decision을 pending으로 유지
- focused listening fill에서 timing은 `acceptable`로 개선됐지만 adjacent repeats `2`, max interval `16` 때문에 phrase continuation `weak`, jazz vocabulary `thin`, decision `needs_followup`으로 기록
- phrase/vocabulary repair sweep에서 focused unique pitch `8`, note count `13`, max active `1`, duplicated 3-note chunk `0`, dead-air `0.333`, adjacent repeats `0`, max interval `7`인 qualified 후보 선택
- phrase/vocabulary repair 후보를 solo/context package로 격리해 range `G4-E5`, phrase span `7.0` beats, max active `1`, final `C5` over `Fm7` chord tone, context decision `keep_for_focused_listening` 확인
- context keep 후보를 focused listening notes template으로 넘기고 timing, phrase continuation, landing, vocabulary, final decision을 pending으로 유지
- focused listening fill에서 timing `acceptable`, phrase continuation `acceptable`, jazz vocabulary `acceptable`, final decision `keep`으로 기록하되 human/audio proof와 분리
- margin-recovered evidence keep candidate를 정리하고 broad trained-model quality, human/audio preference, Brad style adaptation은 아직 미검증으로 유지
- phrase/vocabulary sweep 96개 후보 중 qualified `2`개를 확인하고 selected keep 외 qualified peer가 seed `61`에도 있음을 분리
- qualified peer 후보도 focused context package에서 max active `1`, phrase span `7.0` beats, final `C5` over `Fm7` chord tone, context decision `keep_for_focused_listening` 확인
- qualified peer 후보를 focused listening notes template으로 넘기고 selected keep과 같은 risk boundary를 보존
- qualified peer 후보도 focused listening fill 기준 decision `keep`으로 기록해 selected keep 외 fallback keep 후보 확보
- constrained/postprocessed generation의 strict review gate 통과
- objective-clean focused candidates `6/6`
- listening review pending `6`

## 구현 범위 요약

| 구분 | 내용 |
|---|---|
| 만든 것 | symbolic MIDI 생성 모델의 dataset, tokenization, training, generation, decode, objective review, proxy review pipeline |
| 겪은 문제 | `.mid` 파일 존재만으로 성공 판단 불가, one-note collapse, long sustain block, chord block, dead-air outlier, seed-level margin 부족 |
| 해결 방식 | duration-explicit token 구조, grammar/coverage/chord-aware probe, overlap-free postprocess, repeatability sweep, dead-air diagnostics, proxy review scoring, repair candidate selection |
| 검증 결과 | raw generation local gate 통과, 6-file 5-sample recovery strict `12/15`, margin-recovered fallback focused keep `0/3`, pitch-vocab focused context `keep_for_focused_listening`, timing/repetition repair qualified `2/96`, phrase/vocabulary focused context `keep_for_focused_listening`, phrase/vocabulary focused fill `keep` |
| 주장 경계 | reviewable MIDI 후보 생성 검증 파이프라인까지 가능, human listening preference / Brad style adaptation / broad production quality는 미검증 |

Issue #288 기준 current margin-recovered evidence keep candidate:

| 항목 | 결과 |
|---|---|
| candidate | `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` |
| decision path | objective repair -> focused context -> focused listening notes -> evidence fill |
| filled decision | `keep` |
| note count | `13` |
| unique pitch count | `8` |
| range | `G4-E5` |
| phrase span | `7.000` beats |
| max active notes | `1` |
| dead-air ratio | `0.333` |
| sustained coverage | `0.594` |
| adjacent pitch repeats | `0` |
| max interval | `7` |
| final landing | `C5` over `Fm7`, chord tone |
| remaining risk | `sustained_coverage_review` |
| stability boundary | qualified `2/96`, source `2`, selected/peer filled keep, broad repeatability 미검증 |

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
| dead-air outlier control | 부분 검증 / candidate selection gate, pitch vocabulary gate, timing/repetition repair 추가 |
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
