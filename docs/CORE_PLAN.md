# Core Plan

작성일: 2026-05-21

이 문서는 이 저장소의 기준 문서다.

흩어진 PR/issue/doc 내용을 하나로 묶어서, 지금 무엇을 만들고 있고 왜 그 순서로 가는지 판단하는 데 사용한다.

## 1. 최종 목표

최종 목표는 symbolic MIDI 기반 jazz piano improvisation model을 만드는 것이다.

장기적으로 만들고 싶은 시스템:

- house/techno/dance groove 위에서 쓸 수 있는 jazz piano solo MIDI generator
- 입력: BPM, chord progression, section, energy, density, optional recent MIDI context
- 출력: 1-2 bar jazz piano solo MIDI
- 사용처: FL Studio, Ableton, piano VST, future live controller
- 방향: generic jazz pianist base를 먼저 만들고, 이후 Brad Mehldau 같은 특정 pianist style adaptation을 검토한다

이 프로젝트는 raw audio generation이 아니다.
지금 단계에서는 DAW plugin, Spring Boot backend, SaaS, UI가 핵심이 아니다.

## 2. 현재 MVP 목표

현재 MVP는 제품 MVP가 아니라 model-core MVP다.

MVP 정의:

> 구조적으로 valid하고 리뷰 가능한 1-2 bar jazz piano solo-line MIDI를 생성하는 symbolic MIDI training/generation/evaluation pipeline.

MVP가 끝났다고 볼 수 있는 조건:

- MIDI dataset을 audit하고 train/val split을 관리할 수 있다.
- MIDI를 short phrase/window records로 만들 수 있다.
- tokenized records가 model vocab에 안전하게 들어간다.
- tiny-overfit training이 정상 동작한다.
- generated token을 MIDI로 decode할 수 있다.
- 생성된 MIDI가 단순 파일 생성이 아니라 review gate를 통과한다.
- one-note/two-note output, long sustain block, chord block, empty MIDI를 성공으로 처리하지 않는다.
- 여러 seed/sample에서 pass-rate를 보고 품질을 판단한다.

현재 MVP의 성공 기준은 "멋진 솔로"가 아니다.
먼저 "말이 되는 solo-line 후보"를 안정적으로 만드는 것이다.

## 3. 지금까지의 핵심 판단

### 3.1 Stage A는 실패했다

`control_v1` Stage A는 runnable pipeline으로는 검증됐지만, musical output은 실패했다.

관찰된 문제:

- note count가 너무 적음
- 긴 sustain block
- chord block처럼 보이는 출력
- solo-line으로 볼 수 없는 구조
- deterministic generation에서 one-note collapse

따라서 Stage A를 더 세게 postprocess하거나 broad training으로 키우지 않는다.

### 3.2 Stage B로 간 이유

Stage B는 REMI/Jazz Transformer 계열 판단을 따른다.

핵심은 모델보다 representation이다.

Stage B에서 명시하는 것:

- `BAR`
- `POSITION`
- `CHORD_ROOT`
- `CHORD_QUALITY`
- `NOTE_PITCH`
- `NOTE_DURATION`
- `VELOCITY`
- tempo/role control

이 방향은 임의로 만든 것이 아니라, REMI, Jazz Transformer, MidiTok 계열의 공통 판단과 맞다.

현재 실패는 Transformer architecture 자체보다 다음 문제에 가깝다.

- NOTE_ON/OFF representation이 duration을 안정적으로 만들지 못함
- full-song sequence가 너무 김
- chord/position/phrase 정보를 모델이 명시적으로 보기 어려움
- 작은 Brad dataset만으로 style을 scratch 학습하기 어려움

### 3.3 지금은 SOTA 재현 단계가 아니다

현재는 Aria, Moonbeam, MidiTok 기반 pretrained model을 붙인 SOTA 구현 단계가 아니다.

지금 하는 일은:

- local tokenizer contract 검증
- phrase/window dataset 검증
- Music Transformer training/generation path 검증
- MIDI decode 검증
- review gate 검증
- collapse/failure mode 측정

즉, 레퍼런스의 원칙을 따른 engineering probe 단계다.

## 4. 현재 상태

현재 main 기준으로 완료된 단계:

1. 전체 jazz piano dataset audit path 작성
2. Brad Mehldau subset audit
3. Stage A `control_v1` training/generation probe
4. Stage A failure review
5. Stage B tokenization spec/test
6. Stage B role dataset preparation
7. Stage B 2-bar phrase/window dataset
8. Stage B vocab/model training path 연결
9. Stage B generation/decode probe
10. Stage B grammar-constrained generation
11. Stage B overlap/dedup postprocess gate
12. Stage B multi-sample review-gate probe
13. Stage B collapse diagnostics and sampling sweep
14. Stage B strict collapse-aware review gate
15. Stage B 2-file Brad generation probe
16. Stage B temporal coverage diagnostics
17. Stage B coverage-aware constrained generation probe
18. Stage B coverage-aware A/B sweep
19. Stage B candidate ranking report
20. Stage B ranking harmonic/repetition gate
21. Stage B chord-aware pitch constrained generation
22. Stage B coverage_chord candidate review export
23. Stage B longer 4-bar coverage_chord phrase probe
24. Stage B phrase contour/repeated-pitch diagnostics
25. Stage B root bias diagnostics
26. Stage B `tones` vs `tones_tensions` pitch-mode comparison

가장 최근 의미 있는 결과:

- Issue #43은 candidate MIDI를 직접 읽어 harmonic/repetition diagnostics를 추가했다.
- Issue #43 result: candidates `18`, strict candidates `12`, viable unflagged candidates `0`, flagged candidates `18`
- Issue #45는 constrained generation에서 `NOTE_PITCH` 후보군을 current bar chord 기준으로 제한했다.
- Issue #45 result: candidates `27`, strict candidates `21`, viable unflagged candidates `9`, flagged candidates `18`
- top candidate: `coverage_chord`, groups/bar `4`, sample `2`, score `96.6964`
- top candidate harmonic diagnostics: chord-tone `0.750`, bar chord-tone `0.875`, min bar chord-tone `0.800`, dominant pitch `0.375`, repeated pitch `0.250`
- Issue #47 exported the top 6 `coverage_chord` MIDI candidates to `outputs/stage_b_review_candidates/harness_stage_b_chord_aware_probe`
- Manual piano-roll review found that these candidates can look like melodic fragments, but are still too short and feel unfinished.
- Issue #49 extends the same coverage+chord-aware setup to a `4` bar probe with `32` note groups per sample and exports direct review candidates from the generation probe report.
- Issue #49 fixes the length problem structurally, but repeated-pitch dependence remains a listening-review risk.
- Issue #51 shows this is not adjacent same-note collapse: adjacent repeated pitch ratio is `0.000`, average direction change ratio is around `0.689`, and max longest same pitch run is `1`.
- Issue #53 shows the perceived "root-heavy" line is not pure root collapse: average root tone ratio is around `0.271`, top candidate root ratio is around `0.219`, but tension ratio is `0.000`.
- 이것은 아직 unconstrained model quality나 Brad style adaptation 성공을 의미하지 않는다.

중요한 해석:

> 지금은 "모델이 된다"가 아니라 "어떤 representation/generation constraint에서 reviewable MIDI가 되는지 측정할 수 있게 됐다"가 성과다.

## 5. 현재 가장 큰 위험

가장 큰 위험은 postprocess와 constrained generation으로 gate만 통과시키고, 실제 모델 품질은 좋아지지 않는 것이다.

현재 결과는 이 위험을 보여준다.

- grammar gate는 통과할 수 있다.
- MIDI 파일도 생성된다.
- overlap postprocess 후 review gate를 일부 통과한다.
- 하지만 `top_k=1`에서는 같은 position/pitch 반복 collapse가 발생한다.

따라서 다음 단계도 곧바로 broad training이 아니다.
다음 단계는 manual listening/piano-roll review다. 이제 우선순위 review 대상은 짧은 2-bar package가 아니라 `outputs/stage_b_review_candidates/harness_stage_b_longer_phrase_probe/midi/`의 4-bar candidates다. 이 후보가 귀로도 solo-line phrase sketch라면 generic jazz base 후보 학습 설계로 넘어가고, 아니면 phrase/motif-level constraint 또는 pretrained symbolic base를 먼저 검토한다.

## 6. 다음 단계 로드맵

### Phase 1. Collapse Diagnostics

목표:

- 반복 position/pitch collapse를 metric으로 잡는다.
- postprocess 후 살아남은 note 수만 보지 않는다.
- 생성 전 token 수준과 생성 후 MIDI 수준을 모두 분석한다.

구현 후보:

- repeated `POSITION + NOTE_PITCH` pair ratio
- repeated pitch ratio
- unique position count
- unique pitch count
- per-bar note distribution
- postprocess removal ratio
- sample diversity score

통과 기준:

- collapse report가 `report.json`에 들어간다.
- invalid 샘플의 이유가 "note count low"보다 더 구체적으로 나온다.
- `top_k=1`, `top_k=2` 실패 차이를 숫자로 설명할 수 있다.

### Phase 2. Sampling Sweep

목표:

- 한 checkpoint에서 sampling parameter가 품질에 주는 영향을 측정한다.

비교 후보:

- `top_k=1`
- `top_k=2`
- `top_k=4`
- temperature `0.7`, `0.9`, `1.1`

통과 기준:

- 각 설정별 sample count, grammar pass rate, valid pass rate를 비교한다.
- best sample 하나가 아니라 pass-rate table로 판단한다.
- MIDI를 들어볼 후보를 자동으로 고른다.

### Phase 3. Stage B 2-File Brad Probe

목표:

- one-file tiny smoke를 넘어서 Brad 2-file Stage B probe를 실행한다.
- Stage A에서 실패했던 2-file 조건을 Stage B representation으로 다시 비교한다.

통과 기준:

- train/val split이 명확하다.
- 2-file window dataset이 정상 생성된다.
- 여러 seed/sample에서 최소 pass-rate를 만족한다.
- piano roll에서 one-note/chord-block/sustain-block이 아니다.

실패하면:

- postprocess를 더 세게 하지 않는다.
- tokenization 또는 model/data scale 문제로 본다.

### Phase 3.5. Temporal Coverage and Coverage-Aware Generation

목표:

- 2-file Brad probe의 dead-air failure를 token-level temporal coverage로 설명한다.
- sparse onset, tail/head empty span, sustained empty run을 sample report에 기록한다.
- constrained generation의 `POSITION` 선택만 coverage-aware로 바꿔 review gate 통과 가능성을 검증한다.

현재 결과:

- temporal diagnostics: grammar `3/3`, basic `0/3`, strict `0/3`, max longest sustained empty run `11`
- coverage-aware constrained generation: grammar `3/3`, basic `3/3`, strict `3/3`, max longest sustained empty run `6`

다음 통과 기준:

- completed: plain constrained vs coverage-aware constrained A/B sweep을 같은 checkpoint 조건에서 비교했다.
- completed: `note_groups_per_bar=4/6/8`을 비교했다.
- next: pass-rate가 좋아져도 이것을 style learning 성공으로 표현하지 않고 candidate ranking 기준을 추가한다.

### Phase 3.6. Harmonic Candidate Gate and Pitch Control

목표:

- strict gate를 통과했지만 실제 piano roll에서 solo-line이 아닌 후보를 걸러낸다.
- bar-level chord-tone ratio, dominant pitch ratio, repeated pitch ratio, repeated onset-template ratio를 ranking에 반영한다.
- ranking만 보정하지 않고 generation-side pitch/harmony 제어로 넘어갈 기준을 만든다.

현재 결과:

- completed: ranking이 candidate MIDI를 직접 읽어 harmonic/repetition diagnostics를 계산한다.
- completed: low chord-tone/repeated pitch/mechanical template 후보에 review flags를 붙인다.
- latest result: `18` candidates 중 viable unflagged candidate `0`.
- completed: chord-aware pitch constrained generation을 추가했다.
- latest chord-aware result: `27` candidates 중 viable unflagged candidate `9`.

다음 통과 기준:

- generated top candidate를 실제로 듣고 piano roll로 확인한다.
- 그 후보가 one-note/two-note/chord-block/long-sustain/repeated-template 실패가 없어야 한다.

### Phase 3.7. Longer Phrase Review

목표:

- 2-bar 후보가 "만들다 만 단어"처럼 들리는 문제를 직접 검증한다.
- 같은 coverage+chord-aware 제약을 유지하되, review 후보를 `4` bar로 늘린다.
- 단순 note count 증가가 아니라 phrase로 들을 수 있는 길이와 coverage를 확보한다.

현재 결과:

- completed: `4` bar generation probe를 실행했다.
- completed: sample마다 `32` complete note groups를 생성했다.
- completed: `3/3` samples가 grammar/basic/strict gate를 통과했다.
- completed: generation probe report를 review export 입력으로 직접 사용할 수 있게 했다.
- current risk: repeated pitch ratio가 높기 때문에 motif로 들리는지 기계적 재사용으로 들리는지 확인해야 한다.

다음 통과 기준:

- exported 4-bar candidates를 piano roll과 귀로 확인한다.
- 단편이 아니라 최소한 call/continuation/landing 느낌의 phrase sketch인지 본다.
- 여전히 짧거나 기계적이면 broad generic training 전 phrase/motif-level constraint를 먼저 설계한다.

### Phase 3.8. Phrase Contour Diagnostics

목표:

- repeated pitch ratio 하나만 보고 샘플을 오판하지 않는다.
- adjacent same-note collapse, long same-pitch run, low direction change, low interval variety를 분리해서 본다.
- 후보를 자동 탈락시키기보다 review export에서 risk flag로 표시한다.

현재 결과:

- completed: generated sample report에 `phrase_contour`를 추가했다.
- completed: review export에 `risk_flags`를 추가했다.
- latest result: repeated pitch ratio는 높지만 adjacent repeated pitch ratio는 `0.000`이다.
- latest result: direction change ratio는 약 `0.689`이고 longest same pitch run은 `1`이다.

해석:

- 현재 후보는 한 음을 길게 반복하는 collapse가 아니다.
- 제한된 chord-tone pitch set을 많이 재사용하는 상태다.
- 다음 manual review는 이 pitch reuse가 motif로 들리는지, constrained cycling으로 들리는지 판단해야 한다.

### Phase 3.9. Root Bias and Tension Diagnostics

목표:

- "근음을 계속 친다"는 청취 피드백을 수치화한다.
- root tone, non-root chord tone, tension, non-chord tone 비율을 분리해서 본다.
- root collapse인지, chord-tone-only 안전함인지 판단한다.

현재 결과:

- completed: generated sample report에 `pitch_roles`를 추가했다.
- completed: review export에 `root`, `tension` columns를 추가했다.
- latest result: average root tone ratio는 약 `0.271`이다.
- latest result: top candidate root tone ratio는 약 `0.219`이다.
- latest result: tension ratio는 `0.000`이다.
- Issue #55 result: `tones_tensions`는 root tone ratio를 약 `0.271`에서 `0.135`로 낮췄고, tension ratio를 `0.000`에서 `0.313`으로 올렸다.
- Issue #55 result: 양쪽 모두 strict valid `3/3`이지만, `tones_tensions` 후보는 repeated/dominant pitch risk가 여전히 높다.

해석:

- 현재 후보는 root-only collapse가 아니다.
- 오히려 `chord_pitch_mode=tones` 때문에 tension이 전혀 없는 안전한 chord-tone-only line이다.
- `tones_tensions`는 no-tension 문제를 줄였지만, 더 좋은 solo phrase라고 바로 판단할 단계는 아니다.
- 다음 비교는 unrestricted tension보다 passing/approach pitch policy 또는 motif/contour constraint 쪽이 맞다.

### Phase 4. Generic Jazz Base 후보 학습

목표:

- Brad-only scratch training이 아니라 generic jazz pianist prior를 만든다.

조건:

- Stage B 2-file probe가 최소한 reviewable MIDI를 만든 뒤에만 진행한다.
- dataset audit 결과를 사용해 non-Brad generic jazz split을 만든다.
- Brad subset은 adaptation/holdout으로 분리한다.

통과 기준:

- generic split에서 train/val leakage가 없다.
- broad training 결과가 Brad-only tiny probe보다 안정적이다.
- generated MIDI가 여러 sample에서 review gate를 통과한다.

### Phase 5. Brad Style Adaptation

목표:

- generic jazz base 위에 Brad subset adaptation을 검토한다.

조건:

- generic base가 먼저 valid solo-line MIDI를 만들 수 있어야 한다.
- Brad 72 files 전체를 scratch로 학습하는 방향은 우선순위가 낮다.

후보:

- adapter fine-tuning
- LoRA on real pretrained/base checkpoint
- retrieval/motif memory
- style token conditioning

### Phase 6. Product/Serving MVP

목표:

- 모델 core가 reviewable output을 만들 때만 backend/API로 확장한다.

후순위 작업:

- FastAPI inference server
- request schema
- MIDI download path
- job status
- Spring Boot backend
- DAW/live integration

지금은 하지 않는다.

## 7. 레퍼런스 기준으로 맞는가

현재 방향은 레퍼런스와 대체로 맞다.

맞는 부분:

- Music Transformer 계열 symbolic sequence model을 사용한다.
- REMI/Jazz Transformer처럼 bar/position/chord/duration을 명시한다.
- full-song sequence 대신 phrase/window dataset으로 줄인다.
- tiny-overfit과 decode/review gate를 먼저 통과시키려 한다.
- 작은 Brad dataset만으로 style을 scratch 학습하지 않으려 한다.

아직 부족한 부분:

- MidiTok 같은 검증된 tokenizer library를 직접 사용하지 않았다.
- pretrained symbolic MIDI model을 아직 평가하지 않았다.
- Compound Word/Octuple 같은 grouped representation은 아직 구현하지 않았다.
- chord inference/lead-sheet alignment는 아직 약하다.
- musical listening review loop가 자동화되어 있지 않다.

판단:

> 지금은 "논문 구현체 복제"가 아니라 "논문들이 말하는 실패 방지 순서에 맞춘 local engineering path"다.

## 8. 앞으로 하지 말아야 할 것

다음은 금지하거나 뒤로 미룬다.

- one passing MIDI를 보고 broad training으로 바로 넘어가기
- postprocess를 더 세게 해서 모델 성공처럼 보이게 만들기
- Spring Boot/API/UI를 다시 MVP 중심으로 가져오기
- Brad-only tiny dataset으로 "style model"이라고 주장하기
- `valid .mid file exists`를 성공으로 처리하기
- exact artist clone처럼 공개적으로 표현하기
- SOTA 모델 이름만 붙이고 evaluation 없이 진행하기

## 9. 다음 바로 할 일

완료된 바로 전 작업:

- `Stage B temporal coverage diagnostics 추가`
- 결과: 2-file Brad Stage B window dataset은 `137` samples, train `123`, val `14`로 정상 생성됐다.
- 결과: generated samples는 grammar gate `3/3`, collapse warning `0/3`이었다.
- 결과: basic valid `0/3`, strict valid `0/3`이었다.
- 결과: avg onset coverage `0.167`, avg sustained coverage `0.417`, max longest sustained empty run `11` steps였다.
- 결론: Stage B grammar와 collapse는 2-file probe에서 즉시 병목이 아니며, 현재 병목은 sparse onset과 long empty span으로 인한 dead-air/temporal coverage다.

완료된 바로 전 작업:

```text
Stage B chord-aware pitch constrained generation 추가
```

작업 범위:

- constrained generation에서 pitch 후보군을 현재 bar chord에 맞게 제한하거나 가중한다.
- chord tones와 허용 tension을 구분한다.
- dominant pitch repetition과 low pitch variety를 동시에 줄인다.
- temporal coverage는 유지하면서 bar-level chord-tone ratio가 개선되는지 비교한다.
- 과한 postprocess로 모델 성공처럼 보이게 만들지 않는다.

이 작업이 끝난 뒤 판단:

- chord-aware pitch probe에서 viable unflagged candidate가 `9`개 나왔다.
- 바로 broad training으로 가지 않고 top `coverage_chord` MIDI를 귀와 piano roll로 리뷰했다.
- 2-bar 후보는 melodic fragment일 수는 있어도 너무 짧고 미완성처럼 느껴졌다.
- 그래서 다음 작업은 4-bar longer phrase probe로 잡았다.
- 리뷰가 괜찮으면 generic jazz base 후보 학습 설계로 넘어간다.
- 리뷰가 여전히 기계적이면 rhythm/motif-level constraint 또는 pretrained symbolic base를 먼저 검토한다.

## 10. 한 문장 요약

이 프로젝트의 현재 핵심은 다음이다.

> Brad-style jazz MIDI model을 바로 만드는 것이 아니라, reviewable jazz solo-line MIDI를 만들 수 있는 symbolic representation, dataset window, generation, decoding, and evaluation pipeline을 먼저 증명하는 것이다.
