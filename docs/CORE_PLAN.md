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

Long-term reference:

- Live Music Diffusion Models, LMDM, is relevant to the final live-AI-instrument direction.
- The useful ideas are block-wise generation, sliding context, live input/output scheduling, and long-horizon drift control.
- It does not change the current MVP: the present work remains symbolic MIDI jazz solo grammar, reviewability, and phrase quality.

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
27. Stage B 8-bar approach phrase probe
28. Stage B swing/motif phrase grammar probe
29. Stage B real phrase reference statistics
30. Stage B data-derived motif template extraction
31. Stage B data-derived motif baseline generation
32. Stage B data motif review export
33. Stage B chord-context and straight-grid review export
34. Stage B straight-grid guide-tone/cadence review candidate
35. Stage B data-motif rhythm plus guide-tone/cadence pitch hybrid
36. Stage B reference pitch-role landing statistics and chord-coverage gate
37. Stage B chord progression coverage audit
38. Stage B chord-labeled evaluation subset contract
39. Stage B generated candidate chord-labeled eval bridge
40. Stage B data-guide hybrid generated chord evaluation
41. Stage B review markdown chord eval summary
42. Stage B listening review notes schema
43. Stage B filled listening review aggregate
44. Stage B full review manifest listening notes
45. Stage B objective MIDI note review
46. Stage B objective flags review flow
47. Stage B overlap-free solo-line review export
48. Stage B duration variation review baseline
49. Stage B phrase/cadence review baseline
50. Stage B phrase naturalness objective metrics
51. Stage B phrase recovery review baseline
52. Stage B data motif phrase recovery baseline
53. Stage B objective clean review package
54. Stage B clean context phrase diagnostics
55. Stage B clean listening review notes template
56. Stage B clean MIDI-note proxy review
57. Stage B data-derived contour/cadence landing repair probe
58. Stage B contour repair MIDI-note proxy review
59. Stage B rhythm/phrase vocabulary variation probe

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
- Issue #75 shows reference pitch-role stats cannot be trusted yet because known chord note ratio is `0.000`.
- Issue #77 audits the local dataset for chord progression annotations and finds no usable candidate: role meta `2812` scanned with `0` hits, sidecars `0`, MIDI files scanned for text events `120` with `0` chord-text candidates.
- Issue #79 adds a tiny chord-labeled eval contract so known chord labels can produce pitch-role sanity summaries without pretending real Brad/reference labels already exist.
- Issue #81 connects generated candidate reports with known chord metadata to the chord-labeled evaluator.
- Issue #83 applies that bridge to actual data-guide hybrid review candidates and shows `data_motif_guide_tones` has higher chord-tone ratio than `data_motif`.
- Issue #85 writes a combined review markdown so MIDI paths, rhythm metrics, and chord-role metrics can be reviewed together.
- Issue #87 creates a structured listening review notes schema so subjective review can be recorded consistently instead of as loose comments.
- Issue #89 aggregates filled listening review notes into next-step signals and refuses to change generation rules when all candidates are still pending.
- Issue #91 builds listening review notes from the full review manifest so all 15 review candidates, including timing references, have file paths and pending review fields.
- Issue #93 reads generated MIDI notes directly and reports objective flags for overlap/polyphony, grid alignment, scalar/chromatic motion, duration collapse, and chord-role ratios.
- Issue #95 connects objective flags to listening review notes and aggregate priority so problem/warning candidates are visible before manual listening.
- Issue #97 exports overlap-free solo-line review MIDI variants while preserving original sample paths, reducing objective `overlap_polyphonic` from `9` to `0`.
- Issue #99 adds varied-duration review baselines, reducing objective `duration_pattern_collapse` from `6` to `0` while keeping `overlap_polyphonic=0`.
- Issue #101 adds a phrase/cadence review baseline, reducing `chromatic_walk` from `7` to `1` and `too_stepwise_or_scalar` from `6` to `0` in the next review set.
- Issue #103 adds phrase naturalness metrics and reveals that all `12` Issue #101 review candidates have `unresolved_large_leaps`.
- Issue #105 adds a phrase recovery baseline, reducing `phrase_recovery` unresolved large leap ratio to `0.000-0.048`.
- Issue #107 combines data-derived motif rhythm with phrase recovery pitch grammar and keeps `data_motif_phrase_recovery` objective-clean.
- Issue #109 extracts only the objective-clean `data_motif_phrase_recovery` candidates into a focused listening review package.
- Issue #109 result: `3` clean candidates, all with context MIDI paths, note count `63`, unique pitch count `19-23`, unresolved large leap ratio `0.000-0.045`, and tension ratio `0.476-0.524`.
- Issue #111 reads those clean context candidates back at MIDI note-level and reports `3/3` as `listen_with_context`, with no diagnostic flags.
- Issue #113 creates clean listening review notes for those `3` objective-clean context candidates.
- 2026-05-24 MIDI-note proxy review marks the `3` candidates as `needs_followup=2`, `reject=1`, `keep=0`, with contour/landing and rhythm stiffness as the next blockers.
- Issue #115 adds `data_motif_contour_landing_repair`, improving final resolved landing from `1/3` to `3/3` and reducing max interval from `13` to `7` in the comparison harness.
- Issue #115 objective MIDI review reports candidate count `6` and objective flag counts `{}` for the repair-vs-baseline review set.
- Issue #116 contour repair MIDI-note proxy review marks the `6` repair-vs-baseline candidates as `needs_followup=5`, `reject=1`, `keep=0`.
- Issue #116 contour repair aggregate reports `too_stiff=6`, `too_mechanical=6`, `too_repetitive=6`, and recommends phrase vocabulary, timing grid, and motif variation follow-ups.
- Issue #118 adds `data_motif_rhythm_phrase_variation`, improving syncopation `0.625 -> 0.694`, duration diversity `0.062 -> 0.097`, and IOI diversity `0.079 -> 0.115` while keeping objective MIDI flag counts `{}`.
- Issue #118 preserves final landing `3/3`, reduces max interval `7 -> 6`, and keeps unresolved large leap ratio `0.000` for the variation candidates.
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
이제 다음 단계는 rhythm/phrase variation 후보의 MIDI-note proxy review다. Issue #118은 objective rhythm metrics를 개선했지만 note count 감소와 tension ratio 하락이 있어 실제 review boundary가 필요하다.

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
- Issue #57 result: 8-bar `approach_tensions`는 strict valid `3/3`, root ratio `0.000`, approach resolution ratio `1.000`을 만들었다.
- Issue #57 review premise: 이전보다 나아졌지만 아직 jazz solo가 아니라 다이아토닉 코드톤/근음 기반 초급 melodic exercise처럼 들린다.
- Issue #59 result: `swing_motif_approach`는 strict valid `3/3`을 유지하면서 syncopated onset ratio를 `0.500`에서 `0.750`으로 올렸다.
- Issue #59 result: unique bar-position pattern ratio는 `0.125`에서 `0.500`으로 올랐고, most-common duration ratio는 `0.552`에서 `0.380`으로 낮아졌다.
- Issue #61 result: real jazz phrase windows `57`개 기준 syncopation mean은 `0.736`, unique bar-position pattern mean은 `0.996`, duration diversity mean은 `0.379`, IOI diversity mean은 `0.341`이다.
- Issue #61 result: `swing_motif_approach`는 syncopation은 reference에 가까우나 bar-position variation, duration diversity, IOI diversity가 아직 크게 부족하다.
- Issue #63 result: real Stage B windows에서 `803`개 strictly-increasing solo-line motif를 추출했고, rhythm templates `520`, contour templates `328`, full templates `526`개를 만들었다.
- Issue #63 result: top rhythm support는 `0.009`, top contour support는 `0.012`, top full motif support는 `0.002`라서 one best motif 복붙이 아니라 distribution sampling이 필요하다.
- Issue #65 result: data-derived motif baseline도 strict `3/3`을 통과했다.
- Issue #65 result: hand-written swing 대비 bar-position variation은 `+0.500`, duration diversity는 `+0.016`, IOI diversity는 `+0.016` 개선됐지만 syncopation은 `-0.125` 낮아졌다.
- Issue #67 result: `data_motif`와 `hand_written_swing` 후보를 mode/sample/rank가 드러나는 named MIDI review package로 export했다.
- Issue #69 result: chord/bass guide가 들어간 context MIDI와 straight-grid timing reference를 추가했다.
- Issue #71 result: `straight_guide_tones` 후보를 추가해 swing timing 문제와 chromatic/scale pitch 문제를 분리했다.
- Issue #71 result: `straight_guide_tones`는 note count `64`, unique pitch count `26-29`, chord-tone ratio `0.656`, tension ratio `0.172`, root-tone ratio `0.000`이지만 straight reference용 dead-air gate 때문에 strict `0/3`이다.
- Issue #73 result: `data_motif_guide_tones` 후보를 추가해 data-derived rhythm/duration template과 guide-tone/cadence pitch grammar를 결합했다.
- Issue #73 result: `data_motif_guide_tones`는 strict `3/3`, note count `63`, unique pitch count `23-24`, chord-tone ratio `0.797`, tension ratio `0.062`, root-tone ratio `0.000`, unique bar-position pattern ratio `1.000`이다.
- Issue #75 result: reference pitch-role landing 통계를 시도했지만 known chord note ratio가 `0.000`이라 pitch-role reference는 아직 사용할 수 없다.
- Issue #75 result: 현재 비교 가능한 것은 rhythm reference뿐이며, pitch vocabulary 조정 전에 chord annotation coverage audit이 필요하다.
- Issue #77 result: role metadata `2812`개, raw sidecar `0`개, text event를 검사한 MIDI file `120`개를 scan했지만 chord progression hit는 `0`이다.
- Issue #77 result: 현재 local dataset에는 바로 쓸 수 있는 chord progression annotation이 없으므로 reference pitch-role comparison은 아직 불가능하다.
- Issue #79 result: `inline_notes` tiny fixture `2` samples, `32` notes로 chord-labeled eval contract를 검증했다.
- Issue #79 result: fixture chord-tone ratio는 `0.844`, tension ratio는 `0.156`, outside ratio는 `0.000`이다.
- Issue #81 result: generated candidate bridge fixture `1` sample, `16` notes로 report-to-evaluator 연결을 검증했다.
- Issue #81 result: fixture chord-tone ratio는 `1.000`, tension ratio는 `0.000`, outside ratio는 `0.000`이다.
- Issue #83 result: data-guide hybrid generated chord eval은 `6` candidates, `192` notes를 평가했다.
- Issue #83 result: aggregate chord-tone ratio는 `0.656`, tension ratio는 `0.120`, outside ratio는 `0.000`이다.
- Issue #83 result: `data_motif` chord-tone ratio는 `0.500`, `data_motif_guide_tones` chord-tone ratio는 `0.812`이다.
- Issue #85 result: combined review markdown is written to `outputs/stage_b_generated_chord_eval/harness_stage_b_review_markdown_chord_eval/review_candidates_with_chord_eval.md`.
- Issue #87 result: listening review notes template contains `6` pending candidates and validates phrase quality, timing, chord fit, issue flags, and decision enums.
- Issue #89 result: listening review aggregate reports `6` pending candidates, `0` reviewed candidates, and only recommends `collect_listening_reviews`.
- Issue #91 result: full review manifest notes contain `15` pending candidates with `review_midi_path`, `context_midi_path`, mode, rank, sample, and rhythm/timing metrics.
- Issue #93 result: objective MIDI review flags `chromatic_walk=7`, `duration_pattern_collapse=9`, `overlap_polyphonic=9`, and `too_stepwise_or_scalar=4`.
- Issue #95 result: objective review priority reports `15` candidates, `6` warning/reviewable candidates, and `9` problem candidates before subjective listening.
- Issue #97 result: overlap-free review export reports `15` reviewable candidates, `5` clean candidates, `10` warning candidates, and `overlap_polyphonic=0`.
- Issue #99 result: duration variation review reports `15` reviewable candidates, `8` clean candidates, `7` warning candidates, `duration_pattern_collapse=0`, and `overlap_polyphonic=0`.
- Issue #101 result: phrase/cadence review reports `12` reviewable candidates, `11` clean candidates, `1` warning candidate, `chromatic_walk=1`, and `too_stepwise_or_scalar=0`.
- Issue #103 result: phrase naturalness review reclassifies the same `12` candidates as warnings because `unresolved_large_leaps=12`.
- Issue #105 result: phrase recovery review reports `phrase_cadence` candidates as `3` warnings and `phrase_recovery` candidates as `3` clean candidates.
- Issue #107 result: data motif phrase recovery review reports `data_motif_guide_tones` as `3` warnings and `data_motif_phrase_recovery` as `3` clean candidates.
- Issue #109 result: objective clean review package keeps only the `3` `data_motif_phrase_recovery` candidates and writes `outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package/clean_review_package.md`.
- Issue #111 result: clean context diagnostics reports `3` candidates, diagnostic flags `{}`, bar coverage `8/8`, off-grid ratio `0.000`, max duration `1.000` beat, and decision hint `listen_with_context`.
- Issue #113 result: clean listening review notes template covers the `3` objective-clean context candidates and validates review enums/summary output.
- 2026-05-24 Codex MIDI-note proxy review result: `needs_followup=2`, `reject=1`, `keep=0`; the strongest candidate is still `timing=stiff`, `jazz_vocabulary=thin`.
- Issue #115 result: `data_motif_contour_landing_repair` is strict `3/3`, final landing resolved `3/3`, max interval `7`, abrupt resets `0`, and objective MIDI flag counts `{}`.
- Issue #115 comparison: `data_motif_phrase_recovery` is still strict `3/3`, but final landing resolved is `1/3` and max interval is `13`.
- Issue #116 contour repair MIDI-note proxy review result: `reviewed=6`, `needs_followup=5`, `reject=1`, `keep=0`.
- Issue #116 aggregate result: `phrase=1`, `fragment=4`, `exercise=1`, `too_stiff=6`, `fits=4`, `unclear=2`.
- Issue #118 result: `data_motif_rhythm_phrase_variation` is strict `3/3`, final landing resolved `3/3`, max interval `6`, objective flags `{}`, and pitch range floor `>=51`.
- Issue #118 rhythm result: syncopation `0.694`, duration diversity `0.097`, IOI diversity `0.115`, compared with contour repair `0.625`, `0.062`, `0.079`.

해석:

- 현재 후보는 root-only collapse가 아니다.
- 오히려 `chord_pitch_mode=tones` 때문에 tension이 전혀 없는 안전한 chord-tone-only line이다.
- `tones_tensions`는 no-tension 문제를 줄였지만, 더 좋은 solo phrase라고 바로 판단할 단계는 아니다.
- `approach_tensions`는 pitch-level resolution을 만들지만, 이 또한 jazz vocabulary 자체는 아니다.
- `swing_motif_approach`는 기계적인 grid 반복을 줄였지만, 이 또한 jazz vocabulary 자체는 아니다.
- real phrase reference stats와 motif extraction 기준으로 보면 다음은 hand-written rhythm rule 확장이 아니라 data-derived motif/cadence control 쪽이 맞다.
- 다만 pitch-role 쪽은 real reference chord label이 아직 없으므로, 다음 개선은 실제 청취 결과를 notes에 채운 뒤 issue distribution으로 후속 generation rule을 분기하는 순서가 맞다.
- clean package의 context MIDI review boundary는 proxy review까지 진행됐다.
- proxy review는 실제 오디오 청취가 아니므로 최종 subjective quality proof가 아니다.
- Issue #115는 contour continuity와 final landing objective target을 개선했다.
- contour repair MIDI-note proxy review 결과, 다음 병목은 landing이 아니라 rhythm stiffness, repeated duration/rest template, thin phrase vocabulary다.
- Issue #118은 그 병목 중 rhythm objective metrics와 register floor를 개선했지만, listening/proxy review는 아직 pending이다.

### Phase 3.10. Swing/Motif Phrase Grammar

목표:

- pitch-only approach/tension constraint의 한계를 확인한다.
- 같은 checkpoint에서 baseline approach grammar와 swing/motif rhythm grammar를 비교한다.
- rhythm profile을 candidate ranking과 review export에 넣는다.

현재 결과:

- completed: `jazz_rhythm_position_tokens()`와 `jazz_rhythm_duration_tokens()`를 추가했다.
- completed: `approach_baseline`과 `swing_motif_approach`를 같은 checkpoint에서 비교했다.
- latest result: 두 grammar 모두 strict valid `3/3`이다.
- latest result: syncopated onset ratio는 `0.500`에서 `0.750`으로 좋아졌다.
- latest result: unique bar-position pattern ratio는 `0.125`에서 `0.500`으로 좋아졌다.
- latest result: direct MIDI inspection에서 baseline의 반복 IOI/template 문제가 확인됐다.

해석:

- 현재 후보는 one-note/two-note/chord-block failure가 아니다.
- rhythmic template 반복은 줄었다.
- 하지만 아직 실제 jazz solo vocabulary라고 볼 근거는 부족하다.

다음 통과 기준:

- generated rhythm profile을 real jazz MIDI window 통계와 비교한다.
- pitch motif cell, cadence/landing, phrase memory 중 하나를 다음 issue로 분리한다.
- rule이 아니라 data-derived constraint로 넘어갈지 판단한다.

### Phase 3.11. Real Phrase Reference Statistics

목표:

- generated MIDI가 "이전보다 나음"인지 "실제 jazz phrase 통계에 가까움"인지 분리한다.
- real Stage B phrase windows에서 rhythm/contour reference metrics를 만든다.
- generated candidate report와 comparable metric key를 맞춘다.

현재 결과:

- completed: `scripts/run_stage_b_reference_stats.py`를 추가했다.
- completed: `4`개 MIDI 파일에서 `57`개 8-bar real phrase windows를 분석했다.
- latest result: real syncopated onset ratio mean은 `0.736`이다.
- latest result: real unique bar-position pattern ratio mean은 `0.996`이다.
- latest result: real duration diversity ratio mean은 `0.379`이다.
- latest result: real IOI diversity ratio mean은 `0.341`이다.
- latest result: Issue #59 `swing_motif_approach`는 syncopation은 reference와 거의 맞지만 bar-position/duration/IOI diversity는 부족하다.

해석:

- Issue #59는 baseline보다 나아졌지만 아직 real jazz window 통계에는 미달한다.
- 특히 every-bar pattern variation이 부족하다.
- 다음 단계는 hand-written swing pattern을 더 추가하기보다 dataset에서 phrase motif templates를 추출하는 것이다.

다음 통과 기준:

- real window에서 rhythm/motif templates를 추출한다.
- generated candidate가 reference p25-p75 범위 안에 들어오는 metric을 늘린다.
- phrase ending/cadence도 reference 기반으로 비교한다.

### Phase 3.12. Data-Derived Motif Template Extraction

목표:

- hand-written swing/motif rule을 더 늘리지 않는다.
- real Stage B phrase windows에서 rhythm, contour, full motif templates를 추출한다.
- chord-block 또는 same-onset voicing이 solo-line motif catalog를 오염시키지 않도록 기본 필터를 둔다.
- 다음 generation probe가 data-derived rhythm/contour distribution을 사용할 수 있게 만든다.

현재 결과:

- completed: `scripts/run_stage_b_motif_template_extraction.py`를 추가했다.
- completed: same-onset/non-increasing onset motif를 기본적으로 제외한다.
- completed: `4`개 MIDI 파일에서 만든 Stage B 8-bar windows 기준 `803`개 motif를 추출했다.
- latest result: source records `56`, rhythm templates `520`, contour templates `328`, full templates `526`.
- latest result: top full motif support가 `0.002`라서 full motif를 그대로 복사하는 방식은 맞지 않다.

해석:

- 실제 jazz phrase material은 매우 분산되어 있다.
- 다음 단계는 top motif 하나를 쓰는 것이 아니라 rhythm template과 contour template을 분리해서 sampling하는 것이다.
- 이 단계는 생성 품질을 바로 올리는 작업이 아니라, beginner-like hand-written line에서 data-derived phrase material로 넘어가는 준비다.

다음 통과 기준:

- data-derived motif catalog를 constrained generation의 position/duration/contour 후보로 연결한다.
- generated candidate의 duration diversity와 IOI diversity가 Issue #59보다 좋아지는지 본다.
- reference p25-p75 범위에 가까워지는 metric을 늘린다.
- piano roll에서 chord-tone 나열이 아니라 phrase contour로 들리는지 review export로 확인한다.

### Phase 3.13. Data-Derived Motif Baseline Generation

목표:

- Issue #63 motif catalog를 실제 8-bar generation baseline에 연결한다.
- hand-written `swing_motif_approach`와 data-derived motif baseline을 같은 조건에서 비교한다.
- 생성이 strict gate를 통과하는지, rhythm diversity가 나아지는지 본다.

현재 결과:

- completed: `scripts/run_stage_b_data_motif_generation_compare.py`를 추가했다.
- completed: extracted rhythm template을 position/duration 후보로 사용한다.
- completed: extracted contour template을 pitch interval 후보로 사용한다.
- completed: duration을 다음 onset 전까지 제한해 overlap/postprocess removal을 막는다.
- latest result: `hand_written_swing` strict `3/3`, `data_motif` strict `3/3`.
- latest result: data-derived baseline은 bar-position variation을 `0.500`에서 `1.000`으로 올렸다.
- latest result: duration repetition ratio를 `0.750`에서 `0.375`로 낮췄다.
- latest result: syncopation은 `0.750`에서 `0.625`로 낮아졌다.

해석:

- data-derived motif baseline은 hand-written baseline보다 pattern variation 면에서 낫다.
- 하지만 syncopation 하락과 낮은 diversity 상승폭 때문에 바로 더 좋은 jazz solo라고 말할 수 없다.
- 다음은 MIDI review export와 listening/piano-roll 비교가 필요하다.

다음 통과 기준:

- data_motif 후보를 hand_written_swing 후보와 파일명으로 구분해 review export한다.
- 실제 piano roll에서 phrase contour가 초급 scale exercise보다 나은지 확인한다.
- data_motif가 review 가치가 있으면 model constrained generation 쪽에 연결하고, 아니면 contour/cadence extraction을 더 강화한다.

### Phase 3.14. Data Motif Review Export

목표:

- `data_motif`와 `hand_written_swing` 후보를 mode/sample/rank 기준으로 구분한다.
- piano-roll/listening review가 가능하도록 named MIDI package를 만든다.
- review markdown에 핵심 metric을 같이 남긴다.

현재 결과:

- completed: `review_manifest.json`과 `review_candidates.md`를 생성한다.
- completed: `named_midi/` 아래에 mode가 드러나는 MIDI 파일명을 만든다.
- latest result: review candidates `6`.
- latest result: `data_motif` strict `3/3`, `hand_written_swing` strict `3/3`.

해석:

- 이제 숫자 비교가 아니라 실제 청취 리뷰가 가능하다.
- syncopation 하락이 체감상 나쁜지, duration repetition 감소가 실제로 더 나은지 확인해야 한다.

다음 통과 기준:

- named MIDI 후보를 직접 듣고 piano roll로 확인한다.
- data_motif가 더 자연스러우면 motif sampling을 model constrained generation에 연결한다.
- 둘 다 초급 scale exercise면 cadence/phrase-ending extraction을 먼저 강화한다.

### Phase 3.15. Chord Context and Straight-Grid Review

목표:

- solo-only MIDI 리뷰의 한계를 줄인다.
- chord/bass guide가 포함된 context MIDI를 생성한다.
- swing/motif timing이 문제인지 확인하기 위해 straight-grid reference를 같이 export한다.

현재 결과:

- completed: `chord_guide.mid`를 생성한다.
- completed: candidate별 `*_with_context.mid`를 생성한다.
- completed: `straight_grid` baseline mode를 추가했다.
- latest result: `data_motif` strict `3/3`, `hand_written_swing` strict `3/3`.
- latest result: `straight_grid`는 timing reference로 export한다.

해석:

- 이제 chord progression 위에서 line이 in인지 out인지 들을 수 있다.
- swing/motif가 musical swing이 아니라 timing drift처럼 들리는지 비교할 수 있다.
- straight_grid는 더 좋은 솔로가 아니라 timing 기준점이다.

다음 통과 기준:

- context MIDI를 직접 듣고 `data_motif`, `hand_written_swing`, `straight_grid`를 비교한다.
- swing이 거슬리면 generated output은 straight quantized grid를 기본으로 둔다.
- chord context 위에서도 phrase가 초급스럽다면 cadence/phrase-ending extraction을 먼저 강화한다.

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

```text
Stage B rhythm/phrase vocabulary variation probe
```

결과:

- docs: `docs/STAGE_B_RHYTHM_PHRASE_VARIATION_2026-05-25.md`
- harness: `bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation`
- compare output: `outputs/stage_b_data_motif_compare/harness_stage_b_rhythm_phrase_variation/data_motif_compare_report.md`
- review package: `outputs/stage_b_data_motif_review/harness_stage_b_rhythm_phrase_variation/review_candidates.md`
- candidate count: `6`
- `data_motif_rhythm_phrase_variation`:
  - strict: `3/3`
  - final landing resolved: `3/3`
  - max interval: `6`
  - objective flags: `{}`
  - unresolved large leap ratio: `0.000`
  - repeated pitch interval ratio: `0.000`
  - syncopation: `0.694`
  - duration diversity: `0.097`
  - IOI diversity: `0.115`
- comparison `data_motif_contour_landing_repair`:
  - syncopation: `0.625`
  - duration diversity: `0.062`
  - IOI diversity: `0.079`
- tradeoff:
  - note count: `60`
  - average tension ratio: `0.371`
  - listening notes still pending

다음 작업:

- 다음 issue는 `Stage B rhythm/phrase variation MIDI-note proxy review`로 잡는다.
- 새 variation 후보가 실제로 `too_stiff`와 `too_repetitive` proxy 판단을 줄이는지 확인한다.
- tension ratio 하락이 too-safe 문제로 들리는지 확인한다.
- note count 60이 sparse하게 느껴지는지 확인한다.
- objective clean 후보라도 broad training으로 넘어가지 않는다.
- real Brad/reference chord label은 아직 임의로 넣지 않는다.
- LMDM/audio diffusion은 장기 live instrument reference로만 남기고, 현재 MVP를 audio로 pivot하지 않는다.

## 10. 한 문장 요약

이 프로젝트의 현재 핵심은 다음이다.

> Brad-style jazz MIDI model을 바로 만드는 것이 아니라, reviewable jazz solo-line MIDI를 만들 수 있는 symbolic representation, dataset window, generation, decoding, and evaluation pipeline을 먼저 증명하는 것이다.
