# Jazz Piano MIDI-to-Solo 모델 검증 파이프라인

## 개요

Symbolic MIDI 기반 jazz piano solo-line 생성 모델의 학습, 생성, 디코딩, 검증 흐름을 작은 실험 단위로 검증한 model-core 프로젝트.

현재 범위는 완성형 재즈 연주 모델이 아니라, 입력 MIDI를 context로 변환하고 model-conditioned generation과 constrained decoding을 거쳐 ranked solo MIDI/WAV 후보를 만드는 실행 경로다.

## 현재 상태

| 항목 | 상태 |
|---|---|
| pipeline MVP | 완료 |
| MIDI-to-solo execution path | 입력 MIDI -> context -> ranked MIDI -> WAV technical path 검증 |
| current evidence boundary | `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_path_complete` |
| generation source | `controlled_scale_checkpoint_generation_probe` |
| full generic window preparation | train `154136` / val `21845` tokenized records |
| scale checkpoint training smoke | train `128` / val `32`, best validation loss `5.9031`, checkpoint `1` |
| sequence budget repair | max sequence `96 -> 160`, direct note capacity `17 -> 33` |
| model-direct 8-bar gate | grammar / valid / strict `3 / 3 / 3` |
| contour phrase repeatability | generated / qualified `6 / 6`, flags / overlap `0 / 0` |
| rendered review WAV | `6` files, duration `18.865s-19.000s` |
| listening review input | pending fields `4 / 6 / 18` |
| controlled training scale smoke | train / val `512 / 128`, max sequence `160`, best validation loss `5.1061`, checkpoint `1` |
| controlled scale checkpoint generation probe | sample `3`, valid / strict / grammar `0 / 0 / 3`, collapse warning `3`, repair decision 필요 |
| controlled scale checkpoint repair decision | selected target `target_density_collapse_postprocess_repair`, next density/collapse repair probe |
| controlled density/collapse repair probe | note-count failure `3 -> 0`, collapse warning `3 -> 0`, avg postprocess removal `0.8090 -> 0.2292`, valid / strict / grammar `0 / 0 / 3` |
| controlled dead-air remaining blocker decision | selected target `dead_air_sustained_coverage_repair`, dead-air failure `3`, next dead-air repair probe |
| controlled dead-air repair probe | note groups/bar `12`, valid / strict / grammar `3 / 3 / 3`, dead-air failure `3 -> 0`, repeatability 필요 |
| controlled dead-air repair repeatability probe | seeds `44/52/60`, valid / strict / grammar `7 / 7 / 9`, seed `60` partial failure, temperature guard decision 필요 |
| controlled dead-air repeatability temperature guard decision | selected target `lower_temperature_repeatability_guard_repair`, source/selected temp `0.9 -> 0.75`, top_k `4` 유지 |
| controlled dead-air repeatability temperature guard repair probe | temp `0.75`, seeds `44/52/60`, valid / strict / grammar `9 / 9 / 9`, dead-air/collapse failure `0 / 0` |
| controlled dead-air repeatability temperature guard repair consolidation | objective MIDI support `true`, audio review package required `true`, quality claim `false` |
| controlled dead-air repeatability temperature guard audio review package | rendered WAV `3`, duration `6.747s-6.861s`, technical validation `true`, preference claim `false` |
| controlled dead-air repeatability temperature guard listening review | review template `true`, pending fields `4 / 3 / 9`, preference fill `false` |
| controlled dead-air repeatability temperature guard objective next | objective path support `true`, valid / strict / grammar `9 / 9 / 9`, next training scale decision |
| controlled scale checkpoint training scale decision | selected train / val `2048 / 512`, current `512 / 128`, local bounded smoke |
| controlled scale checkpoint training scale smoke | train / val `2048 / 512`, best validation loss `3.0396`, checkpoint `1` |
| controlled scale checkpoint training scale generation probe | sample `3`, valid / strict / grammar `0 / 0 / 2`, collapse warning `3` |
| controlled scale checkpoint training scale repair decision | selected target `target_density_grammar_collapse_postprocess_repair`, next density/grammar/collapse repair probe |
| controlled scale checkpoint training scale density/grammar/collapse repair probe | valid / strict / grammar `1 / 1 / 3`, note-count/grammar/collapse failure `0 / 0 / 0`, repeatability 필요 |
| controlled scale checkpoint training scale density/grammar/collapse repeatability probe | seeds `47/52/60`, valid / strict / grammar `2 / 2 / 9`, dead-air failure `7`, next dead-air decision |
| human/audio preference | 미검증 |
| MIDI-to-solo musical quality | 미검증 |
| broad trained-model quality | 미주장 |
| Brad style adaptation | 미진행 |
| realtime DAW/plugin, backend/API, SaaS/UI | 범위 밖 |

최신 판단:

- evidence boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_path_complete`
- documentation status: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe`
- next engineering boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_dead_air_remaining_blocker_decision`
- objective MIDI repeatability path support: `true`
- objective temperature guard path support: `true`
- controlled training scale smoke ready: `true`
- selected next training scale: `2048 / 512`
- selected scale training smoke result: validation loss `3.0396`, checkpoint `1`
- selected scale generation probe result: valid / strict / grammar `0 / 0 / 2`
- selected scale repair target: `target_density_grammar_collapse_postprocess_repair`
- selected scale repair probe result: valid / strict / grammar `1 / 1 / 3`, note-count/grammar/collapse failure `0 / 0 / 0`
- selected scale repair repeatability result: seeds `47/52/60`, valid / strict / grammar `2 / 2 / 9`, dead-air failure `7`
- input MIDI to ranked candidate technical path: `true`
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
| Input MIDI context | bar/position/chord/bass context row 추출, empty bar chord carry-forward |
| Generation probe | checkpoint 기반 raw generation, constrained generation, coverage-aware position, duration token 적용 |
| Decode / postprocess | generated token sequence MIDI 복원, overlap-free solo-line 후보 생성 |
| Candidate ranking | objective gate 기반 ranked MIDI candidate export |
| Audio render package | rendered WAV technical metadata 검증 |
| Objective MIDI review | note count, unique pitch, phrase coverage, dead-air, max active notes, long-note ratio, repeated cell, max interval 검증 |
| Repair boundary | sequence budget, pitch contour, timing/dead-air, jazz phrase vocabulary, contour phrase-shape target 분리 |
| Repeatability sweep | seed 범위 확장, aggregate pass-rate, failure reason, claim boundary 기록 |
| Listening review guard | review input 부재 시 preference fill과 musical quality claim 차단 |
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
| 8-bar direct generation budget 부족 | 8-bar / 24-note contract tokens `123`, previous max sequence `96` | sequence budget `160`으로 repair | direct note capacity `17 -> 33`, strict valid `3/3` |
| direct candidate contour failure | max interval `82`, wide interval/register flags `3/3` | pitch contour repair | max interval `82 -> 9`, wide interval/register flags `0/0` |
| timing/dead-air failure | max dead-air ratio `0.6522` | timing phrase repair | max dead-air ratio `0.6522 -> 0.2258`, dead-air flags `3 -> 0` |
| songlike melody 문제 | user listening review `reject_all`, primary failure `songlike_melody_not_soloing` | jazz phrase vocabulary repair target 분리 | fixed-density/four-note/duration/IOI/interval-cap/four-bar-cycle flags `0/0/0/0/0/0` |
| stepwise contour bias | contour bias `3/3` | contour phrase-shape repair | stepwise contour bias `3 -> 0`, max interval `11` |
| density 부족 | note-count failure `3/3` | constrained note-group density와 coverage-aware position 적용 | density/coverage repair valid/strict/grammar `1/1/3` |
| long-note ratio failure | long-note failures `2` | jazz duration token과 duration/long-note repair 적용 | duration repair valid/strict/grammar `2/2/3`, long-note failure delta `2` |
| dead-air 잔여 병목 | dead-air failure `1`, sustained coverage regression 관측 | sustained coverage/dead-air repair, constrained note groups per bar `8` | repair valid/strict/grammar `3/3/3`, dead-air/long-note `0/0` |
| 단일 seed 과장 위험 | objective gate support가 single seed set에 한정 | objective gate repeatability sweep 추가 | seeds `44/52/60`, valid/strict/grammar `9/9/9`, failure reasons none |
| MIDI-to-solo 반복성 과장 위험 | contour phrase candidate 3개만으로는 반복성 부족 | repeatability sweep과 consolidation 추가 | generated/qualified `6/6`, flags/overlap `0/0`, pass rate `1.0000` |
| controlled checkpoint raw generation 실패 | sample `3`, valid/strict/grammar `0/0/3`, collapse warning rate `1.0`, avg/max postprocess removal `0.8090/0.8636` | generation probe와 repair decision 경계 분리 | note count `3-4 < 6`, quality claim 제외 |
| repair target 혼선 위험 | grammar gate `3/3` 통과, valid/strict `0/0`, postprocess removal high | postprocess-only/training-scale/audio-review 제외, density/collapse/postprocess repair target 선택 | selected target `target_density_collapse_postprocess_repair` |
| controlled density/collapse repair 후 잔여 병목 | note-count failure `0`, collapse warning `0`, dead-air failure `3` | coverage-aware position, chord-aware pitch, jazz rhythm/duration token, duration fill 적용 | avg postprocess removal `0.8090 -> 0.2292`, avg onset/sustained `0.0833/0.1667 -> 0.4583/0.7188`, strict gate 미회복 |
| dead-air repair target 분리 필요 | density/collapse target support `true`, strict gate recovered `false`, dead-air failure `3/3` | audio review/training-scale change 제외, dead-air sustained coverage repair target 선택 | next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe` |
| controlled dead-air repair 반복성 미검증 | 단일 seed-set에서 valid/strict/grammar `3/3/3`, dead-air failure `3 -> 0` | note groups/bar `8 -> 12`, 같은 chord/rhythm/duration guard 유지 | avg onset/sustained `0.4583/0.7188 -> 0.5729/0.7292`, next repeatability probe |
| controlled dead-air repair 반복성 partial | seeds `44/52/60`, strict `7/9`, collapse warning `1` | 동일 #562 조건으로 seed sweep 실행 | seed `60` failure `2`, next temperature guard decision |
| controlled dead-air repeatability temperature guard 필요 | source temp/top_k `0.9/4`, strict shortfall `2`, failed seed `[60]` | temp `0.75`, top_k `4` 고정 guard 선택 | next temperature guard repair probe |
| controlled dead-air repeatability temperature guard repair | temp `0.75`, top_k `4`, seeds `44/52/60` | lower-temperature guard 조건으로 seed sweep 재실행 | valid/strict/grammar `9/9/9`, dead-air/collapse `0/0`, next consolidation |
| controlled dead-air repeatability temperature guard support 정리 | strict shortfall `2 -> 0`, dead-air/collapse `2/1 -> 0/0` | objective MIDI support와 quality claim boundary 분리 | audio review package required `true`, musical quality claim `false` |
| controlled dead-air repeatability temperature guard audio review | seed별 대표 MIDI 후보 `3`개 | fluidsynth 기반 WAV 렌더와 technical metadata 검증 | rendered WAV `3`, duration `6.747s-6.861s`, listening review pending |
| controlled dead-air repeatability listening review pending | WAV 후보 `3`개, validated review input `false` | review input template 생성, preference fill 차단 | pending fields `4/3/9`, next objective-only decision |
| controlled temperature guard objective path 정리 | strict `9/9`, dead-air/collapse `0/0`, validated review input `false` | preference/quality claim 차단 상태로 objective-only 경계 완료 | next boundary `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision` |
| controlled training scale 확장 필요 | current smoke `512/128`, objective path support `true`, full records `154136/21845` | local bounded `2048/512`, max_sequence `160`, 1 epoch 선택 | full training/cloud spend 제외, next training smoke |
| selected training scale 실행 필요 | selected `2048/512`, max_sequence `160`, 1 epoch | local bounded training smoke 실행 | returncode `0`, best validation loss `3.0396`, checkpoint `1`, next generation probe |
| selected scale generation 실패 | sample `3`, valid/strict `0/0`, collapse warning `3/3` | checkpoint generation probe 결과를 repair decision으로 라우팅 | postprocess removal avg/max `0.7909/0.8`, next repair decision |
| selected scale repair target 분리 | valid/strict/grammar `0/0/2`, note-count/collapse `3/3`, grammar failure `1` | postprocess-only/audio/additional scale 제외, density/grammar/collapse/postprocess repair target 선택 | selected target `target_density_grammar_collapse_postprocess_repair`, next repair probe |
| selected scale density/grammar/collapse repair | note-count/collapse/grammar failure `3/3/1`, avg postprocess removal `0.7909` | constrained note-group density, coverage-aware position, chord-aware pitch, jazz rhythm/duration token 적용 | valid/strict/grammar `1/1/3`, postprocess removal `0.1875`, next repeatability |
| selected scale repair 반복성 | sample `3` 기준 target support만으로는 부족 | seeds `47/52/60` repeatability sweep 실행 | density/grammar/collapse support `true`, valid/strict/grammar `2/2/9`, dead-air failure `7`, next dead-air decision |
| 음악 품질 claim 과장 위험 | objective MIDI gate와 청감 품질의 분리 필요 | listening review guard와 claim boundary 문서화 | pending fields `4/6/18`, musical quality/human preference/broad quality claim `false` |

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
| input context bars / events | `8 / 128` |
| inferred / carried-forward / unknown chord bars | `4 / 4 / 0` |
| model-direct sequence max | `160` |
| direct 8-bar minimum contract tokens | `123` |
| direct note capacity | `33` |
| direct 8-bar grammar / valid / strict | `3 / 3 / 3` |
| pitch contour max interval | `82 -> 9` |
| timing repair max dead-air ratio | `0.6522 -> 0.2258` |
| jazz phrase repair generated MIDI | `3` |
| contour phrase stepwise bias | `3 -> 0` |
| contour phrase repeatability generated / qualified | `6 / 6` |
| contour phrase repeatability flags / overlap | `0 / 0` |
| contour phrase repeatability pass rate | `1.0000` |
| contour phrase repeatability rendered WAV | `6` |
| listening review pending fields | `4 / 6 / 18` |
| controlled scale smoke selected train / val records | `512 / 128` |
| controlled scale smoke max sequence | `160` |
| controlled scale smoke best validation loss | `5.1061` |
| controlled scale smoke checkpoint count | `1` |
| controlled checkpoint generation probe | sample `3`, valid/strict/grammar `0/0/3` |
| controlled checkpoint collapse warning | count/rate `3/1.0` |
| controlled checkpoint avg/max postprocess removal | `0.809042809042809 / 0.8636363636363636` |
| controlled checkpoint repair decision | selected target `target_density_collapse_postprocess_repair` |
| controlled density/collapse repair probe | sample `3`, valid/strict/grammar `0/0/3`, note-count/collapse failure `0/0`, dead-air failure `3` |
| controlled density/collapse repair deltas | note-count failure `3`, collapse warning `3`, postprocess removal `0.5798761423761424` |
| controlled density/collapse coverage delta | onset/sustained `0.375 / 0.5520833333333334` |
| controlled dead-air remaining blocker decision | selected target `dead_air_sustained_coverage_repair`, audio/training-scale selected `false/false` |
| controlled dead-air repair probe | sample `3`, valid/strict/grammar `3/3/3`, note-count/dead-air/collapse failure `0/0/0` |
| controlled dead-air repair deltas | dead-air failure `3`, valid/strict sample `3/3`, postprocess removal `+0.10416666666666666` |
| controlled dead-air repair repeatability probe | seeds `44/52/60`, sample `9`, valid/strict/grammar `7/7/9`, collapse warning `1` |
| controlled dead-air repeatability failure reasons | `dead-air ratio too high: 0.800 >= 0.800; collapse=postprocess_removed_majority`: `1`, `dead-air ratio too high: 0.846 >= 0.800`: `1` |
| controlled dead-air repeatability temperature guard decision | selected target `lower_temperature_repeatability_guard_repair`, source/selected temp `0.9/0.75`, top_k `4` |
| controlled dead-air temperature guard evidence | strict shortfall `2`, failed seed `[60]`, dead-air failure `2`, collapse warning `1` |
| controlled dead-air temperature guard repair probe | temp `0.75`, seeds `44/52/60`, valid/strict/grammar `9/9/9`, dead-air/collapse `0/0` |
| controlled dead-air temperature guard consolidation | objective support `true`, audio review package required `true`, quality claim `false` |
| controlled dead-air temperature guard audio review package | rendered WAV `3`, sample rate `44100`, duration `6.747s-6.861s` |
| controlled dead-air temperature guard listening review | template written `true`, pending status/candidate/field `4/3/9` |
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
| input MIDI -> context -> ranked MIDI -> WAV technical path | 검증 |
| full generic manifest window preparation | 검증 |
| scale checkpoint training smoke | 검증 |
| raw checkpoint generation 실패 감지 | 검증 |
| constrained objective repair path | 검증 |
| model-direct 8-bar candidate generation | objective gate 범위 검증 |
| model-direct contour phrase repeatability | generated/qualified `6/6` 범위 검증 |
| controlled training scale smoke | `512/128`, max_sequence `160`, checkpoint `1` 범위 검증 |
| controlled scale checkpoint generation/decode path | sample `3`, grammar `3/3` 범위 검증 |
| controlled scale checkpoint review gate | valid/strict `0/0`, repair decision 필요 |
| controlled scale checkpoint repair target | density/collapse/postprocess repair 범위 결정 |
| controlled scale checkpoint density/collapse repair target | note-count/collapse/postprocess 개선, dead-air 잔여 병목 분리 |
| controlled scale checkpoint dead-air repair target | dead-air sustained coverage repair target 선택 |
| controlled scale checkpoint dead-air repair single-seed support | valid/strict `3/3`, repeatability 미검증 |
| controlled scale checkpoint dead-air repeatability boundary | seed `60` partial failure 분리, temperature guard decision 완료 |
| controlled scale checkpoint temperature guard target | lower-temperature repeatability guard 선택, source/selected temp `0.9 -> 0.75` |
| controlled scale checkpoint temperature guard repair target | temp `0.75`, top_k `4` 조건에서 strict `9/9`, failure reasons none |
| controlled scale checkpoint temperature guard support | objective MIDI 범위 통과, audio review package required |
| controlled scale checkpoint audio review package | WAV technical validation 통과, human/audio preference 미검증 |
| controlled scale checkpoint listening review boundary | review input pending, preference fill 차단 |
| `.mid` 파일 존재 기반 성공 판정 제거 | 검증 |
| one-note / long sustain / chord block 실패 감지 | 검증 |
| human/audio preference | 미검증 |
| MIDI-to-solo musical quality | 미검증 |
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

MIDI-to-solo repeatability objective decision:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repeatability-objective-next
```

MIDI-to-solo controlled training scale smoke:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-training-scale-smoke
```

MIDI-to-solo controlled scale checkpoint generation probe:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-generation-probe
```

MIDI-to-solo controlled scale checkpoint repair decision:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-repair-decision
```

MIDI-to-solo controlled scale checkpoint density/collapse repair probe:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-density-collapse-repair-probe
```

MIDI-to-solo controlled scale checkpoint dead-air remaining blocker decision:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-remaining-blocker-decision
```

MIDI-to-solo controlled scale checkpoint dead-air repair probe:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repair-probe
```

MIDI-to-solo controlled scale checkpoint dead-air repair repeatability probe:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repair-repeatability-probe
```
