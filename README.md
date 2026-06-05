# Jazz Piano MIDI-to-Solo 검증 파이프라인

Symbolic MIDI 기반 jazz piano solo-line 생성 파이프라인.

현재 목표는 완성형 연주 모델이 아니라, 입력 MIDI를 받아 context 추출, 후보 생성, constrained decoding, ranking, MIDI/WAV export, objective review까지 이어지는 model-core MVP 검증이다.

## 현재 상태

- latest evidence boundary: `stage_b_midi_to_solo_phrase_bank_listening_review_package`
- current evidence boundary: `stage_b_midi_to_solo_phrase_bank_listening_review_package`
- current MVP evidence support: `true`
- technical model-core MVP completed: `true`
- selected quality gap target: `model_conditioned_input_path_quality_alignment`
- model-conditioned input path aligned: `false`
- model-conditioned candidate source available: `true`
- model-conditioned audio technical path available: `true`
- model-conditioned ranked input-path export contract matched: `true`
- fallback replacement candidate export ready: `true`
- model-conditioned ranked audio render completed: `true`
- fallback replacement technical path ready: `true`
- fallback replacement ready: `true`
- listening review package required: `true`
- listening review package ready: `true`
- phrase-bank retrieval baseline completed: `true`
- phrase-bank source records / motifs: `56 / 803`
- phrase-bank exported / qualified MIDI candidates: `3 / 3`
- phrase-bank best notes / unique pitches / max simultaneous: `64 / 22 / 1`
- phrase-bank rendered WAV files: `3`
- phrase-bank audio technical validation: `true`
- phrase-bank listening review package ready: `true`
- phrase-bank listening review items: `3`
- validated review input: `false`
- input MIDI -> context -> ranked MIDI -> WAV technical path: `true`
- selected-scale objective repair path complete: `true`
- musical quality MVP completed: `false`
- product MVP completed: `false`
- human/audio preference claim: `false`
- MIDI-to-solo musical quality claim: `false`
- broad trained-model quality claim: `false`
- Brad style adaptation claim: `false`

현재 README는 아래 범위까지만 주장한다.

- 입력 MIDI 기반 context row 생성 가능
- ranked MIDI candidate export 가능
- technical WAV render 가능
- objective MIDI gate 기반 실패/개선 분리 가능
- selected-scale checkpoint repair path의 objective evidence 정리 가능
- model-conditioned strict MIDI/WAV technical evidence 존재
- model-conditioned ranked MIDI candidate export 가능
- model-conditioned ranked WAV technical render 가능
- 입력 MIDI context 기반 phrase-bank retrieval 후보 export 가능
- phrase-bank 후보의 WAV technical render 가능
- phrase-bank 후보의 listening review package 생성 가능

현재 README가 주장하지 않는 것.

- 최종 jazz solo 품질
- 사용자 청음 선호
- phrase-bank 후보의 청음 품질
- broad training 완료 모델 품질
- Brad Mehldau style adaptation
- realtime DAW/plugin 또는 product-ready improviser

## 현재 evidence

MVP completion audit.

- technical model-core MVP completed: `true`
- input to ranked MIDI completed: `true`
- input to rendered WAV completed: `true`
- selected-scale objective repair completed: `true`
- musical quality MVP completed: `false`
- human/audio preference completed: `false`
- product MVP completed: `false`

Quality gap decision.

- selected target: `model_conditioned_input_path_quality_alignment`
- fallback path active: `true`
- model-conditioned input path alignment required: `true`
- human review required now: `false`

Model-conditioned input path alignment.

- selected probe target: `replace_fallback_with_model_conditioned_input_path_probe`
- model-conditioned input path aligned: `false`
- fallback replacement probe required: `true`
- human review required now: `false`

Model-conditioned input path probe.

- model-conditioned source: `model_checkpoint_direct_constrained`
- model-conditioned candidate source available: `true`
- model-conditioned audio technical path available: `true`
- same input context as fallback: `true`
- ranked input-path export contract matched: `false`
- fallback replacement ready: `false`
- candidate export required: `true`

Model-conditioned input path candidate export.

- generation source: `model_checkpoint_direct_constrained`
- ranked MIDI candidates exported: `true`
- ranked input-path export contract matched: `true`
- exported candidate count: `3`
- best note / unique pitch / max simultaneous: `24 / 20 / 1`
- fallback replacement candidate export ready: `true`
- fallback replacement ready: `false`
- candidate audio render required: `true`

Model-conditioned input path audio render package.

- rendered audio file count: `3`
- technical WAV validation: `true`
- model-conditioned ranked audio render completed: `true`
- fallback replacement technical path ready: `true`
- fallback replacement ready: `true`
- WAV duration range: `19.585s - 22.390s`
- audio rendered quality claimed: `false`
- human/audio preference claimed: `false`

Model-conditioned input path replacement consolidation.

- model-conditioned input to ranked MIDI completed: `true`
- model-conditioned input to ranked WAV completed: `true`
- fallback replacement technical path ready: `true`
- listening review package required: `true`
- exported/rendered count: `3 / 3`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

Model-conditioned input path listening review package.

- package ready: `true`
- review item count: `3`
- validated review input: `false`
- review WAV files: `rank_01_sample_01.wav`, `rank_02_sample_02.wav`, `rank_03_sample_03.wav`
- human/audio preference claimed: `false`

Phrase-bank retrieval baseline.

- generation source: `phrase_bank_data_motif_retrieval`
- source records / motif count: `56 / 803`
- unique rhythm / contour templates: `520 / 328`
- candidate count: `9`
- qualified candidate count: `3`
- exported / exported qualified MIDI candidates: `3 / 3`
- best note / unique pitch / max simultaneous: `64 / 22 / 1`
- best dead-air / phrase coverage: `0.5873015873015873 / 1.0`
- MIDI-to-solo MVP claimed: `false`
- human/audio preference claimed: `false`

Phrase-bank audio render package.

- rendered WAV files: `3`
- technical WAV validation: `true`
- rank 1 duration / sample rate / sha256 prefix: `18.985s / 44100 / 07a95cfe5c4b`
- rank 2 duration / sample rate / sha256 prefix: `18.984s / 44100 / a3a3efc8a9e1`
- rank 3 duration / sample rate / sha256 prefix: `18.997s / 44100 / d3550541fe41`
- audio rendered quality claimed: `false`
- human/audio preference claimed: `false`

Phrase-bank listening review package.

- package ready: `true`
- review item count: `3`
- validated review input: `false`
- review WAV files: `rank_01_seed_635.wav`, `rank_02_seed_632.wav`, `rank_03_seed_638.wav`
- human/audio preference claimed: `false`
- MIDI-to-solo musical quality claimed: `false`

MIDI-to-solo input contract.

- candidate count: `32`
- exported MIDI candidates: `3`
- target solo bars: `8`
- min note count: `24`
- min unique pitch count: `8`
- max simultaneous notes: `1`
- fallback path: `phrase_retrieval_data_motif_hybrid`

Input context extraction.

- context bars / events: `8 / 128`
- positions per bar: `16`
- inferred / carry-forward / unknown chord bars: `4 / 4 / 0`
- low-confidence chord bars: `4`
- bass-note bars: `4`

Ranked MIDI generation.

- generation source: `context_conditioned_fallback`
- exported / qualified candidates: `3 / 3`
- best note count: `60`
- best unique pitch count: `14`
- best max simultaneous notes: `1`
- best chord tone ratio: `1.0`

Technical WAV render.

- rendered WAV files: `3`
- sample rate: `44100`
- duration range: `18.617s-18.991s`
- technical WAV validation: `true`

Selected-scale objective repair.

- final boundary: `stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_path_complete`
- sample / seed count: `9 / 3`
- valid / strict / grammar: `9 / 9 / 9`
- dead-air / collapse failure count: `0 / 0`
- avg / max postprocess removal ratio: `0.21759259259259262 / 0.2916666666666667`
- target avg postprocess removal ratio: `0.3`
- validated review input present: `false`
- preference fill allowed: `false`

## 구현 범위

Dataset and split.

- readable MIDI audit
- candidate file audit
- generic train/val split
- Brad holdout split
- duplicate hash guard

Stage B representation.

- `BAR`
- `POSITION`
- `CHORD_ROOT`
- `CHORD_QUALITY`
- `NOTE_PITCH`
- `NOTE_DURATION`
- `VELOCITY`

Training and checkpoint probe.

- full generic window preparation
- local bounded training smoke
- checkpoint artifact validation
- max sequence budget repair
- broad training claim guard

MIDI-to-solo execution path.

- input MIDI fixture generation
- bar/position/chord/bass context extraction
- checkpoint-conditioned or fallback candidate generation
- constrained monophonic decoding
- objective gate based ranking
- top MIDI export
- local WAV render

Review and repair path.

- note count gate
- unique pitch gate
- max simultaneous note gate
- dead-air ratio gate
- long-note ratio gate
- interval guard
- phrase coverage gate
- collapse warning
- repeatability sweep
- pending listening review guard

## 문제 / 조치 / 관측 결과

`.mid` 존재만으로 성공 판단 위험.

- 관측: one-note collapse, long sustain block, chord block 후보 발생
- 조치: `.mid exists` 성공 조건 제외, objective MIDI review gate 추가
- 결과: note-level failure reason 분리

Stage A representation 한계.

- 관측: `NOTE_ON/OFF` 중심 구조에서 duration/phrase 제어 부족
- 조치: Stage B duration-explicit tokenization 전환
- 결과: `POSITION`, `NOTE_DURATION`, chord context 기반 generation probe 가능

8-bar direct generation sequence budget 부족.

- 관측: 8-bar / 24-note contract tokens `123`, previous max sequence `96`
- 조치: max sequence `160` smoke
- 결과: direct note capacity `17 -> 33`, direct 8-bar strict valid `3/3`

Model-direct candidate phrase failure.

- 관측: max interval `82`, wide interval/register flags `3/3`
- 조치: pitch contour repair
- 결과: max interval `82 -> 9`, wide interval/register flags `0/0`

Model-direct timing/dead-air failure.

- 관측: max dead-air ratio `0.6522`, dead-air flags `3`
- 조치: timing phrase repair
- 결과: max dead-air ratio `0.6522 -> 0.2258`, dead-air flags `3 -> 0`

User listening rejection.

- 관측: preferred rank `3`, overall decision `reject_all`, primary failure `songlike_melody_not_soloing`
- 조치: jazz phrase vocabulary repair target 분리
- 결과: fixed-density / four-note template / duration monotony / IOI monotony / safe interval compression / 4-bar cycle flags `0/0/0/0/0/0`

Selected-scale checkpoint raw generation failure.

- 관측: sample `3`, valid / strict / grammar `0 / 0 / 2`, collapse warning `3`
- 조치: density/grammar/collapse/postprocess repair target 선택
- 결과: valid / strict / grammar `1 / 1 / 3`, note-count/grammar/collapse failure `0 / 0 / 0`

Selected-scale repair repeatability dead-air 병목.

- 관측: seeds `47/52/60`, valid / strict / grammar `2 / 2 / 9`, dead-air failure `7`
- 조치: selected-scale dead-air sustained coverage repair
- 결과: valid / strict / grammar `9 / 9 / 9`, dead-air/collapse `0 / 0`, avg postprocess removal `0.21759259259259262`

청음 claim 과장 위험.

- 관측: rendered WAV와 objective MIDI gate만으로 musical quality 판단 불가
- 조치: validated review input 없을 때 preference fill 차단
- 결과: human/audio preference claim `false`, MIDI-to-solo musical quality claim `false`

## 주요 산출물

Current evidence.

- `docs/STAGE_B_MIDI_TO_SOLO_MVP_CURRENT_EVIDENCE_CONSOLIDATION_2026-06-05.md`
- `outputs/stage_b_midi_to_solo_mvp_current_evidence_consolidation/harness_stage_b_midi_to_solo_mvp_current_evidence_consolidation/stage_b_midi_to_solo_mvp_current_evidence_consolidation.json`

Ranked MIDI candidates.

- `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_01_seed_489.mid`
- `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_02_seed_488.mid`
- `outputs/stage_b_midi_to_solo_conditioned_generation_probe/harness_stage_b_midi_to_solo_conditioned_generation_probe/midi/rank_03_seed_487.mid`

Rendered WAV candidates.

- `outputs/stage_b_midi_to_solo_candidate_audio_render_package/harness_stage_b_midi_to_solo_candidate_audio_render_package/audio/rank_01_seed_489.wav`
- `outputs/stage_b_midi_to_solo_candidate_audio_render_package/harness_stage_b_midi_to_solo_candidate_audio_render_package/audio/rank_02_seed_488.wav`
- `outputs/stage_b_midi_to_solo_candidate_audio_render_package/harness_stage_b_midi_to_solo_candidate_audio_render_package/audio/rank_03_seed_487.wav`

## 검증 명령

기본 회귀.

```bash
bash scripts/agent_harness.sh quick
```

현재 MVP evidence consolidation.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-mvp-current-evidence-consolidation
```

MIDI-to-solo input contract.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-mvp-contract
```

MIDI-to-solo context extraction.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-context-extraction
```

MIDI-to-solo conditioned generation.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-conditioned-generation-probe
```

MIDI-to-solo candidate audio render.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-candidate-audio-render-package
```

Selected-scale objective next decision.

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-objective-next
```

## 다음 작업

- Stage B MIDI-to-solo model-conditioned input path probe
- `context_conditioned_fallback` path와 selected-scale objective repair path의 quality gap 분리
- human listening review 입력 전까지 preference/musical quality claim 차단 유지
