# Jazz Piano MIDI Solo Generator MVP

MIDI 데이터를 token sequence로 변환하고, Music Transformer 계열 symbolic model을 학습해 짧은 재즈 솔로 후보 MIDI를 생성하는 로컬 MVP.

현재 목표는 완성형 재즈 연주 모델이 아니다. 목표는 **학습된 checkpoint를 사용해 2마디 솔로 후보 MIDI/WAV를 만들고, 재즈 솔로 후보로 남길 수 있는 결과의 수율을 높이는 것**이다.

## 무엇을 만들었나

- Stage B tokenized MIDI dataset 기반 Music Transformer 계열 모델 학습
- 학습 checkpoint 저장
- chord context 기반 2마디 솔로 후보 생성
- model logits 기반 token sampling
- MIDI 문법 붕괴를 줄이기 위한 constrained decoding
- 후보별 objective metric 계산
- 상위 MIDI 후보 export
- WAV 렌더링

## 왜 constrained decoding을 썼나

raw model generation은 note grammar가 자주 깨졌다.

- note가 거의 없는 MIDI
- note_on/note_off 구조 붕괴
- 반복 또는 빈 구간 증가
- MIDI review gate 통과 실패

그래서 모델을 버린 것이 아니라, **모델 checkpoint의 logits로 token을 고르되 MIDI 문법과 chord/rhythm 범위를 제한**했다. 이 방식은 규칙만으로 MIDI를 만든 것이 아니라, 학습된 모델의 확률분포를 사용한 constrained generation이다.

## 현재 확인 결과

사용 checkpoint:

- `outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/training_smoke/controlled_2048_512_maxseq160/checkpoints/checkpoint_epoch1.pt`

학습 조건:

- train records: `2048`
- val records: `512`
- train loss: `3.8923`
- validation loss: `3.0396`
- model: `n_layers=1`, `d_model=64`, `max_sequence=160`

생성 조건:

- chords: `Cm7, Fm7, Bb7, Ebmaj7`
- bars: `2`
- bpm: `124`
- candidates: `20`
- decoding: `constrained`
- pitch: chord-aware `approach_tensions`
- rhythm: `swing_motif`

수율:

- grammar-valid candidates: `20 / 20`
- MIDI review valid candidates: `18 / 20`
- strict valid candidates: `18 / 20`
- selected MIDI candidates: `5`
- rendered WAV files: `5`

반복 스윕:

- chord progression cases: `4`
- total candidates: `24`
- grammar-valid candidates: `24 / 24`
- strict valid candidates: `18 / 24`
- rendered WAV files: `8`
- lowest case: `Dm7, G7, Cmaj7, A7` strict `2 / 6`
- lowest case review: invalid `4 / 4` dead-air threshold miss, next target `duration_fill_or_overlap_aftercare`
- dead-air repair: `fill_n10` 기준 strict `2 / 6 -> 6 / 6`, dead-air fail `4 -> 0`
- repaired full sweep: `fill_n10` 기준 strict `24 / 24`, grammar-valid `24 / 24`, rendered WAV `8`
- listening review package: MIDI `8`, WAV `8`, review input template ready
- listening input guard: validated input `false`, preference fill `false`, next `objective_only_next_decision`
- objective-only decision: candidate `8`, selected objective candidates `4`, musical quality claim `false`
- next boundary: `music_transformer_solo_yield_larger_sample_repeatability_sweep`
- larger sample repeatability: total candidates `48`, strict `47 / 48`, grammar-valid `48 / 48`
- larger sample min case strict rate: `0.9167`, rendered WAV files `12`, musical quality claim `false`
- larger sample listening package: MIDI `12`, WAV `12`, review input template ready
- larger sample input guard: validated input `false`, preference fill `false`, pending candidate fields `36`
- next boundary: `music_transformer_solo_yield_objective_only_next_decision`
- larger sample objective-only decision: candidate `12`, selected objective candidates `6`, musical quality claim `false`
- next boundary: `music_transformer_solo_yield_4bar_phrase_expansion_probe`

## 결과 파일

MIDI:

- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/generated/candidate_01_sample_20.mid`
- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/generated/candidate_02_sample_16.mid`
- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/generated/candidate_03_sample_12.mid`
- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/generated/candidate_04_sample_06.mid`
- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/generated/candidate_05_sample_04.mid`

WAV:

- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/audio/candidate_01_sample_20.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/audio/candidate_02_sample_16.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/audio/candidate_03_sample_12.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/audio/candidate_04_sample_06.wav`
- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/audio/candidate_05_sample_04.wav`

Report:

- `outputs/music_transformer_finetune_mvp/stage_b_solo_yield_probe/constrained_chord_swing_20/report.json`
- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/solo_yield_package.json`
- `outputs/music_transformer_finetune_mvp/solo_yield_mvp/constrained_chord_swing_top5/solo_yield_package.md`
- `docs/STAGE_B_MIDI_TO_SOLO_CHORD_PROGRESSION_YIELD_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_YIELD_FAILURE_CASE_REVIEW_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_DEAD_AIR_REPAIR_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_REPAIRED_PROGRESSION_RETRY_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_REPAIRED_CANDIDATE_LISTENING_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_LARGER_SAMPLE_REPEATABILITY_SWEEP_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_LARGER_SAMPLE_LISTENING_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_LARGER_SAMPLE_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_LARGER_SAMPLE_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-11.md`

## 실행 방법

20개 후보 생성:

```bash
.venv/bin/python scripts/run_stage_b_generation_probe.py \
  --output_root outputs/music_transformer_finetune_mvp/stage_b_solo_yield_probe \
  --run_id constrained_chord_swing_20 \
  --checkpoint_dir outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/training_smoke/controlled_2048_512_maxseq160/checkpoints \
  --skip_prepare \
  --skip_train \
  --generation_mode constrained \
  --num_samples 20 \
  --seed 700 \
  --bpm 124 \
  --bars 2 \
  --chords Cm7,Fm7,Bb7,Ebmaj7 \
  --density medium \
  --temperature 0.85 \
  --top_k 8 \
  --max_sequence 160 \
  --constrained_note_groups_per_bar 8 \
  --coverage_aware_positions \
  --coverage_position_window 1 \
  --chord_aware_pitches \
  --chord_pitch_mode approach_tensions \
  --chord_pitch_repeat_window 2 \
  --constrained_pitch_min 55 \
  --constrained_pitch_max 84 \
  --constrained_max_adjacent_interval 7 \
  --jazz_rhythm_positions \
  --jazz_duration_tokens \
  --jazz_rhythm_profile swing_motif \
  --cap_duration_to_next_position \
  --avoid_reused_positions \
  --postprocess_overlap \
  --max_simultaneous_notes 1
```

상위 후보 패키징:

```bash
.venv/bin/python scripts/build_music_transformer_solo_yield_package.py \
  --probe_report outputs/music_transformer_finetune_mvp/stage_b_solo_yield_probe/constrained_chord_swing_20/report.json \
  --output_root outputs/music_transformer_finetune_mvp/solo_yield_mvp \
  --run_id constrained_chord_swing_top5 \
  --top_n 5
```

## 현재 한계

- 사람 기준으로 좋은 재즈 솔로라고 검증된 상태 아님
- Coltrane, Art Tatum, Oscar Peterson 수준의 긴 솔로 생성 아님
- Brad style adaptation 완료 아님
- 입력 MIDI 전체를 이해해 자유롭게 솔로를 만드는 단계 아님
- 현재는 chord context와 constrained decoding을 사용한 2마디 후보 생성 MVP

## 다음 작업

- top 5 WAV 청취 리뷰
- 사람이 rejected로 판단한 후보의 공통 실패 원인 기록
- `Dm7, G7, Cmaj7, A7` progression 실패 원인 분석
- dead-air repair sweep
- `fill_n10` repair variant 기준 full progression retry sweep
- retry top candidates listening review
- listening review input guard
- objective-only next decision
- larger sample repeatability sweep
- larger sample candidate listening review package
- larger sample listening input guard
- larger sample objective-only next decision
- 4bar phrase expansion probe
- 더 긴 4마디 phrase로 확장 검토
