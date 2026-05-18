# Current Status and Revised Plan

작성일: 2026-05-16

이 문서는 현재 프로젝트의 기준 문서다. 기존 `docs/JAMBOT_MIDI_REFACTOR_PLAN.md`는 초기 리팩터링 계획과 2026-02-20 dead-air 실험 결과 기록으로 유지하고, 앞으로의 실행 순서는 이 문서를 기준으로 갱신한다.

MVP 구현을 위한 세부 문서는 `docs/README.md`에서 시작한다.

2026-05-18 코드 리뷰 기준 문서:

- `docs/STAGE_A_CODE_REVIEW_2026-05-18.md`
- 결론: 현재 Stage A 문제는 metric threshold보다 학습/로드/디코딩/conditioning 구조 문제다. 다음 구현은 decoder fix, full checkpoint loading, tiny overfit smoke 순서로 진행한다.
- decoder fix는 완료됐다. `processor.py`의 `_merge_note()`가 `note_off` 처리 후 active note를 소비하도록 수정했고, stray `note_off`가 ghost sustain note를 만들지 않는 테스트를 추가했다.

## 1. 현재 결정 사항

- 메인 라인: Symbolic MIDI 기반 `Music Transformer + LoRA`.
- 보조 리서치 라인: `magenta-realtime/` 및 Magenta RT 문서는 오디오 기반 실험 자료로만 유지.
- 학습 위치: RunPod GPU.
- 생성/실시간 추론 목표 위치: 로컬 머신 + FL Studio/MCP 연동.
- Stage A 기본 생성값: `--primer_max_tokens 64`.

## 2. 현재 구현 상태

현재 브랜치는 `feature/magenta-rt-jazz-finetuning`이다.

완료된 축:

- `scripts/prepare_role_dataset.py`
  - `conditioning.mid`, `target.mid`, `meta.json` 생성.
  - `conditioning + TOKEN_END + target + TOKEN_END` 형식으로 tokenized train/val 생성.
- `scripts/train_qlora.py`
  - Music Transformer에 LoRA를 붙여 학습.
  - best validation 기준으로 `lora_weights.pt` 저장.
- `scripts/generate.py`
  - full `checkpoint_epoch*.pt` 또는 legacy LoRA checkpoint와 conditioning MIDI를 받아 MIDI 샘플 생성.
  - `--lora_path` 아래 최신 `checkpoint_epoch*.pt`를 우선 로드한다.
  - checkpoint에 `model_config`가 있으면 tiny model 구조도 자동 복원한다.
  - Stage A 기본 primer 길이는 64 token.
  - `temperature`, `top_k`, `top_p` sampling 제어값 지원.
- `scripts/run_stage_a_tiny_overfit.py`
  - 1~3개 known-good MIDI phrase를 생성하고 tokenized train/val dataset을 만든다.
  - 작은 Stage A checkpoint를 overfit하고 raw sample 및 MVP gate 결과를 report로 저장한다.
  - 기본은 LoRA wrapper를 유지하되 base model까지 unfreeze하는 tiny-overfit mode다.
  - 2026-05-18 local `dense_overfit_200` run에서 `fallback_used=false`로 MVP gate를 통과했다.
  - 같은 조건의 `--lora_only` run은 best val loss `4.8228`, decoded raw notes `0`으로 실패했다.
- `scripts/eval_offline_metrics.py`
  - note density, dead-air proxy, 4-gram repetition 평가.
- `scripts/analyze_chord_tone_errors.py`
  - sweep JSON에서 낮은 chord-tone ratio 샘플을 선택.
  - MIDI를 다시 읽어 active chord별 non-chord pitch class를 집계.
  - postprocess 조정 또는 Stage B conditioning/token 설계로 넘어가기 전 진단 리포트 생성.
- `scripts/runpod_train_stage_a.sh`
  - prepare/train/generate/eval 단일 실행 파이프라인.
- `scripts/run_dead_air_sweep.sh`
  - primer sweep, split_pitch 재학습 sweep, best candidate 선택, 재검증, archive 생성.
- `inference/app/generator.py`
  - model-first generation contract.
  - 요청 chord progression 기반 conditioning MIDI를 생성해 Stage A primer로 사용.
  - 여러 model candidate를 생성한 뒤 repair/metrics gate 통과 후보 중 quality score가 가장 낮은 MIDI를 선택.
  - candidate score는 dead-air, repetition, target density 이탈, 낮은 chord-tone ratio penalty를 반영한다.
  - raw model output repair 후 gate 검증.
  - 실패 시 fallback MIDI 생성.
- `inference/app/metrics.py`
  - note density, dead-air, repetition, pitch range를 계산.
  - `unique_pitch_count`, `unique_pitch_class_count`, `expected_duration_sec`, `phrase_coverage_ratio`를 계산.
  - request chord progression 기준 `chord_tone_count`, `non_chord_tone_count`, `chord_tone_ratio`를 계산.
  - chord-tone metric은 현재 gate가 아니라 코드 반응성을 보기 위한 관측 지표다.
  - density별 최소 note count gate를 적용한다. 2-bar 기준 sparse `3`, medium `4`, dense `8`.
  - density별 최소 unique pitch gate를 적용한다. sparse `2`, medium `3`, dense `4`.
  - phrase coverage gate를 적용한다. sparse `0.25`, medium `0.35`, dense `0.45`.
  - solo-line gate를 적용한다. max note duration ratio는 sparse `0.85`, medium `0.55`, dense `0.45`이고, max simultaneous notes는 sparse/medium `2`, dense `3`이다.
- `inference/app/conditioning.py`
  - structured request를 low-register chord guide MIDI로 변환.
  - `--conditioning_midi` 명시값이 없을 때 기본 primer로 사용.
- `inference/app/postprocess.py`
  - pitch range octave mapping.
  - 첫 note 기준 phrase 정렬.
  - 요청 bars 기준 trim.
  - dense request에서 16분음표 chord-tone gap fill 적용.
  - medium request에서 큰 공백만 제한적으로 chord-tone 보정.

최근 smoke sweep:

- 범위: `Cm7,Fm7,Bb7,Ebmaj7`, seed `11,13,17`, density `sparse,medium,dense`.
- repair 전 model success: `5/9`.
- dense-only density repair 후 model success: `8/9`.
- fallback: `4/9 -> 1/9`.
- 3개 chord progression x 3 seed x 3 density sweep에서 medium gap repair 후 model success: `27/27`.
- 현재 작은 sweep 기준 fallback: `0/27`.
- request-derived conditioning smoke: 3개 chord progression x 1 seed x medium density에서 model success `3/3`, fallback `0/3`.
- sampling/candidate selection smoke: 3개 chord progression x 1 seed x medium density, `temperature=0.9`, `top_p=0.95`, `model_candidates=2`에서 model success `3/3`, fallback `0/3`.
- sampling/candidate 27-case sweep: 3개 chord progression x 3 seed x 3 density, `temperature=0.9`, `top_p=0.95`, `model_candidates=2`에서 model success `26/27`, fallback `1/27`.
  - dense: `9/9`
  - medium: `9/9`
  - sparse: `8/9`
  - 유일한 fallback: `sweep_model_p2_sparse_s17`, model candidate 2개 모두 dead-air `1.000 >= 0.900`.
  - 평균 generation time: 약 `31.3s/request`.
- in-process runner + `max_sequence=256` smoke:
  - 3개 chord progression x 1 seed x medium density에서 model success `3/3`, fallback `0/3`, 평균 약 `10.1s/request`.
  - 3개 chord progression x 1 seed x 3 density에서 model success `9/9`, fallback `0/9`, 평균 약 `9.4s/request`.
- in-process runner + `max_sequence=256` 27-case sweep:
  - 3개 chord progression x 3 seed x 3 density에서 model success `27/27`, fallback `0/27`.
  - sparse/medium/dense 모두 `9/9`.
  - 평균 generation time: 약 `9.5s/request`.
  - 평균 note density: 약 `2.92`.
  - 평균 dead-air ratio: 약 `0.31`.
  - 산출물: `outputs/sweeps/inprocess_runner_p256_27case_sparse_policy.json`, `outputs/sweeps/inprocess_runner_p256_27case_sparse_policy.md`.
  - 해석: 이전 `23/27` 결과는 sparse 후보의 긴 공백을 dead-air gate가 과도하게 탈락시킨 것이 주된 원인이었다. sparse는 density/pitch/note-count gate를 통과하면 긴 공백을 허용한다.
- MVP demo contract:
  - 실행 명령: `bash scripts/run_mvp_demo.sh`
  - 산출물: `outputs/demo/demo_request.json`, `outputs/demo/generated.mid`, `outputs/demo/metrics.json`, `outputs/demo/result.json`.
  - 기본 demo seed는 `13`.
  - 현재 demo 결과: `COMPLETED`, fallback `false`, model repaired `true`, note count `7`, chord-tone ratio 약 `0.43`.
- chord-tone scored 27-case sweep:
  - 산출물: `outputs/sweeps/chord_scored_p256_27case.json`, `outputs/sweeps/chord_scored_p256_27case.md`.
  - model success `27/27`, fallback `0/27`.
  - 평균 generation time 약 `9.8s/request`.
  - 평균 chord-tone ratio 약 `0.60`.
  - density별 chord-tone 평균: sparse 약 `0.38`, medium 약 `0.57`, dense 약 `0.85`.
- chord-tone error analysis:
  - 산출물: `outputs/sweeps/chord_tone_error_analysis.json`, `outputs/sweeps/chord_tone_error_analysis.md`.
  - threshold `0.5` 이하 low sample: `10/27`.
  - low sample 평균 chord-tone ratio: 약 `0.32`.
  - low sample 분포: sparse `7`, medium `3`, dense `0`.
  - top non-chord pitch classes: `F#`, `B`, `Bb`, `Ab`, `F`.
  - 해석: dense는 chord-tone 안정성이 높고, sparse/medium 일부가 코드 밖 pitch class를 반복한다. 이제 postprocess를 건드리기 전에 수동 청취와 리뷰가 필요하다.
- minimum note-count gate sweep:
  - 산출물: `outputs/sweeps/min_note_gate_p256_27case.json`, `outputs/sweeps/min_note_gate_p256_27case.md`.
  - model success `23/27`, fallback `4/27`, failed `0/27`.
  - 모든 completed output의 최소 note count는 `3`.
  - fallback 전환 사유는 주로 model candidate의 `note count too low`.
  - 평균 chord-tone ratio는 약 `0.64`.
  - 해석: 이전 분석에서 드러난 1-note/2-note 엉성한 phrase를 valid model output으로 세던 문제가 수정됐다.
- minimum note-count gate 이후 chord-tone error analysis:
  - 산출물: `outputs/sweeps/min_note_gate_chord_tone_error_analysis.json`, `outputs/sweeps/min_note_gate_chord_tone_error_analysis.md`.
  - threshold `0.5` 이하 low sample: `8/27`.
  - low sample 평균 chord-tone ratio: 약 `0.23`.
  - low sample 분포: sparse `5`, medium `3`, dense `0`.
- review-ready phrase gate sweep:
  - 산출물: `outputs/sweeps/review_ready_gate_p256_27case.json`, `outputs/sweeps/review_ready_gate_p256_27case.md`.
  - model success `24/27`, fallback `3/27`, failed `0/27`.
  - fallback 전환 사유는 note count 부족 2건, unique pitch 부족 1건이다.
  - 모든 completed output의 최소 note count는 `3`.
  - 모든 completed output의 최소 unique pitch count는 `3`.
  - 모든 completed output의 최소 phrase coverage ratio는 약 `0.76`.
  - 평균 chord-tone ratio는 약 `0.61`.
  - 해석: 1-note/2-note MIDI, 한 pitch만 반복하는 MIDI, 요청 길이 대비 앞부분에만 몰린 MIDI를 리뷰 대상에서 제외하는 기준이 들어갔다. fallback이 늘어난 것은 품질 기준이 올라간 결과다.
- review-ready gate 이후 chord-tone error analysis:
  - 산출물: `outputs/sweeps/review_ready_gate_chord_tone_error_analysis.json`, `outputs/sweeps/review_ready_gate_chord_tone_error_analysis.md`.
  - threshold `0.5` 이하 low sample: `10/27`.
  - low sample 평균 chord-tone ratio: 약 `0.35`.
  - low sample 분포: sparse `7`, medium `3`, dense `0`.
  - top non-chord pitch classes: `Eb`, `Ab`, `C`, `C#`, `E`.
  - 해석: 낮은 chord-tone 문제는 더 이상 1-note/2-note 착시만은 아니다. sparse/medium에서 chord conditioning 자체가 약한 신호로 보고, 강한 postprocess보다 Stage B conditioning/token 설계 검토가 우선이다.
- solo-line gate sweep:
  - 산출물: `outputs/sweeps/solo_line_gate_p256_27case.json`, `outputs/sweeps/solo_line_gate_p256_27case.md`.
  - model success `10/27`, fallback `17/27`, failed `0/27`.
  - density별 model success: dense `6/9`, medium `1/9`, sparse `3/9`.
  - fallback 전환 사유는 주로 `note duration too long`이며, 일부는 `note count too low`, `too many simultaneous notes`, `too many long notes`다.
  - 평균 chord-tone ratio는 약 `0.84`지만, 이 값은 fallback 증가의 영향이 크다.
  - 해석: Stage A model output은 dense 일부를 제외하면 solo-line MIDI로 보기 어렵다. 특히 medium density에서 긴 sustain block과 chord block이 많이 나온다.
- solo-line gate 이후 chord-tone error analysis:
  - 산출물: `outputs/sweeps/solo_line_gate_chord_tone_error_analysis.json`, `outputs/sweeps/solo_line_gate_chord_tone_error_analysis.md`.
  - threshold `0.5` 이하 low sample: `4/27`.
  - low sample 평균 chord-tone ratio: 약 `0.38`.
  - low sample 분포: sparse `3`, medium `1`, dense `0`.
  - 해석: 말이 안 되는 sustain/chord block을 제거하면 chord-tone 문제는 줄어들지만, 그 대가로 model success가 크게 낮아진다. 다음 개선은 postprocess 미세조정보다 Stage B token/duration/conditioning 재설계가 우선이다.

주의할 점:

- `music_transformer/generate.py`는 원본 Music Transformer 계열의 legacy generator다.
- 현재 Stage A에서 사용해야 하는 생성 엔트리포인트는 `scripts/generate.py`다.
- MVP request contract 확인은 `python -m inference.app.generator`를 사용한다.
- 현재 inference 기본값은 `max_sequence=256`이다. 512-token 생성은 더 느린 비교/실험용으로 유지한다.
- full checkpoint loading path는 구현됐다. 현재 `scripts/generate.py`, inference CLI, FastAPI model runner는 `checkpoint_epoch*.pt`를 우선 로드하고, 없을 때만 legacy `lora_weights.pt`로 fallback한다.
- 새 checkpoint는 `model_config`를 저장하므로, tiny overfit처럼 작은 architecture로 학습한 checkpoint도 generation loader가 자동 복원할 수 있다.
- 다만 현재 checkpoint 자체는 작은 데이터/epoch 1 결과라서, Stage A 결과를 실사용 가능한 jazz solo model로 해석하면 안 된다.
- `music_transformer/third_party/midi_processor/processor.py`의 decoder ghost sustain 문제는 수정됐다. 그래도 model output은 아직 note count 부족/긴 note 문제로 자주 fallback된다.
- 로컬 워크트리에는 추적되지 않은 데이터, 샘플, 문서가 많다. 문서/코드 작업 시 기존 산출물을 정리하거나 삭제하지 않는다.

## 3. 현재 로컬 산출물

데이터:

- `data/roles/lead/dataset_summary.json`
  - samples: 18
  - train: 16
  - val: 2
  - transpose_all_keys: false
- `data/roles_sp60/lead/dataset_summary.json`
  - samples: 216
  - train: 194
  - val: 22
  - transpose_all_keys: true

체크포인트:

- `checkpoints/jazz_lora_stage_a/checkpoint_epoch1.pt`
- `checkpoints/jazz_lora_stage_a/lora_weights.pt`
- `checkpoints/jazz_lora_sp60/checkpoint_epoch1.pt`
- `checkpoints/jazz_lora_sp60/lora_weights.pt`

실험 기록:

- README와 `docs/JAMBOT_MIDI_REFACTOR_PLAN.md`에는 2026-02-20 기준 full dead-air sweep 결과가 기록되어 있다.
- 현재 로컬 `samples/dead_air_sweep_smoke/`는 smoke run 결과이며, full sweep 결과와 동일한 기준 데이터로 보지 않는다.

## 4. Stage A 재현 명령

가장 빠른 전체 실행:

```bash
bash scripts/runpod_train_stage_a.sh \
  --mode all \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --transpose_all_keys \
  --overwrite \
  --install_deps
```

기존 checkpoint로 생성/평가만 확인:

```bash
python scripts/generate.py \
  --lora_path ./checkpoints/jazz_lora_stage_a \
  --conditioning_midi ./data/roles/lead/000000/conditioning.mid \
  --primer_max_tokens 64 \
  --num_samples 5 \
  --length 512 \
  --max_sequence 512 \
  --output ./samples/stage_a

python scripts/eval_offline_metrics.py \
  --input ./samples/stage_a \
  --dead_air_threshold_ms 180 \
  --output_json ./samples/stage_a/metrics.json
```

## 5. 다시 잡은 작업 계획

### Phase 0. 문서와 기준선 정리

목표:

- 현재 기준 문서를 만든다.
- README에서 최신 기준 문서로 연결한다.
- legacy generator와 Stage A generator의 역할을 명확히 분리한다.

완료 기준:

- `docs/CURRENT_STATUS_AND_PLAN.md` 존재.
- README의 참고 문서에 현재 기준 문서가 표시됨.

### Phase 1. Stage A 재현성 고정

목표:

- 기존 checkpoint로 `generate -> eval`을 다시 실행해 현재 환경에서 결과가 나오는지 확인한다.
- `samples/stage_a/metrics.json`을 기준 출력으로 재생성한다.
- smoke run과 full run 결과를 구분해 기록한다.

완료 기준:

- `samples/stage_a/*.mid` 생성.
- `samples/stage_a/metrics.json` 생성.
- empty/undecodable MIDI가 있으면 원인 기록.

### Phase 2. 생성 품질 안정화

목표:

- 빈 MIDI 또는 note density 0 결과를 실패로 처리한다.
- `scripts/generate.py`에 sampling 제어값을 추가한다.
- 최소 후보:
  - temperature
  - top-k
  - top-p
  - retry-on-empty
- 평가 스크립트에 fail gate를 추가한다.

완료 기준:

- 생성 결과 중 빈 MIDI 비율이 리포트에 표시됨.
- dead-air, repetition, note density가 모두 gating 기준에 들어감.
- 실패 샘플은 파일명 또는 별도 JSON에 명확히 표시됨.
- 여러 model candidate 중 metrics score 기준으로 최종 MIDI를 선택함.

현재 상태:

- 기본 sampling 후보값은 `temperature=0.9`, `top_p=0.95`, `model_candidates=2`로 둔다.
- 256-token 27-case sweep 기준 model success는 `27/27`.
- inference 기본 `max_sequence`는 256으로 둔다.
- sparse는 silence가 의도인 density mode이므로 dead-air 상한 gate를 적용하지 않는다. 단, note count, note density, pitch range gate는 유지한다.
- 남은 문제는 request당 약 9~10초가 걸리는 autoregressive generation 병목과 실제 음악성/코드 적합성 평가다.
- README의 MVP demo command는 `bash scripts/run_mvp_demo.sh`로 고정한다.
- chord-tone ratio는 metrics JSON에 포함하고 candidate score에 약하게 반영한다. 현재는 실패 기준으로 쓰지 않고, Stage B 또는 postprocess 개선 전후 비교 지표로 사용한다.
- 다음 구현을 시작하기 전에 여기서 리뷰한다. 특히 model output의 duration/note-off 문제가 우선이다. 1-note/2-note phrase, 한 pitch 반복 phrase, 요청 길이 대비 너무 짧게 몰린 phrase, 전체 phrase를 물고 있는 sustain block, 여러 음이 길게 겹친 chord block은 이제 리뷰 대상이 아니라 invalid candidate로 처리한다.

### Phase 3. Conditioning 의미 강화

목표:

- 현재 `TOKEN_END`를 separator처럼 쓰는 구조를 명시적인 control token 구조로 바꾼다.
- 후보 토큰:
  - `ROLE_LEAD`
  - `TEMPO_*`
  - `COND_SEP`
  - `BAR`
- 학습 스크립트를 `train_role_lora.py`로 분리할지, 기존 `train_qlora.py`에 role mode를 둘지 결정한다.

완료 기준:

- tokenized sequence format이 문서화됨.
- 새 포맷으로 작은 데이터셋 학습 1회 성공.
- 기존 Stage A 포맷과 새 포맷이 혼동되지 않음.

### Phase 4. Realtime 런타임 골격

목표:

- 새 패키지 `realtime/` 추가.
- 최소 모듈:
  - `clock.py`
  - `midi_input.py`
  - `prompt_builder.py`
  - `generation_worker.py`
  - `scheduler.py`
  - `main.py`
- 입력, 프롬프트 구성, 생성, 출력 스케줄링을 분리한다.

완료 기준:

- 로컬에서 MIDI in -> generated MIDI out 루프가 돌아감.
- 10분 smoke run 동안 크래시 없음.
- TTFN, queue underrun, generated notes 수가 로그에 남음.

### Phase 5. 실시간 KPI 튜닝

목표:

- TTFN과 jitter를 계측한다.
- 항상 1~2 bar ahead queue를 유지한다.
- queue underrun 시 fallback phrase를 사용한다.

완료 기준:

- Stage A runtime gate:
  - TTFN <= 120ms
  - 10분 연속 재생 크래시 없음
  - dead-air threshold 180ms 기준 악화 없음
- 이후 목표:
  - TTFN < 80ms
  - jitter 5~10ms 범위

### Phase 6. 포트폴리오 산출물

목표:

- 60초 데모 MIDI 생성.
- FL Studio/MCP 연결 흐름 캡처.
- 모델/데이터/결과 지표를 한 페이지로 요약.

완료 기준:

- `samples/demo_60s/` 산출물.
- 최종 README에 실행 명령, 결과 지표, 한계점 반영.

## 6. 다음 실행 순서

가장 먼저 할 일:

1. tiny overfit smoke는 full-model tiny mode에서 통과했고, LoRA-only random-base mode는 실패했다.
2. 현재 Stage A의 public wording에서 "LoRA fine-tuned jazz model" 표현을 피하고, full checkpoint training 또는 pretrained base 확보를 우선한다.
3. 다음 구현은 from-scratch full training path와 adapter training path를 명확히 분리하는 것이다.
4. 그다음 control token 기반 Conditioning 포맷으로 넘어간다. Duration 문제가 다시 나오면 NOTE_ON/OFF 대신 duration-explicit tokenization으로 넘어간다.

즉, 바로 realtime이나 API 확장으로 가지 않는다. 먼저 현재 생성 파이프라인의 디코딩/체크포인트/학습 구조를 바로잡고, 그다음 실시간 런타임으로 연결한다.
