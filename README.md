# Realtime Jazz Solo AI (Stage A MVP)

FL Studio/MCP에서 받은 MIDI 조건 입력을 바탕으로, LoRA 파인튜닝된 Music Transformer가 솔로 MIDI를 생성하는 프로젝트입니다.

핵심 목표는 2가지입니다.
1. 먼저 "무조건 돌아가는" Stage A MVP를 만든다.
2. 그다음 dead-air 같은 품질 지표를 점진적으로 개선한다.

## 이 문서에서 얻는 것

처음 보는 사람도 아래 순서대로 실행하면 됩니다.
1. 환경 확인
2. 데이터 준비
3. 학습
4. 생성
5. 평가
6. (선택) dead-air 자동 스윕

---

## 0) 프로젝트 구조 한눈에 보기

```text
scripts/
  runpod_train_stage_a.sh          # Stage A 일괄 실행(prepare/train/generate/eval)
  prepare_role_dataset.py          # role-conditioned 데이터셋 생성
  train_qlora.py                   # LoRA 학습
  run_stage_a_tiny_overfit.py      # 1-3개 MIDI tiny-overfit smoke
  generate.py                      # 조건부 MIDI 생성
  eval_offline_metrics.py          # dead-air/반복률/밀도 평가
  run_dead_air_sweep.sh            # dead-air 개선 실험 자동화
  select_best_dead_air_candidate.py# 스윕 결과에서 베스트 자동 선택

docs/
  README.md
  CURRENT_STATUS_AND_PLAN.md
  MVP_PRD.md
  ONE_MONTH_ROADMAP.md
  MVP_IMPLEMENTATION_PLAN.md
  SYSTEM_ARCHITECTURE.md
  API_SPEC.md
  ERD.md
  INFERENCE_MODEL_SPEC.md
  QA_ACCEPTANCE_PLAN.md
  CODEX_EXECUTION_GUIDE.md
  JAMBOT_MIDI_REFACTOR_PLAN.md
  MAGENTA_RT_FINETUNING_GUIDE.md
  RUNPOD_GUIDE.md
```

---

## 1) 실행 환경

- Python 3.10+
- `pip`
- 권장: RunPod RTX 4090 (학습), 로컬(생성/실험)

초기 설치:

```bash
pip install -r requirements.txt
```

GPU 확인:

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu :", torch.cuda.get_device_name(0))
PY
```

---

## 2) 가장 빠른 시작 (권장)

`prepare -> train -> generate -> eval`을 한 번에 실행합니다.

```bash
bash scripts/runpod_train_stage_a.sh \
  --mode all \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --transpose_all_keys \
  --overwrite \
  --install_deps
```

성공 기준:
- `checkpoints/jazz_lora_stage_a/checkpoint_epoch*.pt` 생성
- `checkpoints/jazz_lora_stage_a/lora_weights.pt` 생성
- `samples/stage_a/jazz_sample_*.mid` 생성
- `samples/stage_a/metrics.json` 생성

---

## 2-1) MVP 데모 실행

이미 `checkpoints/jazz_lora_stage_a/checkpoint_epoch*.pt`가 있는 로컬 환경에서는 아래 명령으로 현재 MVP inference contract를 바로 확인합니다. `scripts/generate.py`와 inference runner는 full checkpoint를 우선 로드하고, 없을 때만 legacy `lora_weights.pt`로 fallback합니다.

```bash
bash scripts/run_mvp_demo.sh
```

고정 데모 조건:
- BPM: `124`
- Chords: `Cm7,Fm7,Bb7,Ebmaj7`
- Bars: `2`
- Section/Energy/Density: `drop/high/medium`
- Sampling: `temperature=0.9`, `top_p=0.95`, `model_candidates=2`
- Inference length: `max_sequence=256`
- Seed: `13`

생성 산출물:
- `outputs/demo/demo_request.json`
- `outputs/demo/generated.mid`
- `outputs/demo/metrics.json`
- `outputs/demo/result.json`
- `outputs/demo/generated/mvp_demo_medium_cminor.mid`
- `outputs/demo/metrics/mvp_demo_medium_cminor.json`

Demo metrics에는 MIDI validity 지표와 함께 request chord progression 기준 pitch-class hit ratio인 `chord_tone_ratio`가 포함됩니다. 이 값은 현재 gate가 아니라 코드 반응성을 관찰하기 위한 품질 지표입니다.
Model candidate selection은 이 값을 약하게 반영합니다. `chord_tone_ratio`가 `0.55`보다 낮은 valid candidate에는 작은 penalty를 주지만, passing tone과 tension을 고려해 실패 gate로 쓰지는 않습니다.

Review-ready gate 기준선:
- 256-token 27-case sweep: completed `27/27`
- solo-line gate 기준 model success `10/27`
- fallback `17/27`
- medium model success `1/9`
- 모든 completed output 최소 note count `3`
- 모든 completed output 최소 unique pitch count `3`
- 모든 completed output 최소 phrase coverage ratio 약 `0.76`
- 평균 generation time 약 `8.9s/request`

이 기준선은 단순히 MIDI 파일이 쓰였는지가 아니라, 최소한 리뷰할 가치가 있는 phrase인지 확인합니다. 1-note/2-note MIDI, 같은 pitch만 반복하는 MIDI, 2-bar 요청인데 앞부분에만 몰려 있는 MIDI, 전체 phrase를 물고 있는 sustain block, 여러 음이 길게 겹친 chord block은 valid model output으로 세지 않고 다른 model candidate 또는 fallback으로 넘어갑니다.

현재 해석:
- Stage A 모델은 dense 일부를 제외하면 solo-line MIDI를 안정적으로 만들지 못한다.
- 특히 medium density는 duration/note-off가 망가진 sustain block이 많아 fallback 의존도가 높다.
- 따라서 다음 모델 작업은 chord-tone postprocess보다 duration/token/conditioning 구조를 고치는 Stage B 쪽이 우선이다.

이 데모의 목적은 모델 연구 성능을 과장하는 것이 아니라, request-derived MIDI conditioning, model generation, repair, metrics gate, fallback contract가 하나의 재현 가능한 파이프라인으로 동작하는지 확인하는 것입니다.

---

## 2-2) Stage A Tiny Overfit Smoke

현재 우선순위는 더 큰 기능을 붙이는 것이 아니라, 모델이 아주 작은 MIDI solo grammar를 배울 수 있는지 확인하는 것입니다.
아래 명령은 1-3개의 deterministic MIDI solo phrase를 만들고, 작은 Music Transformer checkpoint를 overfit한 뒤, raw model sample과 MVP inference gate 결과를 리포트로 저장합니다.

```bash
python scripts/run_stage_a_tiny_overfit.py \
  --sample_count 3 \
  --epochs 200 \
  --lr 0.001 \
  --max_sequence 128 \
  --primer_max_tokens 24
```

빠른 구조 확인만 할 때:

```bash
python scripts/run_stage_a_tiny_overfit.py \
  --sample_count 1 \
  --epochs 1 \
  --max_sequence 96 \
  --primer_max_tokens 16
```

산출물:
- `outputs/stage_a_tiny_overfit/<run_id>/input_midi/*.mid`
- `outputs/stage_a_tiny_overfit/<run_id>/tokenized/{train,val}/*.npy`
- `outputs/stage_a_tiny_overfit/<run_id>/checkpoints/checkpoint_epoch*.pt`
- `outputs/stage_a_tiny_overfit/<run_id>/raw_samples/jazz_sample_*.mid`
- `outputs/stage_a_tiny_overfit/<run_id>/report.json`
- `outputs/stage_a_tiny_overfit/<run_id>/report.md`

판단 기준:
- `fallback_used=false`가 나오면 현재 tokenization/training path를 더 실험할 가치가 있다.
- 계속 fallback이면 conditioning을 확장하지 말고, NOTE_ON/OFF tokenization 또는 학습 scope를 먼저 다시 봐야 한다.

full-model tiny와 random-base LoRA-only를 같은 조건으로 비교할 때:

```bash
python scripts/compare_stage_a_tiny_modes.py \
  --sample_count 3 \
  --epochs 200 \
  --lr 0.001 \
  --max_sequence 128 \
  --primer_max_tokens 24
```

---

## 3) 단계별 실행 (문제 추적할 때 추천)

### 3-1. 데이터 준비

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir ./data/roles \
  --role lead \
  --transpose_all_keys \
  --overwrite
```

결과물:
- `data/roles/lead/<id>/conditioning.mid`
- `data/roles/lead/<id>/target.mid`
- `data/roles/lead/<id>/meta.json`
- `data/roles/lead/tokenized/train/*.npy`
- `data/roles/lead/tokenized/val/*.npy`

### 3-2. 학습

```bash
python scripts/train_qlora.py \
  --data_dir ./data/roles/lead/tokenized \
  --epochs 3 \
  --batch_size 8 \
  --num_workers 4 \
  --max_sequence 512 \
  --output_dir ./checkpoints/jazz_lora_stage_a
```

로그에서 반드시 확인:
- `Using device: cuda`
- `Saved best LoRA weights`
- `checkpoint_epoch*.pt` 저장

### 3-3. 생성

Stage A 모델을 직접 호출하는 기본 생성 명령입니다.
`--lora_path` 아래에 `checkpoint_epoch*.pt`가 있으면 가장 최신 epoch의 full `model_state_dict`를 우선 로드합니다.

```bash
python scripts/generate.py \
  --lora_path ./checkpoints/jazz_lora_stage_a \
  --conditioning_midi ./data/roles/lead/000000/conditioning.mid \
  --primer_max_tokens 64 \
  --num_samples 10 \
  --length 512 \
  --max_sequence 512 \
  --output ./samples/stage_a_p64
```

MVP inference contract를 검증할 때는 아래 CLI를 사용합니다. 이 경로는 `--conditioning_midi`를 생략하면 요청한 코드 진행으로 low-register conditioning MIDI를 임시 생성한 뒤 모델 primer로 사용합니다.

```bash
python -m inference.app.generator \
  --bpm 124 \
  --chords Cm7,Fm7,Bb7,Ebmaj7 \
  --bars 2 \
  --density medium \
  --energy high \
  --seed 11 \
  --temperature 0.9 \
  --top_p 0.95 \
  --model_candidates 2 \
  --max_sequence 256
```

### 3-4. 평가

```bash
python scripts/eval_offline_metrics.py \
  --input ./samples/stage_a_p64 \
  --dead_air_threshold_ms 180 \
  --output_json ./samples/stage_a_p64/metrics.json
```

핵심 지표 해석:
- `avg_dead_air_ratio`: 낮을수록 좋음
- `avg_repetition_4gram`: 낮을수록 좋음
- `avg_note_density`: 너무 낮거나 높으면 불안정

---

## 4) Dead-Air 개선 (자동 스윕)

아래 스크립트는 dead-air 개선 실험을 자동으로 수행합니다.

- Primer sweep: `64,96,128,160`
- split_pitch sweep: `55,60,64` (1 epoch 재학습)
- 베스트 후보 자동 선정 + 재검증(`num_samples=20`)
- 결과 아카이브 생성

```bash
bash scripts/run_dead_air_sweep.sh \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau"
```

결과 확인:
- `samples/dead_air_sweep/summary.json`
- `samples/dead_air_sweep/summary.md`
- `samples/dead_air_sweep/*/metrics.json`
- `dead_air_sweep_artifacts.tgz`

---

## 5) 빠른 로컬 검증

모델 로딩 없이 request conditioning과 chord timeline만 빠르게 확인합니다.

```bash
python -m unittest tests.test_request_conditioning
```

---

## 6) RunPod에서 돌릴 때 최소 체크리스트

1. Pod 생성: PyTorch 템플릿 + RTX 4090 1장
2. SSH 또는 Web Terminal 접속
3. 레포 클론 후 실행

```bash
git clone https://github.com/ohhalim/fine_tuning_art-tatum_midi_solo.git
cd fine_tuning_art-tatum_midi_solo

bash scripts/runpod_train_stage_a.sh \
  --mode all \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --transpose_all_keys \
  --overwrite \
  --install_deps
```

주의:
- 학습은 RunPod에서, 실시간 추론은 로컬에서 진행하는 흐름을 권장합니다.
- Pod를 `Stop`하면 컨테이너 디스크는 초기화될 수 있으니 아티팩트를 먼저 백업하세요.

---

## 7) 자주 막히는 문제

### `ModuleNotFoundError: pretty_midi`

```bash
pip install -r requirements.txt
```

### GPU를 안 쓰는 것 같음

로그에 `Using device: cuda`가 나와야 정상입니다.

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

### `tokenized_train` 또는 `tokenized_val`이 0

- `--input_dir` 경로 재확인
- MIDI 파일 수 확인
- `prepare_role_dataset.py`에서 `--split_pitch` 조정
- `--overwrite`로 재생성

---

## 8) 다음 단계 (MVP 이후)

자세한 현재 상태와 새 실행 계획은 `docs/CURRENT_STATUS_AND_PLAN.md`를 기준으로 봅니다.

1. 기존 checkpoint로 Stage A `generate -> eval` 재현성 확인
2. 빈 MIDI 또는 density 0 결과를 실패로 처리하는 generation/eval gate 추가
3. control token 기반 conditioning 포맷 설계
4. 이후 realtime 런타임 골격으로 확장

---

## 9) 최신 실험 결과 (2026-02-20)

Dead-air 스윕 결과, Stage A 기본 생성 파라미터는 아래로 확정했습니다.

- 모델: `checkpoints/jazz_lora_stage_a`
- 기본 primer: `--primer_max_tokens 64`
- Dead-air(20샘플): `0.454905`
- 비교군 p96(20샘플): `0.482452`

비교(20샘플):

| Candidate | Dead-air | Repetition | Note Density |
|---|---:|---:|---:|
| p64 (best, 20 samples) | 0.454905 | 0.019829 | 1.040259 |
| p96 (20 samples) | 0.482452 | 0.013617 | 0.990787 |

## 10) 참고 문서

- 문서 인덱스: `docs/README.md`
- 현재 기준 문서: `docs/CURRENT_STATUS_AND_PLAN.md`
- PRD: `docs/MVP_PRD.md`
- 한 달 로드맵: `docs/ONE_MONTH_ROADMAP.md`
- 구현 계획: `docs/MVP_IMPLEMENTATION_PLAN.md`
- 아키텍처/API/ERD: `docs/SYSTEM_ARCHITECTURE.md`, `docs/API_SPEC.md`, `docs/ERD.md`
- Codex 작업 가이드: `docs/CODEX_EXECUTION_GUIDE.md`
- 초기 실행 계획/실험 기록: `docs/JAMBOT_MIDI_REFACTOR_PLAN.md`
- Magenta RT 참고: `docs/MAGENTA_RT_FINETUNING_GUIDE.md`
- RunPod 참고: `docs/RUNPOD_GUIDE.md`

---

## 11) MVP Generation Contract

API 서버를 붙이기 전, 아래 CLI가 항상 valid MIDI와 metrics JSON을 만드는 Python 코어입니다.

```bash
python -m inference.app.generator \
  --bpm 124 \
  --chords Cm7,Fm7,Bb7,Ebmaj7 \
  --bars 2 \
  --section drop \
  --energy high \
  --density medium \
  --temperature 0.9 \
  --top_p 0.95 \
  --model_candidates 2 \
  --max_sequence 256 \
  --output_dir outputs/generated
```

결과:
- `outputs/generated/<job_id>.mid`
- `outputs/metrics/<job_id>.json`
- `outputs/generated/_conditioning/<job_id>_conditioning.mid`

모델 출력은 먼저 pitch range와 phrase length를 repair한 뒤 gate를 통과시키고, 그래도 실패하면 fallback MIDI를 생성합니다. 결과 JSON의 `model_repaired`, `fallback_used`, `model_failure_reason`으로 어떤 경로를 탔는지 확인합니다.

Metrics JSON에는 기본 validity 지표와 chord quality proxy가 함께 저장됩니다.

- `note_count`, `note_density`, `dead_air_ratio`, `repetition_score`
- `pitch_min`, `pitch_max`, `duration_sec`
- `unique_pitch_count`, `unique_pitch_class_count`
- `expected_duration_sec`, `phrase_coverage_ratio`
- `avg_note_duration_sec`, `max_note_duration_sec`, `max_note_duration_ratio`
- `long_note_ratio`, `max_simultaneous_notes`
- `chord_tone_count`, `non_chord_tone_count`, `chord_tone_ratio`

Validity gate는 density별 최소 노트 수, 최소 unique pitch 수, phrase coverage, note duration, simultaneous note count를 확인합니다. 2-bar 기준 최소 note count는 sparse `3`, medium `4`, dense `8`입니다. 최소 unique pitch count는 sparse `2`, medium `3`, dense `4`입니다. Phrase coverage는 실제 note span이 요청 길이의 일정 비율 이상인지 보는 값이며, sparse `0.25`, medium `0.35`, dense `0.45` 미만이면 너무 짧게 뭉친 phrase로 보고 탈락시킵니다.

Solo-line gate:
- max note duration ratio: sparse `0.85`, medium `0.55`, dense `0.45`
- max long-note ratio: sparse `0.50`, medium `0.25`, dense `0.20`
- max simultaneous notes: sparse `2`, medium `2`, dense `3`

FastAPI inference server:

```bash
uvicorn inference.app.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -s -X POST http://localhost:8000/infer/midi \
  -H 'Content-Type: application/json' \
  -d '{
    "bpm": 124,
    "chordProgression": ["Cm7", "Fm7", "Bb7", "Ebmaj7"],
    "bars": 2,
    "section": "drop",
    "energy": "high",
    "density": "medium",
    "useModel": false
  }'
```

Model quality sweep:

```bash
python scripts/eval_generation_contract_sweep.py \
  --seeds 11,13,17 \
  --densities medium \
  --temperature 0.9 \
  --top_p 0.95 \
  --model_candidates 2 \
  --max_sequence 256 \
  --summary_json outputs/sweeps/generation_contract_sweep.json \
  --summary_md outputs/sweeps/generation_contract_sweep.md
```

Chord-tone error analysis:

```bash
python scripts/analyze_chord_tone_errors.py \
  --input_json outputs/sweeps/chord_scored_p256_27case.json \
  --threshold 0.5 \
  --output_json outputs/sweeps/chord_tone_error_analysis.json \
  --output_md outputs/sweeps/chord_tone_error_analysis.md
```

이 리포트는 낮은 `chord_tone_ratio` 샘플에서 어떤 pitch class가 active chord 밖으로 나가는지 확인하기 위한 진단용입니다. 이 결과를 본 뒤 postprocess를 건드릴지, Stage B conditioning/token 설계로 넘어갈지 결정합니다.

빠른 fallback-only smoke check:

```bash
python scripts/eval_generation_contract_sweep.py \
  --no_model \
  --seeds 1 \
  --chord_progressions "Cm7,Fm7,Bb7,Ebmaj7" \
  --densities medium
```
