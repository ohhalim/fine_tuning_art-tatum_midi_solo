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
  generate.py                      # 조건부 MIDI 생성
  eval_offline_metrics.py          # dead-air/반복률/밀도 평가
  run_dead_air_sweep.sh            # dead-air 개선 실험 자동화
  select_best_dead_air_candidate.py# 스윕 결과에서 베스트 자동 선택

docs/
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
- `checkpoints/jazz_lora_stage_a/lora_weights.pt` 생성
- `samples/stage_a/jazz_sample_*.mid` 생성
- `samples/stage_a/metrics.json` 생성

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

### 3-3. 생성

```bash
python scripts/generate.py \
  --lora_path ./checkpoints/jazz_lora_stage_a \
  --conditioning_midi ./data/roles/lead/000000/conditioning.mid \
  --primer_max_tokens 128 \
  --num_samples 10 \
  --length 512 \
  --max_sequence 512 \
  --output ./samples/stage_a_p128
```

### 3-4. 평가

```bash
python scripts/eval_offline_metrics.py \
  --input ./samples/stage_a_p128 \
  --dead_air_threshold_ms 180 \
  --output_json ./samples/stage_a_p128/metrics.json
```

핵심 지표 해석:
- `avg_dead_air_ratio`: 낮을수록 좋음
- `avg_repetition_4gram`: 낮을수록 좋음
- `avg_note_density`: 너무 낮거나 높으면 불안정

---

## 4) Dead-Air 개선 (자동 스윕)

아래 스크립트는 dead-air 개선 실험을 자동으로 수행합니다.

- Primer sweep: `96,128,160`
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

## 5) RunPod에서 돌릴 때 최소 체크리스트

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

## 6) 자주 막히는 문제

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

## 7) 다음 단계 (MVP 이후)

1. dead-air 최적 조합 고정
2. 베스트 설정으로 재검증(`num_samples=20`)
3. KPI 업데이트 (`docs/JAMBOT_MIDI_REFACTOR_PLAN.md`)
4. 이후 Stage B 확장(컨디셔닝 토큰 강화)

---

## 8) 참고 문서

- 실행 계획: `docs/JAMBOT_MIDI_REFACTOR_PLAN.md`
- Magenta RT 참고: `docs/MAGENTA_RT_FINETUNING_GUIDE.md`
- RunPod 참고: `docs/RUNPOD_GUIDE.md`
