# Real-time Jazz Solo MVP (Stage A)

이 프로젝트는 FL Studio + MCP 기반 실시간 MIDI 즉흥 시스템을 목표로 합니다.

현재 메인 라인은 `Symbolic MIDI Transformer + LoRA + role-conditioned dataset` 입니다.
`magenta-realtime/`은 오디오 실험용 참고 라인으로 유지합니다.

## MVP 목표 (Stage A)

- 입력: chord/gesture MIDI
- 모델: 조건부 심볼릭 Transformer (LoRA fine-tune)
- 출력: 생성된 solo MIDI
- 학습은 RunPod, 실시간 추론은 로컬 GPU 기준

상세 계획 문서: `docs/JAMBOT_MIDI_REFACTOR_PLAN.md`

## 빠른 시작 (학습 전 준비)

1. 의존성 설치

```bash
pip install -r requirements.txt
```

2. role-conditioned 데이터셋 생성

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir ./data/roles \
  --role lead \
  --overwrite
```

생성 결과:
- `data/roles/lead/<sample_id>/conditioning.mid`
- `data/roles/lead/<sample_id>/target.mid`
- `data/roles/lead/<sample_id>/meta.json`
- `data/roles/lead/tokenized/train/*.npy`
- `data/roles/lead/tokenized/val/*.npy`

3. RunPod 자동화 스크립트 확인

```bash
bash scripts/runpod_train_stage_a.sh --help
```

## Stage A 학습/생성 명령

학습:

```bash
python scripts/train_qlora.py \
  --data_dir ./data/roles/lead/tokenized \
  --epochs 3 \
  --batch_size 8 \
  --num_workers 0 \
  --max_sequence 512 \
  --output_dir ./checkpoints/jazz_lora_stage_a
```

조건부 생성:

```bash
python scripts/generate.py \
  --lora_path ./checkpoints/jazz_lora_stage_a \
  --conditioning_midi ./data/roles/lead/000000/conditioning.mid \
  --num_samples 3 \
  --length 512 \
  --max_sequence 512 \
  --output ./samples/stage_a
```

오프라인 평가:

```bash
python scripts/eval_offline_metrics.py \
  --input ./samples/stage_a \
  --dead_air_threshold_ms 180 \
  --output_json ./samples/stage_a/metrics.json
```

## RunPod 원클릭

```bash
bash scripts/runpod_train_stage_a.sh \
  --mode all \
  --role lead \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --max_files 200 \
  --overwrite
```

모드 분리 실행:
- `--mode prepare`
- `--mode train`
- `--mode generate`
- `--mode eval`

## 기존 문서

- 심볼릭 시스템 리팩터 계획: `docs/JAMBOT_MIDI_REFACTOR_PLAN.md`
- Magenta RT 가이드(오디오 실험용): `docs/MAGENTA_RT_FINETUNING_GUIDE.md`
- RunPod 가이드(구버전): `docs/RUNPOD_GUIDE.md`
