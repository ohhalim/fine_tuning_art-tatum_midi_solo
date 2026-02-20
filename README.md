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

## Dead-Air 스프린트 자동화

아래 스크립트는 계획된 dead-air 개선 스프린트를 한 번에 실행합니다.

- Phase 1: primer sweep (`96,128,160`)
- Phase 2: split_pitch sweep (`55,60,64`, 1epoch 재학습)
- Phase 3: 베스트 후보 자동 선택
- Phase 4: 베스트 후보 `num_samples=20` 재검증
- Phase 5: 아티팩트 tar 백업

```bash
bash scripts/run_dead_air_sweep.sh \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau"
```

주요 결과물:
- `samples/dead_air_sweep/summary.json`
- `samples/dead_air_sweep/summary.md`
- `samples/dead_air_sweep/*/metrics.json`
- `dead_air_sweep_artifacts.tgz`

## 기존 문서

- 심볼릭 시스템 리팩터 계획: `docs/JAMBOT_MIDI_REFACTOR_PLAN.md`
- Magenta RT 가이드(오디오 실험용): `docs/MAGENTA_RT_FINETUNING_GUIDE.md`
- RunPod 가이드(구버전): `docs/RUNPOD_GUIDE.md`
# Jazz Piano AI Generator

실시간 재즈 피아노 즉흥연주 AI - Art Tatum 스타일 솔로 생성

## 주요 기능

### Music Transformer QLoRA Fine-tuning
- **QLoRA (Quantized LoRA)**: MultiHeadAttention의 out_proj 레이어에 LoRA 적용 (학습 파라미터 0.71%)
- **데이터 파이프라인**: 재즈 MIDI → Music Transformer 토큰 시퀀스 변환
- **RunPod 최적화**: RTX 3090 (24GB VRAM)에서 60-70% 활용률

### Magenta RealTime (진행 중)
- **실시간 스트리밍**: 2초 청크 단위 오디오 생성
- **스타일 제어**: 텍스트/오디오 프롬프트로 스타일 조절
- **Fine-tuning**: 재즈 피아노 데이터로 스타일 커스터마이징

## 프로젝트 구조

```
├── midi_dataset/           # 재즈 피아노 MIDI 데이터 (2,777개)
├── music_transformer/      # Music Transformer 모델
├── magenta-realtime/       # Magenta RT 모델 (실시간 생성)
├── scripts/
│   ├── preprocess_jazz.py  # MIDI → 토큰 전처리
│   ├── train_qlora.py      # QLoRA 학습 스크립트
│   ├── generate.py         # MIDI 생성
│   └── runpod_setup.sh     # RunPod 환경 설정
├── data/                   # 전처리된 데이터
└── docs/                   # 문서
```

## 빠른 시작

### 1. 환경 설정
```bash
git clone https://github.com/ohhalim/fine_tuning_art-tatum_midi_solo.git
cd fine_tuning_art-tatum_midi_solo
pip install -r requirements.txt
```

### 2. 데이터 전처리
```bash
python scripts/preprocess_jazz.py
```

### 3. QLoRA 학습
```bash
python scripts/train_qlora.py --epochs 10 --batch_size 8
```

### 4. MIDI 생성
```bash
python scripts/generate.py --output output.mid
```

## RunPod 실행 (원클릭)
```bash
bash scripts/runpod_setup.sh
```

## 결과 (RunPod RTX 3090)

| 항목 | 값 |
|------|-----|
| VRAM 사용량 | ~60-70% |
| 학습 시간 | 10 epoch / 40분 |
| Validation Loss | ~5.0014 |
| 학습 파라미터 | 0.71% |

## 브랜치

| 브랜치 | 설명 |
|--------|------|
| `main` | 안정 버전 |
| `feature/music-transformer-qlora` | QLoRA 구현 |
| `feature/magenta-rt-jazz-finetuning` | Magenta RT 실시간 생성 |

## 다음 단계

- [ ] 사전학습 가중치 기반 Fine-tuning
- [ ] Magenta RT 재즈 스타일 Fine-tuning
- [ ] 실시간 MIDI 입력 → AI 응답 연동
- [ ] FL Studio DAW 연동

## 기술 스택

- **AI/ML**: PyTorch, JAX, Hugging Face, LoRA/QLoRA
- **오디오**: Magenta, Music Transformer
- **인프라**: RunPod, Google Colab
- **언어**: Python

## 라이선스

MIT License
