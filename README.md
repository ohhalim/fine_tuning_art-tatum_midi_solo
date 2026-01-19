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
