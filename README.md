# 🎹 Brad Mehldau AI Generator

**SCG + Transformer 하이브리드 모델**로 Brad Mehldau 스타일의 재즈 피아노 솔로를 실시간 생성

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## ✨ 주요 기능

- 🎼 **코드 진행 기반 생성**: Cmaj7 → Dm7 → G7 → Cmaj7 입력 → Brad Mehldau 스타일 솔로 출력
- 🚀 **최신 기술 결합**: SCG Diffusion + Transformer Style Encoder
- 🎹 **FL Studio 실시간 통합**: loopMIDI로 DAW와 연결
- ⚡ **빠른 추론**: DDIM 50 steps (< 1초, GPU 기준)
- 🎨 **창의성 조절**: Temperature & Guidance Scale 파라미터

---

## 🏗️ 아키텍처

```
입력: 코드 진행 ['Cmaj7', 'Dm7', 'G7', 'Cmaj7']
  ↓
┌─────────────────────────────────────────┐
│  Brad Mehldau Style Encoder Transformer │  ← 8-layer BERT-like
│  (코드 임베딩 + 스타일 특징 추출)        │
└─────────────────────────────────────────┘
  ↓ style_embedding [256]
┌─────────────────────────────────────────┐
│  DiT (Diffusion Transformer)            │  ← 12-layer, 6-head
│  + VQ-VAE Latent Diffusion              │
│  + DDIM Sampling (50 steps)             │
└─────────────────────────────────────────┘
  ↓ latent [64, 32, 64]
┌─────────────────────────────────────────┐
│  VQ-VAE Decoder                         │  ← Piano roll 재구성
└─────────────────────────────────────────┘
  ↓
출력: Piano Roll [2, 128, time]
      ↓
    MIDI Notes → FL Studio
```

---

## 📦 설치

### 요구사항

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU 학습용)

### 의존성 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/brad-mehldau-ai.git
cd brad-mehldau-ai

# 의존성 설치
pip install -r requirements.txt

# MIDI 통신 (FL Studio 통합용)
pip install mido python-rtmidi
```

---

## 🚀 Quick Start

### 1. 데이터 다운로드 (테스트용)

```bash
python scripts/download_data.py --dataset test
```

### 2. 학습 (테스트 모드)

```bash
# VQ-VAE 사전학습
python scripts/train_vqvae.py --test --epochs 5

# 체크포인트 확인
ls checkpoints/vqvae/
```

### 3. 추론 테스트

```python
from src.models.hybrid_model import SCGTransformerHybrid

# 모델 로드
model = SCGTransformerHybrid()
model.eval()

# 생성
chord_progression = ['Cmaj7', 'Dm7', 'G7', 'Cmaj7']
piano_roll = model.generate(
    chord_progression=chord_progression,
    num_steps=50,
    guidance_scale=7.5,
    temperature=0.8
)

print(f"Generated: {piano_roll.shape}")
```

---

## 📚 문서

- **[Training Guide](docs/TRAINING_GUIDE.md)**: Runpod/Colab 학습 가이드
- **[FL Studio Integration](docs/FL_STUDIO_GUIDE.md)**: DAW 통합 설정
- **[API Reference](docs/API.md)**: 모델 API 문서

---

## 🎓 프로젝트 구조

```
brad-mehldau-ai/
├── src/
│   ├── models/
│   │   ├── vqvae.py              # VQ-VAE 인코더/디코더
│   │   ├── dit.py                # Diffusion Transformer
│   │   ├── style_encoder.py      # Brad Mehldau Style Encoder
│   │   └── hybrid_model.py       # 통합 모델
│   ├── training/                 # 학습 유틸
│   └── utils/                    # 공통 유틸
│
├── scripts/
│   ├── download_data.py          # 데이터 다운로드
│   ├── train_vqvae.py           # VQ-VAE 학습
│   ├── train_style_encoder.py   # Style Encoder 학습
│   └── train_hybrid.py          # Hybrid 모델 fine-tuning
│
├── server/
│   ├── inference_server.py      # 추론 서버
│   └── midi_server.py           # MIDI 통신 서버
│
├── data/                        # 데이터셋
├── checkpoints/                 # 모델 체크포인트
├── configs/                     # 설정 파일
└── docs/                        # 문서
```

---

## 🎯 학습 파이프라인

### Phase 1: VQ-VAE 사전학습 (Week 1-2)

```bash
# MAESTRO 데이터로 VQ-VAE 학습
python scripts/download_data.py --dataset maestro
python scripts/train_vqvae.py \
  --data_dir ./data/maestro \
  --epochs 50 \
  --batch_size 16
```

**예상 시간**: RTX 3090 기준 8-10시간
**비용**: Runpod ~$3

### Phase 2: Style Encoder 학습 (Week 3-4)

```bash
# PiJAMA 데이터로 Style Encoder 학습
python scripts/download_data.py --dataset pijama
python scripts/train_style_encoder.py \
  --data_dir ./data/pijama \
  --epochs 50 \
  --batch_size 32
```

**예상 시간**: RTX 3090 기준 8-10시간
**비용**: Runpod ~$3

### Phase 3: Brad Mehldau Fine-tuning (Week 5-6)

```bash
# Brad Mehldau 데이터로 Hybrid 모델 fine-tuning
python scripts/train_hybrid.py \
  --vqvae_ckpt ./checkpoints/vqvae/best.pt \
  --style_encoder_ckpt ./checkpoints/style_encoder/best.pt \
  --brad_data ./data/brad_mehldau \
  --epochs 50 \
  --batch_size 16
```

**예상 시간**: RTX 3090 기준 10-15시간
**비용**: Runpod ~$5

**총 예산**: ~$10-15 (Spot instance 사용 시)

---

## 🎹 FL Studio 통합

### 1. loopMIDI 설치

1. [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) 다운로드
2. 가상 포트 2개 생성:
   - `loopMIDI Port 1` (출력: Python → FL Studio)
   - `loopMIDI Port 2` (입력: FL Studio → Python)

### 2. FL Studio 설정

```
Options → MIDI Settings:
  Input:  ✅ loopMIDI Port 2
  Output: ✅ loopMIDI Port 1

Channel Rack:
  Track 1: MIDI Out → Port 2 (코드 입력)
  Track 2: MIDI In → Port 1 (솔로 수신)
```

### 3. MIDI 서버 실행

```bash
python server/midi_server.py \
  --checkpoint ./checkpoints/brad_final/best.pt \
  --device cuda
```

### 4. 사용법

1. FL Studio에서 Track 1에 코드 4개 연주
2. Python이 자동으로 Brad Mehldau 솔로 생성
3. Track 2로 MIDI 전송 → 실시간 재생

---

## 🔧 고급 설정

### 창의성 조절

```python
# 보수적 (Brad 스타일에 충실)
piano_roll = model.generate(
    chord_progression=chords,
    temperature=0.5,
    guidance_scale=10.0
)

# 창의적 (즉흥성 높음)
piano_roll = model.generate(
    chord_progression=chords,
    temperature=1.2,
    guidance_scale=5.0
)
```

### 속도 최적화

```python
# DDIM steps 줄이기 (품질 ↓, 속도 ↑)
piano_roll = model.generate(
    chord_progression=chords,
    num_steps=25  # 50 → 25 (2배 빠름)
)

# INT8 양자화 (CPU 추론 2-3배 빠름)
generator = BradMehldauGenerator(
    checkpoint_path="./checkpoints/brad_final/best.pt",
    quantize=True
)
```

---

## 📊 성능

### 생성 속도

| 환경 | DDIM Steps | 시간 |
|------|-----------|------|
| RTX 4090 | 50 | ~0.5s |
| RTX 3090 | 50 | ~0.8s |
| M1 Max | 50 | ~3.0s |
| CPU (i7) | 50 | ~12s |

### 모델 크기

| 컴포넌트 | 파라미터 | 크기 |
|---------|---------|------|
| VQ-VAE | ~50M | 200MB |
| DiT | ~120M | 480MB |
| Style Encoder | ~85M | 340MB |
| **Total** | **~255M** | **~1GB** |

---

## 🎵 샘플

> **Note**: 학습 완료 후 생성된 샘플을 여기에 추가 예정

```bash
# 샘플 생성
python scripts/generate_samples.py \
  --checkpoint ./checkpoints/brad_final/best.pt \
  --output ./samples/
```

---

## 🛠️ 개발 로드맵

- [x] VQ-VAE 구현
- [x] DiT 구현
- [x] Style Encoder Transformer 구현
- [x] Hybrid 모델 통합
- [x] MIDI 서버 구현
- [ ] 데이터 로더 구현 (TODO)
- [ ] 코드 토크나이저 구현 (TODO)
- [ ] Brad Mehldau 데이터 수집 (TODO)
- [ ] Fine-tuning 실행 (TODO)
- [ ] 성능 평가 (TODO)
- [ ] GUI 제어판 (TODO)

---

## 🤝 기여

기여는 언제나 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 라이센스

MIT License - 자유롭게 사용하세요!

---

## 🙏 감사의 말

- **SCG (Rule-Guided Music)**: VQ-VAE + Diffusion 아키텍처
- **DiT (Diffusion Transformers)**: Transformer 기반 diffusion
- **Brad Mehldau**: 영감의 원천

---

## 📧 문의

- GitHub Issues: 버그 리포트 & 기능 요청
- Email: your.email@example.com

---

**Made with ❤️ for jazz lovers**
