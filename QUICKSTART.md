# TatumFlow Quick Start Guide

## 🚀 5분 안에 시작하기

### 1. 설치 (1분)

```bash
# 저장소 클론
git clone https://github.com/ohhalim/fine_tuning_art-tatum_midi_solo.git
cd fine_tuning_art-tatum_midi_solo

# 의존성 설치
pip install torch numpy scipy mido pyyaml tqdm tensorboard matplotlib

# 또는 전체 설치
pip install -r requirements.txt
```

### 2. 모델 테스트 (1분)

```python
import sys
sys.path.insert(0, 'src')

from tatumflow import create_tatumflow_model, TatumFlowTokenizer
import torch

# 토크나이저 생성
tokenizer = TatumFlowTokenizer()
print(f"Vocabulary: {tokenizer.vocab_size} tokens")

# 모델 생성
model = create_tatumflow_model('base', vocab_size=tokenizer.vocab_size)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 테스트
tokens = torch.randint(0, tokenizer.vocab_size, (1, 128))
outputs = model(tokens, timestep=torch.tensor([500]))
print(f"Output shape: {outputs['logits'].shape}")
print("✅ TatumFlow ready!")
```

### 3. 데이터 준비 (2분)

```bash
# MIDI 파일 준비
mkdir -p data/midi
# 여기에 MIDI 파일 복사

# 또는 PiJAMA 다운로드 (Art Tatum 데이터)
# git clone https://github.com/SonyCSLParis/pijama data/pijama
```

### 4. 학습 시작 (1분)

```bash
# 설정 파일 확인/수정
nano config.yaml

# 학습 실행
python scripts/train_tatumflow.py

# TensorBoard로 모니터링
tensorboard --logdir logs/
```

## 📖 주요 명령어

### 음악 생성

```bash
# 1. Continuation (곡 이어쓰기)
python scripts/generate_music.py \
  --checkpoint checkpoints/best.pt \
  --mode continuation \
  --prompt input.mid \
  --output output.mid \
  --num_tokens 512

# 2. Improvisation (즉흥 변주)
python scripts/generate_music.py \
  --checkpoint checkpoints/best.pt \
  --mode improvise \
  --prompt input.mid \
  --output improv.mid \
  --num_variations 5 \
  --creativity 0.7

# 3. Style Transfer (스타일 변환)
python scripts/generate_music.py \
  --checkpoint checkpoints/best.pt \
  --mode style_transfer \
  --prompt classical.mid \
  --target_style jazz.mid \
  --output jazz_version.mid
```

### Python API

```python
from tatumflow import TatumFlowGenerator, load_model_from_checkpoint

# 모델 로드
model, tokenizer = load_model_from_checkpoint('checkpoints/best.pt')
generator = TatumFlowGenerator(model, tokenizer)

# 생성
generated = generator.generate_continuation(
    prompt_midi='input.mid',
    num_tokens=512,
    temperature=1.0
)

# 저장
generator.tokens_to_midi(generated, 'output.mid')
```

## 🎯 주요 파라미터

### 학습

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `batch_size` | 4 | 배치 크기 |
| `learning_rate` | 1e-4 | 학습률 |
| `num_epochs` | 100 | 에폭 수 |
| `diffusion_prob` | 0.7 | Diffusion 사용 확률 |

### 생성

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `temperature` | 1.0 | 샘플링 온도 (높을수록 다양) |
| `top_k` | 50 | Top-k 샘플링 |
| `top_p` | 0.95 | Nucleus 샘플링 |
| `creativity` | 0.5 | 창의성 (0=보수적, 1=매우 창의적) |

## 🔧 문제 해결

### CUDA Out of Memory

```yaml
# config.yaml
training:
  batch_size: 2  # 4에서 2로 감소
  gradient_accumulation_steps: 8  # 4에서 8로 증가
```

### 학습이 너무 느림

```yaml
# 작은 모델 사용
model:
  size: "small"  # base 대신

# 데이터 워커 증가
data:
  num_workers: 8  # 4에서 8로
```

### 생성 품질이 낮음

```python
# Temperature 조정
generator.generate_continuation(
    ...,
    temperature=0.9,  # 1.0에서 낮춤 (더 보수적)
    top_k=40,         # 50에서 낮춤
)

# 또는 더 많은 에폭 학습
```

## 📊 모델 크기

| Size | Parameters | VRAM | 학습 시간 (100 epoch) |
|------|-----------|------|----------------------|
| Small | 45M | 4GB | ~1일 |
| Base | 110M | 8GB | ~3일 |
| Large | 350M | 16GB | ~7일 |

## 💡 팁

1. **Art Tatum 스타일 학습**:
   ```yaml
   data:
     pijama_dir: "data/pijama"
     artist_filter: "art tatum"
   ```

2. **스타일 혼합**:
   ```python
   style_a, _, _ = model.encode_style(tokens_a)
   style_b, _, _ = model.encode_style(tokens_b)
   mixed = 0.5 * style_a + 0.5 * style_b
   ```

3. **실시간 생성**:
   ```python
   # 짧은 청크로 생성
   for i in range(10):
       chunk = generator.generate_continuation(
           prompt_midi=last_output,
           num_tokens=64  # 512 대신
       )
   ```

## 🆘 도움말

- **Documentation**: [docs/tatumflow_architecture.md](docs/tatumflow_architecture.md)
- **Model Analysis**: [docs/model_analysis.md](docs/model_analysis.md)
- **Issues**: https://github.com/ohhalim/fine_tuning_art-tatum_midi_solo/issues

## ✅ 체크리스트

학습 전:
- [ ] PyTorch 설치 확인 (`python -c "import torch; print(torch.__version__)"`)
- [ ] CUDA 사용 가능 확인 (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] MIDI 데이터 준비 (`data/midi/` 폴더에 `.mid` 파일)
- [ ] `config.yaml` 설정 확인

생성 전:
- [ ] 모델 학습 완료 (`checkpoints/best.pt` 존재)
- [ ] 입력 MIDI 파일 준비
- [ ] 출력 디렉토리 생성

---

**Happy Improvising! 🎹🎵**
