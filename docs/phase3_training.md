# Phase 3: 본격 훈련 🚀

**목표**: 전체 데이터셋으로 프로덕션 품질 모델을 훈련합니다.

**예상 시간**: 1-2주 (GPU 성능에 따라)
**난이도**: ⭐⭐⭐⭐☆

---

## 📋 체크리스트

- [ ] 프로덕션 설정 파일 작성
- [ ] 하이퍼파라미터 설정
- [ ] AMP + EMA 활성화
- [ ] 훈련 시작 및 모니터링
- [ ] 체크포인트 관리
- [ ] Early stopping 설정

---

## 1. 프로덕션 설정

### config.yaml (Full Model)

```yaml
model:
  vocab_size: 2048
  hidden_dim: 512
  latent_dim: 256
  num_layers: 12
  num_heads: 8
  max_seq_len: 2048
  diffusion_steps: 1000
  num_style_dims: 64

training:
  batch_size: 8          # GPU 메모리에 맞게 조정
  epochs: 100
  learning_rate: 0.0001
  warmup_steps: 1000
  weight_decay: 0.01
  gradient_clip: 1.0
  use_amp: true          # 2x 빠름
  use_ema: true          # 품질 향상
  ema_decay: 0.9999

data:
  train_dir: 'data/art_tatum_midi/train'
  val_dir: 'data/art_tatum_midi/val'
  max_seq_len: 2048
  num_workers: 4

checkpoint:
  save_every: 1000       # steps
  keep_last_n: 5         # 최근 5개만
  checkpoint_dir: 'checkpoints/production'

logging:
  log_every: 100
  tensorboard_dir: 'logs/tensorboard'
```

---

## 2. 훈련 시작

### 기본 훈련

```bash
python scripts/train_tatumflow.py --config config.yaml
```

### AMP + EMA 훈련 (권장)

```bash
python scripts/phase3_train_production.py \
  --config config.yaml \
  --use_amp \
  --use_ema \
  --device cuda
```

### 예상 출력

```
TatumFlow Training
==================
Model: 125M parameters
Data: 70 train files, 9 val files
GPU: NVIDIA A100 (40GB)
AMP: Enabled
EMA: Enabled (decay=0.9999)

Epoch 1/100
  Step 100: Loss=5.234, Recon=3.456, Diff=1.234, KL=0.012 (2.3s/step)
  Step 200: Loss=4.789, Recon=3.123, Diff=1.098, KL=0.011 (2.1s/step)
  ...
  Validation Loss: 4.123
  Checkpoint saved: checkpoints/production/step_1000.pt

Epoch 2/100
  Step 1100: Loss=4.234, Recon=2.789, Diff=0.987, KL=0.010 (2.0s/step)
  ...
```

---

## 3. 모니터링

### TensorBoard

```bash
tensorboard --logdir=logs/tensorboard
```

**주요 메트릭**:

1. **Loss/train_total**: 전체 훈련 손실 (↓)
2. **Loss/val_total**: 검증 손실 (↓, 하지만 train보다 높음)
3. **Loss/reconstruction**: 재구성 손실 (빠르게 ↓)
4. **Loss/diffusion**: 디퓨전 손실 (천천히 ↓)
5. **Loss/kl_divergence**: KL 발산 (낮게 유지)
6. **Learning_rate**: 워밍업 후 감소
7. **Grad_norm**: Gradient 크기 (1.0 이하로 클립됨)

### GPU 모니터링

```bash
# 실시간 GPU 사용률
watch -n 1 nvidia-smi

# GPU 메모리 부족 시
# config.yaml에서 batch_size 줄이기
```

---

## 4. 하이퍼파라미터 튜닝

### Batch Size

**큰 batch**:
- ✅ 빠름
- ✅ 안정적
- ❌ 메모리 많이 필요

**작은 batch**:
- ✅ 메모리 절약
- ❌ 느림
- ❌ 불안정

**권장**:
- A100 (40GB): batch_size=8-16
- V100 (16GB): batch_size=4-8
- T4 (16GB): batch_size=2-4

### Learning Rate

**너무 높음** (>0.001):
- Loss 진동
- NaN 발생

**너무 낮음** (<0.00001):
- 느린 학습
- 수렴 안됨

**권장**: 0.0001 (Adam 기준)

### Warmup Steps

처음 N steps는 LR을 천천히 올림

**효과**:
- 훈련 초기 안정성 ↑
- Loss spike 방지

**권장**: 1000 steps

---

## 5. Early Stopping

### 언제 멈출까?

**좋은 신호** (계속):
- Validation loss 계속 감소
- Train/Val gap 작음

**나쁜 신호** (중단):
- Validation loss 5 epochs 연속 증가 → **오버피팅**
- Loss가 NaN → **터짐**
- Loss가 안 떨어짐 → **하이퍼파라미터 조정**

### 자동 Early Stopping

```python
# scripts/phase3_train_production.py에 추가
patience = 5
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    val_loss = validate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint('best.pt')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 6. 체크포인트 관리

### 저장 주기

- **너무 자주**: 디스크 낭비
- **너무 적게**: 좋은 모델 놓침

**권장**: 1000 steps마다

### 중요 체크포인트

1. **best.pt**: 최고 검증 Loss
2. **latest.pt**: 가장 최근
3. **epoch_X.pt**: Epoch별

### 디스크 절약

```yaml
checkpoint:
  keep_last_n: 5  # 최근 5개만
  delete_old: true
```

---

## 7. 훈련 재개 (Resume)

### 중단된 훈련 이어하기

```bash
python scripts/phase3_train_production.py \
  --config config.yaml \
  --resume checkpoints/production/latest.pt
```

**자동으로 복원**:
- 모델 가중치
- Optimizer 상태
- Epoch 번호
- Learning rate scheduler

---

## 🎓 학습 내용

### Mixed Precision (AMP)

**FP32** (기본):
- 정밀도: 높음
- 속도: 느림
- 메모리: 많이 사용

**FP16** (AMP):
- 정밀도: 약간 낮음 (음악에 무시 가능)
- 속도: **2배 빠름**
- 메모리: **50% 절약**

**동작 원리**:
```python
with torch.cuda.amp.autocast():
    output = model(input)  # FP16 연산
loss = criterion(output, target)
scaler.scale(loss).backward()  # FP32 gradient
```

### EMA (Exponential Moving Average)

모델 가중치의 이동 평균을 유지합니다.

**효과**:
- 생성 품질 향상
- 훈련 안정성 증가

**공식**:
```
θ_ema = decay * θ_ema + (1 - decay) * θ
```

**권장 decay**: 0.9999

### Gradient Clipping

Gradient가 너무 크면 clip합니다.

**문제**: Exploding gradient → NaN
**해결**: `gradient_clip=1.0`

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 🚨 문제 해결

### 문제 1: CUDA Out of Memory

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결**:
```yaml
training:
  batch_size: 4  # 줄이기
  gradient_accumulation: 2  # 추가 (effective batch = 4*2=8)
```

### 문제 2: Loss가 NaN

**원인**: Exploding gradient

**해결**:
```yaml
training:
  gradient_clip: 0.5  # 더 낮게
  learning_rate: 0.00005  # 줄이기
```

### 문제 3: 훈련이 너무 느림

**체크**:
1. GPU 사용률 100%인가? (`nvidia-smi`)
2. AMP 활성화했나?
3. num_workers=4 설정했나?

**개선**:
```yaml
data:
  num_workers: 4  # CPU 코어 활용
training:
  use_amp: true   # 2x 빠름
```

---

## ✅ Phase 3 완료 체크

- [ ] 10+ epochs 훈련 완료
- [ ] Validation loss < 2.0
- [ ] TensorBoard에서 정상 학습 곡선 확인
- [ ] 체크포인트 여러 개 저장됨
- [ ] GPU 효율적으로 사용 (>80%)

---

## 다음 단계

**Phase 4: 평가 및 개선**으로 이동:
```bash
cat docs/phase4_evaluation.md
```

**잘 하셨습니다! 이제 모델을 평가해봅시다! 📊**
