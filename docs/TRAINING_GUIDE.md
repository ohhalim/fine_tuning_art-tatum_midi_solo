# Brad Mehldau AI - Training Guide

Complete guide for training the SCG + Transformer hybrid model

## 📋 목차

1. [환경 선택](#환경-선택)
2. [Runpod 학습](#runpod-학습)
3. [Google Colab 학습](#google-colab-학습)
4. [로컬 환경 학습](#로컬-환경-학습)
5. [Training Timeline](#training-timeline)
6. [비용 최적화](#비용-최적화)

---

## 환경 선택

| 환경 | GPU | 비용 | 속도 | 추천 |
|------|-----|------|------|------|
| **Runpod** | RTX 3090/4090 | $0.34-0.79/hr | ⭐⭐⭐⭐⭐ | Phase 3 (Brad fine-tuning) |
| **Colab Pro** | T4/V100 | $10/month | ⭐⭐⭐ | Phase 1-2 (기본 학습) |
| **로컬 (M1/M2)** | M1/M2 GPU | Free | ⭐⭐ | 개발 & 추론만 |
| **로컬 (NVIDIA)** | RTX 3060+ | Free | ⭐⭐⭐⭐ | 시간 여유 있으면 |

### 추천 전략 (예산 $20)

```
Week 1-4: Google Colab Pro ($10/month)
  - VQ-VAE 사전학습
  - Style Encoder 학습
  - DiT 기본 학습

Week 5-6: Runpod ($10)
  - Brad Mehldau fine-tuning ONLY
  - RTX 3090 spot instance

Week 7+: 로컬 환경
  - 추론 & FL Studio 통합
```

---

## Runpod 학습

### Step 1: Runpod 계정 생성

1. https://runpod.io 접속
2. 계정 생성 & 크레딧 추가 ($10-20)
3. GPU Pod 선택:
   - **RTX 3090**: $0.34/hr (Spot), $0.44/hr (On-Demand)
   - **RTX 4090**: $0.69/hr (Spot), $0.79/hr (On-Demand)

### Step 2: Pod 생성

```bash
# Template 선택
Template: PyTorch 2.0+
GPU: RTX 3090 (1x)
Disk: 50GB
Volume: 100GB (영구 저장용)

# Auto-stop 설정 (비용 절약!)
Idle Timeout: 30 minutes
```

### Step 3: 코드 클론 & 의존성 설치

```bash
# SSH 접속 후
cd /workspace

# 프로젝트 클론
git clone https://github.com/yourusername/brad-mehldau-ai.git
cd brad-mehldau-ai

# 의존성 설치
pip install -r requirements.txt

# CUDA 확인
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 4: 데이터 다운로드

```bash
# MAESTRO 데이터 (VQ-VAE 사전학습용)
python scripts/download_data.py --dataset maestro --data_dir ./data

# PiJAMA 데이터 (Brad Mehldau)
python scripts/download_data.py --dataset pijama --data_dir ./data
```

### Step 5: VQ-VAE 사전학습 (~10시간)

```bash
# VQ-VAE 학습
python scripts/train_vqvae.py \
  --data_dir ./data/maestro \
  --save_dir ./checkpoints/vqvae \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --device cuda

# 백그라운드 실행 + 로그 저장
nohup python scripts/train_vqvae.py \
  --data_dir ./data/maestro \
  --save_dir ./checkpoints/vqvae \
  --epochs 50 \
  --batch_size 16 > vqvae_train.log 2>&1 &

# 로그 모니터링
tail -f vqvae_train.log
```

### Step 6: Checkpoint 백업 (중요!)

```bash
# rclone 설정 (Google Drive)
rclone config

# 체크포인트 백업
rclone copy ./checkpoints gdrive:brad-mehldau-checkpoints/ -P

# 자동 백업 스크립트
while true; do
  rclone copy ./checkpoints gdrive:brad-mehldau-checkpoints/ -P
  sleep 3600  # 1시간마다
done &
```

### Step 7: Brad Mehldau Fine-tuning (~15시간)

```bash
# Hybrid 모델 fine-tuning
python scripts/train_hybrid.py \
  --vqvae_ckpt ./checkpoints/vqvae/best.pt \
  --brad_data ./data/brad_mehldau \
  --epochs 50 \
  --batch_size 16 \
  --lr 5e-6 \
  --device cuda \
  --wandb_project "brad-scg-transformer"
```

### 비용 계산

```
VQ-VAE (10시간) + Fine-tuning (15시간) = 25시간
RTX 3090 Spot: 25 × $0.34 = $8.5
RTX 4090 Spot: 25 × $0.69 = $17.25

⚠️  Spot instance는 중간에 끊길 수 있음
→ 자동 체크포인트 저장 필수!
```

---

## Google Colab 학습

### Step 1: Colab Pro 구독

- Colab Pro: $10/month
- GPU: T4 (무료), V100 (Pro)
- 연속 실행: ~12시간

### Step 2: Colab Notebook 설정

```python
# GPU 확인
!nvidia-smi

# 프로젝트 클론
!git clone https://github.com/yourusername/brad-mehldau-ai.git
%cd brad-mehldau-ai

# 의존성 설치
!pip install -r requirements.txt

# Google Drive 마운트 (체크포인트 저장용)
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: 데이터 다운로드

```python
# MAESTRO 다운로드
!python scripts/download_data.py --dataset maestro --data_dir ./data

# 또는 Google Drive에서 복사
!cp -r /content/drive/MyDrive/brad-data/maestro ./data/
```

### Step 4: 학습 실행

```python
# VQ-VAE 학습
!python scripts/train_vqvae.py \
  --data_dir ./data/maestro \
  --save_dir /content/drive/MyDrive/brad-checkpoints/vqvae \
  --epochs 30 \
  --batch_size 8 \
  --device cuda

# ⚠️  12시간 제한 주의!
# → 30 epochs씩 나눠서 학습
```

### Step 5: Checkpoint 저장

```python
# 자동으로 Google Drive에 저장
# save_dir을 Drive 경로로 설정

# 학습 중간에 저장
import shutil
shutil.copy('./checkpoints/vqvae/best.pt',
            '/content/drive/MyDrive/brad-checkpoints/vqvae_backup.pt')
```

### Colab 제한사항

```
✅ 장점:
- 저렴 ($10/month)
- 설정 간단
- GPU 무료 (제한적)

❌ 단점:
- 12시간 연속 실행 제한
- 중간에 끊김
- GPU 할당 불확실 (특히 무료)

💡 해결책:
- 학습을 10-20 epoch 단위로 나눔
- checkpoint에서 resume 기능 필수
- 자동 저장 스크립트 사용
```

---

## 로컬 환경 학습

### NVIDIA GPU (RTX 3060 이상)

```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch 설치 (CUDA 11.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 의존성 설치
pip install -r requirements.txt

# 학습 실행
python scripts/train_vqvae.py \
  --data_dir ./data/maestro \
  --save_dir ./checkpoints/vqvae \
  --epochs 50 \
  --batch_size 8 \
  --device cuda
```

### Apple Silicon (M1/M2)

```bash
# PyTorch MPS 지원 확인
python -c "import torch; print(torch.backends.mps.is_available())"

# 의존성 설치
pip install -r requirements.txt

# 학습 실행 (MPS)
python scripts/train_vqvae.py \
  --data_dir ./data/maestro \
  --save_dir ./checkpoints/vqvae \
  --epochs 50 \
  --batch_size 4 \
  --device mps

# ⚠️  MPS는 CUDA보다 느림 (2-3배)
# → 개발용으로만 권장
```

---

## Training Timeline

### Phase 1: VQ-VAE 사전학습 (Week 1-2)

```bash
# MAESTRO 데이터 다운로드
python scripts/download_data.py --dataset maestro

# VQ-VAE 학습 (RTX 3090 기준: 8-10시간)
python scripts/train_vqvae.py \
  --data_dir ./data/maestro \
  --epochs 50 \
  --batch_size 16

# 체크포인트: ./checkpoints/vqvae/best.pt
```

### Phase 2: Style Encoder 사전학습 (Week 3-4)

```bash
# PiJAMA 데이터 다운로드
python scripts/download_data.py --dataset pijama

# Style Encoder 학습 (RTX 3090: 8-10시간)
python scripts/train_style_encoder.py \
  --data_dir ./data/pijama \
  --epochs 50 \
  --batch_size 32

# 체크포인트: ./checkpoints/style_encoder/best.pt
```

### Phase 3: Brad Mehldau Fine-tuning (Week 5-6)

```bash
# Brad Mehldau 데이터 필터링
python scripts/filter_brad_mehldau.py

# Hybrid 모델 fine-tuning (RTX 3090: 10-15시간)
python scripts/train_hybrid.py \
  --vqvae_ckpt ./checkpoints/vqvae/best.pt \
  --style_encoder_ckpt ./checkpoints/style_encoder/best.pt \
  --brad_data ./data/brad_mehldau \
  --epochs 50 \
  --batch_size 16

# 최종 체크포인트: ./checkpoints/brad_final/best.pt
```

---

## 비용 최적화

### 1. Spot Instance 사용 (50% 절감)

```bash
# Runpod Spot instance
RTX 3090: $0.34/hr (vs $0.44 On-Demand)

# 주의: 중간에 끊길 수 있음
→ 자동 체크포인트 저장 필수
```

### 2. Mixed Precision Training (2배 빠름)

```python
# train_vqvae.py에 추가
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    recon, vq_loss, perplexity = model(piano_roll)
    loss = recon_loss + vq_loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Gradient Accumulation (큰 batch size 효과)

```python
# 메모리 부족 시
accumulation_steps = 4
batch_size = 4  # effective batch = 16

for i, batch in enumerate(train_loader):
    loss = train_step(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Checkpoint Pruning (저장 공간 절약)

```python
# 최근 5개 checkpoint만 유지
import glob
checkpoints = sorted(glob.glob('./checkpoints/*.pt'))
if len(checkpoints) > 5:
    os.remove(checkpoints[0])  # 가장 오래된 것 삭제
```

---

## 문제 해결

### Out of Memory (OOM)

```python
# batch_size 줄이기
--batch_size 8  # 또는 4

# gradient accumulation 사용
--gradient_accumulation 2

# mixed precision training
--mixed_precision fp16
```

### Runpod Pod 끊김

```bash
# 자동 재시작 스크립트
#!/bin/bash
while true; do
    python scripts/train_hybrid.py \
      --resume ./checkpoints/brad_final/latest.pt \
      ...

    if [ $? -eq 0 ]; then
        break
    fi

    echo "Training interrupted, restarting in 10s..."
    sleep 10
done
```

### 학습 느림

```bash
# DataLoader workers 늘리기
num_workers=4  # 또는 8

# Pin memory 사용
pin_memory=True

# Prefetch factor
prefetch_factor=2
```

---

## 다음 단계

학습 완료 후:

1. **모델 검증**: `scripts/evaluate.py`로 성능 평가
2. **추론 테스트**: `server/inference_server.py`로 생성 테스트
3. **FL Studio 통합**: `docs/FL_STUDIO_GUIDE.md` 참고

---

**Questions?** GitHub Issues에 문의해주세요!
