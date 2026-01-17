#!/bin/bash
# RunPod 환경 설정 및 학습 실행 스크립트
# 
# Usage:
#   1. RunPod에서 RTX 3090 인스턴스 생성
#   2. 코드 업로드 후 실행: bash scripts/runpod_setup.sh

set -e

echo "=== RunPod Music Transformer QLoRA Training Setup ==="

# 1. 시스템 패키지 업데이트
echo "[1/5] Updating system packages..."
apt-get update -qq

# 2. Python 패키지 설치
echo "[2/5] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 3. 데이터 전처리
echo "[3/5] Preprocessing MIDI data..."
python scripts/preprocess_jazz.py \
    --input_dir ./midi_dataset/midi \
    --output_dir ./data/jazz_processed \
    --max_seq 2048

# 4. Pretrained 체크포인트 다운로드 (있는 경우)
CHECKPOINT_URL="https://drive.google.com/drive/folders/1qS4z_7WV4LLgXZeVZU9IIjatK7dllKrc"
echo "[4/5] Note: Download pretrained checkpoint from: $CHECKPOINT_URL"
echo "      Place in ./checkpoints/pretrained/ if using pretrained weights"

# 5. 학습 시작
echo "[5/5] Starting LoRA training..."
python scripts/train_qlora.py \
    --data_dir ./data/jazz_processed \
    --epochs 3 \
    --batch_size 2 \
    --lr 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir ./checkpoints/jazz_lora

echo ""
echo "=== Training Complete ==="
echo "LoRA weights saved to: ./checkpoints/jazz_lora/"
