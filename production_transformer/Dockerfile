# Production Dockerfile for Music Transformer
# Multi-stage build for optimized image size

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 작업 디렉토리
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 업그레이드
RUN pip3 install --upgrade pip setuptools wheel

# 의존성 복사 및 설치 (캐싱 최적화)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 코드 복사
COPY . .

# 포트 노출 (Gradio)
EXPOSE 7860

# 기본 명령어
CMD ["python3", "inference/gradio_demo.py", "--checkpoint", "/app/checkpoints/best_model", "--port", "7860", "--server_name", "0.0.0.0"]
