#!/bin/bash

# 필수 논문 다운로드 스크립트
# Usage: bash download_papers.sh

echo "📚 필수 논문 다운로드 중..."
echo ""

# 디렉토리 생성
mkdir -p papers
cd papers

# 1. Attention Is All You Need (Transformer)
echo "1️⃣ Downloading: Attention Is All You Need (2017)"
wget -q --show-progress https://arxiv.org/pdf/1706.03762.pdf -O 1_Transformer_AttentionIsAllYouNeed.pdf

# 2. Music Transformer
echo "2️⃣ Downloading: Music Transformer (2018)"
wget -q --show-progress https://arxiv.org/pdf/1809.04281.pdf -O 2_MusicTransformer.pdf

# 3. LoRA
echo "3️⃣ Downloading: LoRA (2021)"
wget -q --show-progress https://arxiv.org/pdf/2106.09685.pdf -O 3_LoRA.pdf

# 4. Perceiver
echo "4️⃣ Downloading: Perceiver (2021)"
wget -q --show-progress https://arxiv.org/pdf/2103.03206.pdf -O 4_Perceiver.pdf

# 5. QLoRA
echo "5️⃣ Downloading: QLoRA (2023)"
wget -q --show-progress https://arxiv.org/pdf/2305.14314.pdf -O 5_QLoRA.pdf

echo ""
echo "✅ 모든 논문 다운로드 완료!"
echo ""
echo "📁 위치: papers/"
ls -lh

echo ""
echo "📖 읽기 순서:"
echo "  1️⃣ Attention Is All You Need (필수! Transformer 기초)"
echo "  2️⃣ Music Transformer (음악 생성 기초)"
echo "  3️⃣ LoRA (효율적인 fine-tuning)"
echo "  4️⃣ Perceiver (선택, 고급 아키텍처)"
echo "  5️⃣ QLoRA (최신 SOTA, 취업 준비생 필수)"
echo ""
echo "💡 PAPERS_TO_READ.md를 참고하세요!"
