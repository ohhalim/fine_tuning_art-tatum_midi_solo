# 필수 논문 리스트 📚

**Brad Mehldau MIDI Generator 프로젝트를 위한 핵심 논문들**

이 문서는 프로젝트의 4가지 접근 방식을 이해하는데 필수적인 논문들을 정리합니다.

---

## 🎯 왜 논문을 읽어야 하는가?

### 1. **면접 대비**
- "어떤 논문을 기반으로 구현했나요?"
- "Transformer와 RNN의 차이를 설명해주세요"
- "LoRA가 왜 효율적인가요?"

→ 논문을 읽으면 깊이있게 답변 가능!

### 2. **구현 이해**
- 단순히 코드를 베끼는 것 vs 원리를 이해하고 구현
- 논문을 읽으면 왜 그렇게 구현했는지 이해됨
- 문제가 생겼을 때 디버깅 가능

### 3. **최신 트렌드 파악**
- AI 분야는 빠르게 발전
- 2017 Transformer → 2021 LoRA → 2023 QLoRA
- 다음은 무엇이 나올까?

---

## 📑 카테고리별 필수 논문

### 🎵 1. Music Generation (음악 생성)

#### ⭐ Music Transformer (Google Magenta, 2018)
**제목**: Music Transformer: Generating Music with Long-Term Structure

**저자**: Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Noam Shazeer, Ian Simon, Curtis Hawthorne, Andrew M. Dai, Matthew D. Hoffman, Monica Dinculescu, Douglas Eck

**출판**: ICLR 2019

**arXiv**: https://arxiv.org/abs/1809.04281

**관련 브랜치**:
- ✅ 브랜치 3: Perceiver + Music Transformer
- ✅ 브랜치 4: Production Transformer

**핵심 기여**:
1. **Relative Positional Encoding** for music
   - 절대 위치 대신 상대적 위치 사용
   - 음악의 반복 패턴을 더 잘 학습

2. **Long-term Structure**
   - 수천 개의 토큰 시퀀스 처리
   - 분 단위 음악 생성 가능

3. **Memory Efficiency**
   - Relative attention의 메모리를 O(L²D) → O(LD)로 감소

**왜 중요한가**:
- 음악 생성에 Transformer를 성공적으로 적용한 첫 사례
- Relative attention이 symbolic music에 필수적임을 증명
- 우리 프로젝트의 기반 아키텍처

**읽는 법**:
1. Section 3 (Relative Attention) 집중
2. Figure 2 (Relative Positional Encoding) 이해
3. Section 5 (Experiments) 결과 분석

**면접 질문 예상**:
- Q: "왜 Music Transformer는 상대 위치를 사용하나요?"
- A: "음악은 절대 위치보다 상대적 간격이 중요합니다. 예를 들어 'C-E-G'는 어떤 옥타브든 C major chord입니다. Relative attention은 이런 전이 불변성(translational invariance)을 학습할 수 있습니다."

---

### 🤖 2. Transformer Architecture (기본 아키텍처)

#### ⭐⭐⭐ Attention Is All You Need (Google, 2017)
**제목**: Attention Is All You Need

**저자**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

**출판**: NeurIPS 2017

**arXiv**: https://arxiv.org/abs/1706.03762

**관련 브랜치**:
- ✅ **모든 브랜치!** (Transformer 기반)

**핵심 기여**:
1. **Self-Attention Mechanism**
   - RNN 없이 순서 데이터 처리
   - 병렬 처리 가능 → 빠름

2. **Multi-Head Attention**
   - 여러 관점에서 attention 계산
   - 다양한 패턴 학습

3. **Positional Encoding**
   - 위치 정보 주입
   - sin/cos 함수 사용

**왜 중요한가**:
- **현대 AI의 기초** (GPT, BERT, 모든 LLM의 기반)
- 21세기 가장 많이 인용된 논문 (173,000+ 인용)
- 이 논문을 모르면 Transformer를 이해할 수 없음

**읽는 법**:
1. Section 3.2 (Attention) - **가장 중요!**
2. Figure 1 (Architecture) 완전히 이해
3. Section 3.3 (Multi-Head Attention)
4. Section 3.5 (Positional Encoding)

**면접 질문 예상**:
- Q: "Transformer가 RNN보다 나은 이유는?"
- A: "1) 병렬 처리 가능 (RNN은 순차 처리 필수), 2) Long-range dependency를 더 잘 포착 (attention으로 멀리 있는 토큰도 직접 연결), 3) Vanishing gradient 문제 없음"

**꼭 암기해야 할 공식**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Q: Query (무엇을 찾을까?)
K: Key (어디를 볼까?)
V: Value (무엇을 가져올까?)
```

---

### 🔧 3. Efficient Fine-tuning (효율적인 학습)

#### ⭐⭐ LoRA (Microsoft, 2021)
**제목**: LoRA: Low-Rank Adaptation of Large Language Models

**저자**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen

**출판**: ICLR 2022

**arXiv**: https://arxiv.org/abs/2106.09685

**GitHub**: https://github.com/microsoft/LoRA

**관련 브랜치**:
- ✅ 브랜치 2: Moonbeam + LoRA
- ✅ 브랜치 3: Perceiver + QLoRA
- ✅ 브랜치 4: Production Transformer

**핵심 기여**:
1. **Low-Rank Decomposition**
   - 전체 weight matrix를 학습하지 않음
   - 작은 두 행렬(A, B)만 학습: ΔW = BA

2. **효율성**
   - 학습 파라미터 10,000배 감소
   - GPU 메모리 3배 감소
   - 성능은 full fine-tuning과 동등

3. **Inference 시 overhead 없음**
   - LoRA를 merge할 수 있음: W' = W + BA
   - 추론 속도 동일

**왜 중요한가**:
- **2021-2024 가장 많이 쓰이는 fine-tuning 방법**
- Stable Diffusion, ChatGPT 개인화 모두 LoRA 사용
- Consumer GPU에서도 LLM fine-tuning 가능하게 함

**수식**:
```
원래: W ∈ R^(d×k) 전체 학습 (d×k개 파라미터)

LoRA: W = W_0 + ΔW = W_0 + BA
      A ∈ R^(d×r), B ∈ R^(r×k), r << min(d,k)
      학습 파라미터: r(d+k) << dk

예시: d=4096, k=4096, r=8
      원래: 16,777,216개 파라미터
      LoRA: 65,536개 파라미터 (0.39%!)
```

**면접 질문 예상**:
- Q: "LoRA는 어떻게 적은 파라미터로 같은 성능을 낼 수 있나요?"
- A: "Pre-trained 모델의 weight 변화는 intrinsic rank가 낮습니다. 즉, 대부분의 변화는 저차원 부공간에서 일어납니다. LoRA는 이 insight를 활용해 변화량 ΔW를 rank r로 제한합니다. 실험 결과 r=8 정도면 충분했습니다."

---

#### ⭐⭐⭐ QLoRA (University of Washington, 2023)
**제목**: QLoRA: Efficient Finetuning of Quantized LLMs

**저자**: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer

**출판**: NeurIPS 2023

**arXiv**: https://arxiv.org/abs/2305.14314

**GitHub**: https://github.com/artidoro/qlora

**관련 브랜치**:
- ✅ 브랜치 3: Perceiver + QLoRA
- ✅ 브랜치 4: Production Transformer (핵심!)

**핵심 기여**:
1. **4-bit NormalFloat (NF4)**
   - 일반 4-bit보다 정보 이론적으로 최적
   - 정규분포 weight에 특화

2. **Double Quantization**
   - Quantization constant도 quantize
   - 메모리 추가 절약

3. **Paged Optimizers**
   - GPU 메모리 부족 시 CPU로 spill
   - OOM 방지

**성능**:
- 65B 모델을 **48GB GPU 1개**로 fine-tuning!
- 원래는 A100 8개 필요 (~$100,000)
- RTX 3090 1개로 가능 (~$1,500)

**왜 중요한가**:
- **2023-2025 SOTA fine-tuning 방법**
- Consumer GPU로도 LLM fine-tuning 가능
- Hugging Face PEFT 라이브러리의 핵심

**면접 질문 예상**:
- Q: "QLoRA가 LoRA보다 나은 점은?"
- A: "QLoRA는 base model을 4-bit로 quantize해서 메모리를 75% 줄입니다. 이렇게 절약한 메모리로 더 큰 모델을 학습하거나, 더 큰 batch size를 사용할 수 있습니다. 성능 저하는 거의 없습니다(< 1%)."

**우리 프로젝트에서**:
```python
# Production Transformer 브랜치
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "key", "value"]
)
```

---

### 🧠 4. Advanced Architectures (고급 아키텍처)

#### ⭐ Perceiver (DeepMind, 2021)
**제목**: Perceiver: General Perception with Iterative Attention

**저자**: Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, Joao Carreira

**출판**: ICML 2021

**arXiv**: https://arxiv.org/abs/2103.03206

**관련 브랜치**:
- ✅ 브랜치 3: Perceiver + Music Transformer

**핵심 기여**:
1. **Asymmetric Attention**
   - Input → Latent: Cross-attention
   - 입력 크기에 무관한 복잡도

2. **O(N) Complexity**
   - 일반 Transformer: O(N²)
   - Perceiver: O(N) + O(M²), M << N

3. **Modality-Agnostic**
   - 이미지, 오디오, 비디오 모두 처리
   - Symbolic music도 가능

**수식**:
```
일반 Transformer:
  Self-Attention: O(N²D)
  N = 시퀀스 길이, D = hidden dim

Perceiver:
  Cross-Attention (N → M): O(NMD)
  Self-Attention (M): O(M²D)
  M << N이면 O(ND) + O(M²D) ≈ O(ND)

예시: N=50,000 (pixels), M=512 (latents)
      Transformer: 2.5B operations
      Perceiver: 25M operations (100배 빠름!)
```

**왜 중요한가**:
- 긴 시퀀스를 효율적으로 처리
- MIDI는 수천~수만 개 이벤트 → Perceiver 적합
- 브랜치 3에서 가장 효율적인 이유

---

## 📊 논문별 우선순위

### 🔥 필수 (반드시 읽기)
1. **Attention Is All You Need** - Transformer 기본
2. **LoRA** - Fine-tuning 기본
3. **Music Transformer** - 음악 생성 기본

### 🌟 강력 추천 (취업 준비생)
4. **QLoRA** - 2023-2025 SOTA
5. **Perceiver** - 효율적인 아키텍처

### 📚 심화 (시간 있으면)
6. VQ-VAE (Diffusion 이해)
7. Diffusion Transformer
8. BERT (Style Encoder 이해)

---

## 🎯 브랜치별 관련 논문

### 브랜치 1: SCG + Transformer
**필수 논문**:
- Attention Is All You Need (Transformer)
- VQ-VAE
- Denoising Diffusion Probabilistic Models (DDPM)

### 브랜치 2: Moonbeam + LoRA
**필수 논문**:
- Attention Is All You Need (Transformer)
- LoRA
- Music Transformer (참고)

### 브랜치 3: Perceiver + Music Transformer + QLoRA ⭐
**필수 논문**:
- Attention Is All You Need (Transformer)
- Music Transformer
- Perceiver
- LoRA
- QLoRA

### 브랜치 4: Production Transformer ⭐⭐⭐
**필수 논문**:
- Attention Is All You Need (Transformer)
- Music Transformer
- LoRA
- QLoRA

---

## 📖 논문 읽는 법

### 1차 독서 (30분)
1. **Abstract** - 무엇을 한 논문인가?
2. **Introduction** - 왜 이게 중요한가?
3. **Conclusion** - 결과가 어떤가?
4. **Figures** - 시각적으로 이해

### 2차 독서 (2시간)
1. **Method** - 어떻게 구현했나?
2. **Experiments** - 어떤 실험을 했나?
3. **Related Work** - 다른 방법과의 비교

### 3차 독서 (코드 보면서)
1. 논문의 수식을 코드로 대응
2. Figure를 재현
3. Ablation study 이해

---

## 💡 면접 대비 핵심 질문

### Transformer
**Q**: Self-Attention의 시간 복잡도는?
**A**: O(N²D). N개 토큰이 각각 N개를 attention → N². 이것이 긴 시퀀스의 bottleneck.

**Q**: Positional Encoding은 왜 필요한가?
**A**: Attention은 순서 정보가 없음 (permutation invariant). PE를 더해서 위치 정보 주입.

### LoRA
**Q**: LoRA의 rank r은 어떻게 정하나?
**A**: 실험적으로 결정. 보통 r=8이면 충분. 너무 작으면 표현력 부족, 너무 크면 효율성 감소.

**Q**: LoRA를 어느 layer에 적용하나?
**A**: 보통 attention의 Q, K, V projection. FFN은 선택적. 실험 결과 attention만으로 충분.

### Music Transformer
**Q**: Absolute vs Relative position의 차이는?
**A**: Absolute은 절대 위치 (0, 1, 2, ...), Relative는 상대 거리 (i-j). 음악은 전이 불변성이 중요해서 relative가 적합.

---

## 📥 논문 다운로드

모든 논문은 **arXiv**에서 무료로 다운로드 가능:

```bash
# 예시: LoRA 논문 다운로드
wget https://arxiv.org/pdf/2106.09685.pdf -O LoRA.pdf

# QLoRA
wget https://arxiv.org/pdf/2305.14314.pdf -O QLoRA.pdf

# Music Transformer
wget https://arxiv.org/pdf/1809.04281.pdf -O MusicTransformer.pdf

# Attention Is All You Need
wget https://arxiv.org/pdf/1706.03762.pdf -O Transformer.pdf

# Perceiver
wget https://arxiv.org/pdf/2103.03206.pdf -O Perceiver.pdf
```

---

## 🎓 추가 학습 리소스

### 블로그 포스트
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Transformer 시각화
- [LoRA Explained](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch) - LoRA 상세 설명
- [Annotated Music Transformer](https://gudgud96.github.io/2020/04/01/annotated-music-transformer/) - 코드와 함께

### 강의
- Stanford CS224N - NLP with Deep Learning
- MIT 6.S191 - Introduction to Deep Learning
- Fast.ai - Practical Deep Learning

### 구현
- [HuggingFace Transformers Course](https://huggingface.co/course)
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

---

## ✅ 학습 체크리스트

### Week 1: Transformer 기초
- [ ] "Attention Is All You Need" 1차 독서
- [ ] Self-attention 수식 이해
- [ ] Multi-head attention 구현
- [ ] Positional encoding 구현

### Week 2: Music Transformer
- [ ] "Music Transformer" 1차 독서
- [ ] Relative attention 이해
- [ ] Event-based MIDI representation 구현

### Week 3: LoRA
- [ ] "LoRA" 논문 읽기
- [ ] Low-rank decomposition 이해
- [ ] LoRA layer 구현

### Week 4: QLoRA
- [ ] "QLoRA" 논문 읽기
- [ ] 4-bit quantization 이해
- [ ] PEFT 라이브러리 사용

### Week 5: Perceiver (Optional)
- [ ] "Perceiver" 논문 읽기
- [ ] Cross-attention 이해
- [ ] Complexity 분석

---

## 🏆 목표

이 논문들을 읽고 나면:

### 면접에서
✅ Transformer를 깊이있게 설명 가능
✅ 최신 fine-tuning 방법 (LoRA, QLoRA) 이해
✅ 구현 결정을 논문 기반으로 정당화 가능

### 실무에서
✅ 새로운 모델 빠르게 이해
✅ 논문을 코드로 구현 가능
✅ 최적화 방향 결정 가능

### 커리어에서
✅ Research Engineer 역량
✅ ML Architect 기초
✅ 평생 학습 능력

---

**"논문을 읽는 것은 어렵지만, 읽고 나면 세상이 다르게 보입니다."** 🚀

지금 바로 시작하세요!
