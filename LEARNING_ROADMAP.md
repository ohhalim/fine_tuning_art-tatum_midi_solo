# 학습 로드맵: "나 + AI(나) = JAM!" 실현하기

**목표**: 내 연주 스타일을 학습한 AI와 실시간 즉흥연주하는 시스템 구축

**기간**: 3개월 (집중 학습) + 3개월 (구현 & 실험)

---

## 🎯 최종 목표 분해

```
나와 가상의 내가 JAM!
    ↓
┌─────────────────────────────────────────┐
│ 1. AI가 내 스타일 학습                  │
│    → Fine-tuning 기술 필요              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. 실시간 생성                          │
│    → Real-time generation 이해 필요     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. MIDI로 작동                          │
│    → Audio → MIDI 변환 기술 필요        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. 상호작용                             │
│    → Audio/MIDI injection 구현 필요     │
└─────────────────────────────────────────┘
```

---

## 📚 Phase 1: 기초 이론 (2주)

### 1.1 Transformer 완벽 이해 ⭐⭐⭐

**왜 배워야?**
- Magenta RT의 핵심 = Transformer
- Fine-tuning 이해하려면 기본 구조 필수
- 모든 LLM/생성 모델의 기초

**무엇을 배울까?**

```python
# 1. Self-Attention 메커니즘
Q, K, V = query, key, value
Attention(Q,K,V) = softmax(QK^T / √d_k)V

# 왜 중요?
# → 음악의 "맥락"을 이해하는 핵심!
# → "이전 4마디를 기억하고 다음 프레이즈 생성"

# 2. Multi-Head Attention
# 여러 관점에서 동시에 분석
head_1 = Attention(Q1, K1, V1)  # 멜로디 관계
head_2 = Attention(Q2, K2, V2)  # 화성 관계
head_3 = Attention(Q3, K3, V3)  # 리듬 관계

# 3. Positional Encoding
# 토큰의 순서 정보 추가
# → 음악에서 타이밍이 중요!

# 4. Encoder-Decoder 구조
Encoder: 입력 처리 (과거 10초 음악)
Decoder: 생성 (다음 2초 예측)
```

**학습 리소스**:
```
1. 논문 읽기:
   - "Attention Is All You Need" (Vaswani et al.)
   - PAPERS_TO_READ.md 참조

2. 코드 실습:
   - PyTorch Transformer tutorial
   - HuggingFace Transformers 기본 예제

3. 시각화:
   - http://jalammar.github.io/illustrated-transformer/
   - 한국어: Transformer 설명 블로그들

4. 실습 과제:
   - 간단한 seq2seq Transformer 구현
   - Text generation으로 먼저 연습
   - 그 다음 MIDI sequence generation
```

**체크리스트**:
- [ ] Self-attention 수식 유도 가능
- [ ] Multi-head attention 구조 그릴 수 있음
- [ ] PyTorch로 간단한 Transformer 구현
- [ ] Positional encoding 필요성 설명 가능
- [ ] Encoder-decoder 차이 명확히 이해

**예상 시간**: 1주 (매일 2-3시간)

---

### 1.2 Audio/MIDI Tokenization ⭐⭐⭐

**왜 배워야?**
- 오디오를 Transformer가 이해할 수 있는 형태로 변환
- SpectroStream (Audio) 이해 → MIDI tokenizer 개발

**무엇을 배울까?**

```python
# 1. Audio Tokenization (SpectroStream)

# Audio waveform → Discrete tokens
audio_48khz = [48000 samples/sec × 2 channels]
    ↓ Encoder (Neural codec)
audio_tokens = [25 frames/sec × 64 RVQ levels]
    ↓ Decoder
reconstructed_audio ≈ original_audio

# 핵심 개념:
# - RVQ (Residual Vector Quantization)
# - Codebook (vocabulary of audio patterns)
# - Perceptual loss (사람이 듣기에 자연스럽게)

# 2. MIDI Tokenization (내가 구현할 것!)

# Event-based representation:
midi_events = [
    NOTE_ON(pitch=60, velocity=80, time=0.0),
    NOTE_OFF(pitch=60, time=0.5),
    NOTE_ON(pitch=64, velocity=75, time=0.5),
    NOTE_OFF(pitch=64, time=1.0),
]
    ↓ Tokenize
tokens = [
    TOKEN_NOTE_ON_60,
    TOKEN_VELOCITY_80,
    TOKEN_TIME_SHIFT_500ms,
    TOKEN_NOTE_OFF_60,
    TOKEN_NOTE_ON_64,
    ...
]

# REMI (REpresentation of MIDi):
tokens = [
    BAR_START,
    POSITION_0,
    PITCH_60,
    VELOCITY_80,
    DURATION_8,  # 8분음표
    POSITION_2,
    PITCH_64,
    ...
]

# 3. 비교: Audio vs MIDI tokens

Audio (SpectroStream):
  - 2초 = 50 frames × 64 RVQ = 3,200 tokens
  - 모든 음향 정보 포함 (음색, 잔향, 노이즈)
  - 무거움

MIDI:
  - 2초 = ~100 events = ~100 tokens
  - Note on/off, velocity, timing만
  - 가벼움 (30배!)
  - → 더 빠른 생성 가능!
```

**학습 리소스**:
```
1. Audio Codec:
   - SoundStream paper (Google)
   - EnCodec paper (Meta)
   - SpectroStream code 분석

2. MIDI Tokenization:
   - Miditok library 문서
   - "This Time with Feeling" (MIDI tokenization survey)
   - Music Transformer paper (REMI representation)

3. 코드 실습:
   - Miditok 라이브러리 사용해보기
   - MIDI 파일 → tokens → MIDI 복원
   - 다양한 tokenization 방식 비교

4. 실습 과제:
   - 내 MIDI 파일 tokenize해보기
   - Token vocabulary 크기 분석
   - Reconstruction 품질 평가
```

**체크리스트**:
- [ ] RVQ 개념 설명 가능
- [ ] Audio codec이 왜 필요한지 이해
- [ ] MIDI event-based vs REMI 비교 가능
- [ ] Miditok으로 MIDI tokenize/detokenize 가능
- [ ] 내 프로젝트에 맞는 tokenization 방식 선택

**예상 시간**: 3-4일 (매일 2-3시간)

---

### 1.3 Music Generation 기초 ⭐⭐

**왜 배워야?**
- 음악 생성의 도메인 지식
- Magenta RT가 해결한 문제들 이해

**무엇을 배울까?**

```python
# 1. Music Language Modeling

# Text LM과 유사:
"The cat sat on the ___" → "mat" (예측)

# Music LM:
[C, E, G, ___ ] → "C" or "E" (예측)
# 하지만 음악은:
# - Polyphonic (여러 음이 동시에)
# - Hierarchical (멜로디 + 화성 + 리듬)
# - Long-term structure (16마디, 32마디 구조)

# 2. 음악 생성의 도전과제

# Challenge 1: Long-term coherence
# → 4마디는 괜찮은데 32마디는 산만해짐
# → Solution: Hierarchical generation, Planning

# Challenge 2: Multiple attributes
# → 멜로디, 화성, 리듬 동시 컨트롤
# → Solution: Multi-conditioning, Disentanglement

# Challenge 3: Evaluation
# → "좋은 음악"을 어떻게 측정?
# → Solution: Perplexity, 사람 평가, Musicality metrics

# 3. Conditioning (컨디셔닝)

# Unconditional:
model.generate()  # 랜덤 생성

# Conditional:
model.generate(
    genre="jazz",
    tempo=120,
    key="C major",
    style_embedding=my_style
)

# 4. Sampling strategies

# Greedy: 항상 가장 확률 높은 것
# → 안전하지만 지루함

# Top-k: 상위 k개 중 샘플링
# → 적당한 다양성

# Temperature: 확률 분포 조절
# temp=0.1: 보수적 (안전)
# temp=1.0: 표준
# temp=2.0: 모험적 (랜덤)
```

**학습 리소스**:
```
1. 논문:
   - Music Transformer (Google Magenta)
   - MuseNet (OpenAI)
   - Jukebox (OpenAI)

2. 실습:
   - Magenta.js 웹 데모들 체험
   - Music Transformer Colab 실행
   - 다양한 sampling parameter 실험

3. 음악 이론:
   - 기본 화성학 (코드 진행)
   - 리듬 패턴
   - 곡 구조 (AABA, verse-chorus)
```

**체크리스트**:
- [ ] Music LM vs Text LM 차이 설명 가능
- [ ] Polyphonic music 생성의 어려움 이해
- [ ] Conditioning 방식들 비교 가능
- [ ] Temperature, top-k의 효과 체험
- [ ] 음악 생성 샘플 품질 평가 가능

**예상 시간**: 3-4일

---

## 📚 Phase 2: Magenta RealTime 깊이 이해 (2주)

### 2.1 Magenta RT Architecture 완벽 분해 ⭐⭐⭐

**왜 배워야?**
- 이 시스템을 MIDI로 개조해야 함
- 각 컴포넌트의 역할 이해 필수

**무엇을 배울까?**

```python
# 전체 파이프라인 분해

# ═══════════════════════════════════════
# Component 1: MusicCoCa (Style Embedding)
# ═══════════════════════════════════════

class MusicCoCa:
    """
    목적: Text와 Audio를 같은 공간에 임베딩

    왜 중요?
    - "jazz piano" (text) = my_recording.wav (audio)
    - 내 연주를 prompt로 사용 가능!
    """

    def __init__(self):
        self.audio_encoder = ViT(12_layers)  # Vision Transformer
        self.text_encoder = Transformer(12_layers)
        self.text_decoder = Transformer(3_layers)  # Regularization

    def encode_audio(self, audio_10s):
        # Audio → Log-mel spectrogram
        spectrogram = to_mel(audio_10s)  # 128 channels × 992 frames

        # ViT로 처리
        embedding_768d = self.audio_encoder(spectrogram)

        # Quantize to 12 tokens
        tokens_12 = self.quantize(embedding_768d)
        return tokens_12

    def encode_text(self, text):
        # Text → Tokens
        text_tokens = tokenize(text)  # max 128 tokens

        # Transformer로 처리
        embedding_768d = self.text_encoder(text_tokens)

        # Quantize to 12 tokens
        tokens_12 = self.quantize(embedding_768d)
        return tokens_12

    def blend_prompts(self, prompts):
        """Multiple prompts weighted average"""
        embeddings = []
        weights = []

        for prompt, weight in prompts:
            if isinstance(prompt, str):
                emb = self.encode_text(prompt)
            else:
                emb = self.encode_audio(prompt)
            embeddings.append(emb)
            weights.append(weight)

        # Weighted average
        blended = sum(w * e for w, e in zip(weights, embeddings))
        blended = blended / sum(weights)
        return blended

# 배워야 할 것:
# - Contrastive learning (CoCa)
# - Joint embedding space
# - Quantization (768D → 12 tokens)
# - Attention pooling

# ═══════════════════════════════════════
# Component 2: SpectroStream (Audio Codec)
# ═══════════════════════════════════════

class SpectroStream:
    """
    목적: Audio ↔ Discrete tokens

    왜 중요?
    - Transformer는 discrete tokens 처리
    - Continuous audio를 language-like로 변환
    """

    def encode(self, audio_2s):
        # Audio → Latent
        latent = self.encoder_nn(audio_2s)

        # Latent → RVQ tokens
        tokens = []
        residual = latent
        for level in range(64):  # 64 RVQ levels
            codes, residual = self.vq_layers[level](residual)
            tokens.append(codes)

        # Shape: [50 frames, 64 RVQ levels]
        # → 3,200 tokens for 2 seconds
        return tokens

    def decode(self, tokens):
        # RVQ tokens → Latent
        latent = 0
        for level, codes in enumerate(tokens):
            latent += self.vq_layers[level].lookup(codes)

        # Latent → Audio
        audio = self.decoder_nn(latent)
        return audio

# 배워야 할 것:
# - RVQ (Residual Vector Quantization)
# - Codebook learning
# - Perceptual loss
# - Hierarchical structure (coarse → fine)

# ═══════════════════════════════════════
# Component 3: Encoder-Decoder Transformer
# ═══════════════════════════════════════

class MagentaRTTransformer:
    """
    목적: 과거 context → 다음 2초 생성
    """

    def __init__(self, config):
        self.encoder = T5Encoder(config)  # Bidirectional
        self.decoder = T5Decoder(config)  # Causal

    def generate_chunk(self, history_10s, style_12tokens):
        """
        Inputs:
        - history_10s: 5 chunks × 50 frames × 4 RVQ = 1000 tokens
        - style_12tokens: 12 tokens

        Output:
        - next_chunk: 50 frames × 16 RVQ = 800 tokens
        """

        # 1. Encoder: Process context
        encoder_input = torch.cat([
            history_10s,     # 1000 tokens (coarse)
            style_12tokens   # 12 tokens
        ])  # Total: 1012 tokens

        encoder_output = self.encoder(encoder_input)

        # 2. Decoder: Generate next chunk
        # Two-stage architecture:

        # Stage 1: Temporal module (frame-level)
        temporal_context = []
        for frame_idx in range(50):  # 2s = 50 frames
            frame_emb = self.temporal_module(
                encoder_output,
                frame_idx
            )
            temporal_context.append(frame_emb)

        # Stage 2: Depth module (RVQ-level)
        chunk_tokens = []
        for frame_idx in range(50):
            frame_tokens = []
            for rvq_level in range(16):
                token = self.depth_module(
                    temporal_context[frame_idx],
                    previous_rvq_tokens=frame_tokens
                )
                frame_tokens.append(token)
            chunk_tokens.append(frame_tokens)

        return chunk_tokens  # [50, 16]

# 배워야 할 것:
# - T5 architecture
# - Bidirectional vs Causal attention
# - Two-stage decoding (temporal + depth)
# - KV-cache for efficiency

# ═══════════════════════════════════════
# Component 4: Chunk-based Generation Loop
# ═══════════════════════════════════════

class StreamingGenerator:
    """
    목적: 무한 스트림 생성
    """

    def generate_stream(self, initial_style):
        state = {
            'chunks': [],  # 최근 5 chunks 유지
            'style': initial_style
        }

        while True:  # 무한 생성!
            # 1. 과거 10초 추출 (coarse)
            if len(state['chunks']) < 5:
                # Cold start: padding
                history = pad_to_5chunks(state['chunks'])
            else:
                # 최근 5 chunks
                history = state['chunks'][-5:]

            # Coarse context (4 RVQ levels만)
            history_coarse = [
                chunk[:, :, :4]  # [frames, RVQ] → [frames, 4]
                for chunk in history
            ]

            # 2. 다음 2초 생성 (16 RVQ levels)
            next_chunk = self.model.generate_chunk(
                history_coarse,
                state['style']
            )

            # 3. Audio로 변환 & 재생
            audio_2s = self.codec.decode(next_chunk)
            play(audio_2s)

            # 4. State 업데이트
            state['chunks'].append(next_chunk)

            # Sliding window (최대 5 chunks)
            if len(state['chunks']) > 5:
                state['chunks'].pop(0)

            # 5. Style 업데이트 (사용자가 변경했다면)
            if user_changed_style:
                state['style'] = new_style

# 배워야 할 것:
# - Stateless generation
# - Sliding window context
# - Cold start handling
# - Style transition smoothing
```

**학습 리소스**:
```
1. 논문 정독:
   - Live Music Models (arxiv 2508.04651)
   - LIVE_MUSIC_MODELS_ANALYSIS.md

2. 코드 분석:
   - github.com/magenta/magenta-realtime
   - 각 컴포넌트 코드 읽기
   - 데이터 플로우 추적

3. Colab 실습:
   - 공식 Colab 데모 실행
   - 각 단계별 intermediate 결과 출력
   - Parameter 변경해보며 효과 관찰

4. 시각화:
   - Architecture diagram 직접 그리기
   - Data flow 차트 작성
   - Tensor shapes 추적
```

**체크리스트**:
- [ ] MusicCoCa의 3개 컴포넌트 설명 가능
- [ ] SpectroStream encoding/decoding 과정 이해
- [ ] Two-stage decoder 구조 그릴 수 있음
- [ ] Chunk-based generation loop 코드 작성 가능
- [ ] Coarse context vs full generation 차이 설명 가능
- [ ] 전체 파이프라인을 순서도로 그릴 수 있음

**예상 시간**: 1주 (매일 3-4시간)

---

### 2.2 Real-time Generation 기술 ⭐⭐⭐

**왜 배워야?**
- 실시간 듀엣의 핵심 = 낮은 레이턴시
- RTF (Real-Time Factor) 최적화 필요

**무엇을 배울까?**

```python
# 1. Latency 분석

total_latency = (
    encoding_time +      # Audio → Tokens
    model_inference +    # Tokens → Next tokens
    decoding_time +      # Tokens → Audio
    audio_buffer         # 재생 버퍼
)

# 목표:
# Audio: ~800ms (Magenta RT)
# MIDI: <50ms (우리 목표!)

# 2. Optimization 기법들

# ═══ Model Optimization ═══

# A. Quantization
model_fp32 = load_model()  # 32-bit float
model_fp16 = quantize(model_fp32, 'fp16')  # 16-bit: 2배 빠름
model_int8 = quantize(model_fp32, 'int8')  # 8-bit: 4배 빠름

# B. KV-Cache (Transformer 최적화)
class TransformerWithCache:
    def forward(self, x, cache=None):
        if cache is None:
            # First step: compute all
            k = self.compute_keys(x)
            v = self.compute_values(x)
        else:
            # Subsequent steps: reuse cache
            k = torch.cat([cache['k'], self.compute_keys(x[-1:])])
            v = torch.cat([cache['v'], self.compute_values(x[-1:])])

        attention = self.attention(q, k, v)
        return attention, {'k': k, 'v': v}

# → 매번 전체 계산 안 함! (10배 빠름)

# C. Model Compilation
import torch
model = torch.compile(model, mode='reduce-overhead')
# → PyTorch 2.0 compilation (2배 빠름)

# D. Batch Size = 1
# Real-time에서는 batching 불가능
# → Single sample inference 최적화 필요

# ═══ Hardware Optimization ═══

# A. GPU Selection
# RTX 3060 (8GB): 괜찮음
# RTX 3090 (24GB): 완벽
# TPU v2-8: Colab 무료!

# B. Mixed Precision Training
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
# → FP16 연산으로 2배 빠름

# ═══ Algorithmic Optimization ═══

# A. Coarse Context (Magenta RT 핵심!)
# Context: 4 RVQ levels (coarse)
# Generation: 16 RVQ levels (fine)
# → 4배 메모리 절약, 속도 향상

# B. Chunk Size Tuning
# 작은 chunk: 낮은 latency, 불안정
# 큰 chunk: 높은 latency, 안정적
# Magenta RT: 2초 (최적 지점)

# 3. MIDI의 이점 (우리 프로젝트!)

# Audio tokenization:
audio_2s = 50 frames × 64 RVQ = 3,200 tokens
encoding_time = 50ms
decoding_time = 100ms

# MIDI tokenization:
midi_2s = ~100 events = 100 tokens
encoding_time = 5ms (10배 빠름!)
decoding_time = 10ms (10배 빠름!)

# → Total latency: 800ms → <50ms 가능!

# 4. Profiling & Benchmarking

import time

def profile_model():
    times = {
        'encoding': [],
        'model': [],
        'decoding': []
    }

    for _ in range(100):
        # Encoding
        t0 = time.time()
        tokens = encode(audio)
        times['encoding'].append(time.time() - t0)

        # Model inference
        t0 = time.time()
        output = model(tokens)
        times['model'].append(time.time() - t0)

        # Decoding
        t0 = time.time()
        audio = decode(output)
        times['decoding'].append(time.time() - t0)

    # 분석
    for stage, ts in times.items():
        print(f"{stage}: {np.mean(ts)*1000:.1f}ms ± {np.std(ts)*1000:.1f}ms")

    # RTF 계산
    chunk_duration = 2.0  # seconds
    total_time = sum(np.mean(ts) for ts in times.values())
    rtf = chunk_duration / total_time
    print(f"RTF: {rtf:.2f}x")

# 5. Cold Start 문제

# 첫 번째 chunk 생성 시:
# - Context가 없음
# - Model을 GPU로 로딩
# - 첫 inference는 느림

# 해결:
def warm_up_model(model):
    """Model warm-up으로 첫 latency 줄이기"""
    dummy_input = torch.zeros(1, 1012).to('cuda')
    for _ in range(5):
        _ = model(dummy_input)
    # → 이후 inference는 빠름!
```

**학습 리소스**:
```
1. 최적화 기술:
   - PyTorch Performance Tuning Guide
   - NVIDIA TensorRT 문서
   - torch.compile 가이드

2. Profiling 도구:
   - PyTorch Profiler
   - NVIDIA Nsight
   - Python cProfile

3. 실습:
   - Magenta RT 코드 profiling
   - 각 단계별 시간 측정
   - Bottleneck 찾기
   - Optimization 적용 & 재측정

4. 벤치마크:
   - 다양한 모델 크기 비교
   - GPU별 성능 비교
   - Quantization 효과 측정
```

**체크리스트**:
- [ ] Latency 구성 요소 설명 가능
- [ ] RTF 계산 및 해석 가능
- [ ] Quantization (FP16, INT8) 적용 가능
- [ ] KV-cache 구현 이해
- [ ] 모델 profiling 실행 가능
- [ ] Optimization 전후 성능 비교 가능
- [ ] MIDI가 Audio보다 빠른 이유 설명 가능

**예상 시간**: 4-5일

---

### 2.3 Audio Injection 메커니즘 ⭐⭐

**왜 배워야?**
- 실시간 상호작용의 핵심!
- MIDI injection 구현 시 필요

**무엇을 배울까?**

```python
# Audio Injection의 작동 원리

class AudioInjectionGenerator:
    """
    목적: 사용자 입력을 실시간으로 모델에 주입

    핵심 아이디어:
    1. 사용자 audio를 캡처
    2. 모델 output과 믹싱
    3. 믹싱된 것을 다음 context로 사용
    4. 모델이 사용자 입력에 "반응"
    """

    def __init__(self):
        self.model = MagentaRT()
        self.codec = SpectroStream()
        self.audio_buffer = []

    def inject_and_generate(self, user_audio, mix_ratio=0.3):
        """
        Args:
            user_audio: 사용자가 연주한 audio (2초)
            mix_ratio: 0.0-1.0, 사용자 audio 비율
        """

        # 1. 현재 context (과거 10초)
        context = self.get_context()

        # 2. 모델로 다음 2초 생성
        model_output = self.model.generate_chunk(context)
        model_audio = self.codec.decode(model_output)

        # 3. 사용자 audio와 믹싱
        mixed_audio = (
            mix_ratio * user_audio +
            (1 - mix_ratio) * model_audio
        )

        # ⚠️ 중요: 사용자 audio는 직접 재생 안 됨!
        # 대신 mixed_audio를 다음 context로 사용

        # 4. Mixed audio를 tokenize
        mixed_tokens = self.codec.encode(mixed_audio)

        # 5. Context 업데이트
        self.audio_buffer.append(mixed_tokens)

        # 6. Model output 재생 (not mixed!)
        play(model_audio)

        return model_audio

# 왜 이렇게 복잡하게?

# ═══ Naive approach (안 좋음) ═══
# User plays → Encode → Add to prompt
# → 모델이 사용자 입력을 "따라하기만" 함
# → 반응이 아닌 모방

# ═══ Audio injection (좋음!) ═══
# User plays → Mix with model output → Context
# → 모델이 "내가 방금 사용자와 함께 연주했다"고 인식
# → 자연스러운 대화/반응

# 실제 사용 예시:

def interactive_session():
    generator = AudioInjectionGenerator()

    while True:
        # 1. 사용자 입력 캡처 (마이크)
        user_audio = capture_from_mic(duration=2.0)

        # 2. AI 생성 + 믹싱
        ai_audio = generator.inject_and_generate(
            user_audio,
            mix_ratio=0.3  # 30% 사용자, 70% AI
        )

        # 3. AI만 재생 (사용자는 이미 들음)
        play_to_speaker(ai_audio)

        # 4. 다음 반복
        # → AI는 "user + 이전 AI"를 context로 본 상태
        # → 사용자 입력에 영향받은 다음 출력!

# MIDI Injection 구상 (우리가 구현할 것!)

class MIDIInjectionGenerator:
    """
    Audio injection → MIDI injection 변환
    """

    def inject_and_generate(self, user_midi_events, mix_ratio=0.3):
        """
        Args:
            user_midi_events: 사용자 MIDI 입력 (2초)
            mix_ratio: 사용자 입력 반영 비율
        """

        # 1. Context
        context = self.get_midi_context()  # 과거 10초 MIDI

        # 2. AI 생성
        ai_midi = self.model.generate_chunk(context)

        # 3. "믹싱" (MIDI의 경우)
        # Audio처럼 단순 mix 불가능
        # → 두 가지 방법:

        # Method A: Interleaving (교차)
        mixed_midi = []
        for user_note, ai_note in zip(user_midi_events, ai_midi):
            if random.random() < mix_ratio:
                mixed_midi.append(user_note)
            else:
                mixed_midi.append(ai_note)

        # Method B: Harmonic blending
        mixed_midi = []
        for user_note, ai_note in zip(user_midi_events, ai_midi):
            # 사용자 음에 AI가 화성 추가
            mixed_midi.append(user_note)
            if is_harmonically_compatible(user_note, ai_note):
                mixed_midi.append(ai_note)

        # Method C: Velocity blending
        mixed_midi = []
        for t in time_steps:
            user_notes = get_notes_at(user_midi_events, t)
            ai_notes = get_notes_at(ai_midi, t)

            # 사용자 음: 그대로
            # AI 음: velocity 낮춰서 background
            for note in user_notes:
                mixed_midi.append(note)
            for note in ai_notes:
                if note not in user_notes:
                    note.velocity *= 0.5  # 배경으로
                    mixed_midi.append(note)

        # 4. Context 업데이트
        self.midi_buffer.append(mixed_midi)

        # 5. AI만 출력
        return ai_midi

# 핵심 통찰:

# Audio injection:
# - 물리적 믹싱 가능 (waveform 더하기)
# - 자연스러움

# MIDI injection:
# - 물리적 믹싱 불가능 (discrete events)
# - 음악적 규칙 필요 (화성, 타이밍)
# - 더 창의적 접근 필요!

# 배워야 할 것:
# 1. Audio mixing 원리
# 2. Context management (sliding window)
# 3. MIDI event merging 전략
# 4. Harmonic compatibility 판단
# 5. Real-time MIDI routing
```

**학습 리소스**:
```
1. Audio processing:
   - Librosa 라이브러리
   - Audio mixing 기초
   - Real-time audio I/O (PyAudio)

2. MIDI processing:
   - Mido 라이브러리
   - Real-time MIDI I/O
   - MIDI event timing

3. 실습:
   - Magenta RT audio injection 데모
   - 직접 audio mixing 코드 작성
   - MIDI event merging 실험
   - Harmonic compatibility 함수 작성

4. 음악 이론:
   - 화성 이론 (어떤 음이 함께 어울리는가)
   - Voicing (음 배치)
   - Call-response 패턴
```

**체크리스트**:
- [ ] Audio injection 작동 원리 설명 가능
- [ ] 왜 mixing이 필요한지 이해
- [ ] Audio mixing 코드 작성 가능
- [ ] MIDI event merging 3가지 방법 설명 가능
- [ ] Harmonic compatibility 판단 함수 작성
- [ ] Real-time MIDI I/O 코드 작성 가능

**예상 시간**: 3-4일

---

## 📚 Phase 3: Fine-tuning 기술 (1.5주)

### 3.1 Transfer Learning & Fine-tuning 기초 ⭐⭐⭐

**왜 배워야?**
- Magenta RT를 "내 스타일"로 만들기
- 처음부터 학습 vs Fine-tuning 차이 이해

**무엇을 배울까?**

```python
# Transfer Learning의 철학

# ═══ From Scratch (처음부터) ═══
model = MagentaRT(random_weights=True)
model.train(
    data=my_100_hours_recordings,
    epochs=1000,
    time=3_months,
    cost=$10000
)
# → 데이터, 시간, 비용 엄청남!

# ═══ Transfer Learning (전이 학습) ═══
pretrained_model = MagentaRT.from_pretrained(
    'magenta-rt-large',  # 190,000시간 데이터로 학습됨!
)
my_model = pretrained_model.finetune(
    data=my_10_hours_recordings,  # 100시간 → 10시간!
    epochs=50,                     # 1000 → 50!
    time=1_week,                   # 3개월 → 1주!
    cost=$50                       # $10000 → $50!
)
# → 압도적으로 효율적!

# 왜 가능한가?

# Pretrained model이 이미 배운 것:
# - 음악의 기본 구조 (멜로디, 화성, 리듬)
# - 일반적인 패턴 (코드 진행, 리듬 패턴)
# - Audio tokenization (SpectroStream)
# - Long-term dependencies

# Fine-tuning으로 추가로 배우는 것:
# - 나만의 voicing 습관
# - 나만의 리듬 패턴
# - 나만의 프레이즈
# - 나만의 "언어"

# ═══ Fine-tuning Strategies ═══

# Strategy 1: Freeze lower layers
pretrained = load_pretrained()
for layer in pretrained.encoder.layers[:8]:  # 하위 8개 layer
    layer.requires_grad = False  # 얼림 (학습 안 함)

# 상위 4개 layer만 학습
for layer in pretrained.encoder.layers[8:]:
    layer.requires_grad = True

# 왜?
# - 하위 layer: 일반적인 특징 (모든 음악에 공통)
# - 상위 layer: 구체적인 특징 (스타일 특화)
# - 하위는 유지, 상위만 내 스타일로!

# Strategy 2: Different learning rates
optimizer = AdamW([
    {'params': pretrained.encoder.parameters(), 'lr': 1e-5},  # 작게
    {'params': pretrained.decoder.parameters(), 'lr': 1e-4},  # 크게
])

# 왜?
# - Encoder: 약간만 수정 (일반 지식 유지)
# - Decoder: 많이 수정 (내 스타일 학습)

# Strategy 3: Gradual unfreezing
# Epoch 1-10: 상위 2개 layer만
# Epoch 11-20: 상위 4개 layer
# Epoch 21-30: 상위 6개 layer
# ...
# → 점진적으로 더 많이 학습

# ═══ Catastrophic Forgetting 방지 ═══

# 문제:
pretrained_model.quality = 90/100  # 훌륭한 일반 음악 생성
my_finetuned.quality_on_my_style = 95/100  # 내 스타일 완벽
my_finetuned.quality_on_general = 30/100   # 일반 음악은 못 만듦!
# → 일반 능력을 "잊어버림" (catastrophic forgetting)

# 해결책 1: Regularization
loss = (
    style_loss(output, my_data) +           # 내 스타일 학습
    0.1 * general_loss(output, general_data)  # 일반 능력 유지
)

# 해결책 2: Small learning rate
lr = 1e-5  # 천천히 학습 (급격한 변화 방지)

# 해결책 3: Early stopping
# Validation loss 증가하면 멈춤
# → 과적합 방지

# ═══ Data Augmentation (내 데이터 늘리기) ═══

# 내 녹음: 10시간 → 부족!
# Data augmentation으로 확장:

def augment_midi(midi_file):
    """1개 MIDI → 36개로 증식!"""
    augmented = []

    # 1. Transposition (조옮김)
    for semitones in range(-2, 3):  # -2 ~ +2 semitones
        transposed = transpose(midi_file, semitones)
        augmented.append(transposed)
    # → 5배

    # 2. Tempo variation (템포 변화)
    for tempo_ratio in [0.9, 1.0, 1.1]:
        tempo_changed = change_tempo(midi_file, tempo_ratio)
        augmented.append(tempo_changed)
    # → 3배

    # 3. 조합
    for trans in range(-2, 3):
        for tempo in [0.9, 1.0, 1.1]:
            aug = transpose(midi_file, trans)
            aug = change_tempo(aug, tempo)
            augmented.append(aug)
    # → 15배

    return augmented
# 10시간 × 36 = 360시간 equivalent!

# 주의: 과도한 augmentation은 역효과
# - Transposition: ±2 semitones (자연스러움)
# - Tempo: ±10% (자연스러움)
# - 너무 많으면 내 "진짜" 스타일 희석
```

**학습 리소스**:
```
1. 이론:
   - Transfer learning 기초 (CS231n Lecture)
   - Fine-tuning best practices
   - Catastrophic forgetting 논문

2. 실습:
   - HuggingFace 모델 fine-tuning 튜토리얼
   - PEFT 라이브러리 예제
   - 작은 모델로 실험 (GPT-2 등)

3. 도메인 지식:
   - 음악 데이터 augmentation 기법
   - Style transfer in music
   - Personalization in generative models

4. 실습 과제:
   - Pretrained 모델 로드
   - 일부 layer freeze
   - 작은 데이터로 fine-tune
   - 품질 비교 (before/after)
```

**체크리스트**:
- [ ] Transfer learning 장점 설명 가능
- [ ] Layer freezing 전략 이해
- [ ] Learning rate 차별화 이유 설명
- [ ] Catastrophic forgetting 방지법 3가지
- [ ] Data augmentation 코드 작성
- [ ] Fine-tuning 실험 실행 가능

**예상 시간**: 4-5일

---

### 3.2 LoRA & QLoRA 완벽 이해 ⭐⭐⭐

**왜 배워야?**
- GPU 메모리 부족 해결
- Fine-tuning 효율 10,000배 향상!
- Magenta RT를 RTX 3060에서 학습 가능

**무엇을 배울까?**

```python
# LoRA: Low-Rank Adaptation

# ═══ 기존 Fine-tuning (Full) ═══

# Magenta RT Large: 760M parameters
# Fine-tuning: 모든 760M 파라미터 학습
# GPU 메모리: 40GB+ 필요
# 학습 시간: 며칠
# 비용: $$$

class FullFineTuning:
    def __init__(self, pretrained_model):
        self.model = pretrained_model  # 760M params

    def train(self, my_data):
        # 모든 파라미터 업데이트!
        for param in self.model.parameters():
            param.requires_grad = True  # 760M개 모두!

        optimizer = AdamW(self.model.parameters())
        # Optimizer state: 760M × 2 = 1.5B values
        # → 메모리 폭발! 💥

# ═══ LoRA (Low-Rank Adaptation) ═══

# 핵심 아이디어:
# "Full fine-tuning은 사실 low-rank space에서 일어난다"
# → 실제로는 작은 부분공간만 변화
# → 그 부분공간만 학습하자!

class LoRALayer:
    """
    Original: W (d × k)
    LoRA: W + ΔW, where ΔW = A @ B
    - A: (d × r)
    - B: (r × k)
    - r << d, k (rank가 작음!)
    """

    def __init__(self, original_layer, rank=8):
        self.W = original_layer.weight  # (d, k)
        self.d, self.k = self.W.shape
        self.r = rank

        # LoRA matrices (trainable!)
        self.A = nn.Parameter(torch.randn(self.d, self.r))
        self.B = nn.Parameter(torch.randn(self.r, self.k))

        # Original은 freeze
        self.W.requires_grad = False

    def forward(self, x):
        # Original output + LoRA adaptation
        return x @ self.W + x @ self.A @ self.B
        #      ^^^^^^^^     ^^^^^^^^^^^^^^
        #      Frozen       Trainable!

# 예시: Transformer attention layer

# Original:
W_q = torch.randn(768, 768)  # 768 × 768 = 589,824 params

# LoRA with rank=8:
A = torch.randn(768, 8)  # 768 × 8 = 6,144 params
B = torch.randn(8, 768)  # 8 × 768 = 6,144 params
# Total: 12,288 params (98% 감소!)

# Full model:
# Magenta RT: 760M params to train

# With LoRA (r=8):
# Only: ~2M params to train (99.7% 감소!)
# → GPU 메모리: 40GB → 8GB
# → 학습 속도: 3배 빠름
# → RTX 3060으로 가능! ✅

# LoRA 적용 코드:

from peft import LoraConfig, get_peft_model

# 1. Config
lora_config = LoraConfig(
    r=8,                    # Rank (핵심 파라미터!)
    lora_alpha=16,          # Scaling factor
    target_modules=[        # 어느 layer에 적용?
        "q_proj",           # Query projection
        "v_proj",           # Value projection
        "k_proj",           # Key projection (optional)
        "o_proj",           # Output projection (optional)
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 2. 적용
model = MagentaRT.from_pretrained('large')
model = get_peft_model(model, lora_config)

# 3. 학습 가능한 파라미터 확인
model.print_trainable_parameters()
# Output:
# trainable params: 2,097,152 / 760,000,000 = 0.28%

# 4. 학습 (일반 fine-tuning과 동일!)
optimizer = AdamW(model.parameters(), lr=1e-4)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()

# 5. 저장 (작음!)
model.save_pretrained('my_lora_weights')
# File size: ~10MB (vs 3GB for full model!)

# ═══ QLoRA (Quantized LoRA) ═══

# LoRA + Quantization = 극한의 효율!

# Quantization:
# - FP32 (32-bit): 1 param = 4 bytes
# - FP16 (16-bit): 1 param = 2 bytes
# - INT8 (8-bit): 1 param = 1 byte
# - INT4 (4-bit): 1 param = 0.5 bytes

# QLoRA = 4-bit quantization + LoRA

from transformers import BitsAndBytesConfig

qlora_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",      # Normal Float 4
    bnb_4bit_use_double_quant=True
)

model = MagentaRT.from_pretrained(
    'large',
    quantization_config=qlora_config,
    device_map='auto'
)
# → Model size: 3GB → 750MB (75% 감소!)

model = get_peft_model(model, lora_config)

# Total GPU memory:
# Model (quantized): 750MB
# LoRA params: 10MB
# Optimizer: 20MB
# Activations: ~3GB
# Total: ~4GB
# → RTX 3060 (8GB) 충분! ✅✅✅

# ═══ Rank 선택 가이드 ═══

# Rank = LoRA의 핵심 하이퍼파라미터

# r=4: 매우 작음
# - 메모리: 최소
# - 학습 데이터: 1-5시간
# - 스타일 학습: 제한적
# - 사용 케이스: 빠른 실험

# r=8: 표준 (추천!)
# - 메모리: 작음
# - 학습 데이터: 10-50시간
# - 스타일 학습: 충분
# - 사용 케이스: 대부분의 경우

# r=16: 큼
# - 메모리: 중간
# - 학습 데이터: 50-100시간
# - 스타일 학습: 매우 정교
# - 사용 케이스: 고품질 필요 시

# r=32: 매우 큼
# - 메모리: 큼
# - 학습 데이터: 100시간+
# - 스타일 학습: 과적합 위험
# - 사용 케이스: 특수한 경우

# 실험:
# 1. r=8로 시작
# 2. Validation loss 확인
# 3. 너무 높으면 r 증가
# 4. 과적합되면 r 감소

# ═══ LoRA Merging (배포 시) ═══

# Fine-tuning 후:
# Base model (3GB) + LoRA weights (10MB)

# Inference 시 두 가지 옵션:

# Option 1: 분리 (개발/실험)
base_model = load('magenta-rt-large')
lora_weights = load('my_lora_weights')
output = base_model(input) + lora_weights(input)

# Option 2: Merge (배포)
model = merge_lora_weights(base_model, lora_weights)
# → Single model (3GB)
# → 더 빠른 inference
output = model(input)

# Merge 코드:
def merge_lora():
    model = MagentaRT.from_pretrained('large')
    model = PeftModel.from_pretrained(model, 'my_lora_weights')

    # Merge!
    merged = model.merge_and_unload()
    merged.save_pretrained('my_finetuned_model')

# ═══ 실전 예시: 내 프로젝트 ═══

# 내 환경:
# - GPU: RTX 3060 (8GB)
# - 데이터: 내 연주 20시간
# - 목표: "ohhalim style" 학습

# 1. QLoRA config (4-bit)
qlora_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# 2. LoRA config (r=8)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

# 3. Model 로드
model = MagentaRT.from_pretrained(
    'large',
    quantization_config=qlora_config
)
model = get_peft_model(model, lora_config)

# 4. 학습 데이터
my_data = load_my_recordings('my_20_hours/')
my_data = augment(my_data)  # 20시간 → 200시간 equivalent

# 5. 학습!
trainer = Trainer(
    model=model,
    train_dataset=my_data,
    args=TrainingArguments(
        output_dir='ohhalim-style',
        num_train_epochs=50,
        per_device_train_batch_size=1,  # GPU 메모리 고려
        learning_rate=1e-4,
        save_steps=1000,
    )
)
trainer.train()

# 6. 저장
model.save_pretrained('ohhalim-lora')
# → 10MB file! (공유 쉬움)

# 7. 사용
my_model = load_model_with_lora('magenta-rt-large', 'ohhalim-lora')
output = my_model.generate(style_prompt="ohhalim style")
# → 내 스타일로 연주! 🎹
```

**학습 리소스**:
```
1. 논문:
   - LoRA (Hu et al., 2021) - PAPERS_TO_READ.md
   - QLoRA (Dettmers et al., 2023)

2. 코드:
   - HuggingFace PEFT 라이브러리
   - QLoRA GitHub repository
   - LoRA 튜토리얼들

3. 실습:
   - 작은 모델로 LoRA 실험 (GPT-2)
   - Rank 변화에 따른 효과 측정
   - QLoRA로 큰 모델 학습 (Llama 등)

4. 비교 실험:
   - Full fine-tuning vs LoRA
   - LoRA vs QLoRA
   - 다양한 rank (r=4, 8, 16, 32)
```

**체크리스트**:
- [ ] LoRA 수식 이해 및 유도 가능
- [ ] Low-rank 개념 설명 가능
- [ ] Rank 선택 가이드라인 이해
- [ ] QLoRA = 4-bit + LoRA 설명 가능
- [ ] PEFT 라이브러리로 LoRA 적용 가능
- [ ] Trainable params 계산 가능
- [ ] LoRA merge 코드 작성 가능
- [ ] 메모리 절감 계산 가능 (760M → 2M)

**예상 시간**: 5-6일

---

## 📚 Phase 4: 실전 구현 (2주)

### 4.1 MIDI Tokenization 설계 & 구현 ⭐⭐⭐

**왜 배워야?**
- SpectroStream → MIDI tokenizer 대체
- 30배 효율 향상의 핵심!

**무엇을 배울까?**

```python
# MIDI Tokenizer 설계

# ═══ 요구사항 분석 ═══

# SpectroStream (Audio):
# - 2초 = 50 frames × 64 RVQ = 3,200 tokens
# - Vocabulary: 1,024 per RVQ level
# - Hierarchical (coarse → fine)

# Our MIDI Tokenizer:
# - 2초 = ~100 events = ~100 tokens (30배 적음!)
# - Vocabulary: ~500 tokens
# - Expressive (velocity, timing, pedal)

# ═══ Tokenization 방식 비교 ═══

# Method 1: Event-based (추천!)
# 장점: 직관적, 표현력 높음
# 단점: Variable length

# Example:
events = [
    ("NOTE_ON", 60, 80, 0.0),      # (type, pitch, velocity, time)
    ("NOTE_ON", 64, 75, 0.0),      # Chord: C-E
    ("NOTE_ON", 67, 75, 0.0),      # Chord: C-E-G
    ("NOTE_OFF", 60, 0, 0.5),
    ("NOTE_OFF", 64, 0, 0.5),
    ("NOTE_OFF", 67, 0, 0.5),
    ("NOTE_ON", 65, 70, 0.5),      # F
    ("NOTE_OFF", 65, 0, 1.0),
]

# Tokenize:
tokens = [
    TOKEN_NOTE_ON_60,    # Base: 0-127
    TOKEN_VELOCITY_80,   # Base: 128-255
    TOKEN_TIME_0,        # Base: 256
    TOKEN_NOTE_ON_64,
    TOKEN_VELOCITY_75,
    TOKEN_TIME_0,
    TOKEN_NOTE_ON_67,
    TOKEN_VELOCITY_75,
    TOKEN_TIME_0,
    TOKEN_TIME_SHIFT_500ms,  # Base: 300
    TOKEN_NOTE_OFF_60,   # Base: 400-527
    ...
]

# Vocabulary design:
vocab = {
    # Note events: 0-255
    "NOTE_ON_0": 0,
    "NOTE_ON_1": 1,
    ...
    "NOTE_ON_127": 127,
    "NOTE_OFF_0": 128,
    ...
    "NOTE_OFF_127": 255,

    # Velocity: 256-383 (128 bins)
    "VEL_0": 256,    # ppp
    "VEL_1": 257,
    ...
    "VEL_127": 383,  # fff

    # Time shifts: 384-511 (128 bins)
    "TIME_0ms": 384,
    "TIME_10ms": 385,
    "TIME_20ms": 386,
    ...
    "TIME_2000ms": 511,

    # Special tokens: 512-527
    "BAR": 512,
    "POSITION_0": 513,
    ...
    "POSITION_15": 528,  # 16분음표 단위

    # Total vocabulary: ~530 tokens
}

# Method 2: REMI (Representation of MIDi)
# 장점: 구조화됨, quantized timing
# 단점: 덜 표현력

tokens_remi = [
    BAR_START,        # 512
    POSITION_0,       # 513 (16분음표 위치)
    PITCH_60,         # 0-127
    VELOCITY_80,      # 256-383
    DURATION_8,       # 530-545 (8분음표)
    POSITION_2,       # 515
    PITCH_64,
    VELOCITY_75,
    DURATION_8,
    ...
]

# ═══ 구현: EventBasedMIDITokenizer ═══

class EventBasedMIDITokenizer:
    """
    MIDI events ↔ Token IDs
    """

    def __init__(self):
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Time quantization
        self.time_bins = np.linspace(0, 2000, 128)  # 0-2000ms, 128 bins

    def _build_vocab(self):
        vocab = {}
        idx = 0

        # NOTE_ON: 0-127
        for pitch in range(128):
            vocab[f"NOTE_ON_{pitch}"] = idx
            idx += 1

        # NOTE_OFF: 128-255
        for pitch in range(128):
            vocab[f"NOTE_OFF_{pitch}"] = idx
            idx += 1

        # VELOCITY: 256-383
        for vel in range(128):
            vocab[f"VEL_{vel}"] = idx
            idx += 1

        # TIME_SHIFT: 384-511
        for i in range(128):
            vocab[f"TIME_{i}"] = idx
            idx += 1

        # POSITION: 512-527 (16분음표 단위)
        for pos in range(16):
            vocab[f"POS_{pos}"] = idx
            idx += 1

        # Special tokens
        vocab["PAD"] = idx; idx += 1
        vocab["BOS"] = idx; idx += 1  # Begin of sequence
        vocab["EOS"] = idx; idx += 1  # End of sequence
        vocab["BAR"] = idx; idx += 1

        return vocab

    def encode(self, midi_file, duration=2.0):
        """
        MIDI file → Token IDs

        Args:
            midi_file: path to MIDI file
            duration: chunk duration in seconds

        Returns:
            tokens: List[int]
        """
        # 1. Parse MIDI
        midi = mido.MidiFile(midi_file)
        events = []
        current_time = 0

        for msg in midi:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                events.append({
                    'type': 'note_on',
                    'pitch': msg.note,
                    'velocity': msg.velocity,
                    'time': current_time
                })
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                events.append({
                    'type': 'note_off',
                    'pitch': msg.note,
                    'time': current_time
                })

        # 2. Filter by duration
        events = [e for e in events if e['time'] <= duration]

        # 3. Convert to tokens
        tokens = [self.vocab["BOS"]]
        prev_time = 0

        for event in events:
            # Time shift
            time_delta = event['time'] - prev_time
            if time_delta > 0:
                time_bin = np.digitize(time_delta * 1000, self.time_bins)
                time_bin = min(time_bin, 127)
                tokens.append(self.vocab[f"TIME_{time_bin}"])

            # Note event
            if event['type'] == 'note_on':
                tokens.append(self.vocab[f"NOTE_ON_{event['pitch']}"])
                tokens.append(self.vocab[f"VEL_{event['velocity']}"])
            else:
                tokens.append(self.vocab[f"NOTE_OFF_{event['pitch']}"])

            prev_time = event['time']

        tokens.append(self.vocab["EOS"])
        return tokens

    def decode(self, tokens):
        """
        Token IDs → MIDI events

        Args:
            tokens: List[int]

        Returns:
            midi_events: List[dict]
        """
        events = []
        current_time = 0
        current_velocity = 64  # Default

        for token_id in tokens:
            token = self.id_to_token[token_id]

            if token.startswith("TIME_"):
                # Time shift
                bin_idx = int(token.split("_")[1])
                time_delta = self.time_bins[bin_idx] / 1000  # ms → s
                current_time += time_delta

            elif token.startswith("NOTE_ON_"):
                pitch = int(token.split("_")[2])
                events.append({
                    'type': 'note_on',
                    'pitch': pitch,
                    'velocity': current_velocity,
                    'time': current_time
                })

            elif token.startswith("NOTE_OFF_"):
                pitch = int(token.split("_")[2])
                events.append({
                    'type': 'note_off',
                    'pitch': pitch,
                    'time': current_time
                })

            elif token.startswith("VEL_"):
                current_velocity = int(token.split("_")[1])

        return events

    def to_midi_file(self, events, output_path, tempo=120):
        """
        Events → MIDI file
        """
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)

        # Tempo
        track.append(mido.MetaMessage('set_tempo',
                                      tempo=mido.bpm2tempo(tempo)))

        # Convert events to messages
        prev_time = 0
        for event in events:
            delta_time = int((event['time'] - prev_time) * 480)  # ticks

            if event['type'] == 'note_on':
                track.append(mido.Message(
                    'note_on',
                    note=event['pitch'],
                    velocity=event['velocity'],
                    time=delta_time
                ))
            elif event['type'] == 'note_off':
                track.append(mido.Message(
                    'note_off',
                    note=event['pitch'],
                    velocity=0,
                    time=delta_time
                ))

            prev_time = event['time']

        midi.save(output_path)

# ═══ 사용 예시 ═══

# 1. Tokenizer 생성
tokenizer = EventBasedMIDITokenizer()
print(f"Vocabulary size: {tokenizer.vocab_size}")  # ~530

# 2. MIDI → Tokens
tokens = tokenizer.encode("my_improvisation.mid", duration=2.0)
print(f"2 seconds = {len(tokens)} tokens")  # ~100 tokens

# 3. Tokens → MIDI
events = tokenizer.decode(tokens)
tokenizer.to_midi_file(events, "reconstructed.mid")

# 4. 품질 확인
original = mido.MidiFile("my_improvisation.mid")
reconstructed = mido.MidiFile("reconstructed.mid")
# → 청취 & 비교

# ═══ SpectroStream와 통합 ═══

# Magenta RT에서 SpectroStream 대신 MIDI tokenizer 사용

class MagentaRTMIDI:
    """
    Magenta RT adapted for MIDI
    """

    def __init__(self):
        self.tokenizer = EventBasedMIDITokenizer()
        self.encoder = MagentaRTEncoder()  # 그대로 사용!
        self.decoder = MagentaRTDecoder()  # 그대로 사용!
        # SpectroStream → tokenizer로 대체

    def generate_chunk(self, history_midi, style):
        """
        Input: MIDI events (10초)
        Output: MIDI events (2초)
        """
        # 1. MIDI → Tokens
        history_tokens = []
        for chunk_midi in history_midi:
            tokens = self.tokenizer.encode(chunk_midi)
            # Coarse: 전체 tokens (MIDI는 이미 충분히 작음)
            history_tokens.append(tokens)

        # 2. Encoder
        encoder_input = torch.cat(history_tokens + [style])
        encoder_output = self.encoder(encoder_input)

        # 3. Decoder
        next_tokens = self.decoder.generate(
            encoder_output,
            max_length=100  # ~2초 분량
        )

        # 4. Tokens → MIDI
        next_events = self.tokenizer.decode(next_tokens)

        return next_events

# 효율 비교:
# Audio (SpectroStream):
# - 2초 = 3,200 tokens
# - Encoding: 50ms
# - Decoding: 100ms

# MIDI (EventBased):
# - 2초 = 100 tokens (32배 적음!)
# - Encoding: 5ms (10배 빠름!)
# - Decoding: 10ms (10배 빠름!)

# → Total latency: 800ms → 50ms! 🚀
```

**학습 리소스**:
```
1. MIDI 기초:
   - Mido 라이브러리 문서
   - MIDI specification
   - Pretty_midi 라이브러리

2. Tokenization 연구:
   - Miditok 라이브러리 분석
   - "This Time with Feeling" 논문
   - Music Transformer tokenization

3. 실습:
   - 다양한 MIDI 파일로 테스트
   - Tokenize → Detokenize 품질 평가
   - Vocabulary size 최적화
   - 내 연주 데이터로 실험

4. 통합:
   - Magenta RT 코드 읽기
   - SpectroStream 사용 부분 찾기
   - MIDI tokenizer로 교체 계획
```

**체크리스트**:
- [ ] Event-based vs REMI 비교 설명
- [ ] Vocabulary 설계 논리 이해
- [ ] EventBasedMIDITokenizer 구현 완료
- [ ] Encode/decode 동작 확인
- [ ] Reconstruction 품질 평가
- [ ] SpectroStream 대체 계획 수립
- [ ] 효율 개선 계산 (32배)

**예상 시간**: 5-6일

---

### 4.2 Real-time MIDI Generation System ⭐⭐⭐

**왜 배워야?**
- 모든 것을 통합하는 최종 시스템!
- "나와 가상의 내가 JAM!" 실현

**무엇을 배울까?**

```python
# 전체 시스템 설계

# ═══ System Architecture ═══

"""
┌────────────────────────────────────────────────┐
│  Input Layer (Real-time MIDI Input)            │
│  - MIDI Keyboard                               │
│  - Ableton/FL Studio                          │
│  - Virtual MIDI ports                          │
└───────────────┬────────────────────────────────┘
                ↓
┌────────────────────────────────────────────────┐
│  Capture & Buffer Layer                        │
│  - 2-second chunks                            │
│  - Thread-safe queue                           │
│  - Timing synchronization                      │
└───────────────┬────────────────────────────────┘
                ↓
┌────────────────────────────────────────────────┐
│  Analysis Layer                                │
│  - Chord detection                             │
│  - Rhythm analysis                             │
│  - Style extraction                            │
└───────────────┬────────────────────────────────┘
                ↓
┌────────────────────────────────────────────────┐
│  AI Generation Layer                           │
│  - MagentaRT-MIDI model                       │
│  - My style (fine-tuned)                      │
│  - Context: 10s history                        │
└───────────────┬────────────────────────────────┘
                ↓
┌────────────────────────────────────────────────┐
│  MIDI Injection Layer                          │
│  - Merge user + AI MIDI                       │
│  - Harmonic blending                           │
│  - Timing sync                                 │
└───────────────┬────────────────────────────────┘
                ↓
┌────────────────────────────────────────────────┐
│  Output Layer                                  │
│  - Virtual MIDI out                            │
│  - DAW routing                                 │
│  - Synth/VST                                  │
└────────────────────────────────────────────────┘
"""

# ═══ Implementation ═══

import threading
import queue
import time
import mido
import torch

class RealTimeMIDIDuetSystem:
    """
    나와 가상의 내가 JAM!
    """

    def __init__(self, model_path, input_port, output_port):
        # 1. Model
        self.model = self.load_finetuned_model(model_path)
        self.tokenizer = EventBasedMIDITokenizer()

        # 2. MIDI I/O
        self.input_port = mido.open_input(input_port)
        self.output_port = mido.open_output(output_port)

        # 3. State
        self.context_buffer = []  # 최근 10초 MIDI
        self.user_playing = False
        self.ai_enabled = True

        # 4. Threading
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # 5. Timing
        self.chunk_duration = 2.0  # seconds
        self.latency_target = 0.05  # 50ms

        # 6. Style
        self.my_style = self.load_my_style()

    def load_finetuned_model(self, path):
        """Load fine-tuned model"""
        base_model = MagentaRTMIDI.from_pretrained('large')
        model = PeftModel.from_pretrained(base_model, path)
        model.eval()
        model.to('cuda')
        return model

    def load_my_style(self):
        """Load my style embedding"""
        # Option 1: From text
        style = self.model.encode_text("ohhalim jazz piano style")

        # Option 2: From audio reference
        # my_recording = "my_best_improv.wav"
        # style = self.model.encode_audio(my_recording)

        return style

    def start(self):
        """Start the duet system"""
        # Start threads
        input_thread = threading.Thread(target=self.input_loop)
        generation_thread = threading.Thread(target=self.generation_loop)
        output_thread = threading.Thread(target=self.output_loop)

        input_thread.start()
        generation_thread.start()
        output_thread.start()

        print("🎹 Real-time MIDI Duet System Started!")
        print("Play on your MIDI keyboard...")
        print("Press Ctrl+C to stop")

        try:
            input_thread.join()
            generation_thread.join()
            output_thread.join()
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            self.stop()

    def input_loop(self):
        """Thread 1: Capture MIDI input"""
        chunk = []
        chunk_start_time = time.time()

        for msg in self.input_port:
            current_time = time.time()
            elapsed = current_time - chunk_start_time

            # Add to current chunk
            chunk.append({
                'message': msg,
                'time': elapsed
            })

            # Check if chunk is complete
            if elapsed >= self.chunk_duration:
                # Put chunk in queue
                self.input_queue.put(chunk)

                # Start new chunk
                chunk = []
                chunk_start_time = current_time

    def generation_loop(self):
        """Thread 2: AI generation"""
        while True:
            # Wait for input chunk
            user_chunk = self.input_queue.get()

            # Detect if user is playing
            self.user_playing = len(user_chunk) > 0

            if self.ai_enabled:
                # Generate AI response
                t0 = time.time()

                # 1. Tokenize user input
                user_tokens = self.midi_chunk_to_tokens(user_chunk)

                # 2. Update context
                self.context_buffer.append(user_tokens)
                if len(self.context_buffer) > 5:  # Keep last 10s
                    self.context_buffer.pop(0)

                # 3. Generate AI response
                ai_tokens = self.model.generate_chunk(
                    history=self.context_buffer,
                    style=self.my_style,
                    temperature=1.0,
                    top_k=40
                )

                # 4. Decode to MIDI
                ai_events = self.tokenizer.decode(ai_tokens)

                # 5. MIDI Injection (blend with user)
                blended_events = self.midi_injection(
                    user_chunk,
                    ai_events,
                    mix_ratio=0.3
                )

                # 6. Update context with blended
                blended_tokens = self.events_to_tokens(blended_events)
                self.context_buffer[-1] = blended_tokens

                # 7. Put AI events to output queue
                self.output_queue.put(ai_events)

                # 8. Measure latency
                latency = time.time() - t0
                if latency > self.latency_target:
                    print(f"⚠️ Latency: {latency*1000:.1f}ms "
                          f"(target: {self.latency_target*1000:.0f}ms)")

    def output_loop(self):
        """Thread 3: MIDI output"""
        while True:
            # Wait for AI events
            ai_events = self.output_queue.get()

            # Send to MIDI output
            for event in ai_events:
                # Convert to mido message
                if event['type'] == 'note_on':
                    msg = mido.Message(
                        'note_on',
                        note=event['pitch'],
                        velocity=event['velocity']
                    )
                elif event['type'] == 'note_off':
                    msg = mido.Message(
                        'note_off',
                        note=event['pitch']
                    )

                # Send!
                self.output_port.send(msg)

                # Wait for timing
                if 'delta_time' in event:
                    time.sleep(event['delta_time'])

    def midi_injection(self, user_chunk, ai_events, mix_ratio=0.3):
        """
        Blend user MIDI with AI MIDI

        Strategy: Harmonic blending
        - User notes: 100% kept
        - AI notes: Added if harmonically compatible
        """
        blended = []

        # Extract user notes (time → notes mapping)
        user_notes_by_time = self.group_by_time(user_chunk)
        ai_notes_by_time = self.group_by_time(ai_events)

        # Merge
        all_times = sorted(set(
            list(user_notes_by_time.keys()) +
            list(ai_notes_by_time.keys())
        ))

        for t in all_times:
            user_notes = user_notes_by_time.get(t, [])
            ai_notes = ai_notes_by_time.get(t, [])

            # Add all user notes
            blended.extend(user_notes)

            # Add AI notes if:
            # 1. No user notes at this time (AI solo)
            # 2. Harmonically compatible with user
            if len(user_notes) == 0:
                # AI solo
                blended.extend(ai_notes)
            else:
                # Check harmony
                user_pitches = [n['pitch'] for n in user_notes]
                for ai_note in ai_notes:
                    if self.is_harmonically_compatible(
                        ai_note['pitch'],
                        user_pitches
                    ):
                        # Compatible! Add as background
                        ai_note['velocity'] = int(ai_note['velocity'] * 0.6)
                        blended.append(ai_note)

        return blended

    def is_harmonically_compatible(self, ai_pitch, user_pitches):
        """
        Check if AI note is harmonically compatible with user notes
        """
        # Simple heuristic: interval check
        for user_pitch in user_pitches:
            interval = abs((ai_pitch - user_pitch) % 12)

            # Dissonant intervals: m2, M7, tritone
            if interval in [1, 6, 11]:
                return False

        # Consonant!
        return True

    def stop(self):
        """Stop the system"""
        self.input_port.close()
        self.output_port.close()
        print("✅ System stopped")

# ═══ 사용 예시 ═══

# 1. MIDI 포트 확인
print("Available MIDI inputs:")
print(mido.get_input_names())
print("\nAvailable MIDI outputs:")
print(mido.get_output_names())

# 2. 시스템 시작
system = RealTimeMIDIDuetSystem(
    model_path='ohhalim-lora',
    input_port='MIDI Keyboard',
    output_port='Virtual MIDI Bus 1'
)

# 3. 듀엣 시작!
system.start()

# → 이제 MIDI 키보드로 연주하면
# → AI가 실시간으로 반응! 🎹✨

# ═══ DAW 통합 (Ableton 예시) ═══

"""
Setup in Ableton:

Track 1: "Me" (Real piano)
  - Input: MIDI Keyboard
  - Monitor: In
  - Instrument: Piano VST

Track 2: "AI Me" (Virtual me)
  - Input: Virtual MIDI Bus 1 (from our system)
  - Monitor: In
  - Instrument: Electric Piano VST

→ 두 트랙이 동시에 연주!
→ 나와 가상의 내가 JAM! 🎉
"""

# ═══ Performance Tuning ═══

class OptimizedDuetSystem(RealTimeMIDIDuetSystem):
    """
    Optimized version with profiling
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Profiling
        self.latencies = []
        self.rtf_values = []

        # Optimizations
        self.model = torch.compile(self.model)  # PyTorch 2.0
        self.model.half()  # FP16

        # KV-cache warm-up
        self.warmup_model()

    def warmup_model(self):
        """Warm-up to reduce first inference latency"""
        print("🔥 Warming up model...")
        dummy_input = torch.zeros(1, 100).long().to('cuda')
        for _ in range(5):
            _ = self.model(dummy_input)
        print("✅ Model ready!")

    def generation_loop(self):
        """With profiling"""
        while True:
            user_chunk = self.input_queue.get()

            if self.ai_enabled:
                t0 = time.time()

                # Generation
                ai_tokens = self.model.generate_chunk(
                    history=self.context_buffer,
                    style=self.my_style
                )

                latency = time.time() - t0
                self.latencies.append(latency)

                # RTF
                rtf = self.chunk_duration / latency
                self.rtf_values.append(rtf)

                # Print stats every 10 chunks
                if len(self.latencies) % 10 == 0:
                    avg_latency = np.mean(self.latencies[-10:])
                    avg_rtf = np.mean(self.rtf_values[-10:])
                    print(f"📊 Latency: {avg_latency*1000:.1f}ms, "
                          f"RTF: {avg_rtf:.1f}x")

                # ... rest of generation loop
```

**학습 리소스**:
```
1. Real-time programming:
   - Python threading & multiprocessing
   - Queue & synchronization
   - Low-latency best practices

2. MIDI I/O:
   - Mido library advanced usage
   - Virtual MIDI ports (loopMIDI on Windows)
   - rtmidi library

3. DAW integration:
   - Ableton MIDI routing
   - FL Studio MIDI setup
   - Logic Pro MIDI environment

4. 실습:
   - 간단한 MIDI echo 프로그램
   - Latency 측정 도구
   - MIDI injection 프로토타입
   - 전체 시스템 통합

5. Debugging:
   - MIDI monitor 도구
   - Latency profiling
   - Thread synchronization 이슈
```

**체크리스트**:
- [ ] Multi-threading 구조 이해
- [ ] MIDI I/O 코드 작성 가능
- [ ] Chunk-based capture 구현
- [ ] AI generation loop 구현
- [ ] MIDI injection 구현
- [ ] DAW 통합 설정 완료
- [ ] Latency profiling 구현
- [ ] 전체 시스템 동작 확인
- [ ] "나와 가상의 내가 JAM!" 성공! 🎉

**예상 시간**: 7일 (가장 중요!)

---

## 🗓️ 주차별 학습 플랜

### Week 1-2: 기초 이론
```
Day 1-3: Transformer 이해
Day 4-5: Audio/MIDI tokenization
Day 6-7: Music generation 기초
Weekend: 복습 & 실습 프로젝트
```

### Week 3-4: Magenta RealTime
```
Day 1-3: Architecture 분해
Day 4-5: Real-time generation 기술
Day 6-7: Audio injection 메커니즘
Weekend: Colab 데모 완전 분석
```

### Week 5-6: Fine-tuning
```
Day 1-3: Transfer learning 기초
Day 4-6: LoRA & QLoRA 마스터
Day 7: 작은 실험 (10개 MIDI)
Weekend: 데이터 수집 계획 수립
```

### Week 7-8: 실전 구현
```
Day 1-3: MIDI tokenizer 구현
Day 4-7: Real-time system 구현
Weekend: 통합 테스트
```

### Week 9-10: 데이터 & Fine-tuning
```
Week 9: 내 연주 20시간 녹음
Week 10: Fine-tuning 실행
Weekend: 품질 평가 & 개선
```

### Week 11-12: 완성 & 테스트
```
Week 11: 실시간 듀엣 시스템 완성
Week 12: 테스트, 디버깅, 최적화
Weekend: 🎉 First JAM with AI me!
```

---

## 📚 학습 리소스 총정리

### 논문
- [ ] Attention Is All You Need (Transformer)
- [ ] Music Transformer (Google Magenta)
- [ ] LoRA (Hu et al.)
- [ ] QLoRA (Dettmers et al.)
- [ ] Live Music Models (arxiv 2508.04651)
- [ ] SoundStream / EnCodec (Audio codecs)

### 코드 & 라이브러리
- [ ] github.com/magenta/magenta-realtime
- [ ] HuggingFace Transformers
- [ ] HuggingFace PEFT
- [ ] Miditok
- [ ] Mido
- [ ] PyTorch

### 튜토리얼 & 강의
- [ ] The Illustrated Transformer (Jay Alammar)
- [ ] HuggingFace NLP Course
- [ ] Fast.ai Practical Deep Learning
- [ ] Stanford CS224N (NLP)
- [ ] MIT 6.S191 (Deep Learning)

### 도구
- [ ] Colab (무료 TPU)
- [ ] Weights & Biases (실험 tracking)
- [ ] TensorBoard (시각화)
- [ ] MIDI Monitor (디버깅)
- [ ] DAW (Ableton / FL Studio)

---

## 🎯 성공 기준

### Phase 1 완료 (2주)
- [ ] Transformer 수식 유도 가능
- [ ] MIDI tokenization 코드 작성
- [ ] Music LM 개념 설명 가능

### Phase 2 완료 (2주)
- [ ] Magenta RT architecture 완전 이해
- [ ] Chunk-based generation 구현 가능
- [ ] Colab 데모 수정 가능

### Phase 3 완료 (1.5주)
- [ ] LoRA 적용하여 fine-tuning 실행
- [ ] 내 10시간 데이터로 실험 성공
- [ ] "ohhalim style" 학습 확인

### Phase 4 완료 (2주)
- [ ] MIDI tokenizer 완성
- [ ] Real-time duet system 작동
- [ ] Latency <50ms 달성

### 최종 목표 달성 (3개월)
- [ ] 나와 가상의 내가 실시간 JAM! 🎹
- [ ] 10분 이상 안정적 듀엣 연주
- [ ] AI가 내 스타일로 반응
- [ ] 음악적으로 만족스러움

---

## 💪 실행 팁

### 매일 학습
- **시간**: 2-3시간 (집중)
- **방식**: 이론 1시간 + 실습 2시간
- **기록**: 매일 배운 것 정리 (노트)

### 주말 프로젝트
- **시간**: 4-6시간
- **방식**: 한 주 배운 것 통합
- **목표**: 작동하는 코드 완성

### 막힐 때
1. 공식 문서 다시 읽기
2. 코드 디버깅 (print 찍기)
3. 간단한 버전부터 (MVP)
4. 커뮤니티 질문 (Stack Overflow, Reddit)

### 동기부여
- **비전 상기**: "나와 가상의 내가 JAM!"
- **작은 성공**: 매일 작은 진전 축하
- **음악 듣기**: Brad Mehldau 등 영감
- **휴식**: 번아웃 방지!

---

## 🎉 마지막 메시지

**3개월 후 당신의 모습:**

```
나: [MIDI 키보드로 연주 시작]
AI: [내 스타일로 자연스럽게 반응]
나: "오, 이건 내가 3년 전에 쳤던 프레이즈네!"
AI: [계속 대화하듯 연주]
나: "진짜 나랑 듀엣하는 것 같아... 신기하다"

→ 목표 달성! 🎹✨
```

**Let's make it happen!** 💪

**지금 시작하세요:**
1. 이 문서를 프린트하거나 북마크
2. Week 1 Day 1 시작: "Attention Is All You Need" 논문 읽기
3. 매일 조금씩, 꾸준히!

**You got this!** 🚀
