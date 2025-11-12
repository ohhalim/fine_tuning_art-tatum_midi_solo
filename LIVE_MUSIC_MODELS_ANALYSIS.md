# Live Music Models - 논문 분석

**Paper**: Live Music Models (NeurIPS 2025 Creative AI Track)
**Authors**: Lyria Team, Google DeepMind
**ArXiv**: 2508.04651

---

## 🎯 핵심 요약 (3줄)

1. **실시간 음악 생성**: 사용자 입력에 즉각 반응하는 연속적 음악 스트림
2. **Magenta RealTime**: 오픈소스, 760M 파라미터, RTF 1.8x (실시간보다 1.8배 빠름)
3. **Chunk-based generation**: 2초 청크 단위로 무한히 생성, 10초 컨텍스트 유지

---

## 📖 1. 논문의 핵심 개념

### 1.1 Live Music Model이란?

**기존 AI 음악 생성 (Offline)**:
```
사용자: "재즈 피아노 음악 만들어줘"
    ↓ 기다림 (10-30초)
AI: [완성된 30초 음악 파일]
```

**Live Music Model (Real-time)**:
```
사용자: [실시간으로 컨트롤 입력]
    ↓ 즉각 반응 (<1초 지연)
AI: [끊임없이 흐르는 음악 스트림]
    ↓ 사용자가 컨트롤 변경
AI: [음악이 자연스럽게 변화]
```

### 1.2 Live Music Model의 3가지 필수 조건

1. **Real-time generation**: RTF ≥ 1x (실시간보다 빠르게 생성)
2. **Causal streaming**: 연속적으로 생성, 과거 출력을 기반으로 다음 생성
3. **Responsive controls**: 낮은 지연시간 (사용자 입력에 즉각 반응)

---

## 🏗️ 2. Magenta RealTime 아키텍처

### 전체 파이프라인

```
User Input (Text/Audio Prompt)
    ↓
┌─────────────────────────────────────────┐
│  1. MusicCoCa (Style Embedding)         │
│  - Text → 768D vector                   │
│  - Audio → 768D vector                  │
│  - Quantized to 12 tokens              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. Encoder-Decoder Transformer         │
│                                          │
│  Encoder (Bidirectional):               │
│  - 10s audio context (4 RVQ depth)      │
│  - 12 style tokens                      │
│  - Total: 1012 tokens                   │
│                                          │
│  Decoder (Causal):                      │
│  - "Temporal" module: frame-level       │
│  - "Depth" module: RVQ prediction       │
│  - Generates 2s chunk (16 RVQ depth)    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. SpectroStream (Audio Codec)         │
│  - Discrete tokens → Audio              │
│  - 48kHz stereo, 16kbps                 │
│  - RVQ: 25Hz frame rate, 64 depth       │
└─────────────────────────────────────────┘
    ↓
Output: 2 seconds of audio
    ↓
Append to context → Generate next chunk...
```

---

## 🔬 3. 핵심 기술 상세

### 3.1 SpectroStream Audio Codec

**목적**: 오디오를 discrete tokens으로 변환 (LLM처럼 처리하기 위해)

```python
# Audio → Tokens
audio = [48000 samples/sec × 2 channels]  # 48kHz stereo
tokens = SpectroStream.encode(audio)

# Token 구조:
# - Frame rate: 25Hz (초당 25 프레임)
# - RVQ depth: 64 levels (hierarchical quantization)
# - Vocabulary: 1024 tokens per level
# - Bandwidth: 16kbps

# 2초 오디오 = 50 frames × 64 RVQ levels = 3200 tokens
```

**RVQ (Residual Vector Quantization) 계층 구조**:
```
Level 1-4:   Coarse (가장 중요한 정보 - 피치, 리듬)
Level 5-16:  Medium (음색, 하모닉스)
Level 17-64: Fine (세밀한 디테일, 노이즈)
```

**실시간 최적화**:
- Training: 6 RVQ levels만 사용
- Context: 4 RVQ levels (coarse only)
- Generation: 16 RVQ levels (high fidelity)
- → 메모리와 속도 최적화!

### 3.2 MusicCoCa (Style Embedding Model)

**목적**: Text와 Audio를 같은 공간에 임베딩 → 스타일 컨트롤

```python
# Architecture:
MusicCoCa = {
    'audio_tower': ViT-12layers,      # Vision Transformer for mel-spectrogram
    'text_tower': Transformer-12layers,
    'text_decoder': Transformer-3layers,  # Regularization용
    'embedding_dim': 768,
    'quantized_tokens': 12,
    'codebook_size': 1024
}

# Usage:
text_embedding = MusicCoCa.text("jazz piano, upbeat")
audio_embedding = MusicCoCa.audio("reference_track.mp3")

# Weighted mixing:
style = 0.7 * text_embedding + 0.3 * audio_embedding

# → 12 discrete tokens로 quantization
style_tokens = quantize(style)  # [12 tokens]
```

**입력 사양**:
- Audio: 10초, 16kHz, log-mel spectrogram (128 channels)
- Text: 최대 128 tokens
- Output: 768D vector → 12 discrete tokens

**장점**:
```python
# Embedding arithmetic 가능!
techno = embed("techno")
flute = embed("flute")
techno_flute = 0.5 * techno + 0.5 * flute
# → "techno with flute" 스타일!

# Multiple prompts blending:
style = (
    2.0 * embed("brad mehldau piano") +
    1.0 * embed("bebop jazz") +
    0.5 * embed(my_audio_sample)
) / 3.5
```

### 3.3 Chunk-based Autoregression

**문제**: 무한히 긴 음악을 어떻게 생성?

**기존 방식 (Sliding Window)**:
```
[토큰1, 토큰2, 토큰3, ..., 토큰10000] → 메모리 폭발!
```

**Magenta RT 방식 (Chunk-based)**:
```python
# Chunk = 2초 단위
# Context = 최근 5 chunks (10초)

state = None
while True:  # 무한 생성!
    # 1. Encoder: 10초 컨텍스트 + 스타일 처리
    encoder_input = [
        Coarse(chunk_i-5),  # 10초 전
        Coarse(chunk_i-4),  # 8초 전
        Coarse(chunk_i-3),  # 6초 전
        Coarse(chunk_i-2),  # 4초 전
        Coarse(chunk_i-1),  # 2초 전 (바로 직전)
        style_tokens        # 12 tokens
    ]  # Total: 1012 tokens

    # 2. Decoder: 다음 2초 생성
    chunk_i = decoder.generate(encoder_input)
    # → 50 frames × 16 RVQ = 800 tokens

    # 3. Audio 변환 & 재생
    audio_2s = SpectroStream.decode(chunk_i)
    play(audio_2s)

    # 4. Context 업데이트 (sliding)
    # 가장 오래된 chunk 버림, 새 chunk 추가
```

**장점**:
1. **무한 생성**: 컨텍스트가 고정 길이 (1012 tokens)
2. **Stateless**: 각 청크가 독립적 (error accumulation 감소)
3. **유연한 컨트롤**: 청크마다 스타일 변경 가능
4. **메모리 효율**: 10초 컨텍스트만 유지

### 3.4 Encoder-Decoder Transformer

**T5 아키텍처 기반**:

```
┌─────────────────────────────────────────┐
│  ENCODER (Bidirectional)                │
│                                          │
│  Input: 1012 tokens                     │
│  - Audio context: 1000 tokens           │
│    (5 chunks × 50 frames × 4 RVQ)       │
│  - Style: 12 tokens                     │
│                                          │
│  T5 Base: 220M params                   │
│  T5 Large: 770M params                  │
│                                          │
│  Output: Encoded representation         │
└─────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────┐
│  DECODER (Causal)                       │
│                                          │
│  Two-stage architecture:                │
│                                          │
│  1. Temporal Module:                    │
│     - Process frame by frame            │
│     - 50 frames (2초)                   │
│                                          │
│  2. Depth Module:                       │
│     - Predict 16 RVQ tokens per frame   │
│     - Autoregressive within frame       │
│                                          │
│  Output: 800 tokens (50×16)             │
└─────────────────────────────────────────┘
```

**실시간 성능**:
- **T5 Large (770M)**: RTF = 1.8x on H100 GPU
- 2초 음악을 1.1초에 생성 (1.8배 빠름)
- → 실시간 생성 가능!

---

## 📊 4. 성능 평가

### 4.1 기존 모델들과 비교

| Model | Live? | Sample Rate | Params | FD↓ | KL↓ | CLAP↑ |
|-------|-------|-------------|--------|-----|-----|-------|
| **Magenta RT** | ✅ | 48kHz | 760M | **72.14** | **0.47** | 0.35 |
| Stable Audio | ❌ | 44.1kHz | 1.1B | 96.51 | 0.55 | **0.41** |
| MusicGen Large | ❌ | 32kHz | 3.3B | 190.47 | 0.52 | 0.31 |

**결과**:
- **FD (Fréchet Distance)**: 낮을수록 좋음 → Magenta RT 압도적 1위
- **KL (Kullback-Leibler)**: 낮을수록 좋음 → Magenta RT 1위
- **CLAP Score**: 높을수록 좋음 → Stable Audio가 약간 우세
- **파라미터 수**: Magenta RT가 가장 작음 (효율적!)

### 4.2 Prompt Transition (실시간 컨트롤 변화)

**실험**: Prompt A → Prompt B로 60초 동안 점진적 전환

```python
# Example:
prompt_A = "calm piano ballad"
prompt_B = "energetic jazz piano"

# 10초마다 interpolation
for t in [0, 10, 20, 30, 40, 50, 60]:
    alpha = t / 60
    style = (1 - alpha) * embed(A) + alpha * embed(B)
    generate_chunk(style)
```

**결과**:
- 매끄러운 스타일 전환 (smooth transition)
- 이전 컨텍스트의 영향으로 자연스러운 변화
- Cosine similarity가 선형적으로 변화

**의미**: 실시간 연주 중 스타일 변경 가능!

---

## 🎮 5. 컨트롤 방식

### 5.1 Text Prompt

```python
# Simple text
style = embed("jazz piano, upbeat, bebop style")

# Multiple weighted prompts
style = weighted_avg([
    (2.0, "brad mehldau style"),
    (1.0, "modal jazz"),
    (0.5, "ambient")
])
```

**특징**:
- 장르, 악기, 무드, 템포 등 high-level 컨트롤
- 직관적이지만 세밀한 컨트롤 어려움

### 5.2 Audio Prompt

```python
# Reference audio로 스타일 지정
reference = "my_favorite_track.mp3"
style = embed(reference)

# Text + Audio blending
style = weighted_avg([
    (1.0, "jazz piano"),
    (2.0, reference_audio)  # Audio가 더 강한 영향
])
```

**특징**:
- 말로 표현하기 어려운 스타일도 가능
- Training 조건과 유사 → 더 효과적
- **내 연주를 reference로 사용 가능!** ← 중요!

### 5.3 Audio Injection (혁신!)

**개념**: 실시간으로 오디오 입력을 모델에 주입

```python
while generating:
    # 1. 사용자 입력 캡처 (마이크/MIDI)
    user_audio = capture_input()

    # 2. 모델 출력과 믹싱
    mixed = mix(user_audio, model_output, ratio=0.3)

    # 3. 믹싱된 오디오를 tokenize
    mixed_tokens = SpectroStream.encode(mixed)

    # 4. 다음 청크 생성 시 컨텍스트로 사용
    next_chunk = generate(context=mixed_tokens, style=style)
```

**작동 방식**:
```
User plays: [C E G]
    ↓ (mix with model output)
Model sees: [Previous output + User's C E G]
    ↓ (generate continuation)
Model output: [Responds to user's phrase...]
```

**효과**:
- 모델이 사용자 입력을 "듣고" 반응
- Call-response improvisation 가능!
- 사용자 오디오는 직접 재생 안 됨 (모델이 해석해서 반영)

---

## 🎹 6. 당신의 프로젝트에 적용하기

### 6.1 현재 Magenta RT의 한계

1. **Audio 기반**: MIDI 아님 (오디오로 생성)
2. **RTF 1.8x**: 빠르지만 MIDI가 더 빠를 수 있음
3. **48kHz stereo**: 고품질이지만 무거움

### 6.2 MIDI 버전으로 개조하기

**아이디어**: SpectroStream 대신 MIDI tokenizer 사용

```python
# Original (Audio):
SpectroStream: Audio → RVQ tokens (3200 tokens/2s)

# Your MIDI version:
MIDITokenizer: MIDI → Event tokens (~100 tokens/2s)
# → 30배 가벼움!
# → RTF 50x 이상 예상 (초고속!)
```

**아키텍처 수정**:
```
┌─────────────────────────────────────────┐
│  1. MusicCoCa                           │
│  - Audio prompt: 내 연주 녹음           │
│  - Text prompt: "ohhalim jazz style"   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. Encoder-Decoder Transformer         │
│  - Context: 10초 MIDI events           │
│  - Generate: 2초 MIDI events           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. MIDI Output (NOT SpectroStream)     │
│  - Event tokens → MIDI messages         │
│  - Latency: ~20ms (초고속!)            │
└─────────────────────────────────────────┘
```

### 6.3 Fine-tuning 전략

**Step 1: 내 연주 녹음 → MIDI**
```python
# 100시간 녹음 (목표)
my_recordings = [
    "improvisation_01.mid",
    "improvisation_02.mid",
    ...
]

# Audio도 함께 저장 (MusicCoCa용)
my_audio = [
    "improvisation_01.wav",
    "improvisation_02.wav",
    ...
]
```

**Step 2: MusicCoCa Fine-tuning**
```python
# 내 스타일 학습
MusicCoCa_personal = finetune(
    MusicCoCa_pretrained,
    audio_samples=my_audio,
    text_labels=["ohhalim style", "my jazz piano", ...]
)

# 결과: embed("ohhalim style") → 나만의 벡터!
```

**Step 3: Transformer Fine-tuning (QLoRA)**
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

model = apply_qlora_to_model(
    magenta_rt_transformer,
    lora_config
)

trainer.train(
    model=model,
    train_data=my_midi_tokenized,
    style_embeddings=embed("ohhalim style"),
    epochs=50
)
```

### 6.4 실시간 듀엣 시스템

```python
from magenta_rt_midi import PersonalDuetSystem

duet = PersonalDuetSystem(
    my_model="ohhalim_finetuned.ckpt",
    input_device="MIDI Keyboard",
    output_device="DAW MIDI In"
)

duet.start_session(
    mode="call_response",  # 또는 "simultaneous"
    latency_target=50,     # ms
    context_window=10      # seconds
)

# 실시간 연주:
while True:
    # 1. 내 연주 캡처
    my_phrase = duet.capture_input()

    # 2. AI 응답 생성 (내 스타일로!)
    ai_phrase = duet.generate_response(
        context=my_phrase,
        style=embed("ohhalim style")
    )

    # 3. AI 연주 출력
    duet.play_output(ai_phrase)
```

---

## 💡 7. 핵심 인사이트

### 7.1 논문에서 배울 점

1. **Chunk-based generation**: 무한 스트림 생성 핵심!
   - 2초 청크
   - 10초 컨텍스트
   - Stateless (간단!)

2. **Coarse context**: 메모리/속도 최적화
   - Context: 4 RVQ levels (coarse)
   - Generation: 16 RVQ levels (high-fi)
   - → 4배 효율 향상!

3. **MusicCoCa embedding**: 유연한 컨트롤
   - Text + Audio blending
   - Weighted averaging
   - **내 연주를 prompt로!**

4. **Audio injection**: 실시간 상호작용
   - 사용자 입력을 context에 믹싱
   - 모델이 "듣고" 반응
   - → Call-response 가능!

### 7.2 당신의 프로젝트에 적용

**목표**: "나 + AI(나) = JAM!"

```python
# Phase 1: 나를 녹음
my_data = record_myself(100_hours)

# Phase 2: 나를 학습
my_ai = finetune(
    magenta_rt,
    my_data,
    style_name="ohhalim"
)

# Phase 3: 나와 듀엣
while jamming:
    i_play()
    my_ai_responds()  # 내 스타일로!
    i_respond_back()
    my_ai_continues()

    # → Musical dialogue!
```

**핵심 차이점**:
- ❌ Brad Mehldau 모방 (유명인)
- ✅ 나 자신 학습 (personal)
- ❌ Offline 생성 (턴제)
- ✅ Real-time 듀엣 (라이브)
- ❌ Audio 생성 (무거움)
- ✅ MIDI 생성 (빠름)

---

## 🚀 8. 다음 단계

### 8.1 즉시 시도 (이번 주)

```bash
# 1. Magenta RT Colab 실행
open https://github.com/magenta/magenta-realtime

# 2. Audio prompt 테스트
# 내 연주 10분 녹음 → Audio prompt로 사용
# → AI가 내 스타일 흉내?

# 3. Audio injection 데모
# 실시간으로 마이크 입력하며 AI 반응 확인
```

### 8.2 연구할 내용 (이번 달)

1. **MIDI tokenizer 개발**
   - SpectroStream 대신
   - Event-based MIDI representation
   - Target: ~100 tokens/2s

2. **MusicCoCa fine-tuning**
   - 내 연주 10-20개로 실험
   - "ohhalim style" 임베딩 학습
   - Effectiveness 측정

3. **Latency 최적화**
   - Target: <50ms
   - MIDI가 audio보다 빠름
   - Quantization (INT8, FP16)

### 8.3 구현 로드맵 (3개월)

**Month 1: 기초**
- Magenta RT 코드 분석
- MIDI tokenizer 개발
- Chunk-based generation 구현

**Month 2: Fine-tuning**
- 내 연주 50시간 녹음
- MusicCoCa + Transformer fine-tuning
- 품질 평가

**Month 3: Real-time System**
- Audio injection → MIDI injection
- Latency <50ms 달성
- Live duet demo!

---

## 📚 9. 논문의 철학적 메시지

### "Music as a verb" (음악은 동사다)

**기존 AI 음악**:
- Music as a noun (명사)
- 완성된 작품 생성
- Static, 고정됨

**Live Music Models**:
- Music as a verb (동사)
- 진행 중인 행위
- Dynamic, 살아있음
- **Process > Product**

**당신의 프로젝트와 일치**:
```
"난 내가 만든 인공지능과 즉흥연주하고 싶어"
                      ^^^^
                    (동사!)

→ 완성된 곡이 아닌
→ 함께 연주하는 과정!
```

### Human-in-the-loop

**기존**: Human → AI → Output (일방향)

**Live**: Human ⇄ AI ⇄ Output (양방향)
```
나: [프레이즈]
AI: [응답] ← 나를 학습한 스타일로!
나: [반응]
AI: [계속...]

→ 진짜 대화!
```

---

## 🎯 10. 최종 요약

### 논문의 핵심 기여

1. **Live music model 정의**: 실시간, 연속, 반응형
2. **Magenta RealTime**: 오픈소스, 760M, RTF 1.8x
3. **Chunk-based autoregression**: 무한 스트림 생성
4. **Audio injection**: 실시간 상호작용 메커니즘
5. **SOTA 성능**: 적은 파라미터로 높은 품질

### 당신의 프로젝트에 주는 의미

```
Magenta RT = 완벽한 출발점!

1. Architecture: ✅ Chunk-based (검증됨)
2. Open-source: ✅ 코드 & weights 공개
3. Fine-tunable: ✅ 내 스타일 학습 가능
4. Real-time: ✅ 라이브 듀엣 가능

수정할 부분:
- SpectroStream → MIDI tokenizer
- Audio injection → MIDI injection
- Style: Brad Mehldau → Ohhalim!

→ 완벽하게 실현 가능! 🚀
```

---

## 💪 실행 계획

### Today (1시간):
```bash
# Magenta RT GitHub 클론
git clone https://github.com/magenta/magenta-realtime.git

# Colab 데모 실행
# → 실시간 생성 체험

# 논문 다시 읽기
# → 아키텍처 완전 이해
```

### This Week:
```python
# 1. 내 연주 10분 녹음
record_my_playing("improvisation_test.wav")

# 2. Audio prompt로 테스트
style = embed(my_audio="improvisation_test.wav")
generate(style=style)
# → AI가 내 스타일 흉내내나?

# 3. MIDI tokenizer 설계 시작
design_midi_tokenizer()
```

### This Month:
```
Week 1: Architecture understanding
Week 2: MIDI tokenizer implementation
Week 3: Fine-tuning experiments (10 files)
Week 4: Real-time MIDI generation prototype
```

---

**"Live Music Models" 논문 = 당신의 비전을 실현할 완벽한 blueprint! 🎹✨**

**다음 단계: 코드 분석 & MIDI 버전 개발 시작!**
