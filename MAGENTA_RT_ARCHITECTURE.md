# Magenta RealTime 아키텍처 분석

## 🏗️ 시스템 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Magenta RealTime System                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐│
│  │   Input      │────▶│   Style      │────▶│   Output     ││
│  │              │     │   Control    │     │              ││
│  │ Text/Audio   │     │              │     │   Audio      ││
│  └──────────────┘     └──────────────┘     └──────────────┘│
│         │                    │                    ▲         │
│         │                    │                    │         │
│         ▼                    ▼                    │         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           MusicCoCa (Style Encoder)                  │  │
│  │  - Text → Embedding                                  │  │
│  │  - Audio → Embedding                                 │  │
│  │  - RVQ Tokenization (6 layers × 1024 codebook)      │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           │ Style Tokens (6,)               │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Depthformer LLM (T5X-based)                  │  │
│  │                                                       │  │
│  │  Encoder Input:                                      │  │
│  │  ┌────────────────────────────────────────┐         │  │
│  │  │ Context Tokens (250 frames × 4 RVQ)    │         │  │
│  │  │        + Style Tokens (6)               │         │  │
│  │  │  = 1006 tokens total                    │         │  │
│  │  └────────────────────────────────────────┘         │  │
│  │                                                       │  │
│  │  Decoder Output:                                     │  │
│  │  ┌────────────────────────────────────────┐         │  │
│  │  │ Generated Tokens (50 frames × 16 RVQ)  │         │  │
│  │  │  = 800 tokens per chunk                 │         │  │
│  │  └────────────────────────────────────────┘         │  │
│  │                                                       │  │
│  │  Sampling Parameters:                                │  │
│  │  - Temperature: 1.1                                  │  │
│  │  - Top-K: 40                                         │  │
│  │  - Classifier-Free Guidance: 5.0                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           │ RVQ Tokens (50, 16)             │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        SpectroStream Codec (Audio Decoder)           │  │
│  │  - RVQ → Spectrogram                                 │  │
│  │  - 16 RVQ layers × 1024 codebook                     │  │
│  │  - 25 fps frame rate                                 │  │
│  │  - 48kHz stereo output                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           │ Audio (96000 samples, 2 ch)     │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Crossfade + State Update                   │  │
│  │  - 40ms crossfade between chunks                     │  │
│  │  - Context sliding window (10 seconds)               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📊 데이터 플로우

### 1. Style Embedding (MusicCoCa)

```python
# Input
text_or_audio = "fast tempo jazz piano"
                 또는
                 Waveform(samples, sample_rate=48000)

# MusicCoCa 처리
style_embedding = magenta_rt.embed_style(text_or_audio)
# shape: (512,) - 512차원 벡터

# RVQ Tokenization
style_tokens = musiccoca.tokenize(style_embedding)
# shape: (6,) - 6개의 이산 토큰 (각 0-1023 범위)

# LLM 입력용 변환
style_tokens_lm = utils.rvq_to_llm(
    style_tokens,
    codebook_size=1024,
    offset=17140  # vocab_style_offset
)
# shape: (6,) - 범위 [17140, 23554)
```

### 2. Context 준비

```python
# 이전 생성 결과 (초기: 빈 상태)
context_tokens = state.context_tokens
# shape: (250, 16) - 10초 분량의 컨텍스트
#   250 frames = 10초 (at 25 fps)
#   16 = RVQ depth

# Encoder용으로 일부만 사용
context_for_encoder = context_tokens[:, :4]  # 처음 4 RVQ layers만
# shape: (250, 4) → flatten → (1000,)

# LLM 입력용 변환
codec_tokens_lm = utils.rvq_to_llm(
    context_for_encoder,
    codebook_size=1024,
    offset=2  # vocab_codec_offset
)
# shape: (1000,) - 범위 [2, 16386)
```

### 3. LLM 추론

```python
# Encoder 입력 구성
encoder_inputs_pos = np.concatenate([
    codec_tokens_lm,    # (1000,) - 컨텍스트
    style_tokens_lm     # (6,) - 스타일
])
# shape: (1006,)

# Classifier-Free Guidance용 negative prompt
encoder_inputs_neg = encoder_inputs_pos.copy()
encoder_inputs_neg[-6:] = MASK_TOKEN  # 스타일 마스킹

# Batch 구성
encoder_inputs = np.stack([
    encoder_inputs_pos,  # Conditioned
    encoder_inputs_neg   # Unconditioned
])
# shape: (2, 1006)

# LLM 생성
generated_tokens, _ = llm(
    encoder_input_tokens=encoder_inputs,  # (2, 1006)
    decoder_input_tokens=zeros(2, 800),   # 시작 토큰들
    max_decode_steps=800,  # 50 frames × 16 RVQ
    temperature=1.1,
    topk=40,
    guidance_weight=5.0
)
# shape: (2, 800)

# CFG 결합
# output = uncond + guidance_weight × (cond - uncond)
final_tokens = generated_tokens[1] + 5.0 * (
    generated_tokens[0] - generated_tokens[1]
)
# shape: (800,) → reshape → (50, 16)
```

### 4. Audio 디코딩

```python
# RVQ 토큰 → LLM 토큰 역변환
rvq_tokens = utils.llm_to_rvq(
    final_tokens,
    codebook_size=1024,
    offset=2
)
# shape: (50, 16) - 각 값 0-1023 범위

# Crossfade를 위한 이전 프레임 추가
xfade_tokens = np.concatenate([
    state.context_tokens[-1:],  # 마지막 1 frame (40ms)
    rvq_tokens                  # 새로 생성된 50 frames
])
# shape: (51, 16)

# SpectroStream 디코딩
audio_with_xfade = codec.decode(xfade_tokens)
# shape: (97920, 2) - 51 frames × 1920 samples/frame
#   = 2.04초 (2초 chunk + 40ms crossfade)

# Crossfade 적용
chunk = audio_with_xfade[:-1920]  # 2초 chunk
xfade_samples = audio_with_xfade[-1920:]  # 40ms overlap

# Equal-power crossfade
ramp = crossfade_ramp(1920, style='eqpower')
chunk[:1920] *= ramp
chunk[:1920] += state.crossfade_samples * (1 - ramp)

# Output
# shape: (96000, 2) - 정확히 2초 @ 48kHz stereo
```

### 5. State 업데이트

```python
# 컨텍스트 슬라이딩 윈도우 (FIFO)
state.context_tokens = np.concatenate([
    state.context_tokens[50:],  # 오래된 50 frames 제거
    rvq_tokens                  # 새로운 50 frames 추가
])
# 항상 (250, 16) 유지 = 10초 컨텍스트

# Crossfade 샘플 저장
state.crossfade_samples = xfade_samples

# Chunk 인덱스 증가
state.chunk_index += 1
```

## 🔧 핵심 컴포넌트

### MagentaRTConfiguration

```python
@dataclass
class MagentaRTConfiguration:
    chunk_length: float = 2.0              # 한 번에 생성할 길이 (초)
    context_length: float = 10.0           # LLM이 참조할 이전 오디오 (초)
    crossfade_length: float = 0.04         # Chunk 간 크로스페이드 (초)

    codec_sample_rate: int = 48000         # 오디오 샘플레이트
    codec_frame_rate: float = 25.0         # Codec 프레임 레이트
    codec_num_channels: int = 2            # 스테레오
    codec_rvq_codebook_size: int = 1024    # RVQ 코드북 크기

    encoder_codec_rvq_depth: int = 4       # Encoder에 사용할 RVQ layers
    decoder_codec_rvq_depth: int = 16      # Decoder에서 생성할 RVQ layers

    encoder_style_rvq_depth: int = 6       # Style 토큰 개수
    style_rvq_codebook_size: int = 1024    # Style RVQ 코드북 크기
```

**계산 예시**:
```python
chunk_length_frames = 2.0초 × 25 fps = 50 frames
chunk_length_samples = 2.0초 × 48000 Hz = 96000 samples

context_length_frames = 10.0초 × 25 fps = 250 frames
context_length_samples = 10.0초 × 48000 Hz = 480000 samples

crossfade_length_frames = 0.04초 × 25 fps = 1 frame
crossfade_length_samples = 0.04초 × 48000 Hz = 1920 samples

encoder_input_size = (250 frames × 4 RVQ) + 6 style = 1006 tokens
decoder_output_size = 50 frames × 16 RVQ = 800 tokens

vocab_size = 2 (PAD+MASK) + 16384 (codec) + 1024 (unused) + 6144 (style) = 23554
```

### MagentaRTState

```python
class MagentaRTState:
    context_tokens: np.ndarray          # (250, 16) 이전 오디오 토큰
    crossfade_samples: Waveform         # (1920, 2) 마지막 40ms 오디오
    chunk_index: int                    # 현재 생성 중인 chunk 번호

    def update(self, chunk_tokens, crossfade_samples):
        # FIFO: 오래된 데이터 제거, 새 데이터 추가
        self.context_tokens = np.concatenate([
            self.context_tokens[chunk_tokens.shape[0]:],
            chunk_tokens
        ])
        self.crossfade_samples = crossfade_samples
        self.chunk_index += 1
```

## 🎯 핵심 알고리즘

### Classifier-Free Guidance (CFG)

```python
# 두 가지 조건으로 생성
cond_output = model(context, style=style_tokens)        # 스타일 조건부
uncond_output = model(context, style=MASK)              # 스타일 무조건부

# 스타일 신호 증폭
final = uncond + guidance_weight × (cond - uncond)

# guidance_weight = 5.0
# → 스타일 특성이 5배 강하게 반영됨
```

**효과**: Text/Audio prompt의 영향력을 조절하여 더 명확한 스타일 제어

### Equal-Power Crossfade

```python
def crossfade_ramp(n_samples, style='eqpower'):
    t = np.linspace(0, 1, n_samples)
    if style == 'eqpower':
        # Equal power law: 에너지 보존
        return np.sqrt(t)
    elif style == 'linear':
        return t

# 사용 예시
fade_in = ramp              # 0 → 1
fade_out = 1 - ramp         # 1 → 0

# 두 chunk 결합
output = chunk_A * fade_out + chunk_B * fade_in
```

**효과**: Chunk 경계에서 클릭 노이즈 없이 부드러운 전환

### RVQ Token 변환

```python
def rvq_to_llm(rvq_tokens, codebook_size, offset):
    """RVQ 다층 토큰을 LLM vocab으로 변환

    Input: (frames, depth)
    각 값: 0 ~ codebook_size-1

    Output: (frames, depth)
    각 값: offset + (layer_idx * codebook_size) + token_value
    """
    depth = rvq_tokens.shape[-1]
    layer_offsets = np.arange(depth) * codebook_size
    return offset + layer_offsets + rvq_tokens

# 예시:
# rvq = [[5, 10, 15, 20]]  # 1 frame, 4 layers
# codebook_size = 1024
# offset = 2
#
# layer 0: 2 + 0×1024 + 5 = 7
# layer 1: 2 + 1×1024 + 10 = 1036
# layer 2: 2 + 2×1024 + 15 = 2063
# layer 3: 2 + 3×1024 + 20 = 3094
#
# llm_tokens = [[7, 1036, 2063, 3094]]
```

**이유**: 각 RVQ layer는 독립적인 정보를 인코딩하므로, vocab 공간에서 분리 필요

## 💡 설계 철학

### 1. **실시간 스트리밍 최적화**
- Chunk 단위 생성 (2초)
- 컨텍스트 윈도우 제한 (10초)
- Stateful 설계 (이전 상태 재사용)

### 2. **고품질 오디오**
- 16-layer RVQ → 높은 fidelity
- 48kHz 스테레오
- Equal-power crossfade → 무손실 결합

### 3. **유연한 스타일 제어**
- Text + Audio 프롬프트 지원
- 다중 프롬프트 블렌딩
- CFG로 제어 강도 조절

### 4. **효율적인 토큰 사용**
- Encoder: 4 RVQ layers (압축)
- Decoder: 16 RVQ layers (고품질)
- Style: 6 토큰만으로 스타일 표현

## 🔬 성능 특성

### 지연시간 (Latency)
```
Chunk 생성: ~1-2초 (GPU/TPU)
└─ Style embedding: ~50ms
└─ LLM inference: ~800ms
└─ Audio decoding: ~100ms
└─ Crossfade: ~10ms

실시간 Factor: 1-2× (2초 생성에 1-2초 소요)
```

### 메모리 사용량
```
Model weights: ~1.5GB (large), ~500MB (base)
State: ~400KB
  ├─ context_tokens: (250, 16, 4 bytes) = 16KB
  └─ crossfade_samples: (1920, 2, 4 bytes) = 15KB
JAX 컴파일 캐시: ~2GB
```

### 품질 지표
```
Sample rate: 48kHz (CD 품질 초과)
Bit depth: 32-bit float
Channels: 2 (stereo)
RVQ layers: 16 (매우 높은 압축률 대비 품질)
Frequency response: 20Hz - 24kHz (Nyquist)
```

## 🎼 사용 예시

### 기본 생성
```python
from magenta_rt import system

# 시스템 초기화
magenta_rt = system.MagentaRT(tag='large', device='gpu')

# 스타일 임베딩
style = magenta_rt.embed_style("upbeat jazz piano solo")

# 30초 생성
chunks = []
state = None
for i in range(15):  # 15 chunks × 2초 = 30초
    chunk, state = magenta_rt.generate_chunk(
        state=state,
        style=style,
        temperature=1.1,  # 다양성 조절
        topk=40,          # 샘플링 범위
        guidance_weight=5.0  # 스타일 강도
    )
    chunks.append(chunk)

# 결합 및 저장
output = audio.concatenate(chunks)
output.write("output.wav")
```

### 다중 프롬프트 블렌딩
```python
# 여러 스타일 결합
styles = [
    magenta_rt.embed_style("classical piano"),
    magenta_rt.embed_style("jazz improvisation"),
    magenta_rt.embed_style("ambient pads")
]

# 가중 평균
weights = np.array([0.5, 0.3, 0.2])
blended_style = np.average(styles, axis=0, weights=weights)

# 생성
chunk, state = magenta_rt.generate_chunk(style=blended_style)
```

### 오디오 프롬프트
```python
# 기존 오디오를 스타일 참조로 사용
reference = audio.Waveform.from_file("reference.wav")
style = magenta_rt.embed_style(reference)

# 해당 스타일로 continuation 생성
chunk, state = magenta_rt.generate_chunk(style=style)
```

## 🚀 최적화 팁

### 1. Warm-up 필요
```python
# 첫 생성은 느림 (모델 로딩 + JIT 컴파일)
magenta_rt.warm_start()  # ~30초 소요

# 이후 생성은 빠름
```

### 2. Batch 처리 불가
```python
# 현재 구현은 batch_size=2 고정 (CFG용)
# 여러 스타일을 병렬 생성하려면 multiple instances 필요
```

### 3. TPU 권장
```python
# TPU에서 GPU 대비 2-3배 빠름
magenta_rt = system.MagentaRT(device='tpu')
```

### 4. 메모리 관리
```python
# Lazy loading으로 메모리 절약
magenta_rt = system.MagentaRT(lazy=True)

# 명시적 warm-up
magenta_rt.warm_start()  # 필요할 때만 로딩
```

## 📚 관련 컴포넌트

### SpectroStream (Codec)
- Audio ↔ RVQ 토큰 변환
- 16-layer RVQ
- 25 fps latency
- 파일: `magenta_rt/spectrostream.py`

### MusicCoCa (Style Encoder)
- Text/Audio → 임베딩
- Contrastive learning
- 512차원 임베딩 공간
- 파일: `magenta_rt/musiccoca.py`

### Depthformer (LLM)
- T5X 기반 Transformer
- Encoder-Decoder 구조
- 2가지 크기: base (500M), large (1.5B)
- 파일: `magenta_rt/depthformer/model.py`

## 🔗 참고 자료

- [Magenta RealTime Paper](https://arxiv.org/abs/2501.xxxxx)
- [GitHub Repository](https://github.com/magenta/magenta-realtime)
- [Colab Demo](https://colab.research.google.com/github/magenta/magenta-realtime/blob/main/notebooks/Magenta_RT_Demo.ipynb)
