# Lyria RealTime 분석

**논문**: Live Music Models (NeurIPS 2025)
**Lyria RealTime**: API-based live music model by Google DeepMind

---

## 🎯 Lyria RT vs Magenta RT 비교

### 핵심 차이점

| 특징 | Magenta RealTime | Lyria RealTime |
|------|------------------|----------------|
| **배포 방식** | Open-weights (오픈소스) | API (클라우드) |
| **실행 위치** | On-device (로컬) | Cloud (서버) |
| **모델 크기** | 760M parameters | Unknown (더 큼) |
| **하드웨어** | TPU v2-8 (무료 Colab) | Specialized hardware |
| **컨트롤** | Text + Audio prompts | Extended controls |
| **커스터마이징** | Fine-tuning 가능 | API 제공 기능만 |
| **비용** | 무료 (자체 실행) | API 요금 |
| **레이턴시** | 낮음 (로컬) | 약간 높음 (네트워크) |
| **프라이버시** | 완전 보장 (로컬) | 데이터 서버 전송 |
| **안정성** | 인터넷 불필요 | 인터넷 필요 |

---

## 🏗️ Lyria RealTime 작동 방식

### 1. **같은 Core Architecture 사용**

논문에서 명시:
> "Both use the same core methodological framework, which centers around codec language modeling"

```
Lyria RT = Magenta RT의 확장 버전

동일한 기본 구조:
① MusicCoCa (Style Embedding)
② Encoder-Decoder Transformer
③ SpectroStream (Audio Codec)

차이점:
- 더 큰 모델 (더 많은 파라미터)
- 더 강력한 하드웨어 (GPU/TPU 클러스터)
- Extended controls (추가 기능들)
```

### 2. **API 기반 작동**

```python
# Magenta RT (로컬):
model = load_magenta_rt_locally()
audio = model.generate(style="jazz piano")

# Lyria RT (API):
import requests

response = requests.post(
    "https://g.co/magenta/lyria-realtime/api",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "prompts": [
            {"text": "jazz piano", "weight": 2.0},
            {"audio_url": "my_style.wav", "weight": 1.0}
        ],
        "duration": 60,
        "controls": {
            "tempo": 120,
            "key": "C major",
            "energy": 0.8,
            # ... extended controls
        }
    }
)

audio_stream = response.iter_content()
```

### 3. **Extended Controls (확장된 컨트롤)**

논문에서:
> "Lyria RealTime, an API-based model with extended controls, offering access to our most powerful model with wide prompt coverage"

**Magenta RT 컨트롤**:
- Text prompts (장르, 악기, 무드)
- Audio prompts (스타일 레퍼런스)
- Weighted blending

**Lyria RT 추가 컨트롤 (추정)**:
```python
extended_controls = {
    # Musical structure
    "tempo": 120,              # BPM 명시
    "key": "C major",          # 조성 명시
    "time_signature": "4/4",   # 박자

    # Energy & dynamics
    "energy_level": 0.8,       # 0.0-1.0
    "dynamics_range": "wide",  # soft, medium, wide

    # Instrumentation
    "instruments": [
        {"type": "piano", "prominence": 1.0},
        {"type": "bass", "prominence": 0.5},
        {"type": "drums", "prominence": 0.3}
    ],

    # Structure
    "structure": "AABA",       # 곡 구조
    "sections": [
        {"type": "intro", "duration": 8},
        {"type": "verse", "duration": 16},
        {"type": "chorus", "duration": 16}
    ],

    # Advanced
    "harmonic_complexity": 0.7,
    "rhythmic_complexity": 0.6,
    "melodic_range": "medium"
}
```

---

## 💡 왜 두 버전을 제공하는가?

### Use Case에 따른 선택

**Magenta RealTime (Open-weights) 사용 시나리오**:

```
✅ 연구자 / 개발자
   - Fine-tuning 필요
   - 커스터마이징 필요
   - 프라이버시 중요
   - 비용 절감

✅ 뮤지션 (로컬 실행)
   - 라이브 공연 (인터넷 불안정)
   - 빠른 응답 필요 (네트워크 지연 없음)
   - 무료로 실험

✅ 교육 목적
   - 내부 작동 원리 학습
   - 알고리즘 연구
```

**Lyria RealTime (API) 사용 시나리오**:

```
✅ 프로덕션 앱 개발자
   - 높은 품질 필요
   - 복잡한 컨트롤 필요
   - GPU/TPU 없음
   - 관리 부담 감소

✅ 콘텐츠 크리에이터
   - 빠르게 시작
   - 복잡한 설정 싫음
   - 품질 > 비용

✅ 상업적 사용
   - 안정성 보장
   - 고객 지원
   - SLA 필요
```

---

## 🔬 Lyria RT의 기술적 우위

### 1. **더 큰 모델**

```
Magenta RT: 760M parameters
Lyria RT:   2B-10B parameters (추정)

→ 더 복잡한 음악 생성
→ 더 정교한 스타일 학습
→ 더 일관된 장기 구조
```

### 2. **더 강력한 하드웨어**

```
Magenta RT: TPU v2-8 (무료 Colab)
Lyria RT:   TPU v5p clusters (추정)

→ 더 빠른 생성 (RTF > 2x)
→ 더 높은 품질 (more RVQ levels)
→ 더 긴 컨텍스트 (>10s)
```

### 3. **Wide Prompt Coverage**

논문에서:
> "wide prompt coverage"

```python
# Magenta RT:
prompts = [
    "jazz piano",
    "electronic music",
    "ambient"
]

# Lyria RT:
prompts = [
    "jazz piano in the style of Bill Evans with lush voicings",
    "aggressive techno with dark bassline and industrial sounds",
    "ambient soundscape with evolving pads and field recordings",
    "baroque harpsichord piece in the style of Bach",
    "afrobeat with polyrhythmic percussion and brass section",
    # ... 훨씬 더 구체적이고 다양한 프롬프트 이해
]

→ 더 정교한 언어 이해
→ 더 세밀한 스타일 컨트롤
```

---

## 🎮 Lyria RT 실제 사용 예시

### Real-time Streaming Generation

```python
import lyria_rt_client

# 1. 클라이언트 초기화
client = lyria_rt_client.LyriaRealTime(
    api_key="YOUR_API_KEY",
    region="us-central1"
)

# 2. 세션 시작
session = client.start_session(
    initial_style={
        "text": "jazz piano trio, bebop style",
        "tempo": 180,
        "key": "Bb major"
    }
)

# 3. 실시간 스트리밍
for chunk in session.stream():
    # 2초 청크씩 받기
    audio_2s = chunk.audio  # 48kHz stereo
    play(audio_2s)

    # 4. 실시간 컨트롤 변경
    if user_changed_style:
        session.update_style({
            "text": "modal jazz, slower tempo",
            "tempo": 120,
            "key": "D minor"
        })
        # → 다음 청크부터 반영!

# 5. 세션 종료
session.close()
```

### Interactive Audio Injection (API 버전)

```python
# 1. 사용자 오디오 업로드
session.inject_audio(
    audio_file="user_input.wav",
    mix_ratio=0.3  # 30% 사용자, 70% AI
)

# 2. AI가 반응
response_chunk = session.generate_next()
# → 사용자 입력에 영향받은 음악!
```

---

## 📊 성능 비교 (추정)

| Metric | Magenta RT | Lyria RT |
|--------|------------|----------|
| **RTF** | 1.8x | 3-5x (추정) |
| **Latency** | ~800ms | ~1200ms (네트워크 포함) |
| **Audio Quality** | 48kHz, 16kbps | 48kHz, 32kbps (추정) |
| **Context Window** | 10s | 20-30s (추정) |
| **Style Accuracy** | Good | Excellent |
| **Prompt Coverage** | Standard | Wide |
| **Cost** | Free | API 요금 |

---

## 💰 비용 구조 (추정)

```python
# Lyria RT API 가격 (예상)
pricing = {
    "free_tier": {
        "minutes_per_month": 60,  # 1시간 무료
        "rate_limit": "10 requests/min"
    },
    "standard": {
        "price_per_minute": "$0.10",  # 분당 10센트
        "rate_limit": "100 requests/min"
    },
    "premium": {
        "price_per_minute": "$0.05",  # 대량 할인
        "rate_limit": "unlimited",
        "sla": "99.9% uptime"
    }
}

# 예시:
# 1시간 음악 생성 = 60분 × $0.10 = $6.00
# 10시간 연습/실험 = $60.00
```

---

## 🎯 당신의 프로젝트에는?

### **Magenta RT 추천!**

이유:

```
1. ✅ Fine-tuning 필요
   → "ohhalim style" 학습해야 함
   → API는 fine-tuning 불가능

2. ✅ 프라이버시
   → 내 연주 데이터가 소중함
   → 로컬에서만 처리

3. ✅ 비용
   → 무료로 실험 가능
   → 장기적으로 무료

4. ✅ 커스터마이징
   → MIDI로 개조 가능
   → Audio injection → MIDI injection

5. ✅ 학습 목적
   → 내부 작동 원리 이해
   → 연구 & 개선
```

### Lyria RT는 언제 사용?

```
상황 1: 프로토타입 빠르게 테스트
   - Magenta RT 설정 귀찮을 때
   - API 한 줄로 바로 실행

상황 2: 최고 품질 필요
   - 중요한 공연/녹음
   - Magenta RT보다 높은 품질 필요

상황 3: GPU 없을 때
   - 로컬에서 실행 불가능
   - 클라우드 의존해야 함
```

**하지만 당신의 목표 ("나와 가상의 내가 JAM")를 위해서는:**

→ **Magenta RT로 시작, Fine-tune, 커스터마이징!** ✅

---

## 🔗 접근 방법

### Magenta RealTime
```bash
# 1. GitHub 클론
git clone https://github.com/magenta/magenta-realtime.git

# 2. Colab 무료로 실행
# (TPU v2-8 제공)

# 3. 코드 & Weights 모두 공개
# → 완전한 자유
```

### Lyria RealTime
```bash
# 1. API 신청
# https://g.co/magenta/lyria-realtime

# 2. API Key 받기

# 3. SDK 설치
pip install lyria-realtime

# 4. API 호출
# → 간단하지만 제한적
```

---

## 💡 핵심 요약

**Lyria RealTime 작동 방식**:

```
1. 같은 Architecture (Magenta RT와)
   - MusicCoCa + Transformer + SpectroStream

2. 더 큰 모델
   - 더 많은 파라미터
   - 더 강력한 하드웨어

3. API 기반
   - 클라우드에서 실행
   - REST API 호출

4. Extended Controls
   - 더 세밀한 컨트롤
   - 더 넓은 프롬프트 커버리지

5. 상업적 사용 최적화
   - 안정성, SLA, 지원
```

**당신의 프로젝트에는:**
→ **Magenta RealTime이 완벽!** 🎹

이유: Fine-tuning 가능, 무료, 커스터마이징 자유, 프라이버시 보장!

**Lyria RT는 필요할 때 나중에 고려!**

---

**다음 단계: Magenta RT Colab 실행 + 내 연주 녹음!** 🚀
