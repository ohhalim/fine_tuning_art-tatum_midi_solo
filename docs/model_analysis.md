# ImprovNet vs Magenta Realtime 딥러닝 모델 분석

## 목차
1. [개요](#개요)
2. [ImprovNet 분석](#improvnet-분석)
3. [Magenta Realtime 분석](#magenta-realtime-분석)
4. [비교 분석](#비교-분석)
5. [결론 및 활용 방안](#결론-및-활용-방안)

---

## 개요

본 문서는 음악 생성 딥러닝 모델인 **ImprovNet**과 **Magenta Realtime**을 비교 분석합니다. 두 모델 모두 AI 기반 음악 생성을 목표로 하지만, 접근 방식과 응용 분야에서 차이를 보입니다.

### 분석 대상 모델

| 모델 | 개발 기관 | 주요 목적 | 데이터 형식 |
|------|-----------|-----------|-------------|
| **ImprovNet** | Queen Mary University of London, SUTD | 스타일 전이 기반 즉흥 연주 생성 | Symbolic (MIDI) |
| **Magenta Realtime** | Google Magenta | 실시간 스트리밍 음악 생성 | Audio (Waveform) |

---

## ImprovNet 분석

### 1. 핵심 아키텍처

#### 1.1 Transformer 기반 Encoder-Decoder 구조
- **Encoder**: 12 layers, 8 attention heads, hidden size 512
- **Decoder**: 동일 구성
- **최대 시퀀스 길이**: Encoder 2048 tokens, Decoder 512 tokens
- **처리 단위**: 5초 세그먼트 (최대 11개 세그먼트, 55초)

#### 1.2 토크나이저: Aria Tokenizer
```
특징:
- Chunked absolute onset encoding
- 최소 양자화 (onset/duration: 10ms, velocity: 15 MIDI units)
- 5초 세그먼트로 분할 (<T> 토큰으로 구분)
- 표현력 있는 연주 렌더링 가능
```

**토큰 구조:**
- **Onset Token**: 절대 시작 시간 (10ms 단위)
- **Duration Token**: 음표 지속 시간
- **Pitch-Velocity Token**: 음높이와 세기를 하나의 토큰으로 병합

### 2. 학습 방법론

#### 2.1 Self-Supervised Corruption-Refinement 전략

**핵심 개념:**
1. 원본 세그먼트를 의도적으로 손상(corruption)
2. 모델이 원본으로 복원(refinement)하도록 학습
3. 장르 조건부 학습 (conditional generation)

#### 2.2 9가지 Corruption Functions

| Corruption 함수 | 설명 | 음악적 효과 |
|-----------------|------|-------------|
| **Pitch Velocity Mask** | 음높이-세기 토큰 마스킹 | 멜로디/화성 재생성 |
| **Onset Duration Mask** | 시작 시간-지속 시간 마스킹 | 리듬 변화, 싱코페이션 |
| **Whole Mask** | 전체 세그먼트 마스킹 | Infilling/Continuation |
| **Permute Pitch** | 음높이 순서 섞기 | 멜로디 재구성 |
| **Permute Pitch Velocity** | 음높이-세기 섞기 | 표현력 변화 |
| **Fragmentation** | 20-50% 음표만 유지 | 변주 생성 |
| **Incorrect Transposition** | ±5 반음 전조 | 재즈 스케일, 크로매틱 |
| **Note Modification** | 음표 추가/삭제 | 밀도 조절 |
| **Skyline** | 최고 음만 추출 | 멜로디 하모나이제이션 |

#### 2.3 학습 데이터셋

**Pre-training:**
- ATEPP 데이터셋 (재전사 버전): ~1000시간 클래식 피아노

**Fine-tuning:**
- Maestro: 177시간 클래식 피아노
- PiJAMA (재전사): 200시간+ 재즈 피아노
- Doug McKenzie: 307개 짧은 재즈 MIDI

**학습 설정:**
- Pre-training: 360K steps, batch size 4, lr 1×10⁻⁴
- Fine-tuning: 318K steps, batch size 4 (gradient accumulation 3), lr 5×10⁻⁵

### 3. 생성 방법론

#### 3.1 Iterative Refinement Framework

**수식 정의:**
```
S = (s₁, ..., sₙ)  # 5초 세그먼트 시퀀스
Sᵢₙₚᵤₜ = (sᵢ₋ₗ, ..., sᵢ₋₁, sᵢ, sᵢ₊₁, ..., sᵢ₊ᵣ)  # 좌우 컨텍스트
```

**생성 과정:**
1. 각 패스(pass) P에서 모든 세그먼트 순회
2. Corruption rate α에 따라 세그먼트 변형
3. 목표 장르 조건으로 refinement
4. 여러 패스 반복하여 점진적 스타일 전이

#### 3.2 주요 기능

**1) 스타일 기반 즉흥 연주**
- **Cross-Genre Improvisation (CGI)**: 클래식 → 재즈
- **Intra-Genre Improvisation (IGI)**: 클래식 → 클래식 변주

**2) 하모나이제이션**
- Skyline corruption + Logit constraint
- 단선율에 장르별 화성 추가
- 첫 음표 아래 코드 생성 제약:
  ```python
  zₜ = { zₜ     if 0 ≤ onset(xₜ) - onset(x₁) ≤ 50ms
       { -∞     otherwise
  ```

**3) Short Prompt Continuation**
- 20초 프롬프트 → 15초 생성
- 우측 컨텍스트 제거하여 continuation 모드

**4) Short Infilling**
- 0:20s + 40:60s 제공 → 20:40s 생성
- 좌우 컨텍스트 활용

### 4. 평가 결과

#### 4.1 객관적 평가

**Cross-Genre Improvisation:**
- Whole Mask: 재즈 확률 최고 증가, SSM 상관관계 최저 (구조 변화 큼)
- Skyline/Note Modification: 원곡 구조 잘 보존

**Short Continuation/Infilling (vs AMT):**
| 메트릭 | ImprovNet | AMT | 원본 |
|--------|-----------|-----|------|
| Avg IOI (Continuation) | 0.1244 | 0.1386 | 0.1405 |
| Note Density | 37.49 | 67.66 | 30.85 |
| PCTM Cosine Sim | 0.3470 | 0.3074 | - |
| Pitch Class KL | 1.2500 | 1.6084 | - |

→ ImprovNet이 원본 분포에 더 가까움

**Harmonization:**
- With constraints: Polyphony Rate 0.91, Chord Diversity 18.30
- Without constraints: Polyphony Rate 0.25 (하모나이제이션 실패)

#### 4.2 주관적 평가 (28명 참가자)

| 태스크 | Interestingness | Human-like | Overall | Structural Similarity | Genre Accuracy |
|--------|-----------------|------------|---------|------------------------|----------------|
| **CGI** | 3.36/5 | 2.71/5 | 3.11/5 | 3.21/5 | 79% |
| **IGI** | 3.39/5 | 3.25/5 | 3.21/5 | 4.07/5 | - |
| **Original** | 3.43/5 | 5.0/5 | 3.64/5 | - | - |

**Cross-Genre Harmonization:**
- Interestingness: 3.54/5 (원본 3.38/5보다 높음!)
- 재즈 스타일 식별률: 76% (통계적 유의 p=0.0133)

**Short Continuation:**
- ImprovNet 선호: 56%
- AMT 선호: 20%
- 동등: 24%

### 5. 장점 및 한계

#### 장점
✅ **사용자 제어 가능성**: Corruption 함수, rate, 패스 수 조절
✅ **다목적 통합 모델**: Improvisation, harmonization, infilling, continuation
✅ **표현력**: 최소 양자화로 인간 같은 연주 렌더링
✅ **데이터 효율성**: Self-supervised learning으로 약한 레이블만 필요
✅ **구조 보존**: SSM 기반 preservation으로 원곡 인식 가능

#### 한계
⚠️ **과도한 코드 밀도**: 하모나이제이션 시 너무 많은 음표
⚠️ **불규칙한 리듬**: Onset-duration mask 사용 시 스윙 리듬 불안정
⚠️ **Continuation 길이 제한**: 20초 이상에서 일관성 저하
⚠️ **Human-likeness 낮음**: CGI에서 프레이징 왜곡 발생

---

## Magenta Realtime 분석

### 1. 핵심 아키텍처

#### 1.1 시스템 구성
- **모델**: Google의 Lyria 기반 오디오 생성 모델
- **출력 형식**: 48kHz, 스테레오 오디오 파형
- **생성 단위**: 2초 청크 (Chunk-based generation)
- **컨텍스트 윈도우**: 10초 과거 컨텍스트 활용

#### 1.2 SpectroStream 인코더
```
- 오디오 → 스펙트로그램 변환
- 프레임 길이: 40ms
- Crossfade 시간: 40ms (청크 연결 시)
```

### 2. 학습 방법론

#### 2.1 Text-to-Audio 조건부 생성
- **스타일 임베딩**: 텍스트 프롬프트를 768차원 벡터로 변환
- **조건부 생성**: 스타일 임베딩 + 과거 컨텍스트 → 다음 청크 생성

#### 2.2 학습 데이터
- 주로 서양 기악 음악 (Western instrumental music)
- 보컬 및 다양한 음악 전통에 대한 커버리지 제한적

**비교: Lyria RealTime API**
- 상용 API 버전은 더 넓은 스타일 커버리지 제공
- Magenta RT는 오픈 웨이트 버전

### 3. 생성 방법론

#### 3.1 Stateful Streaming Generation

**핵심 메커니즘:**
```python
chunk, state = model.generate_chunk(
    state=state,           # 이전 상태
    style=embedding,       # 스타일 임베딩
    seed=seed,             # 랜덤 시드
    temperature=1.3,       # 샘플링 온도
    top_k=40,              # Top-k 샘플링
    guidance_weight=5.0    # 가이던스 강도
)
```

**상태 관리:**
1. `state=None`으로 시작
2. 각 청크 생성 후 상태 업데이트
3. 다음 청크 생성 시 이전 상태 전달
4. 10초 컨텍스트 유지

#### 3.2 실시간 스타일 전환

**방법 1: 점진적 프롬프트 변경**
```python
styles = ["synthwave", "disco synthwave", "disco", "disco funk"]
for style in styles:
    chunk, state = model.generate_chunk(state, style_embed(style))
```

**방법 2: 임베딩 공간 보간**
```python
embed_a = model.embed_style("synthwave")
embed_b = model.embed_style("disco funk")
interpolated = embed_a + weight * (embed_b - embed_a)
```

#### 3.3 Weighted Prompts (Multiple Style Mixing)
```python
prompts = [
    ("synthwave", 0.7),
    ("flamenco guitar", 0.3)
]
# 가중 평균으로 스타일 블렌딩
```

### 4. 실시간 생성 시스템

#### 4.1 Buffering 전략
- **기본 버퍼**: 2초 (48000 samples)
- **추가 버퍼링**: 0-4초 조절 가능 (네트워크 안정성)
- **Crossfading**: 40ms 오버랩으로 경계 아티팩트 완화

#### 4.2 비동기 처리 (Colab UI)
```python
class AudioStreamer:
    - generate(): 오디오 청크 생성 (동기)
    - embed_style(): 텍스트 임베딩 (비동기, ThreadPoolExecutor)
    - 오디오 큐: Queue로 청크 버퍼링
```

### 5. 샘플링 파라미터

| 파라미터 | 범위 | 기본값 | 효과 |
|----------|------|--------|------|
| **Temperature** | 0.0-4.0 | 1.3 | 낮음: 안정적, 높음: 실험적 |
| **Top-k** | 0-1024 | 40 | 낮음: 일관성, 높음: 다양성 |
| **Guidance** | 0.0-10.0 | 5.0 | 높음: 프롬프트 충실, 낮음: 자유도 |

### 6. 장점 및 한계

#### 장점
✅ **실시간 생성**: 2초 청크 단위 스트리밍
✅ **오디오 출력**: 직접 재생 가능한 고품질 파형
✅ **즉각적 스타일 전환**: 생성 중 스타일 변경 가능
✅ **간단한 인터페이스**: 텍스트 프롬프트만으로 제어
✅ **오픈 웨이트**: 로컬 실행 가능 (TPU/GPU)
✅ **오디오 프롬프트**: 텍스트뿐 아니라 오디오 레퍼런스 사용 가능

#### 한계
⚠️ **제한된 커버리지**: 주로 서양 기악 음악, 보컬 부족
⚠️ **블랙박스**: 내부 메커니즘 불명확 (기술 리포트 예정)
⚠️ **세밀한 제어 부족**: Symbolic 레벨 편집 불가
⚠️ **음악 이론 제어 불가**: 특정 화성, 리듬 패턴 지정 어려움
⚠️ **하드웨어 요구사항**: TPU v5 lite 또는 A100 GPU 권장

---

## 비교 분석

### 1. 근본적 차이점

| 차원 | ImprovNet | Magenta Realtime |
|------|-----------|------------------|
| **데이터 도메인** | Symbolic (MIDI) | Audio (Waveform) |
| **생성 패러다임** | Corruption-Refinement | Autoregressive Chunking |
| **제어 방식** | 다중 파라미터 (9 corruption, rate, passes) | 텍스트/오디오 프롬프트 |
| **학습 방식** | Self-supervised (weak labels) | Supervised (text-audio pairs) |
| **출력 형식** | MIDI → DAW/VST 필요 | Audio → 즉시 재생 |
| **목표 태스크** | 스타일 전이 즉흥 연주, 하모나이제이션 | 실시간 스트리밍 생성 |

### 2. 아키텍처 비교

#### 모델 구조
```
ImprovNet:
Encoder-Decoder Transformer
├─ Encoder: 12 layers, 8 heads, 512 hidden
├─ Decoder: 12 layers, 8 heads, 512 hidden
└─ Tokens: Onset, Duration, Pitch-Velocity

Magenta Realtime:
Unknown Architecture (Lyria 기반)
├─ SpectroStream Encoder
├─ Text Encoder (768d embedding)
└─ Audio Decoder (48kHz stereo)
```

#### 컨텍스트 처리
- **ImprovNet**: 1-5 세그먼트 좌우 컨텍스트 (5-25초)
- **Magenta RT**: 10초 고정 과거 컨텍스트

### 3. 음악적 제어 비교

| 제어 요소 | ImprovNet | Magenta Realtime |
|-----------|-----------|------------------|
| **장르 전환** | Cross-genre improvisation (클래식↔재즈) | 텍스트 프롬프트 스타일 변경 |
| **화성 제어** | Skyline + Logit constraint | 프롬프트 텍스트로 간접 제어 |
| **리듬 제어** | Onset-duration mask corruption | Temperature/Top-k 조절 |
| **멜로디 제어** | Pitch-velocity mask | 불가능 (오디오 도메인) |
| **구조 보존** | SSM preservation ratio | State 유지로 일관성 |
| **즉흥성 강도** | Corruption rate, 패스 수 | Temperature 파라미터 |

### 4. 생성 프로세스 비교

#### ImprovNet 워크플로우
```
1. 원곡 MIDI 입력
2. 5초 세그먼트 분할
3. [반복] 각 패스마다:
   a. 세그먼트 선택 (corruption rate α)
   b. Corruption 함수 적용
   c. 목표 장르 조건으로 refinement
4. 최종 MIDI 출력
```

#### Magenta RT 워크플로우
```
1. 스타일 프롬프트 입력
2. 텍스트 → 768d 임베딩
3. [반복] 각 청크마다:
   a. 이전 상태 + 스타일 임베딩
   b. 2초 오디오 청크 생성
   c. 40ms crossfade
4. 실시간 스트리밍 출력
```

### 5. 응용 분야 비교

| 사용 사례 | ImprovNet | Magenta Realtime |
|-----------|-----------|------------------|
| **스타일 전이** | ⭐⭐⭐⭐⭐ (세밀한 제어) | ⭐⭐⭐ (텍스트 기반) |
| **실시간 연주** | ⭐⭐ (MIDI → Audio 변환 필요) | ⭐⭐⭐⭐⭐ (즉시 재생) |
| **음악 이론 교육** | ⭐⭐⭐⭐⭐ (하모나이제이션) | ⭐ (이론 제어 불가) |
| **DAW 통합** | ⭐⭐⭐⭐⭐ (MIDI 호환) | ⭐⭐ (오디오 트랙만) |
| **라이브 DJ/공연** | ⭐⭐ (지연 시간) | ⭐⭐⭐⭐ (실시간 스타일 전환) |
| **작곡 보조** | ⭐⭐⭐⭐ (변주, infilling) | ⭐⭐⭐ (프롬프트 continuation) |
| **커스터마이징** | ⭐⭐⭐⭐⭐ (9 corruption 함수) | ⭐⭐ (3 샘플링 파라미터) |

### 6. 평가 방법 비교

#### ImprovNet 평가
- **객관적**: Genre Classifier, SSM Correlation, Pitch Class KL, PCTM
- **주관적**: Listening test (28명), 장르 식별, 구조 유사도
- **Baseline**: AMT (Anticipatory Music Transformer), 원본

#### Magenta Realtime 평가
- **객관적**: (기술 리포트 예정)
- **주관적**: Colab 데모 체험
- **Baseline**: 없음 (오픈 웨이트 최초)

### 7. 장단점 종합 비교

| 관점 | ImprovNet 우위 | Magenta RT 우위 |
|------|----------------|------------------|
| **제어 가능성** | ✅ 9가지 corruption 함수, 다중 파라미터 | ❌ 텍스트 프롬프트 의존 |
| **실시간성** | ❌ Iterative refinement 느림 | ✅ 2초 청크 스트리밍 |
| **음악 이론 적용** | ✅ 하모나이제이션, 전조, 리듬 변형 | ❌ Symbolic 레벨 제어 불가 |
| **사용 편의성** | ❌ 복잡한 파라미터 설정 | ✅ 텍스트만 입력 |
| **출력 품질** | ❌ MIDI → 렌더링 품질 의존 | ✅ 48kHz 스테레오 직접 출력 |
| **스타일 커버리지** | ❌ 클래식/재즈만 | ✅ 다양한 장르 (서양 중심) |
| **데이터 효율성** | ✅ Self-supervised, 약한 레이블 | ❌ Text-audio 페어 필요 |
| **편집 가능성** | ✅ MIDI 편집, DAW 통합 | ❌ 오디오는 편집 어려움 |

### 8. 기술적 혁신 비교

#### ImprovNet 혁신
1. **Corruption-Refinement 패러다임**: 9가지 함수로 다목적 학습
2. **Iterative Generation**: 점진적 스타일 전이
3. **Logit Constraint Harmonization**: 첫 음 제약으로 화성 강제
4. **SSM-based Preservation**: 구조 보존하며 변형

#### Magenta RT 혁신
1. **Stateful Streaming**: 10초 컨텍스트 유지하며 실시간 생성
2. **Style Embedding Interpolation**: 임베딩 공간 보간으로 부드러운 전환
3. **Crossfading**: 40ms 오버랩으로 경계 아티팩트 제거
4. **Weighted Prompts**: 다중 스타일 블렌딩

---

## 결론 및 활용 방안

### 1. 모델 선택 가이드

#### ImprovNet 추천 시나리오
✅ **음악 이론 교육**: 하모나이제이션, 변주 학습
✅ **작곡 워크플로우**: MIDI 편집, DAW 통합
✅ **스타일 전이 연구**: 클래식 → 재즈 변환
✅ **세밀한 제어 필요**: Corruption 함수 커스터마이징
✅ **데이터 부족 상황**: Self-supervised learning

#### Magenta Realtime 추천 시나리오
✅ **실시간 공연/DJ**: 즉각적 스타일 전환
✅ **빠른 프로토타이핑**: 텍스트 프롬프트로 스타일 탐색
✅ **오디오 콘텐츠 제작**: 즉시 재생 가능한 출력
✅ **비전문가 사용**: 간단한 인터페이스
✅ **다양한 장르 탐색**: 서양 기악 음악 전반

### 2. Art Tatum 프로젝트 적용 방안

#### 현재 프로젝트 목표
- Art Tatum 스타일 재즈 솔로 생성
- PiJAMA 데이터셋에서 Art Tatum 필터링
- DAW 연동

#### 제안: Hybrid Approach

**Phase 1: Magenta RT로 초기 생성**
```python
# 1. Art Tatum 스타일 프롬프트
style = "Art Tatum virtuosic stride piano, fast tempo, complex harmonies"
embedding = model.embed_style(style)

# 2. 실시간 생성
chunks = []
for i in range(num_chunks):
    chunk, state = model.generate_chunk(state, embedding)
    chunks.append(chunk)

# 3. Audio → MIDI 변환 (Aria AMT 사용 가능)
midi = aria_amt.transcribe(audio)
```

**Phase 2: ImprovNet으로 정제**
```python
# 1. MIDI를 ImprovNet 입력으로
improvisations = []

# 2. Jazz Intra-Genre Improvisation
for pass_num in range(3):
    output = improvnet.generate(
        input=midi,
        target_genre="jazz",
        corruption_fn="note_modification",  # Art Tatum의 장식음
        corruption_rate=0.5,
        context_size=3
    )

# 3. Harmonization (Art Tatum의 복잡한 화성)
harmonized = improvnet.harmonize(
    melody=extract_melody(midi),
    genre="jazz",
    logit_constraint=True
)
```

**Phase 3: 스타일 분석 및 Fine-tuning**
```python
# 1. Art Tatum MIDI 필터링 (PiJAMA)
art_tatum_midis = filter_artist(pijama, "Art Tatum")

# 2. ImprovNet Fine-tuning
improvnet.finetune(
    data=art_tatum_midis,
    epochs=50,
    corruption_fns=["note_modification", "incorrect_transposition"]
)

# 3. Genre Classifier 재학습
classifier.train(art_tatum_midis, label="art_tatum_style")
```

### 3. 통합 워크플로우

```
[사용자 입력]
    ↓
[Magenta RT: 실시간 스타일 탐색]
    - 텍스트 프롬프트로 여러 변형 생성
    - 빠른 프로토타이핑
    ↓
[Audio → MIDI 변환]
    - Aria AMT 또는 Basic Pitch
    ↓
[ImprovNet: 세밀한 편집]
    - Corruption 함수로 Art Tatum 특징 강화
    - Harmonization으로 복잡한 화성 추가
    - Iterative refinement로 스타일 전이
    ↓
[MIDI Editor / DAW]
    - 수동 편집
    - VST로 렌더링 (Pianoteq 등)
    ↓
[최종 오디오 출력]
```

### 4. 기술적 시너지

| 기능 | Magenta RT 역할 | ImprovNet 역할 |
|------|-----------------|----------------|
| **초기 아이디어** | 텍스트 프롬프트로 빠른 생성 | - |
| **스타일 정제** | - | Corruption 함수로 특징 강화 |
| **화성 추가** | - | Logit constraint harmonization |
| **실시간 변형** | 스타일 임베딩 보간 | - |
| **최종 편집** | - | MIDI 레벨 세밀한 조정 |

### 5. 연구 방향 제안

#### 단기 목표 (1-3개월)
1. **Magenta RT 데모 실행**: Colab에서 Art Tatum 스타일 생성 테스트
2. **PiJAMA 데이터 분석**: Art Tatum 곡 필터링 및 특징 추출
3. **ImprovNet 환경 구축**: Pre-trained 모델로 재즈 improvisation 테스트

#### 중기 목표 (3-6개월)
1. **Hybrid 파이프라인 구축**: Magenta RT → Transcription → ImprovNet
2. **Art Tatum Fine-tuning**: ImprovNet을 Art Tatum 데이터로 fine-tune
3. **DAW 통합**: MIDI 출력을 Ableton/Logic 등과 연동

#### 장기 목표 (6-12개월)
1. **Custom Corruption 함수**: Art Tatum 특유의 스타일 패턴 학습
2. **Genre Classifier 확장**: Art Tatum vs 다른 Stride 피아니스트 구분
3. **실시간 시스템**: Magenta RT처럼 ImprovNet도 실시간 생성 가능하도록 최적화

### 6. 예상 도전 과제

| 도전 과제 | 해결 방안 |
|-----------|-----------|
| **Magenta RT의 재즈 커버리지 부족** | LyriaRT API 사용 또는 오디오 프롬프트 활용 |
| **Audio → MIDI 변환 품질** | Aria AMT (SOTA) 사용, 수동 보정 |
| **ImprovNet의 스윙 리듬 불안정** | Fine-tuning 시 onset-duration 조정 |
| **하드웨어 요구사항** | Magenta RT: Colab TPU, ImprovNet: 로컬 GPU |
| **데이터 부족** | Self-supervised learning, data augmentation |

### 7. 최종 권장 사항

본 프로젝트 **"Art Tatum AI - Fine-tuning"**에는 다음과 같은 접근을 권장합니다:

1. **ImprovNet 중심 개발**
   - Symbolic domain에서 세밀한 제어 가능
   - PiJAMA 데이터로 fine-tuning
   - DAW 통합 용이

2. **Magenta RT 보조 활용**
   - 초기 아이디어 탐색용
   - 오디오 레퍼런스 생성
   - 실시간 데모용

3. **점진적 통합**
   - Phase 1: ImprovNet standalone
   - Phase 2: Magenta RT → ImprovNet 파이프라인
   - Phase 3: 실시간 하이브리드 시스템

**근거:**
- Art Tatum의 특징(복잡한 화성, 빠른 패시지)은 Symbolic 레벨 제어 필요
- PiJAMA 데이터셋은 MIDI 형식 → ImprovNet 직접 활용
- DAW 연동 목표 → MIDI 출력 필수

---

## 참고 자료

### ImprovNet
- **논문**: "ImprovNet - Generating Controllable Musical Improvisations with Iterative Corruption Refinement"
- **저자**: Keshav Bhandari et al., QMUL & SUTD
- **코드**: https://github.com/keshavbhandari/improvnet
- **데이터셋**: ATEPP, Maestro, PiJAMA, Doug McKenzie

### Magenta Realtime
- **블로그**: https://g.co/magenta/rt
- **저장소**: https://github.com/magenta/magenta-realtime
- **HuggingFace**: https://huggingface.co/google/magenta-realtime
- **Colab 데모**: [Magenta_RT_Demo.ipynb](../notebooks/Magenta_RT_Demo.ipynb)

### 관련 모델
- **AMT (Anticipatory Music Transformer)**: Short continuation/infilling baseline
- **Aria Tokenizer**: ImprovNet의 토크나이저
- **Lyria RealTime API**: Magenta RT의 상용 버전

---

**작성일**: 2025-11-18
**버전**: 1.0
**작성자**: AI Analysis (Claude)
