# 2025년 재즈/음악 AI 분야 최신 SOTA 심화 분석

## 개요

본 문서는 2025년 재즈 및 음악 AI 분야의 최신 State-of-the-Art (SOTA) 모델들을 심도 있게 분석한 연구 자료입니다. 기존에 검토된 ImprovNet, ReaLJam, SiMBA 등의 모델에 더해, 2025년에 발표된 혁신적인 신규 모델들과 아키텍처를 포괄적으로 다룹니다.

---

## 1. 실시간 인간-AI 협업 연주 (Real-time Human-AI Co-Creativity)

### 1.1 Aria-Duet / "The Ghost in the Keys" (2025년 11월)

**논문 정보:**
- arXiv: [2511.01663](https://arxiv.org/abs/2511.01663)
- 학회: NeurIPS Creative AI Track 2025, ISMIR 2025
- 저자: Louis Bradshaw, Alexander Spangher, Stella Biderman, Simon Colton (EleutherAI)
- GitHub: https://github.com/EleutherAI/aria

**시스템 개요:**

Aria-Duet은 **Yamaha Disklavier**를 물리적 인터페이스로 활용하여 인간 피아니스트와 Aria 생성 모델이 실시간 듀엣을 수행하는 획기적인 시스템입니다. MIDI 연결을 통해 연주자의 퍼포먼스를 캡처하고, AI가 생성한 연주를 물리적으로 피아노 건반을 통해 재생합니다.

**핵심 기술:**

1. **턴 기반 협업 (Turn-taking Collaboration)**: 사용자가 연주하고 핸드오버를 신호하면, 모델이 일관성 있는 연속 구절을 생성하여 피아노로 연주

2. **대규모 데이터 사전학습**: Aria-MIDI 데이터셋의 정제된 서브셋(100,000시간 이상)으로 사전학습되어 광범위한 장르와 스타일을 포괄

3. **실시간 엔진**: 사용자 제어 흐름, 입출력 관리, 실시간 추론을 관리하는 저지연 시스템

**성능 평가:**

분석 결과, 모델은 **스타일적 의미론(stylistic semantics)을 유지**하고 **응집력 있는 프레이즈 아이디어를 발전**시킬 수 있음을 보여줍니다. 이는 물리적으로 구현된(embodied) 시스템이 음악적으로 정교한 대화에 참여할 수 있음을 입증합니다.

**혁신성:**

- 디지털 오디오가 아닌 **물리적 피아노 건반**을 통한 연주로 더욱 자연스러운 인간-AI 상호작용 실현
- 100,000시간 이상의 대규모 피아노 MIDI 데이터로 학습된 SOTA 생성 모델
- 실제 공연 환경에서 사용 가능한 수준의 저지연 시스템

---

### 1.2 jam_bot (2025년) - MIDI Innovation Award 수상

**시스템 정보:**
- 발표: ISMIR 2025
- 개발: MIT Media Lab × Jordan Rudess (Grammy 수상 키보디스트)
- 수상: 2025 MIDI Innovation Award Finalist (Installation Category)
- 공식 사이트: [MIT Media Lab jam_bot](https://www.media.mit.edu/projects/jordan-rudess-genai/)

**시스템 개요:**

jam_bot은 **Transformer 기반 Music Language Models**를 최적화하여 라이브 공연자와 무대 위에서 실시간으로 자유로운 즉흥 연주를 수행하는 시스템입니다. Grammy 수상 아티스트인 Jordan Rudess의 절충적(eclectic) 즉흥 미학에 맞출 수 있도록 설계되었습니다.

**핵심 기술:**

1. **다양한 음악적 역할 수행**:
   - Lead (선율 주도)
   - Accompany (반주)
   - Call and Response (교환 연주)

   이를 위해 컨텍스트와 조건 신호(conditioning signals)를 수정하여 Music LM을 각 상호작용 전략에 맞게 조정

2. **스타일 특화 Fine-tuning**: 의도적인 즉흥 구조화를 통해 특정 스타일에 맞춤

3. **실시간 최적화**:
   - Music LM을 실시간으로 실행하기 위한 최적화
   - 저지연 멀티스레드 시스템에 임베딩
   - 듣기, 프롬프트, 생성 스케줄링을 매끄럽게 처리

**공연 실적:**

2024년 9월 21일 MIT Media Lab에서 매진 공연으로 첫 공개 시연. Jordan Rudess와 jam_bot의 협연은 관객들에게 강렬한 인상을 남겼으며, 2025 MIDI Innovation Award 후보에 선정되었습니다.

**혁신성:**

- Grammy 수상 연주자의 복잡한 즉흥 스타일을 학습하고 재현
- 다양한 음악적 역할을 실시간으로 전환 가능
- 실제 무대 공연에서 검증된 안정성

---

## 2. 차세대 생성 모델 (Next-Generation Generative Models)

### 2.1 DiffRhythm (2025년 3월)

**논문 정보:**
- arXiv: [2503.01183](https://arxiv.org/abs/2503.01183)
- 출시: 2025년 3월 4일
- 라이센스: Apache 2.0
- GitHub: https://github.com/ASLP-lab/DiffRhythm
- 공식 사이트: https://diffrhythm.com

**모델 개요:**

DiffRhythm은 **세계 최초의 오픈소스 Diffusion 기반 End-to-End 음악 생성 모델**로, 보컬과 반주를 동시에 포함한 완전한 곡을 최대 **4분 45초** 길이로 **단 10초 만에** 생성할 수 있습니다.

**핵심 혁신:**

1. **End-to-End 동시 생성**: 보컬과 반주를 별도로 생성하거나 복잡한 캐스케이딩 아키텍처에 의존하는 기존 모델과 달리, 모든 요소를 동시에 생성

2. **비자기회귀(Non-autoregressive) 구조**: 빠른 추론 속도 보장

3. **단순성과 우아함**:
   - 복잡한 데이터 전처리 불필요
   - 간단한 모델 구조
   - 추론 시 가사와 스타일 프롬프트만 필요

**기술 아키텍처:**

- **Latent Diffusion**: 오디오의 잠재 공간에서 직접 작동
- **VAE (Variational Autoencoder)**: 오디오 신호를 압축/해제
- 원시 오디오 대신 축소된 잠재 공간에서 작업하여 음악 패턴을 더욱 효율적으로 포착

**성능:**

- 10초 내 완전한 곡 생성
- 높은 음악성(musicality)과 명료도(intelligibility) 유지
- 재즈를 포함한 다양한 스타일 지원 ("melancholic jazz ballad" 등)

**오픈소스 기여:**

- 대규모 데이터로 사전학습된 모델 공개
- 완전한 학습 코드 공개
- 재현성과 추가 연구 촉진

---

### 2.2 Mureka O1 (2025년 3월) - 최초의 음악 추론 모델

**출시 정보:**
- 출시: 2025년 3월 26일
- 개발: Kunlun Tech
- 특징: **세계 최초 음악 추론(Reasoning) 대규모 모델**

**핵심 혁신: MusiCoT (Music Chain-of-Thought)**

Mureka O1은 **MusiCoT**라는 음악 생성에 특화된 Chain-of-Thought (CoT) 변형을 도입했습니다:

- **기존 자기회귀 모델**: 오디오를 단계별로(step-by-step) 생성
- **Mureka O1**: 오디오 토큰을 디코딩하여 최종화하기 **전에** 전체 음악 구조를 사전 생성

이는 **구조적 일관성**과 **장기 계획 능력**을 크게 향상시킵니다.

**성능:**

- Suno를 능가하여 **SOTA 달성**
- 전체 곡의 아키텍처를 먼저 "추론"한 후 생성하여 더 응집력 있는 결과 산출

**개인화 기능:**

- 사용자가 자신의 음악을 업로드하여 **개인화된 AI 모델 훈련** 가능
- 고유한 스타일과 취향 서명(taste signatures) 학습
- 사용자 맞춤형 곡 생성

**의의:**

음악 생성에 "추론" 단계를 도입한 것은 패러다임의 전환입니다. 이는 단순히 패턴을 학습하는 것을 넘어 **음악적 구조를 이해하고 계획**하는 능력을 보여줍니다.

---

### 2.3 Quality-aware Masked Diffusion Transformer (QA-MDT) (2025년 6월)

**논문 정보:**
- arXiv: [2405.15863](https://arxiv.org/abs/2405.15863)
- 최종 개정: 2025년 6월 17일
- 학회: IJCAI 2025 (International Joint Conference on Artificial Intelligence)
- 파라미터: 675M (8 decoder layers)

**문제 인식:**

대규모 오픈소스 음악 데이터베이스는 **품질이 불균형**합니다. 이로 인해:
- 저품질 데이터가 모델 성능 저하
- 저품질 캡션(caption)이 텍스트-음악 정렬 저해

**핵심 혁신:**

1. **품질 인식 학습 패러다임 (Quality-aware Training)**:
   - 품질이 불균형한 대규모 데이터셋에서 고품질·고음악성 음악 생성
   - 품질 제어 능력 내재화

2. **3단계 캡션 정제 (Caption Refinement)**:
   - 저품질 캡션 문제 해결
   - 텍스트-음악 정렬 개선

3. **Masked Diffusion Transformer (MDT) 적용**:
   - 텍스트-음악 태스크에 MDT 모델 구현
   - 향상된 음악성 과시

**성능:**

- MusicCaps 벤치마크에서 **SOTA 달성**
- Song-Describer Dataset에서도 SOTA
- 객관적·주관적 메트릭 모두에서 최고 성능

**의의:**

실세계 데이터의 품질 불균형 문제를 직접 다룬 최초의 SOTA 모델. 이는 실용적 배포에 있어 중요한 진전입니다.

---

### 2.4 LiLAC (2025년 6월) - 경량 제어 모델

**논문 정보:**
- arXiv: [2506.11476](https://arxiv.org/abs/2506.11476)
- 제목: "A Lightweight Latent ControlNet for Musical Audio Generation"

**문제 인식:**

텍스트-오디오 Diffusion 모델은 고품질 음악을 생성하지만, 많은 SOTA 모델이 음악 프로덕션에 필수적인 **세밀한 시간-변화 제어(fine-grained, time-varying controls)**가 부족합니다.

**핵심 혁신:**

1. **경량 모듈형 아키텍처**:
   - 파라미터 수를 크게 감소시키면서도 ControlNet과 동등한 오디오 품질 달성
   - 조건 준수도(condition adherence)도 동일

2. **메모리 효율성**:
   - 상당히 낮은 메모리 사용량
   - 더 큰 유연성 제공

3. **세밀한 제어**:
   - 화음 진행(chord progressions), 리듬 패턴 등 미세 조정 가능
   - 음악 프로덕션 워크플로우에 적합

**성능:**

- ControlNet 수준의 품질 유지
- 파라미터 수 대폭 감소
- 메모리 사용량 크게 절감

**의의:**

효율성과 성능의 균형을 추구하는 트렌드를 반영. 실용적 배포와 엣지 디바이스 적용 가능성을 높입니다.

---

## 3. 멀티모달 음악 생성 (Multimodal Music Generation)

### 3.1 MeLFusion (2024/2025) - 이미지-음악 SOTA

**논문 정보:**
- arXiv: [2406.04673](https://arxiv.org/abs/2406.04673)
- 학회: CVPR 2024
- 벤치마크: **MusicCaps에서 이미지-조건 SOTA**

**핵심 개념:**

MeLFusion은 **"Visual Synapse"**라는 새로운 메커니즘을 도입하여 시각 모달리티의 의미론을 생성된 음악에 효과적으로 주입합니다.

**기술:**

- 텍스트-음악 Diffusion 모델 기반
- 이미지 조건부(image-conditioned) 생성
- 시각적 단서와 언어적 단서를 결합하여 음악 합성

**성능:**

- MusicCaps 벤치마크에서 이미지-조건부 텍스트-음악 생성 부문 **SOTA**

**응용:**

- 영화 음악 작곡 (영상 장면에 맞춘 음악)
- 광고 음악 생성
- 게임 적응형 사운드트랙

---

### 3.2 MusFlow (2025년 4월)

**논문 정보:**
- arXiv: [2504.13535](https://arxiv.org/html/2504.13535)
- 제목: "Multimodal Music Generation via Conditional Flow Matching"

**핵심 기술:**

1. **Conditional Flow Matching**: 새로운 생성 패러다임 적용

2. **다중 MLP (Multi-Layer Perceptrons)**:
   - 멀티모달 조건 정보를 오디오의 CLAP 임베딩 공간에 정렬
   - 다양한 모달리티 (텍스트, 이미지, 오디오 등) 통합

3. **Conditional Flow를 통한 Mel-spectrogram 재구성**:
   - 사전학습된 VAE의 잠재 공간에서 압축된 Mel-spectrogram을 재구성

**혁신성:**

- Flow Matching과 멀티모달 조건을 결합한 새로운 접근
- 다양한 입력 모달리티를 통합하여 더욱 풍부한 제어 가능

---

## 4. 아키텍처 효율성 혁신 (Efficiency Innovations)

### 4.1 State-Space Models (Mamba) for Music (2025년 7월)

**논문 정보:**
- arXiv: [2507.06674](https://arxiv.org/html/2507.06674)
- 제목: "Exploring State-Space-Model based Language Model in Music Generation"

**배경:**

**State Space Models (SSMs)**, 특히 **Mamba**의 등장은 Transformer의 강력한 대안으로 자리잡았습니다:
- 선형 복잡도 연산
- 긴 시퀀스 처리에 효율적

**적용:**

- Mamba 기반 아키텍처를 텍스트-음악 생성에 적용
- **Residual Vector Quantization (RVQ)**의 이산 토큰을 모델링 표현으로 채택

**성능:**

- 제한된 학습 자원에서 **Transformer보다 빠른 수렴**
- 생성된 음악이 실제 악보와 더 유사
- 적은 연산량으로 우수한 생성 품질

**의의:**

SiMBA (이미 검토됨)와 함께 State-Space Models가 음악 생성 분야에서 Transformer의 유력한 대안으로 부상하고 있음을 보여줍니다.

---

### 4.2 Stable Audio 2.0/2.5 (2025년)

**개발:**
- Stability AI
- 오픈소스: Stable Audio Open, Stable Audio Tools

**주요 혁신: ARC Post-training**

**Adversarial Relativistic-Contrastive (ARC) Post-training**:
- Diffusion/Flow 모델을 위한 **최초의 적대적 가속 알고리즘**
- **비용이 많이 드는 증류(distillation) 기반이 아님**
- 추론 속도 대폭 향상

**특징:**

- 가장 진보된 제어 메커니즘 제공
- 세밀한 오디오 조작 가능
- 재즈, 클래식, 필름 스코어 등 장르별 높은 스타일 정확도

**버전:**

- **Stable Audio Open 1.0**: 오픈소스, 사운드 디자인 특화
- **Stable Audio Small**: ARC Post-training 적용
- **Stable Audio 2.5**: 최신 상업 버전

**의의:**

고품질 오디오 생성과 효율성의 균형을 추구하는 대표적 사례. 오픈소스 커뮤니티에도 큰 기여.

---

## 5. 상징적 음악 생성 (Symbolic Music Generation)

### 5.1 SMART (2025년 4월)

**논문 정보:**
- arXiv: [2504.16839](https://arxiv.org/html/2504.16839)
- 제목: "Tuning a Symbolic Music Generation System with an Audio Domain Aesthetic Reward"

**핵심 개념: Audio-Domain Reward**

**문제점:**
- 기존 상징적 음악(symbolic music) 모델은 MIDI/악보 도메인에서만 학습
- 실제 오디오 렌더링 시 음질이나 미학적 평가가 반영되지 않음

**혁신:**

1. **Audio-Rendered Outputs as Reward**:
   - MIDI를 오디오로 렌더링
   - 렌더링된 오디오의 품질을 보상(reward)으로 활용

2. **Group Relative Policy Optimization**:
   - 피아노 MIDI 모델을 오디오 도메인 보상으로 fine-tuning
   - 강화학습 적용

3. **아키텍처**:
   - Causal Transformer with Microsoft Phi 3 architecture

**성능:**

- 상징적 표현과 실제 음질의 간극 해소
- 더욱 듣기 좋은 음악 생성

**의의:**

상징적 음악 생성과 오디오 도메인을 연결한 선구적 연구. 실용성을 크게 향상시켰습니다.

---

### 5.2 Giant Music Transformer (2025년)

**모델 정보:**
- 파라미터: 786M
- 시퀀스 길이: 8k
- 정확도: 92%

**특징:**

- **진정한 전체 MIDI 악기 범위 지원**
- 다중 악기 음악(multi-instrumental) 지원
- 다양한 장르 커버

**성능:**

- 950M 파라미터 Transformer 모델과 경쟁
- 튜링 스타일 청취 설문에서 종종 인간 작곡으로 평가됨

**의의:**

상징적 음악 생성에서 대규모 모델의 효과를 입증. 인간 수준의 작곡 능력에 근접하고 있음을 보여줍니다.

---

### 5.3 MIREX 2025 - Traditional Approach with Modern Tools

**대회 정보:**
- MIREX 2025 Symbolic Music Generation Challenge
- 과제: 4마디 피아노 프롬프트에서 12마디 연속 생성

**접근법:**

- **RWKV-7 아키텍처** 활용 (최신 State-Space Model)
- **Aria-MIDI 데이터셋** 활용
- 가설: 토큰화된 원시 음악 데이터에 대한 단순한 다음-토큰 예측이, 피아노 연속 생성 태스크에 특화 학습될 경우 파운데이션 모델을 능가할 수 있음

**데이터셋:**

- **Aria-MIDI**: 최근 공개된 대규모 피아노 MIDI 데이터셋
- 여러 장르를 포괄하는 큐레이트된 100K 파일 서브셋 사용
- **MAESTRO**: 200시간의 전문 클래식 피아노 연주

**의의:**

전통적 접근(다음-토큰 예측)과 최신 도구(RWKV-7, Aria-MIDI)의 결합이 특정 태스크에서 범용 파운데이션 모델을 능가할 수 있음을 시사합니다.

---

## 6. 모델링 패러다임 비교 (Modeling Paradigm Comparison)

### 6.1 Auto-Regressive vs Flow-Matching (2025년 6월)

**논문 정보:**
- arXiv: [2506.08570](https://arxiv.org/abs/2506.08570)
- 제목: "A Comparative Study of Modeling Paradigms for Text-to-Music Generation"

**비교 대상:**

1. **Auto-Regressive Models** (e.g., MusicGen):
   - 토큰을 순차적으로 생성
   - 안정적이고 검증된 접근

2. **Flow-Matching Models** (e.g., MusFlow):
   - 연속적인 흐름을 통해 생성
   - 최근 주목받는 새로운 패러다임

**주요 발견:**

- 두 패러다임 모두 고품질 음악 생성 가능
- **Trade-offs 존재**:
  - Auto-Regressive: 더 나은 구조적 일관성, 느린 추론
  - Flow-Matching: 빠른 추론, 때때로 구조적 불안정성

**훈련 데이터셋, 아키텍처 선택의 중요성:**

모델링 패러다임만큼이나 데이터와 아키텍처가 성능에 큰 영향을 미침을 확인했습니다.

**의의:**

연구자들이 태스크와 제약 조건에 따라 적절한 패러다임을 선택할 수 있도록 가이드를 제공합니다.

---

## 7. 상업 SOTA: Suno vs Udio (2025년 지속 업데이트)

### 7.1 Suno

**강점:**
- **가장 상업적으로 실행 가능한 완전한 곡 생성**
- 최소한의 편집으로 사용 가능한 결과물
- End-to-end 보컬 포함 곡 생성에 강점

**기술:**
- Multimodal Transformer 기반 아키텍처
- 텍스트 프롬프트와 오디오 패턴 모두 처리

---

### 7.2 Udio

**강점:**
- **인간 청취자 선호도에서 일관되게 높은 점수**
- 복잡한 악기 편곡 처리에 탁월
- **강력한 텍스트-오디오 정렬**
- 재즈, 클래식, 필름 스코어 등에서 뛰어난 스타일 정확도

**기술:**
- Transformer 네트워크와 Auto-Regressive 모델의 조합
- 정교한 구조적 인식(structural awareness)

**평가:**
- 2025년 4월 벤치마크 테스트에서 많은 매치업에서 인간 베이스라인을 능가
- 청취자들이 Udio 출력을 인간 작곡보다 선호하는 경우 다수

---

### 7.3 상업 모델의 의의

Suno와 Udio는 연구용 모델들이 실세계에 적용될 때의 벤치마크 역할을 합니다:
- **사용자 친화성**: 복잡한 프롬프트 없이 자연어만으로 생성
- **품질 일관성**: 대부분의 출력이 사용 가능한 수준
- **장르 다양성**: 재즈, 클래식, 팝, 록 등 광범위

---

## 8. 2025년 재즈/음악 AI의 주요 트렌드

### 8.1 멀티모달 접근 증가

- 텍스트, 이미지, 오디오를 결합한 생성 모델 급증
- Vision-to-Music Generation이 중요한 연구 영역으로 부상
- 예: MeLFusion, MusFlow

### 8.2 추론(Reasoning) 능력 도입

- Mureka O1의 MusiCoT: 음악 생성에 Chain-of-Thought 적용
- 단순 패턴 학습을 넘어 **구조 이해 및 계획** 능력
- 장기적 일관성(long-term coherence) 크게 향상

### 8.3 효율성과 품질의 균형

- State-Space Models (Mamba, SiMBA): Transformer 대비 효율성
- LiLAC: 경량화하면서도 품질 유지
- ARC Post-training (Stable Audio): 증류 없는 가속

### 8.4 실시간 인간-AI 협업 성숙

- Aria-Duet: 물리적 악기를 통한 자연스러운 상호작용
- jam_bot: 무대 위에서 실시간 즉흥 연주
- ReaLJam: 강화학습으로 응답성 개선

### 8.5 품질 인식 및 제어

- QA-MDT: 품질 불균형 데이터 처리
- LiLAC: 세밀한 시간-변화 제어
- SMART: 오디오 도메인 보상으로 미학 개선

### 8.6 개인화 및 Few-Shot 학습

- Mureka O1: 사용자 업로드 음악으로 개인화 모델 훈련
- 사용자 스타일·취향 학습
- AI 음악 에이전트: 장기 기억으로 선호도 적응

### 8.7 End-to-End 풀-송 생성

- DiffRhythm: 4분 45초 곡을 10초에 생성
- 보컬+반주 동시 생성
- 복잡한 파이프라인 불필요

### 8.8 오픈소스와 재현성 강조

- DiffRhythm, Stable Audio Open, Aria 등 코드·모델 공개
- 커뮤니티 기여 활성화
- 연구 재현성 향상

---

## 9. 비교 분석: 기존 모델 vs 신규 모델

| 모델/시스템 | 주요 혁신 | 발표 시기 | 특화 영역 |
|------------|---------|----------|----------|
| **ImprovNet** | 반복적 손상-정제 학습 | 2025.02 | 장르 간/내 즉흥 |
| **ReaLJam** | 강화학습 실시간 합주 | 2025.02 | 실시간 상호작용 |
| **Aria-Duet** | Disklavier 물리적 연주 | 2025.11 | 인간-AI 듀엣 |
| **jam_bot** | 다중 역할 Music LM | 2025 | 라이브 무대 즉흥 |
| **DiffRhythm** | End-to-End Diffusion | 2025.03 | 풀-송 빠른 생성 |
| **Mureka O1** | 음악 CoT 추론 | 2025.03 | 구조적 일관성 |
| **QA-MDT** | 품질 인식 학습 | 2025.06 | 불균형 데이터 처리 |
| **LiLAC** | 경량 제어 | 2025.06 | 효율적 세밀 제어 |
| **MeLFusion** | Visual Synapse | 2024/25 | 이미지-음악 SOTA |
| **MusFlow** | Conditional Flow Matching | 2025.04 | 멀티모달 생성 |
| **Mamba for Music** | State-Space Model | 2025.07 | 효율적 시퀀스 모델링 |
| **SMART** | Audio-Domain RL | 2025.04 | 상징적 음악 미학 |
| **Giant Music Transformer** | 786M 다중악기 | 2025 | 상징적 SOTA |
| **Stable Audio 2.5** | ARC Post-training | 2025 | 빠른 고품질 오디오 |
| **Suno/Udio** | 상업 SOTA | 지속 업데이트 | 상용 생성 |

---

## 10. Art Tatum AI 프로젝트에의 시사점

본 프로젝트가 Art Tatum 스타일의 재즈 솔로 생성 AI를 목표로 한다면, 다음 모델들과 기법이 특히 유용할 것입니다:

### 10.1 생성 모델 선택

1. **ImprovNet**: 재즈 즉흥 생성에 특화, 장르 내 스타일 제어 가능
2. **Mureka O1**: MusiCoT로 구조적 일관성 높은 솔로 생성
3. **SMART**: 오디오 도메인 보상으로 미학적 품질 향상

### 10.2 실시간 연주 시스템

1. **Aria-Duet**: Disklavier 활용한 물리적 피아노 연주
2. **jam_bot**: 실시간 즉흥 협연
3. **ReaLJam**: 강화학습 기반 응답성

### 10.3 데이터 및 학습 전략

1. **PiJAMA + Aria-MIDI**: 대규모 재즈 피아노 데이터
2. **Quality-aware Training (QA-MDT)**: 품질 불균형 해결
3. **Audio-Domain Reward (SMART)**: 실제 음질 기반 최적화

### 10.4 아키텍처 고려사항

1. **Mamba/State-Space Models**: 효율적이고 빠른 추론
2. **Masked Diffusion Transformer**: 제어 가능한 고품질 생성
3. **Transformer-XL/RWKV-7**: 긴 시퀀스 처리

### 10.5 스타일 분석 및 제어

1. **Deconstructing Jazz Piano Style**: 연주자별 스타일 분해 및 해석
2. **Explainable Subnetworks**: 멜로디, 화성, 리듬, 다이나믹스 별도 분석
3. **94% 정확도**: Art Tatum 특유 스타일 식별 및 재현 가능

---

## 11. 결론 및 향후 전망

2025년 재즈/음악 AI 분야는 다음과 같은 방향으로 급속히 발전하고 있습니다:

### 주요 성과:

1. **실시간 협업의 성숙**: Aria-Duet, jam_bot 등이 무대 위 실제 연주 가능 수준 달성
2. **생성 품질의 비약적 향상**: Mureka O1, DiffRhythm 등이 상업적 사용 가능 수준
3. **효율성 혁신**: Mamba, LiLAC 등이 품질 유지하며 연산량 대폭 감소
4. **멀티모달 통합**: MeLFusion, MusFlow 등이 다양한 입력 모달리티 지원
5. **추론 능력 획득**: Mureka O1의 MusiCoT가 구조 이해 및 계획 능력 시연

### 향후 전망:

1. **더욱 정교한 스타일 제어**: 특정 아티스트 스타일의 세밀한 재현
2. **실시간 협업의 일반화**: 다양한 악기와 장르에서 인간-AI 협연
3. **개인화의 심화**: Few-shot learning으로 사용자 스타일 빠르게 학습
4. **멀티모달 확장**: 영상, 감정, 맥락 등 더 다양한 조건부 생성
5. **설명 가능성 강화**: AI가 생성한 음악의 음악 이론적 설명 제공

재즈 AI, 특히 Art Tatum과 같은 전설적 연주자의 스타일을 재현하는 프로젝트는 이러한 최신 기술들을 적극 활용함으로써 전례 없는 수준의 음악적 표현력과 즉흥 능력을 달성할 수 있을 것입니다.

---

## 참고 문헌 및 출처

### 논문 및 기술 문서

1. **Aria-Duet**: [arXiv:2511.01663](https://arxiv.org/abs/2511.01663) - The Ghost in the Keys: A Disklavier Demo for Human-AI Musical Co-Creativity
2. **jam_bot**: [ISMIR 2025](https://ismir2025program.ismir.net/poster_321.html) - The jam_bot, a Real-Time System for Collaborative Free Improvisation with Music Language Models
3. **DiffRhythm**: [arXiv:2503.01183](https://arxiv.org/abs/2503.01183) - Blazingly Fast and Embarrassingly Simple End-to-End Full-Length Song Generation
4. **QA-MDT**: [arXiv:2405.15863](https://arxiv.org/abs/2405.15863) - Quality-aware Masked Diffusion Transformer for Enhanced Music Generation
5. **LiLAC**: [arXiv:2506.11476](https://arxiv.org/abs/2506.11476) - A Lightweight Latent ControlNet for Musical Audio Generation
6. **MeLFusion**: [arXiv:2406.04673](https://arxiv.org/abs/2406.04673) - Synthesizing Music from Image and Language Cues using Diffusion Models
7. **MusFlow**: [arXiv:2504.13535](https://arxiv.org/html/2504.13535) - Multimodal Music Generation via Conditional Flow Matching
8. **State-Space Models for Music**: [arXiv:2507.06674](https://arxiv.org/html/2507.06674) - Exploring SSM-based Language Model in Music Generation
9. **SMART**: [arXiv:2504.16839](https://arxiv.org/html/2504.16839) - Tuning a Symbolic Music Generation System with Audio Domain Aesthetic Reward
10. **Auto-Regressive vs Flow-Matching**: [arXiv:2506.08570](https://arxiv.org/abs/2506.08570) - Comparative Study of Modeling Paradigms

### 공식 웹사이트 및 리소스

- [Aria GitHub Repository](https://github.com/EleutherAI/aria)
- [DiffRhythm GitHub](https://github.com/ASLP-lab/DiffRhythm)
- [DiffRhythm Official Site](https://diffrhythm.com)
- [MIT Media Lab - jam_bot](https://www.media.mit.edu/projects/jordan-rudess-genai/)
- [Stable Audio - Stability AI](https://stability.ai/stable-audio)
- [MusicGen - Meta AudioCraft](https://audiocraft.metademolab.com/musicgen.html)
- [Papers with Code - MusicCaps Benchmark](https://paperswithcode.com/sota/text-to-music-generation-on-musiccaps)

### 뉴스 및 블로그

- [Mureka O1 Launch (Yahoo Finance)](https://finance.yahoo.com/news/kunlun-tech-launches-worlds-first-070000484.html)
- [MIDI Innovation Award 2025](https://midi.org/the-jambot-at-ismir-2025)
- [MIT News - A model of virtuosity](https://news.mit.edu/2024/model-virtuosity-jordan-rudess-jam-bot-1119)
- [Beatoven.ai - AI Music Generation Models 2025 Guide](https://www.beatoven.ai/blog/ai-music-generation-models-the-only-guide-you-need/)
- [MIT Technology Review - AI is coming for music](https://www.technologyreview.com/2025/04/16/1114433/ai-artificial-intelligence-music-diffusion-creativity-songs-writer/)

### 벤치마크 및 데이터셋

- MusicCaps Benchmark
- Aria-MIDI Dataset
- MAESTRO Dataset
- PiJAMA Dataset
- Song-Describer Dataset

---

**문서 작성일**: 2025년 11월 23일
**작성 목적**: Art Tatum MIDI Solo Fine-tuning 프로젝트를 위한 2025년 재즈/음악 AI SOTA 심화 리서치
**프로젝트 Repository**: fine_tuning_art-tatum_midi_solo
