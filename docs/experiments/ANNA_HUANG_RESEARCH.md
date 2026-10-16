# 안나 황 교수 연구 정리 (#1766)

작성 2026-10-16. 사용자가 이 분야에서 가장 존경하는 연구자이고, 그 연구 흐름을 이 프로젝트에서 따라가기 위해 정리했습니다. #1765 문서(음악 AI 기술 지도)의 6.5절을 넓히고 바로잡은 문서입니다.

확인 수준 표시

- **[원문]** 논문(arXiv 초록·저자·코멘트), 저장소, 공식 기사를 직접 열어 확인
- **[목록]** Semantic Scholar 또는 arXiv 목록에서 제목·연도·저자 수만 확인
- **[요약]** 검색 결과 요약만 봄
- **[판단]** Claude의 판단. 사실이 아님

---

## 1. 먼저 볼 것

1. **ReaLchords·ReaLJam·GAPT 공식 코드가 공개돼 있습니다.** `lukewys/realchords-pytorch`, MIT 라이선스입니다. 체크포인트는 Hugging Face `lukewys/realchords-pytorch`에 있습니다 [원문]
2. **#1765 6.5.3절을 바로잡습니다.** 그 문서에는 "ReaLchords는 사람이 멜로디, AI가 코드라서 우리와 방향이 반대"라고 적었습니다. 그런데 저장소에는 **코드를 받아 멜로디를 내는 온라인 모델의 학습 레시피**("RL Training for Melody Model")가 있고, 학습된 체크포인트 `rl_melody_gail_4_datasets`도 올라와 있습니다. 방향 문제는 코드 수준에서 이미 풀려 있습니다 [원문]
3. **ReaLJam 서버는 Apple Silicon용 MLX 백엔드를 지원합니다.** `pip install realjam` 후 `realjam-start-server --mlx`로 실행합니다. 이 Mac에서 바로 체험할 수 있습니다. 단, 외부 다운로드라서 승인이 필요합니다 [원문]
4. 남는 차이는 **표현**입니다. 저장소 모델은 16분음표 격자, 단선율 멜로디, 코드 기호 어휘를 씁니다. 우리는 실제 왼손 보이싱, 다성 오른손, Aria 토큰을 씁니다 (7절)
5. ReaLchords는 **±6반음 전조 증강(13배)**을 씁니다. 우리 D1 Arm E는 ×12 증강이 재붕괴를 일으킨다는 결과로 증강을 기각했습니다. 레시피를 그대로 옮기면 이 결정과 충돌합니다 (8절)

---

## 2. 이력

| 시기 | 내용 | 확인 |
|---|---|---|
| 학부 | USC | [원문 홈페이지] |
| 2008 | MIT Media Lab 석사(SM '08). 학위 논문 "Melodic variations: toward cross-cultural transformation" | [원문 MIT News] + [목록] |
| ~2019 | 하버드 박사. 학위 논문 "Deep Learning for Music Composition: Generation, Recommendation and Control"(Semantic Scholar 기록 2019). 하버드 시기 Krzysztof Gajos와 공동 연구(신스 노브 2014, ChordRipple 2016) | [원문 홈페이지] + [목록] |
| 약 8년 | Google Brain → Google DeepMind Magenta. Music Transformer, Coconet·Bach Doodle, MAESTRO, MIDI-DDSP | [원문 홈페이지] |
| — | Mila Canada CIFAR AI Chair, 몬트리올대 겸임 교수 | [원문 홈페이지] |
| 2020–22 | AI Song Contest 조직위원 | [원문 홈페이지] |
| 2024 가을~ | MIT 부교수. 음악·연극(MTA)과 EECS 겸임(EECS는 Schwarzman College of Computing 경유) | [원문 MIT News] |
| 2024 가을~ | 새 대학원 과정 Music Technology and Computation(MTC). 책임자는 Eran Egozy. 2026-05-13 첫 연구 발표회 | [원문 MIT News] |
| 2025–26 | Radcliffe 펠로우 | [요약] |

강의

- 21M.569 "Algorithms and Interactions in Human-AI Partnerships"(2024 가을 신설) [요약]
- 2026 가을: Grisha Coleman 교수와 21M.369/569 "Tuning Attention: Creative Practices in Movement, Sound, and AI" 공동 강의 예정 [원문 MIT News]
- MIT EECS를 통해 Music Technology 박사과정 학생을 받는 겸임 교수 3명 중 한 명(다른 두 명은 Mark Rau, Paris Smaragdis) [원문 MIT News]

2026 강연

- 2026-05-13 MIT Music Technology Research Showcase 기조연설. 제목은 기사 본문에 "In Search of Resonance in Human-AI Interaction", 사진 설명에 "In Search of Human-AI Resonance"로 다르게 적혀 있습니다 [원문 MIT News]
  - 요지: AI 작업의 중심에 사람 음악가를 둔다. 음악 기술은 세계의 여러 음악 전통을 포함해야 한다. 스튜디오에서 음악가와 직접 공동 설계하고, 매주 새로운 것을 시도하며, 창작 과정과 함께 기술을 바꿔 간다 [원문, 기사의 간접 인용]
- 2026-06-22 King's College London MARC Open Day(NIME 위성 행사) 특강, 같은 제목. 이어서 Heidi Lei가 "Gesture Controlled Systems for Live Piano Jamming"을 발표 [요약]

---

## 3. 연구 흐름

### 3.1 네 시기

| 시기 | 무엇을 했나 | 대표 |
|---|---|---|
| A. 학위 시절 (2008–2016) | 선율 변형, 신스 조작 노브 학습, 초보 작곡가에게 코드를 추천하는 인터페이스. 처음부터 **음악 + 사람-컴퓨터 상호작용(HCI)** | ChordRipple(IUI 2016) |
| B. Google Magenta (2017–2022) | 기호 음악 생성 모델의 기초, 대규모 대중 상호작용, 초보자용 조종 도구, 표현적 연주 합성 | Coconet, Music Transformer, MAESTRO, Bach Doodle, Cococo, MIDI-DDSP |
| C. 전환 (2023–2024) | **온라인(동시) 생성 + 강화학습**, 비서구 전통(힌두스타니) | ReaLchords(ICML 2024), GaMaDHaNi(ISMIR 2024) |
| D. MIT (2024–2026) | 실시간 잼 시스템, 보상 해킹 방지, 스트리밍 반주, 실시간 오디오 디퓨전, 즉흥 소통 데이터, 춤·드론·재즈 콤보 | ReaLJam, GAPT, StreamMusicGen, LMDM, H2H, Encypher, JazzSAMBA |

### 3.2 처음부터 이어지는 네 줄기 [판단]

1. **사람이 조종한다**: ChordRipple(2016) → Cococo·AI 조종 도구(CHI 2020) → Expressive Communication(IUI 2022) → Compositional Steering(2022) → Latent Fourier Transform(ICLR 2026 Oral). 모델이 혼자 잘 만드는 것보다 사람이 원하는 방향으로 움직일 수 있는지를 계속 묻습니다
2. **동시에 연주한다**: ReaLchords(2024) → ReaLJam(2025) → GAPT(2025–26) → StreamMusicGen(2025) → LMDM(2026). 상대의 미래를 모르는 상태에서 지금 맞춰 연주하는 문제입니다
3. **여러 음악 전통**: 바흐 코랄, 힌두스타니 성악, 댄스 사이퍼, 재즈 콤보
4. **음악가와 공동 설계하고 상호작용으로 평가한다**: 사용자 연구가 거의 모든 논문에 들어갑니다. H2H(즉흥 소통 모델), Design Space(시스템 184개 분류)로 평가 틀 자체를 만듭니다

### 3.3 연구 구상 — "jam space" [원문 홈페이지 요지]

음악가와 에이전트가 함께 즉흥하는 공간을 구상합니다. 연구자는 자기 생성 에이전트와 보상 모델을 바꿔 끼울 수 있고(OpenAI Gym처럼), 평가는 결과 음악뿐 아니라 상호작용까지 봅니다. 실시간으로 돌아가는 악기와 에이전트를 위한 효율적 ML도 함께 적었습니다. 홈페이지에서는 "envision"이라고 써서 아직 만들어지지 않은 구상으로 표현합니다.

[판단] `realchords-pytorch`의 멜로디 에이전트 대 코드 에이전트 시뮬레이션(MARL)과 ReaLJam 에이전트 인터페이스는 이 구상의 작은 구체 사례로 볼 수 있습니다. 우리 CLAUDE.md §6 목표 3("교체 가능한 어댑터 + 지표가 찍히는 환경")과 같은 방향입니다.

---

## 4. 논문 전체 목록

출처: arXiv 저자 검색("Cheng-Zhi Anna Huang", "Anna Huang") 23편과 Semantic Scholar 저자 기록 4개(7900665, 2316891503, 2392323124, 2396427430)를 합쳤습니다. Semantic Scholar는 한 사람을 여러 기록으로 나눠 두었고, 동명이인 논문도 섞여 있었습니다. 동명이인은 아래 4.2에 따로 적었습니다.

저자 위치는 "몇 번째/전체"입니다. 마지막 저자는 보통 지도·책임 역할입니다.

### 4.1 목록

| 연도 | 제목 | 학회 | 위치 | 한 줄 | 확인 |
|---|---|---|---|---|---|
| 2008 | Melodic variations: toward cross-cultural transformation | MIT 석사 논문 | 단독 | 문화 간 선율 변형 | [목록] |
| 2014 | Active learning of intuitive control knobs for synthesizers using Gaussian processes | IUI 2014 | 6명 중(마지막 Gajos) | 신스 조작 노브를 가우시안 프로세스 능동학습으로 | [목록] |
| 2016 | ChordRipple: Recommending Chords to Help Novice Composers Go Beyond the Ordinary | IUI 2016 | 3명 중(마지막 Gajos) | 초보 작곡가에게 평범하지 않은 코드 추천 | [목록] |
| 2017 | Counterpoint by Convolution (Coconet) | ISMIR 2017 | 1/5 | 바흐 코랄을 순서 없이 채워 넣는 생성(Gibbs 샘플링) | [원문] |
| 2018 | Towards Mixed-initiative generation of multi-channel sequential structure | ICLR 2018 워크숍 | 4명 중(마지막 Eck) | 사람과 모델이 번갈아 여러 성부를 채움 | [목록] |
| 2018 | Visualizing Music Transformer | — | 4명 중(마지막 Eck) | Music Transformer 주의(attention) 시각화 | [목록] |
| 2019 | **Music Transformer** | ICLR 2019 | 1/10 | 상대 위치 주의의 메모리를 줄여 분 단위 기호 음악 생성. 이 저장소의 출발점 | [원문] |
| 2019 | Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset | ICLR 2019 | 5/9 | MAESTRO 데이터셋, 전사 → 기호 생성 → 합성(Wave2Midi2Wave) | [원문] |
| 2019 | The Bach Doodle: Approachable music composition with machine learning at scale | ISMIR 2019 | 1/7 | Coconet을 브라우저에서 대중이 사용한 기록과 데이터 | [원문 저자] + 학회는 [목록]. arXiv 코멘트는 "18th ISMIR"로 Coconet과 같게 적혀 있어 맞지 않음 |
| 2019 | Deep Learning for Music Composition: Generation, Recommendation and Control | 하버드 박사 논문 | 단독 | 생성·추천·제어 | [목록] |
| 2020 | Novice-AI Music Co-Creation via AI-Steering Tools for Deep Generative Models | CHI 2020 | 3/5 | 초보자가 생성 모델을 조종하는 도구(Cococo) 사용자 연구 | [목록] |
| 2020 | Cococo: AI-Steering Tools for Music Novices Co-Creating with Generative Models | IUI 2020 워크숍 | 5명 중 | 위 연구의 워크숍판 | [목록] |
| 2020 | AI Song Contest: Human-AI Co-Creation in Songwriting | ISMIR 2020 | 1/5 | AI 송 콘테스트 참가 팀들의 작업 방식 분석 | [원문] |
| 2022 | MIDI-DDSP: Detailed Control of Musical Performance via Hierarchical Modeling | ICLR 2022 | 8/9 | 음표 → 연주 표현 → 합성 3단계 계층 제어 | [원문] |
| 2022 | Expressive Communication: Evaluating Developments in Generative Models and Steering Interfaces for Music Creation | IUI 2022 | 3/3 | 생성 모델과 조종 인터페이스를 "의도 전달" 기준으로 평가 | [원문] |
| 2022 | Compositional Steering of Music Transformers | IUI 2022 워크숍 | 마지막/5 | Music Transformer를 여러 속성으로 조합 조종 | [목록] |
| 2022 | Improving Source Separation by Explicitly Modeling Dependencies Between Sources | ICASSP 2022 | 3/5 | 음원 분리를 순서 없는 자기회귀(NADE) + Gibbs로 | [원문] |
| 2022 | Redefining Relationships in Music | NeurIPS 2022 워크숍 | 7/16 | 음악 AI가 창작자·청취자·산업 관계를 바꾸는 문제(입장 논문) | [원문] |
| 2023 | Grammar Generative Models for Music Notation | — | 단독 | 성격 미확인 | [목록] |
| 2024 | **Adaptive Accompaniment with ReaLchords** | ICML 2024 | 13/13 | 사람 멜로디에 실시간 코드 반주. MLE → RL(일관성 보상 + 미래를 보는 교사 증류) | [원문] |
| 2024 | Hierarchical Generative Modeling of Melodic Vocal Contours in Hindustani Classical Music (GaMaDHaNi) | ISMIR 2024 | 5/5 | 음높이 윤곽 → 오디오 2단계. 사람 연주에 반응하는 용도 | [원문] |
| 2024 | Exploratory Study Of Human-AI Interaction For Hindustani Music | NeurIPS 2024 Creative AI | 2/2 | 음악가 3명 사용자 연구. 출력 제약 부족과 비일관성이 문제 | [원문] |
| 2025 | **ReaLJam: Real-Time Human-AI Music Jamming with Reinforcement Learning-Tuned Transformers** | CHI EA 2025 | 8/8 | 실시간 잼 인터페이스와 프로토콜. 에이전트가 계획을 미리 화면에 보여 줌(anticipation) | [원문] |
| 2025 | FLAM: Frame-Wise Language-Audio Modeling | ICML 2025 | 4/8 | 프레임 단위 개방 어휘 소리 사건 위치 찾기(Adobe 공동) | [원문] |
| 2025 | Live Music Models | arXiv 2508.04651 | 13/36 | Magenta RealTime(공개)·Lyria RealTime(API) | [원문] |
| 2025 | The jam_bot, a Real-Time System for Collaborative Free Improvisation with Music Language Models | ISMIR 2025 | 6/7 | Jordan Rudess와 실시간 자유 즉흥(AMT 미세조정) | [원문 로컬 전문] |
| 2025 | **Streaming Generation for Music Accompaniment** | arXiv 2510.22105 | 8/8 | 오디오 스트리밍 반주. 미래 가시성 t_f와 청크 길이 k의 절충 | [원문] |
| 2025 | MIDI-LLM: Improving Text-to-MIDI Music Generation via Adapting Large Language Models | ISMIR 2026 | 6/6 | Llama 3.2 1B 어휘에 MIDI 토큰 추가. Hookpad 사용자 58명 실사용 평가 | [원문] |
| 2025 | **Generative Adversarial Post-Training Mitigates Reward Hacking in Live Human-AI Music Interaction (GAPT)** | ICLR 2026 | 9/9 | RL 사후학습의 다양성 붕괴를 판별자로 막음 | [원문 저자] + 학회는 이전 세션 확인 |
| 2026 | A Design Space for Live Music Agents | CHI 2026 | 9/10 | 실시간 음악 에이전트 184개를 분류한 설계 공간 | [원문] |
| 2026 | Stemphonic: All-at-once Flexible Multi-stem Music Generation | ICASSP 2026 | 4/5 | 여러 stem을 한 번에 생성(Adobe 공동) | [원문] |
| 2026 | **Latent Fourier Transform** | ICLR 2026 Oral | 2/2 | 잠재 공간에서 푸리에 변환해 시간 척도별로 음악 구조 제어("구조용 이퀄라이저") | [원문] |
| 2026 | **Live Music Diffusion Models (LMDM)** | arXiv 2605.22717 | 11/11 | 실시간 오디오 디퓨전의 효율적 미세조정·사후학습 | [원문] |
| 2026 | The Moving Drone: Negotiating Agency Between the Voice and the Virtual | NIME 2026 음악 트랙 | 3/3 | 힌두스타니 탄푸라 드론을 루퍼 + GaMaDHaNi로 움직이게 한 공연. 일부러 저품질 출력을 씀 | [원문] |
| 2026 | **H2H Music Improv: A Communication Model and Audio-Visual Dataset for Music Improvisation** | ISMIR 2026 | 5/7 | 즉흥 듀오의 소통 모델 + 시청각 데이터셋 | [원문] |
| 2026 | Encypher: Shared Agency and Social Presence in Collaborative Music Generation for Dance Cyphers | CHI 2027 심사 중 | 2/2 | 여러 사람의 움직임 → 텍스트 프롬프트 → 실시간 음악 생성 | [원문] |
| 2026 | JazzSAMBA: A Synchronous and Asynchronous Multi-take Band Audio Dataset of Jazz Standards for Live Music Models | ICASSP 2027 투고 | 10/12 | 재즈 콤보 스탠더드 76곡, 악기별 오디오·MIDI, 마디·코드·구간·솔로이스트 주석 | [원문] |

### 4.2 이 사람의 논문이 아닌 것

- Semantic Scholar 기록에 섞인 동명이인: "You Only Need Minimal RLVR Training"(LLM), "TMD-TTS"(티베트어 TTS), "Capturing Dynamic User Interests…"(추천), arXiv "Unremarkable to Remarkable AI Agent"(CSCW 2025, CMU 노인 돌봄 연구의 다른 Anna Huang) [원문 저자·초록으로 판별]
- MusicLM, SingSong, MusicRL: arXiv 저자 검색 결과에 없습니다. 이 분 논문으로 쓰지 않습니다 [원문 검색 결과]

### 4.3 자주 함께 쓰는 사람 [원문 저자 목록에서 집계, 지도 관계는 미확인]

| 사람 | 함께 쓴 것 |
|---|---|
| Yusong Wu | MIDI-DDSP, GaMaDHaNi, ReaLchords, ReaLJam, FLAM, StreamMusicGen, GAPT. `realchords-pytorch` 저장소 주인(lukewys) |
| Stephen Brade | jam_bot, StreamMusicGen, GAPT, Design Space, LMDM, JazzSAMBA, MIT 발표회 고래 노래 디퓨전 공연 |
| Nithya Shikarpur | GaMaDHaNi, Hindustani 사용자 연구, LMDM, Moving Drone, JazzSAMBA. MIT 박사과정(MIT News) |
| Shih-Lun Wu | StreamMusicGen, MIDI-LLM, Stemphonic |
| Aleksandra Teng Ma | GAPT, H2H(1저자), JazzSAMBA |
| Mason Wang | StreamMusicGen, Latent Fourier Transform(1저자) |
| Lancelot Blanchard | jam_bot(1저자), StreamMusicGen |
| Zhixing Chen | Encypher(1저자) |
| Heidi Lei | StreamMusicGen, KCL 피아노 잼 발표 |
| Natasha Jaques, Aaron Courville | ReaLchords, ReaLJam, GAPT, StreamMusicGen 등 RL·학습 쪽 공동 연구 |

MIT 발표회의 Rachel Loh "Visualizing the Internal State of Music Models for Live Human-AI Improvisation"도 우리 활성값 조종 아이디어(#1765 2순위)와 가깝습니다. 지도교수는 기사에 없습니다 [원문 MIT News].

---

## 5. 따라갈 핵심 방법

### 5.1 ReaLchords (ICML 2024) [원문 초록 + 저장소 설정]

- 문제: 기존 생성 모델은 상대 연주를 끝까지 본 뒤 만드는 오프라인 방식이라 동시에 연주할 수 없다
- 구조
  1. **온라인 디코더**를 MLE로 사전학습. 멜로디와 코드를 프레임마다 번갈아 놓은 한 시퀀스
  2. **RL 미세조정**. 보상은 여러 모델의 묶음
     - 대조(contrastive) 보상 모델 2개, 판별(discriminative) 보상 모델 2개: 화성 일관성
     - 리듬 전용 대조·판별 보상 모델 각 2개(전조 증강 없이 학습): 시간 일관성
     - 규칙 페널티: 조기 종료, 잘못된 출력, 무음, 반복
  3. **미래를 보는 교사 증류**: 멜로디 전체를 보는 인코더-디코더 모델을 기준(anchor)으로 두고 KL로 묶음
- 결과: 낯선 입력에도 잘 적응한다(정량 실험 + 청취 평가)

### 5.2 ReaLJam (CHI EA 2025) [원문 초록 + README]

- 실시간 잼에 필요한 세 가지로 낮은 지연, 계획 전달, 실시간 적응을 꼽습니다
- **anticipation**: 에이전트가 앞으로의 연주를 계속 예측하고, 그 계획을 화면에 보여 줍니다
- 조절값: Initial Beats of Silence(기본 8박), **Lookahead Beats**(얼마나 앞까지 생성할지), **Commit Beats**(미래 버퍼 중 얼마를 확정할지)
- MIDI 키보드와 컴퓨터 키보드 입력, 웹 화면. ONNX(KV 캐시) 가속, Apple Silicon MLX 백엔드

### 5.3 GAPT (ICLR 2026) [원문 초록 + 저장소 설정]

- 문제: 일관성 보상으로 RL을 하면 모델이 보상을 쉽게 받는 단조로운 출력으로 무너진다(보상 해킹). 잼에서는 특히 치명적이다
- 해법: 정책이 만든 궤적과 실제 데이터를 구분하는 **판별자를 함께 학습**(GAIL 방식). 정책은 일관성 보상과 판별자 점수를 같이 최대화
- 설정(멜로디 모델 기준): 판별자 512차원 8층, 판별자 갱신 5스텝마다, 레이블 스무딩, 반복 페널티 임계 4
- 평가: 고정 멜로디와 학습된 멜로디 에이전트로 시뮬레이션 + 전문 연주자 실시간 사용자 연구. 다양성·화성 일관성·적응 속도·사용자 주도감 개선
- 공개 평가 지표: note-in-chord 화성, 규칙 페널티, 코드 엔트로피, **Vendi score**(다양성)

### 5.4 StreamMusicGen (2025) [원문 초록]

- 시스템 지연을 전제로 두 설계 변수를 둡니다
  - **t_f(미래 가시성)**: 출력 재생 시각과 조건으로 쓴 최신 입력 시각의 차이
  - **k(청크 길이)**: 한 번 호출에 내보내는 프레임 수
- 결과: 유효 t_f가 커지면 최근 입력과의 간격이 줄어 일관성이 좋아지지만 추론이 더 빨라야 합니다. k가 커지면 처리량은 늘지만 갱신이 느려져 반주가 나빠집니다
- 단순 MLE 스트리밍 학습으로는 미래를 모르는 상황의 일관된 반주가 부족하다. 예측(anticipatory)·에이전트형 목적이 필요하다

### 5.5 Latent Fourier Transform (ICLR 2026 Oral) [원문 초록]

- 디퓨전 오토인코더의 잠재 시퀀스를 주파수 영역으로 바꾸고, 학습 때 주파수 마스킹을 합니다
- 추론 때 원하는 시간 척도(잠재 주파수 대역)만 남기거나 섞어 변주·혼합을 만듭니다
- 오디오 모델이라 바로 쓰지는 못합니다. "시간 척도별 제어"라는 생각은 기호 모델의 활성값 조종에도 옮겨 볼 수 있습니다 [판단]

### 5.6 조종과 상호작용 평가 줄기

- Cococo(CHI 2020): 초보자가 생성 모델을 예시·의미 축으로 조종 [목록]
- Expressive Communication(IUI 2022): 생성 모델과 인터페이스를 "만든 사람의 의도가 듣는 사람에게 전달되는가"로 평가 [원문 저자·제목]
- H2H(ISMIR 2026): 즉흥 듀오 소통 모델과 시청각 데이터 [원문]
- Design Space(CHI 2026): 사용 맥락·상호작용·기술·생태계 축으로 184개 시스템 분류. 웹에 주석 데이터 공개 [원문]

---

## 6. `realchords-pytorch` 저장소 — 직접 확인한 것

확인 날짜 2026-10-16. 코드는 받지 않고 GitHub API와 원본 파일 보기로만 확인했습니다.

| 항목 | 내용 |
|---|---|
| 주소 | github.com/lukewys/realchords-pytorch (Yusong Wu) |
| 라이선스 | MIT. RL 부분은 OpenRLHF v0.6.1.post1에서 가져온 코드라 Apache 2.0 |
| 생성·최근 갱신 | 2025-11-18 생성, 2026-05-11 마지막 푸시, 별 16개 |
| 포함 | ReaLchords, ReaLJam(서버·웹 화면·ONNX·MLX), GAPT, 멜로디 모델 RL |
| 체크포인트 | Hugging Face `lukewys/realchords-pytorch`에 35개 폴더. `gapt`, `realchords`, `decoder_only_online_melody_4_datasets`, `enc_dec_base_melody_4_datasets`, **`rl_melody_gail_4_datasets`**, 각 보상 모델 |
| 데이터 | Hooktheory 19,086곡, POP909 909곡, Nottingham 1,019곡, Wikifonia 약 6,000곡. 멜로디 모델 혼합 비율 0.5 / 0.2 / 0.2 / 0.1 |
| 토큰 | 박당 4프레임(16분음표 격자). 멜로디는 프레임마다 한 음: SILENCE, NOTE_p(지속), NOTE_ON_p(시작). 코드는 CHORD_이름(지속), CHORD_ON_이름(시작). 최대 길이 512 |
| 모델 | 디코더 512차원, 8층, 8헤드, dropout 0.1, bf16, 30,000스텝, 배치 64. 약 2,500만 파라미터 [판단, 설정값으로 계산] |
| 증강 | ±6반음 전조(13배) |
| RL | 단일 GPU, 48 GB VRAM 기준. rollout 배치 384, 1,000스텝, 100스텝마다 저장·평가 |
| ReaLJam | `pip install realjam`, `realjam-start-server --mlx`(Apple Silicon). 체크포인트는 실행 시 자동 다운로드 |

---

## 7. 우리 프로젝트와의 차이

| | ReaLchords 계열(멜로디 모델 기준) | 우리 |
|---|---|---|
| 사람 입력 | 코드 기호 | 실제 왼손 음(보이싱) |
| AI 출력 | 단선율 멜로디 | 오른손 피아노 솔로(화음 포함 가능) |
| 시간 | 16분음표 격자 프레임 | Aria 토큰(ms 단위 시간) + 블록 계약(4/4·128 BPM 반 마디 937.5 ms) |
| 데이터 | 리드시트 약 27,000곡 | 재즈 피아노 train 189곡 |
| 스타일 | 팝·포크 | 티그랑·비밥, 스윙 아님, 홀수 박자 포함 |
| 모델 | 약 2,500만, 처음부터 학습 | Aria 사전학습 모델 + adapter |
| 학습 | MLE → 보상 모델 → RL + 판별자 | teacher-forced CE(#1760) |
| 자원 | 48 GB GPU | Mac 로컬, 유료 GPU 없음 |

판단

- 방향(코드 → 멜로디)은 이미 풀려 있습니다. 남은 차이는 표현과 데이터입니다
- 저장소 모델을 그대로 쓰려면 왼손 보이싱을 코드 기호로 바꾸는 인식기와 오른손을 맨 윗선 하나로 줄이는 단계가 필요합니다. 둘 다 정보를 버립니다
- #1760에서 Aria + adapter는 현재 코드를 쓰지 못했습니다(b 2.77425 > c 2.76603). ReaLchords 표현은 코드 토큰이 매 프레임 같은 시퀀스 안에 들어갑니다. 이 표현이 조건 사용을 더 쉽게 만드는지는 열린 질문이고, 작은 실험 하나로 확인할 수 있습니다

---

## 8. 따라가기 계획 (제안, 아스트라와 논의)

원칙: 한 실험은 한 질문만 답합니다. 판정 기준은 실행 전에 커밋합니다. 외부 다운로드는 항목별로 승인을 받습니다.

| 순서 | 무엇 | 답할 질문 | 필요한 것 | 비용 |
|---|---|---|---|---|
| F0 | ReaLJam을 이 Mac에서 MLX로 직접 연주해 보기 | 실시간 온라인 반주 에이전트의 lookahead·commit 설계가 연주자에게 어떻게 느껴지는가. 이 Mac에서 지연은 얼마인가 | `realjam` 패키지 + 체크포인트 다운로드 **승인** | 낮음 |
| F1 | 공개 체크포인트 `rl_melody_gail_4_datasets`에 우리 dev 4곡 왼손을 코드 기호로 바꿔 넣기 | 공개된 코드→멜로디 RL 모델이 현재 코드와 틀린 코드를 구별해 따라가는가(#1760 b/c 대조와 같은 판정) | 저장소 코드 + 체크포인트 **승인**, 왼손 → 코드 기호 인식기(자체 정확도 검사 포함) | 낮음~중간 |
| F2 | 평가 지표 이식 | note-in-chord, 엔트로피, Vendi score를 우리 게이트에 추가해 기존 결과를 다시 잴 수 있는가 | 저장소 평가 코드(MIT, 출처 표기) | 낮음 |
| F3 | 우리 데이터로 레시피를 작게 재현. 첫 단계는 온라인 MLE 디코더만 | 프레임 교차 표현(왼손 코드 + 오른손 윗선)의 온라인 모델은 현재 코드를 쓰는가(b < c) | F1의 인식기, 오른손 윗선 추출. 저장소 멜로디 MLE 체크포인트에서 시작할지는 따로 결정 | 중간 |
| F4 | StreamMusicGen 축으로 블록 계약 다시 보기 | 우리 블록 계약을 (t_f, k)로 쓰면 어디에 있는가. ReaLJam의 lookahead·commit과 어떻게 대응하는가 | 문서 작업 + 기존 모델 오프라인 격자 | 낮음 |
| F5 | anticipation 표시 | 연주자가 AI의 다음 계획을 보면 함께 연주하기 쉬워지는가 | 실시간 루프(Phase 1) 이후 | 중간 |
| F6 | 상호작용 평가 | H2H 소통 모델과 Design Space 축으로 우리 시스템을 기술·평가 | Phase 3 녹음 | 낮음 |

추천 순서 [판단]: F0 → F2 → F1 → F3 첫 단계 → F4. F5·F6은 실시간 루프가 생긴 뒤에 합니다.

기존 결정과 충돌하는 점

- **증강**: ReaLchords는 ±6반음 전조 증강을 씁니다. 우리 D1 Arm E는 ×12 증강이 재붕괴를 일으켜 증강을 기각했습니다. F3은 증강 없이 하는 것을 기본으로 합니다. 증강을 쓰려면 별도 질문으로 등록합니다
- **모델 크기**: 저장소 모델은 약 2,500만 파라미터라 "모델 키우지 않는다"(D0)와 충돌하지 않습니다
- **RL 자원**: 저장소 RL은 48 GB GPU 기준입니다. Mac에서는 rollout 배치를 크게 줄여야 합니다. RL 단계는 MLE 단계에서 조건 사용이 확인된 뒤에 검토합니다
- **스타일**: 저장소 데이터는 팝·포크 리드시트입니다. F1 결과의 음악 품질은 판단하지 않습니다. 확인하는 것은 조건을 따라가는지뿐입니다

열린 질문과의 관계(CLAUDE.md §4)

- jam_bot(안나 황 공동저자)은 작은 데이터 과적합을 기능으로 씁니다. GAPT(안나 황 마지막 저자)는 RL 사후학습의 다양성 붕괴를 문제로 보고 판별자로 막습니다. 같은 연구실 안에서도 목적에 따라 다양성을 다르게 다룹니다
- "스타일 밀착과 다양성은 트레이드오프인가"라는 질문에 GAPT가 네 번째 자료가 됩니다. RL은 다양성을 줄이고, 판별자를 더하면 일관성과 다양성을 함께 올릴 수 있다는 결과입니다 [원문 초록]

---

## 9. 확인하지 못한 것

- Infilling Piano Performances(2018 NeurIPS 워크숍으로 알려짐)는 이번 검색에서 찾지 못했습니다
- 2018년 이전 논문은 Semantic Scholar 목록만 봤습니다. 요청 한도에 걸려 전체 저자 목록을 받지 못했습니다
- 박사과정 지도 관계는 확인하지 못했습니다. 4.3은 공동저자 집계입니다
- Radcliffe 펠로우, 21M.569 강의, KCL 특강은 검색 요약만 봤습니다
- GAPT의 ICLR 2026 채택은 이전 세션에서 확인했고, 이번 arXiv 코멘트에는 학회명이 없습니다
- JazzSAMBA 데이터 공개 범위와 라이선스는 확인하지 못했습니다
- `rl_melody_gail_4_datasets` 체크포인트는 돌려 보지 않았습니다
- Google Scholar는 열지 않았고, DBLP API는 막혀 있었습니다

---

## 출처

- 홈페이지: https://czhuang.github.io
- MIT News(2026-06-29): https://news.mit.edu/2026/inaugural-mit-music-technology-research-showcase-celebrates-work-students-0629
- MIT 발표회 중계: https://web.mit.edu/webcast/mta/s26/
- KCL MARC Open Day: https://marc.kcl.ac.uk/2026/05/marc-open-day/
- arXiv 저자 검색: https://export.arxiv.org/api/query?search_query=au:%22Cheng-Zhi%20Anna%20Huang%22
- Semantic Scholar 저자 기록: 7900665, 2316891503, 2392323124, 2396427430
- 저장소: https://github.com/lukewys/realchords-pytorch
- 체크포인트: https://huggingface.co/lukewys/realchords-pytorch
- 논문(arXiv): 1809.04281, 1903.07227, 1810.12247, 1907.06637, 2010.05388, 2111.14951, 2112.09312, 2203.15140, 2212.08038, 2408.12658, 2411.13846, 2502.21267, 2505.05335, 2506.14723, 2508.04651, 2510.22105, 2511.03942, 2511.17879, 2602.05064, 2602.09891, 2604.17986, 2605.22717, 2606.13640, 2608.13957, 2609.18062, 2609.34931
- jam_bot 전문: `jambot_ismir2025_fulltext.txt`(이 워크트리)
