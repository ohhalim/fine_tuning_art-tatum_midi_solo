# 음악 머신러닝·실시간 즉흥연주 선행연구 조사

- 조사 기준일: 2026-09-03
- 최신 검색 구간: 2026-01-01 ~ 2026-09-03
- 검색량: 10개 주제 축, 79개 검색 질의, 후보 결과 950건(중복 포함)
- 선정 원칙: 논문 원문, 학회 proceedings, 공식 프로젝트·코드 우선
- 핵심 범위: 즉흥연주, 실시간 협연, 연주자 스타일, symbolic MIDI, audio generation, 조건 제어, 지연, 평가

## 1. 조사 범위와 한계

“모든 음악 머신러닝 논문”은 생성 외에도 음악 검색·추천, 장르·감정 분류, 자동 채보, 소스 분리, 음원 합성, 저작권 분석 등을 포함하는 수만 편 규모의 분야다. 이 문서는 사용자의 목표에 직접 영향을 주는 연구를 다음 순서로 조사했다.

1. 실시간 인간-AI 즉흥연주 및 반주
2. 개인 연주자·작곡가 스타일 학습과 혼합
3. MIDI·symbolic music 생성
4. 오디오 스트리밍 생성과 MIDI/audio 조건 제어
5. 재즈 즉흥연주 생성 및 데이터셋
6. WAV 입력에 필요한 채보·박자·코드·소스 분리
7. 지연, 스케줄링, DAW 통합
8. 음악성·스타일·복제·사용자 경험 평가

범용 음악 ML은 위 경로에 영향을 준 계보적 연구만 포함했다. 따라서 이 문서는 “음악 ML 전체 논문 목록”이 아니라 사용자의 연구 방향 결정을 위한 체계적 선행연구 지도다.

근거 수준:

- A: 동료심사 학회·저널 원문 또는 공식 proceedings
- B: arXiv 원문 또는 연구팀 공식 기술 공개
- C: 예비 technical report·데모·프로젝트 자료. 결론의 주 근거로 단독 사용하지 않음

## 2. 먼저 바로잡을 연도

### 2.1 Music Transformer와 jam_bot은 같은 연구가 아니다

| 연도 | 연구·사건 | 확인된 내용 |
|---|---|---|
| 2018 | [Music Transformer](https://arxiv.org/abs/1809.04281) | 상대적 self-attention으로 장기 symbolic music 구조 모델링. 생성 아키텍처 연구이며 Jordan Rudess 실시간 시스템이 아님 |
| 2023/2024 | [Anticipatory Music Transformer](https://openreview.net/forum?id=EBNJ33Fcrl) | 미래 조건을 순서 재배치로 처리하는 controllable symbolic infilling 모델. jam_bot의 실제 기반 모델 |
| 2024-09-21 | [Jordan and the jam_bot 공연](https://arts.mit.edu/jordan-and-the-jam_bot/) | MIT Media Lab에서 Jordan Rudess와 첫 공개 공연 |
| 2024-09-25 | [Developing Symbiotic Virtuosity](https://doi.org/10.21428/e4baedd9.69c11de7) | 공연·악기 설계에 관한 비전 및 사례 글 |
| 2025 | [The jam_bot, ISMIR 2025](https://ismir2025program.ismir.net/poster_321.html) | 약 360M AMT를 실시간 symbolic MIDI 협연 시스템으로 바꾼 첫 정식 시스템 논문. 15~45분 MIDI를 12키 전조 증강하고 2,000 step 학습, 약 300 step부터 과적합 관측 |
| 2026 | [Enhancing Expressive Musical Conversation in the jam_bot](https://nime.org/proceedings/2026/nime2026_73.pdf) | velocity, ggml, 3.5시간 call-response 데이터, 가변 템포, MIDI 출력 지연 보상 추가 |

판정:

- `Music Transformer = 2018`은 맞다.
- `jam_bot = 2018년에 완성된 시스템`은 아니다.
- `jam_bot 2026`만 적은 표기도 부정확하다. 2024 공연, 2025 원 시스템, 2026 확장판으로 분리해야 한다.
- 2026 NIME 논문은 실제 최신 논문이다. 다만 핵심 LM 계보는 2018 Music Transformer와 2023/2024 AMT에서 왔다.

### 2.2 2026 jam_bot에서 새로 검증한 것

- 약 400M AMT에 velocity token 추가
- Jordan Rudess가 foot switch로 call/response를 표시한 약 3.5시간 MIDI 수집
- 170M·416M 모델의 ONNX Runtime과 ggml 처리량 비교
- RTX 4090 기준 416M ggml: token당 4.04ms, 247.65 token/s
- M3 Max 기준 416M ggml: token당 26.17ms, 38.21 token/s
- Ryzen 9 7950X CPU 기준 416M ggml: token당 100.17ms, 9.98 token/s
- player piano의 pitch·velocity별 물리 지연 보상: beat-alignment median F1 0.7834
- 논문이 직접 기록한 미해결점: 단일 아티스트 데이터가 다른 연주자·스타일 재사용을 제한

이 수치는 “키를 누른 뒤 한 음이 4.04ms에 완성된다”는 의미가 아니다. 모델 token 처리량이며 note당 4개 token을 쓰고, MIDI 장치·DAW·스케줄러·버퍼 지연은 별도다.

2025 원 시스템이 이미 사용한 runtime 구성:

- ONNX Runtime, 8-bit weight quantization, KV cache
- Clock / Input Capture / Processing / Generation 4개 thread와 thread-safe queue
- internal clock 또는 MIDI Timecode 입력
- external MTC offset을 완화하는 proportional control loop
- 이미 전송된 미래 note를 무효화하는 generation meta-signal

따라서 cache, thread 분리, 외부 clock 동기, queued-note invalidation 자체도 신규성이 아니다.

## 3. 결론: 사용자의 아이디어는 어디까지 선행됐는가

사용자의 원래 아이디어를 다섯 능력으로 분리한다.

| 능력 | 선행연구 상태 | 대표 근거 |
|---|---|---|
| 사용자의 연주를 듣고 즉시 응답 | 이미 오래전부터 구현 | Voyager, GenJam, Continuator, OMax, Somax2 |
| 특정 연주자의 스타일을 학습 | 이미 구현 | Continuator, jam_bot, composer-style models |
| Transformer 기반 실시간 MIDI 협연 | 이미 구현 | ReaLJam, jam_bot, StreamMUSE |
| MIDI/audio/text로 저지연 DAW 악기 제어 | 2026년 현재 매우 앞섬 | Magenta RealTime 2 |
| 여러 연주자의 동의 기반 개인 모델 + DJ 곡의 드롭·구조 문맥 + 편집 가능한 lead MIDI 출력 | 조사 범위에서 하나의 검증된 시스템으로는 미확인 | 각 구성요소는 존재하지만 통합·평가 연구가 남음 |

따라서 다음 주장은 신규성이 없다.

- “AI가 사람과 실시간 즉흥연주”
- “한 사람의 연주를 학습해 그 사람처럼 응답”
- “MIDI를 받아 MIDI 또는 음악을 실시간 생성”
- “Transformer를 DAW에 연결”

연구 가능성이 남는 주장은 더 좁다.

> 동의를 받은 여러 연주자의 소량 MIDI에서 교체 가능한 연주자 어댑터를 학습하고, 이미 존재하는 DJ 트랙의 bars-to-drop 문맥을 조건으로 사용하며, adapter 소유권과 복제 위험을 함께 검증한다.

이 문장도 아직 신규성 확정이 아니라 검증할 연구 가설이다. 편집 가능한 symbolic 출력, deadline 형식화, style classifier는 선행 방법을 재사용하는 시스템 요건이지 각각의 연구 기여가 아니다. Somax2, ImproteK, jam_bot, Composer Vector, Magenta RealTime 2를 직접 비교해야 한다.

## 4. 가장 가까운 기존 시스템 비교

| 시스템 | 입력·출력 | 스타일 적응 | 실시간 | 사용자의 목표와 겹치는 점 | 남는 차이 |
|---|---|---|---|---|---|
| [GenJam, 1994](https://genjam.org/wp-content/uploads/2019/07/bilesicmc94.pdf) | chord progression → jazz solo | 인간의 실시간 평가로 phrase 진화 | 예 | 코드 위 재즈 solo | 개인 연주 코퍼스의 신경망 학습 아님 |
| [Continuator, 2003](https://www.francoispachet.fr/wp-content/uploads/2021/01/pachet-03d.pdf) | MIDI phrase → continuation | 입력 스타일을 실시간 Markov 학습 | 예 | “내가 친 스타일의 또 다른 나”와 매우 가까움 | 장기 신경망 표현·다중 어댑터·DJ 구조 평가 없음 |
| [OMax, 2006](http://articles.ircam.fr/textes/Assayag06d/index.pdf) | live audio/video/MIDI → recombination | 연주에서 corpus를 실시간 구축 | 예 | 사용자 재료 학습, 실시간 협연 | corpus recombination 중심 |
| [ImproteK, 2017](https://hal.science/hal-01380163/document) | corpus + temporal/harmonic scenario → audio/MIDI | corpus 기반 | 예 | 계획된 화성·시간 구조와 반응성 결합 | neural performer adapter 아님 |
| [BachDuet, 2020](http://labsites.rochester.edu/air/publications/benetatos20bachduet.pdf) | human MIDI → counterpoint MIDI | Bach 분포 | 예 | MIDI human-machine duet | 16분음표 grid·monophonic 제약, 개인화 아님 |
| [RL-Duet, 2020](https://doi.org/10.1609/aaai.v34i01.5413) | online melody → accompaniment | RL 정책 | 예 | 즉시 반주 | 개인 스타일 solo가 아님 |
| [BebopNet, 2020](https://archives.ismir.net/ismir2020/paper/000132.pdf) | chord progression → monophonic jazz solo | 청취자 개인 취향에 맞춘 ranking | 아니오 | chord-constrained jazz solo | 연주자 스타일 복제가 아니라 listener preference 개인화 |
| [MINGUS, 2021](https://archives.ismir.net/ismir2021/paper/000051.pdf) | chord·bass·meter → jazz solo | 장르 모델 | 아니오 | monophonic chord-conditioned solo | 실시간·연주자 적응 없음 |
| [Somax2, 2022/2023](https://doi.org/10.3233/faia230106) | 사용자가 고른 audio/MIDI corpus + live input → co-improvisation | 임의 corpus 기반 | 예 | 목적 기능과 가장 폭넓게 중첩 | neural adapter가 아니라 corpus-based generation; 동일 평가 질문은 여전히 유효 |
| [ReaLChords, 2024](https://proceedings.mlr.press/v235/wu24c.html) | online melody → chords | 미래를 보는 teacher에서 online student로 RL/distillation | 예 | 미래를 모르는 상태의 적응형 반주 | solo·연주자 스타일 아님 |
| [ReaLJam, 2025](https://arxiv.org/html/2502.21267) | MIDI melody → chord accompaniment | ReaLChords RL | 예 | Transformer, lookahead, commit, 동기화 | 6명 연구, chord 역할 고정, 스타일 복제 아님 |
| [ImprovNet, 2025](https://arxiv.org/html/2502.04522v4) | 완성 MIDI → style-transferred MIDI | 장르·구조 제어 | 아니오 | 입력곡을 jazz화·변형 | offline iterative refinement, live deadline 없음 |
| [jam_bot, 2025](https://zenodo.org/records/17706584) | live MIDI ↔ symbolic MIDI | Jordan의 15~45분 MIDI를 12키 전조, 약 360M AMT를 2,000 step FT; 약 300 step부터 과적합 | 예 | ONNX, 8-bit, KV cache, 4 thread, MTC 동기, 미래 note 무효화 | 단일 아티스트, 별도 모델, 데이터 비공개, planned improvisation 미해결 |
| [jam_bot 확장, 2026](https://nime.org/proceedings/2026/nime2026_73.pdf) | foot-switch call → MIDI response | Jordan 3.5시간 call-response | 예 | 표현·가변 템포·지연 보상 | 논문이 직접 다중 연주자 재사용 한계 인정 |
| [Magenta RealTime 2, 2026](https://magenta.withgoogle.com/magenta-realtime-2) | MIDI·text·audio prompt → streaming audio | style prompt; 사용자 fine-tuning은 향후 기능으로 표기 | 예 | DAW plugin, 40ms frame, 약 200ms control latency | 출력이 MIDI가 아닌 audio; 현재 공개판 개인 연주자 FT 미완; live audio input도 향후 범위 |
| [StreamMUSE, 2026](https://arxiv.org/html/2606.11886v1) | melody frames → symbolic accompaniment | 일반 accompaniment model | 예 | 200ms 이하 frame-synchronous scheduling 연구 | 개인 solo adapter·DJ transition 아님 |
| [LiveBand, 2026](https://arxiv.org/html/2606.03803v1) | live audio mix → audio accompaniment | 일반 multi-instrument | 예 | causal, 무 lookahead audio accompaniment | MIDI 편집성·개인 연주자 solo 없음 |
| [LMDM, 2026](https://arxiv.org/html/2605.22717v1) | recent audio/sketch/text → future audio | diffusion fine-tuning | 예 | consumer laptop, live generative delay | 약 1초 지연, audio 변형, jazz·note-level harmony 약함 |
| [LDM+MAX/MSP, 2026](https://arxiv.org/html/2604.07612v1) | live audio → instrumental audio stem | 일반 stem model | 예 | sliding lookahead와 지연-품질 trade-off | personal style·symbolic output 없음 |
| [LK_Jam, 2026](https://arxiv.org/html/2606.21018v1) | MIDI → role-aware MIDI | 단계별 학습 주장 | 예 | JUCE/GRU/RTNeural, VST 구조 | C 등급 preliminary report; 정량 음악성·사용자 검증 부족 |

## 5. 지연과 끊김: 현재 선행연구가 실제로 보여준 것

| 시스템 | 확인된 시간 단위 | 해석 |
|---|---|---|
| Continuator | real-time Markov continuation | 작은 통계 모델은 즉시 응답 가능하지만 장기·생성 다양성과 별개 |
| ReaLJam | 단일 장치에서 응답 다수가 100ms 이내 | 150 BPM의 16분음표 frame에 맞추며 lookahead·commit으로 늦은 응답을 가림 |
| jam_bot 2025/2026 | 2025: ONNX·8-bit·KV cache·MTC control loop; 2026: 416M/RTX4090 4.04ms/token, 120-token context | 전체 키 입력→소리 지연과 동일한 지표는 아님 |
| Magenta RT v1 | 2초 frame, control 약 3초 | 연속 재생은 가능하지만 즉흥 대화에는 느린 축 |
| Magenta RT2 | 40ms frame, control 약 200ms | 현재 공개 시스템 중 저지연 audio 생성의 강한 기준. Apple Silicon 실시간 지원 |
| LMDM | block + inference 고정 지연, Jamendo 모델 약 1초 | gapless지만 즉각적 note 대화는 아님; 논문도 sub-second 개선 필요 기록 |
| StreamMUSE | 목표 frame 200ms 이내 | 요청 주기·생성 길이·tempo·RTT의 feasible region을 시스템 문제로 모델링 |

실시간의 최소 정의를 분리해야 한다.

1. 처리량: 1초 분량을 1초보다 빨리 생성하는가
2. 반응 지연: 새 입력이 출력에 반영되기까지 몇 ms인가
3. deadline miss: 예정된 note가 늦거나 비는 비율
4. jitter: note-on 실제 시간과 예정 시간의 편차
5. 안정성: 10~30분 연속 실행에서 drop·stuck note·crash가 없는가

현재 프로젝트의 256-token 약 9.4~9.8초, 512-token 약 31.3초 측정은 실시간 불가 근거다. 하지만 모델 크기만의 문제가 아니라 매 token마다 full prefix를 다시 계산하는 현재 구현의 영향이 크다. Jam_bot과 MRT2 모두 cache·resident native runtime·ahead scheduling을 사용한다.

## 6. 시대별 핵심 연구 계보

### 6.1 규칙·확률·초기 상호작용

| 연도 | 연구 | 전환점 |
|---|---|---|
| 1957 | [ILLIAC Suite](https://distributedmuseum.illinois.edu/exhibit/illiac-suite/) | 규칙과 확률을 이용한 초기 컴퓨터 작곡 |
| 1984 | [The Synthetic Performer](https://web.media.mit.edu/~bv/papers/synthetic%20performer.pdf) | 연주를 추적해 동반하는 score-following performer |
| 1987~2000 | [Voyager](https://doi.org/10.1162/096112100570585) | George Lewis의 자율적 실시간 improvising orchestra |
| 1989 | [Experiments in Musical Intelligence](https://doi.org/10.1080/09298218908570541) | 작품 corpus에서 작곡가 style을 재조합 |
| 1993 | [Interactive Music Systems](https://wp.nyu.edu/robert_rowe/2016/01/07/interactive-music-systems-mit-press-1993-online/) | machine listening/composing 시스템 분류 |
| 1994 | [GenJam](https://genjam.org/wp-content/uploads/2019/07/bilesicmc94.pdf) | chord changes 위 재즈 solo를 유전 알고리즘과 인간 평가로 학습 |
| 2003 | [Continuator](https://doi.org/10.1076/jnmr.32.3.333.16861) | 임의 MIDI 스타일을 실시간 학습해 연속·응답 |
| 2004~2006 | [Factor Oracle/OMax](http://articles.ircam.fr/textes/Assayag06d/index.pdf) | live corpus의 variable-order sequence를 실시간 재조합 |
| 2012~2017 | [ImproteK](https://doi.org/10.1145/3022635) | OMax 계열에 harmony·temporal scenario와 anticipatory scheduling 추가 |

### 6.2 신경망 symbolic music

| 연도 | 연구 | 전환점 |
|---|---|---|
| 2002 | [LSTM music composition](https://www.idsia.ch/~juergen/blues/IDSIA-07-02.pdf) | 장기 의존성을 순환신경망으로 모델링 |
| 2012 | [RNN-RBM](https://arxiv.org/abs/1206.6392) | polyphonic sequence의 시간·동시음 결합 |
| 2016 | [MelodyRNN](https://github.com/magenta/magenta/tree/main/magenta/models/melody_rnn) | Magenta의 접근 가능한 melody generation 도구 |
| 2017 | [PerformanceRNN](https://magenta.withgoogle.com/performance-rnn) | expressive timing·velocity performance event 모델링 |
| 2017 | [Deep Music / AI Duet](https://doi.org/10.1609/aaai.v31i1.10544) | neural call-response 피아노 인터페이스 |
| 2018 | [Music Transformer](https://arxiv.org/abs/1809.04281) | 상대적 attention과 장기 piano generation |
| 2018 | [MusicVAE](https://arxiv.org/abs/1803.05428) | 장기 구조의 hierarchical latent model |
| 2020 | [REMI / Pop Music Transformer](https://arxiv.org/abs/2002.00212) | bar·position·tempo·chord 중심 beat representation |
| 2021 | [Compound Word Transformer](https://doi.org/10.1609/aaai.v35i1.16091) | 한 musical event의 속성을 묶어 token 효율 향상 |
| 2021 | [MusicBERT](https://doi.org/10.18653/v1/2021.findings-acl.70) | OctupleMIDI 기반 대규모 symbolic pretraining |
| 2022/2023 | [FIGARO](https://arxiv.org/abs/2201.10936) | expert·learned description을 이용한 fine-grained control |
| 2023/2024 | [Anticipatory Music Transformer](https://arxiv.org/abs/2306.08620) | 미래 constraint를 만족하는 arbitrary-order infilling |
| 2023 | [Polyffusion](https://arxiv.org/abs/2307.10304) | polyphonic piano-roll diffusion과 내부·외부 control |
| 2023/2025 | [GETMusic](https://arxiv.org/abs/2305.10841) | 임의 track 조합을 조건화하는 unified diffusion |
| 2026 | [Composer Vector](https://arxiv.org/abs/2604.03333) | 재학습 없이 latent direction으로 작곡가 스타일 강도·혼합 제어 |
| 2026 | [Chord-Symbol Time-Series Adaptation](https://arxiv.org/abs/2606.07334) | 11개 장르·5개 적응법 비교. LoRA 출력은 target 분포에 가까워졌지만 10/11 장르에서 고유 chord와 entropy 감소 |
| 2026 | [Diff-Symbo](https://arxiv.org/abs/2608.05222) | text-controlled long symbolic generation의 autoregressive latent diffusion |
| 2026 | [Agogic](https://arxiv.org/abs/2608.03999) | score time이 아닌 performance-time symbolic token 강조 |

### 6.3 재즈 즉흥연주 생성·분석

| 연도 | 연구 | 사용자의 목표와 관계 |
|---|---|---|
| 1994 | [GenJam](https://genjam.org/wp-content/uploads/2019/07/bilesicmc94.pdf) | chord progression 위 solo 생성 |
| 2018 | [JazzGAN](https://musicalmetacreation.org/mume2018/proceedings/Trieu.pdf) | GAN 기반 jazz improvisation |
| 2019 | [Transfer learning for jazz melody](https://arxiv.org/abs/1908.09484) | 적은 jazz data를 위한 transfer learning |
| 2020 | [BebopNet](https://archives.ismir.net/ismir2020/paper/000132.pdf) | monophonic harmony-constrained solo와 listener preference personalization |
| 2020 | [Jazz Transformer](https://arxiv.org/abs/2008.01307) | chord에 맞춘 jazz generation과 attention 해석의 한계 |
| 2021 | [MINGUS](https://archives.ismir.net/ismir2021/paper/000051.pdf) | chord·bass·meter 조건의 monophonic jazz solo |
| 2022 | [Analysis-by-Synthesis jazz model](https://transactions.ismir.net/articles/10.5334/tismir.87) | 생성모델을 인간 improvisation 인지 이론과 평가에 연결 |
| 2023 | [PiJAMA](https://transactions.ismir.net/articles/10.5334/tismir.162) | 2,777 solo-piano performances, 120 pianists, 자동 MIDI 채보 |
| 2023 | [JAZZVAR](https://arxiv.org/abs/2307.09670) | 동일 standard의 여러 solo-piano 변형 쌍 |
| 2024 | [Charlie Parker Aligned Digital Omnibook](https://arxiv.org/abs/2405.16687) | 고품질 monophonic solo·audio alignment |
| 2025/2026 | [Machine learning of artistic fingerprints in jazz](https://www.nature.com/articles/s42256-026-01279-9) | 84시간, 20 pianists, melody·harmony·rhythm·dynamics 분리, 94% performer identification |
| 2025 | [ImprovNet](https://arxiv.org/abs/2502.04522) | complete symbolic work의 controllable genre improvisation |
| 2026 | [Audio-to-Score Jazz Solo Transcription](https://doi.org/10.1109/icassp55912.2026.11461977) | WAV 재즈 solo를 note/score로 바꾸는 최신 입력 경로 |

### 6.4 실시간 neural 협연

| 연도 | 연구 | 전환점 |
|---|---|---|
| 2020 | [BachDuet](http://labsites.rochester.edu/air/publications/benetatos20bachduet.pdf) | RNN 기반 real-time counterpoint MIDI duet |
| 2020 | [RL-Duet](https://arxiv.org/abs/2002.03082) | online accompaniment를 sequential decision/RL로 구성 |
| 2021 | [RNN real-time jazz accompaniment adaptability](https://doi.org/10.3389/frai.2020.508727) | 처음 보는 jazz solo 입력에 대한 적응성 분석 |
| 2022/2023 | [Somax2](https://doi.org/10.3233/faia230106) | audio/MIDI corpus, machine listening, multi-agent co-improvisation |
| 2024 | [ReaLChords](https://proceedings.mlr.press/v235/wu24c.html) | 미래 melody를 못 보는 online model을 RL/distillation으로 보정 |
| 2025 | [ReaLJam](https://arxiv.org/abs/2502.21267) | lookahead·commit·visual plan·100ms 수준 request loop |
| 2025 | [jam_bot](https://zenodo.org/records/17706584) | AMT+ONNX+JUCE+multi-thread로 artist-specific free improvisation 공연 |
| 2026 | [jam_bot expressive extension](https://nime.org/proceedings/2026/nime2026_73.pdf) | velocity, call-response training, ggml, output-device latency compensation |
| 2026 | [StreamMUSE](https://arxiv.org/abs/2606.11886) | frame-synchronous LM inference와 tempo/RTT/scheduling 관계 |
| 2026 | [LK_Jam](https://arxiv.org/abs/2606.21018) | lightweight GRU, RTNeural, JUCE plugin technical report |
| 2026 | [A Design Space for Live Music Agents](https://arxiv.org/abs/2602.05064) | 학술논문·영상의 live music agents 184개를 분류한 가장 넓은 지도 |
| 2026 | [OTIAC](https://nime.org/proceedings/2026/nime2026_33.pdf) | SOM+Factor Oracle를 feedback guitar에 통합한 online agent |
| 2026 | [H2H Music Improv](https://arxiv.org/abs/2608.13957) | 인간 듀오 즉흥의 소통 모델과 audio-visual dataset. 생성기는 아님 |

### 6.5 raw-audio 생성·연주 인터페이스

| 연도 | 연구 | 전환점 |
|---|---|---|
| 2016 | [WaveNet](https://arxiv.org/abs/1609.03499) | raw waveform autoregression |
| 2018 | [SampleRNN music at scale](https://papers.neurips.cc/paper/8023-the-challenge-of-realistic-music-generation-modelling-raw-audio-at-scale) | 긴 raw audio generation의 계산 문제 |
| 2020 | [Jukebox](https://arxiv.org/abs/2005.00341) | hierarchical audio tokens로 장르·아티스트·가사 조건 노래 생성 |
| 2021 | [RAVE](https://arxiv.org/abs/2111.05011) | consumer hardware real-time neural audio synthesis/timbre transfer |
| 2022 | [AudioLM](https://arxiv.org/abs/2209.03143) | semantic·acoustic token을 결합한 장기 audio continuation |
| 2023 | [MusicLM](https://arxiv.org/abs/2301.11325) | text-to-music hierarchical audio generation |
| 2023 | [MusicGen](https://arxiv.org/abs/2306.05284) | 단일-stage controllable audio token LM |
| 2024 | [Music ControlNet](https://doi.org/10.1109/TASLP.2024.3399026) | time-varying melody·rhythm·dynamics control |
| 2024 | [JASCO](https://arxiv.org/abs/2406.10970) | audio와 chord/drum/melody symbolic 조건을 함께 사용 |
| 2025 | [Live Music Models / Magenta RT](https://arxiv.org/abs/2508.04651) | 2초 chunk의 continuous audio와 text/audio style control |
| 2026 | [Magenta RealTime 2](https://magenta.withgoogle.com/magenta-realtime-2) | 40ms frame, 약 200ms control, MIDI/text/audio prompt, DAW plugin |
| 2026 | [LMDM](https://arxiv.org/abs/2605.22717) | diffusion block의 KV cache, ARC-Forcing, 약 1초 generative delay |
| 2026 | [LiveBand](https://arxiv.org/abs/2606.03803) | 미래 입력 없이 causal live audio accompaniment |
| 2026 | [LDM+MAX/MSP](https://arxiv.org/abs/2604.07612) | consistency distillation과 sliding lookahead, 5.4배 sampling speedup |
| 2026 | [Streaming Consistency Distillation](https://arxiv.org/abs/2606.24307) | offline text-to-music diffusion을 single-step streaming instrument로 변환 |
| 2026 | [PLAUD](https://arxiv.org/abs/2608.13724) | 작은 개인 sound corpus의 DDSP/latent Max for Live 악기 |

## 7. WAV 입력에 필요한 인접 연구

WAV를 “들려주면 맞춰 연주”하려면 audio generator보다 먼저 문맥 추출의 실패 모드를 정의해야 한다.

| 하위 문제 | 대표 연구 | 현재 의미 |
|---|---|---|
| piano audio → MIDI | [Onsets and Frames](https://arxiv.org/abs/1710.11153) | PiJAMA 같은 자동 채보 기반 |
| multi-instrument transcription | [MT3](https://arxiv.org/abs/2111.03017) | 여러 악기의 note sequence 추출 |
| jazz audio → score | [Rhythm Perceiver, 2026](https://doi.org/10.1109/icassp55912.2026.11461977) | swing·notation을 고려한 최신 재즈 채보 축 |
| beat/downbeat | [Beat Transformer](https://arxiv.org/abs/2209.07140), [Beat This!](https://arxiv.org/abs/2407.21658) | 마디 경계와 drop deadline 추정 |
| online beat | [Real-Time PLP](https://audiolabs-erlangen.de/resources/MIR/2024-TISMIR-RealTimePLP) | live mode로 확장할 때 필요한 causal clock |
| online chord | [Latency-controlled ACR, 2025](https://smc25.iem.sh/contributions/latency-controlled-chord-recognition/) | chord 정확도와 lookahead latency trade-off |
| source separation | [Spleeter](https://doi.org/10.21105/joss.02154), [Demucs](https://arxiv.org/abs/1909.01174) | bass/drums/harmony stem 분석 전처리 |
| jazz trio separated data | [Jazz Trio Database](https://transactions.ismir.net/articles/10.5334/tismir.186) | 피아노·bass·drums 문맥 분리 연구 |

첫 버전은 live WAV가 아니라 전체 파일 사전분석이 타당하다. 전체 곡을 볼 수 있으므로 beat·chord·section 오류를 사람이 교정하고, 실시간 경로에는 검증된 timeline만 전달할 수 있다. Magenta RT2도 2026-06 공식 페이지에서 live audio streaming input을 향후 확장으로 적고 있다.

## 8. 스타일 학습에서 반드시 구분할 것

“스타일”은 한 단어가 아니다.

1. 음색: 어떤 피아노·신스처럼 들리는가
2. 작곡가 style: harmony·form·motif 분포
3. 연주자 style: timing·dynamics·articulation·voicing·line choice
4. 장르 style: bebop, modal, baroque, EDM 등의 집단 분포
5. 개인 취향: 어떤 출력을 특정 청취자가 선호하는가

BebopNet의 personalization은 5번이다. jam_bot과 사용자의 목표는 주로 3번이다. Magenta RT2의 audio/text style prompt는 1번과 4번 비중이 크다. Composer Vector는 주로 2번이다. 이들을 같은 “개인화”로 묶으면 선행연구 비교가 무너진다.

2026 Nature Machine Intelligence의 [artistic fingerprints](https://www.nature.com/articles/s42256-026-01279-9) 연구는 20명 jazz pianist를 melody·harmony·rhythm·dynamics 네 domain으로 분리해 94%로 식별했다. 이는 생성 SOTA가 아니라 스타일 검증 도구의 강한 baseline이다.

## 9. PiJAMA를 쓸 때의 정확한 연구 경계

- 장점: 2,777 solo-piano performances, 120 pianists, 약 219시간 규모
- 장점: 연주자별 분류·스타일 분석·generation benchmark에 적합
- 한계: 실제 연주 MIDI가 아니라 audio에서 자동 채보한 MIDI
- 한계: full solo-piano texture와 사용자의 Serum용 monophonic lead target이 다름
- 한계: 같은 녹음의 두 자동 채보본은 독립 표본이 아님
- 위험: 랜덤 window split은 같은 곡·앨범·녹음의 근접 구간이 train/test에 섞여 스타일 성능을 부풀릴 수 있음
- 권장 split: performer 내 album/recording 단위 분리 + target performer를 base pretraining에서 제외한 실험 병행
- 권장 target: lead extraction 규칙을 고정하고 원 full-piano와 추출 lead 양쪽의 style signal 손실 측정
- 권장 강건성: 서로 다른 transcription으로 결론 방향 재현 여부 확인

Charlie Parker·Coltrane 같은 관악기 연주자는 PiJAMA의 pianist axis와 다르다. [Weimar Jazz Database/Jazzomat](https://jazzomat.hfm-weimar.de/)와 [Charlie Parker Omnibook](https://arxiv.org/abs/2405.16687)을 별도 monophonic corpus로 다뤄야 한다.

## 10. 추천 연구 문제

### 10.1 추천 제목

`동의 기반 다중 연주자 어댑터와 deadline-aware DJ transition 문맥을 이용한 실시간 symbolic solo generation`

### 10.2 가설

- H1: 단일 base model + 교체형 adapter가 연주자별 full fine-tuned model보다 스타일 분리도와 다양성을 유지하면서 저장·전환 비용 감소
  - 반대 근거: chord-symbol 적응 연구에서 LoRA는 target 분포 정렬을 개선했지만 10/11 장르의 고유 chord 수와 entropy를 감소시켰다. chord-symbol·장르와 note-level·연주자라는 조건 차이를 포함해 직접 재검증한다.
- H2: chord만 주는 조건보다 chord + section + bars-to-drop + energy trajectory가 DJ transition에서 phrase landing과 구조 적합도 개선
- H3: 직접 audio 생성보다 symbolic lead MIDI 생성이 동일 hardware에서 더 낮은 deadline miss와 더 높은 편집 가능성 제공
- H4: 스타일 adapter가 단순 nearest-neighbor/corpus recombination보다 새 문맥에 일반화하면서 training phrase 복제율 제한
- H5: 동일 PiJAMA recording의 full-piano 표현과 추출 lead 표현을 비교했을 때 performer identification signal이 사전 등록 허용치 안에서 보존되는지는 미확정

Weimar와 Charlie Parker Omnibook은 고품질 monophonic pipeline sanity check로 사용할 수
있지만, PiJAMA와 동일 연주자·동일 악기 paired reference가 아니므로 직접적인 identity-retention
참조군으로 간주하지 않는다.

### 10.3 비교 baseline

최소 비교 없이 신규성을 주장하면 안 된다.

1. chord-scale/rule baseline
2. nearest-neighbor phrase retrieval
3. 실행 가능한 Somax2 또는 Factor Oracle/OMax 계열 corpus recombination
4. base symbolic LM without adapter
5. target performer full fine-tuning
6. target performer LoRA/adapter
7. shuffled performer-label control
8. 가능하면 Composer Vector 계열 inference steering

Jam_bot은 공개 데이터가 없어 동일 조건 재학습 비교가 어렵다. 대신 interaction protocol·latency·artist count·model count·planned context를 시스템 수준에서 비교한다.

### 10.4 평가 축

| 축 | 지표·검증 |
|---|---|
| MIDI 유효성 | orphan note, stuck note, note overlap, range |
| 문맥 적합 | strong-beat chord role, phrase landing, tension/release, bars-to-drop alignment |
| 스타일 | held-out performer classifier; melody/harmony/rhythm/dynamics domain별 confusion |
| 일반화 | unseen song·album·key·tempo·chord progression |
| 다양성 | distinct n-gram, pitch/rhythm entropy, mode-collapse rate |
| 비복제 | symbolic fingerprint, longest matching subsequence, nearest training-window similarity, piano-roll SSIMuse |
| 실시간 | input-to-context p50/p95/p99, generation deadline miss, output jitter, underrun, 30분 안정성 |
| 인간 평가 | blind A/B: style identity, 곡 적합, transition usefulness, playability |
| 소유권 | performer consent, adapter ownership, delete/export provenance |

비복제 위반 기준은 생성 결과를 본 뒤 정하지 않는다. train/held-out real-performance의
overlap 분포로 임계값을 먼저 고정하되, 생성물과 동일한 lead-extraction 규칙을 train과
held-out real recording 모두에 적용한다. generated-to-train similarity가 그 상한을 넘는
비율을 보고한다. 비교 기준은 복제를 목적으로 하는 retrieval이 아니라 base LM이다. 1마디 단위
symbolic replication에는 [SSIMuse](https://arxiv.org/abs/2509.13658), split leakage에는
[Lakh MIDI de-duplication](https://ismir2025program.ismir.net/poster_188.html)을 참고한다.
SSIMuse의 grid·window parameter는 원문과 공개 구현을 확인한 뒤 preregistration에서 확정한다.

### 10.5 첫 실험 범위

새 모델을 먼저 크게 학습하지 않는다.

1. direct MIDI transport와 scheduler의 독립 기술 검증
2. 3명 pianist, 동일 데이터 시간 상한
3. album-held-out split
4. retrieval vs base vs LoRA vs full FT
5. style classifier + 복제율 + 곡 문맥 평가
6. 승자 한 개만 resident model runtime으로 이동
7. 마지막에 FL Studio virtual MIDI → Serum 출력 연결

transport/scheduler를 먼저 검증하는 이유는 model 품질과 무관한 악기 안전성·deadline 측정
경계를 확보하기 위해서다. 학습 실험에서는 “교체형 개인 style adapter가 corpus
재조합·full FT보다 실제로 무엇을 얻는가”를 먼저 판정하고 승자만 resident runtime에 연결한다.

## 11. 연구하지 말아야 할 주장

- “최초의 AI 재즈 즉흥연주”
- “최초의 실시간 human-AI jam”
- “최초의 개인 스타일 학습”
- “Music Transformer를 처음 공연에 사용”
- “오디오 SOTA보다 빠르므로 더 좋은 음악 모델”
- objective proxy만으로 “Keith Jarrett처럼 들린다” 주장
- PiJAMA의 자동 채보 오류를 실제 연주자의 의도로 간주
- 한 연주자의 train/test window 분할로 unseen-style generalization 주장

## 12. 최종 판단

사용자의 프로젝트를 중단할 이유는 없다. 그러나 기존 설계의 연구 주장 그대로는 약하다.

- 제품·악기 가치: 있음. 사용자가 실제 DJ set과 VST workflow에서 쓰는 편집 가능한 MIDI 도구는 별도 가치가 있음.
- 컨셉 신규성: 낮음. personalized real-time improviser 자체는 Continuator·Somax2·jam_bot이 선행.
- 연구 신규성 후보: 중간. 다중 연주자 어댑터, consent/ownership, DJ transition deadline, symbolic editability, 복제 방지, 통합 평가의 교집합.
- 가장 강한 상대: corpus 관점의 Somax2/ImproteK, symbolic LM 관점의 jam_bot/StreamMUSE, audio·DAW 관점의 Magenta RT2.
- 다음 문서 작업: Issue #1472 설계를 이 조사에 맞춰 수정. 기존 `direct MIDI` 결정은 유지 가능하지만, 차별점·baseline·style 정의·deadline metric을 다시 작성해야 함.

## 13. 핵심 survey·지도

- [A Design Space for Live Music Agents, 2026](https://arxiv.org/abs/2602.05064): 학술논문과 영상의 184개 시스템 분류
- [A Survey on Music Generation from Single/Cross/Multi-Modal Perspectives, 2026](https://doi.org/10.1145/3800682)
- [Recent Advances in Music Generation, 2026](https://doi.org/10.1007/s10462-026-11582-x)
- [Survey on the Evaluation of Generative Models in Music, 2025](https://arxiv.org/abs/2506.05104)
- [A Survey of Music Generation in the Context of Interaction, 2024](https://arxiv.org/abs/2402.15294)
- [A Survey on Deep Learning for Symbolic Music Generation, 2023](https://doi.org/10.1145/3597493)
- [A Comprehensive Survey on Deep Music Generation, 2020](https://arxiv.org/abs/2011.06801)
- [Computational Systems for Music Improvisation, 2018](https://doi.org/10.1080/14626268.2018.1426613)
- [Musical Agents: A Typology and State of the Art, 2018](https://doi.org/10.1080/09298215.2018.1511736)

## 14. 현재 확인이 더 필요한 항목

- Magenta RT2 공개판 supervised fine-tuning의 실제 배포 시점. 2026-06 공식 페이지·현재 README에는 향후 기능으로 표기
- Composer Vector의 jazz performer와 monophonic lead에 대한 재현성. 논문은 composer-style symbolic generation 중심
- LK_Jam의 정량 latency·음악성·사용자 평가. 현재는 preliminary technical report 성격
- PiJAMA 자동 채보에서 pianist별 lead extraction 후 identity signal 보존 정도
- DJ transition의 bars-to-drop·energy trajectory를 조건으로 한 직접 선행연구 추가 검색
- 고인이 된 연주자 데이터의 저작권·실연자권·퍼블리시티권은 국가별 법률 검토 필요
