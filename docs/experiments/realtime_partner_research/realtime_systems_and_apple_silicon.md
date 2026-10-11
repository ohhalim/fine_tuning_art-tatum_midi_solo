# 실시간 인터랙티브 심볼릭 음악 시스템의 지연(latency) 엔지니어링 + Apple Silicon(MLX/MPS) 추론·학습 근거

조사일: 2026-10-11. 범위: 2016–2026.

태그 규칙:
- [직접 확인]: 이 세션에서 원문(주로 GitHub 코드/README/데이터)을 직접 열어 확인함.
- [요약만]: 검색엔진 스니펫이나 2차 요약만 봤고 원문 PDF는 열지 못함.
- "미확인": 출처가 있는 수치를 찾지 못함.

접근 환경 메모(모든 Gaps에 공통): 이 세션의 egress proxy가 arxiv.org, export.arxiv.org, nime.org, zenodo.org, media.mit.edu, semanticscholar.org, huggingface.co, alphaxiv.org, openreview.net, dl.acm.org, springer, frontiersin, ismir 사이트, *.github.io를 모두 차단했다(WebFetch DNS 실패, curl CONNECT 403). 열린 곳은 github.com(WebFetch와 git clone)과 pypi.org뿐이다. 그래서 논문 본문의 수치는 대부분 [요약만]이고, 코드로 확인할 수 있는 부분만 [직접 확인]이다. 세션 끝 무렵에는 WebSearch 예산(200회/턴, 에이전트 공유)도 소진됐다.

---

## 1. jam_bot (ISMIR 2025) 및 NIME 2026 후속 "Enhancing Expressive Musical Conversation in the jam_bot"

### Takeaway
jam_bot은 MIT Media Lab의 시스템이다. music LM을 역할(lead/accompany/call-and-response)별 context와 conditioning으로 바꿔 쓰고, "listen → prompt → schedule"을 다중 스레드로 돌린다. NIME 2026 후속에서 velocity 모델링, ggml 포팅, call-and-response 학습, 외부 MIDI 출력 지연 보상이 추가됐고, Ryzen 9 7950X(CPU), RTX 4090(CUDA), M3 Max(Metal)에서 tokens/s와 ms/token을 측정했다. 다만 실제 수치와 100 ms / 800 ms 목표치는 원문을 열 수 없어 확인하지 못했다(미확인).

### Cited Findings
- ISMIR 2025 논문: Blanchard, Naseck, Brade, Lecamwasam, Rudess, Huang, Paradiso. 제목은 "The jam_bot, a Real-Time System for Collaborative Free Improvisation with Music Language Models", Proc. 26th ISMIR. 페이지는 출처마다 다르다. MIT 페이지는 755–762쪽, Zenodo 기록은 769–776쪽이다. [요약만] — [ISMIR 2025 poster P7-3](https://ismir2025program.ismir.net/poster_321.html); [MIT Media Lab publication](https://www.media.mit.edu/publications/the-jam_bot-a-real-time-system-for-collaborative-free-improvisation-with-music-language-models/); [Zenodo 17706584](https://zenodo.org/records/17706584)
- 초록은 세 축으로 구성된다. (1) context와 conditioning signal을 바꿔 music LM이 다른 음악적 역할을 맡게 하는 방법, (2) 미세조정 데이터를 위해 Rudess가 즉흥을 구조화한 방식, (3) music LM을 실시간으로 돌리기 위한 최적화와, 듣고(listen) 프롬프트를 만들고(prompt) 생성을 스케줄링하는(schedule) low-latency multi-threaded system에 넣는 방법. [요약만] — [MIT Media Lab publication](https://www.media.mit.edu/publications/the-jam_bot-a-real-time-system-for-collaborative-free-improvisation-with-music-language-models/)
- 학회 발표 요약에는 역할이 lead, accompany, call-and-response로 나열돼 있다. [요약만, 2차 요약] — [ISMIR 2025 poster P7-3](https://ismir2025program.ismir.net/poster_321.html)
- 베이스 모델 서술이 출처마다 다르다. MIT 기사는 "Anna Huang이 개발한 Music Transformer를 출발점으로" 했고 Rudess의 연주(베이스라인~코드~멜로디)로 미세조정했다고 쓴다. MIT 프로젝트 페이지는 학술 전사 데이터셋으로 사전학습한 뒤 Rudess가 이 프로젝트를 위해 녹음한 연주로 미세조정했다고 쓴다. NIME 2026 본문 스니펫은 "expert dataset of MIDI sequences ... annotated calls and responses, and use it to finetune an AMT model"이라고 한다. [요약만] — [MIT "A model of virtuosity"](https://www.media.mit.edu/articles/a-model-of-virtuosity/); [MIT project overview](https://www.media.mit.edu/projects/jordan-rudess-genai/overview/); [NIME 2026 PDF](https://nime.org/proceedings/2026/nime2026_73.pdf)
- 라이브 입력은 Rudess와 Bäckman 악기의 live audio와 MIDI 메시지이고, 이를 바탕으로 언제·어떻게 합류할지 결정한다. 출력은 MIDI이고 MIDI controller로 조작한다. 첫 공개 공연은 2024-09-21 MIT Media Lab이었다. ISMIR 2025 Music session M5에서 재연됐고, 2025 MIDI Innovation Award를 받았다. [요약만] — [MIDI.org jambot at ISMIR 2025](https://midi.org/the-jambot-at-ismir-2025); [MIDI.org Rudess & jam_bot](https://midi.org/jordan-rudess-and-mits-jam_bot-a-new-era-of-human-ai-musical-collaboration)
- NIME 2026 후속: Blanchard, Naseck, Liang, Tan, Lei, Huang, Paradiso. NIME 2026 Article 73(621–626쪽), London, 2026-06-26 "Papers 9A" 세션. DOI는 10.5281/zenodo.20784228이다. 개선은 네 가지다. (1) velocity 모델링, (2) ggml 구현으로 throughput 향상(velocity 때문에 시퀀스가 길어져서 필요), (3) 여러 템포에 걸친 free-form call-and-response 학습 모드, (4) 외부 MIDI 출력 지연 보상(연주자와 리듬 정합 유지). throughput 향상과 지연 보상을 정량 평가했고 MIDI 샘플을 공개했다고 한다. [요약만] — [NIME proc page](https://nime.org/proc/nime2026_73/); [NIME web archive 507](https://nime.org/web_archive/2026/proceedings/507.html); [MIT publication](https://www.media.mit.edu/publications/enhancing-expressive-musical-conversation-in-the-jam_bot/); [session 9A](https://nime.org/web_archive/2026/sessions/papers-9a.html)
- NIME 2026 측정 하드웨어는 CPU AMD Ryzen 9 7950X, CUDA NVIDIA RTX 4090, Apple Metal M3 Max MacBook이다. [요약만] — [NIME 2026 PDF](https://nime.org/proceedings/2026/nime2026_73.pdf)
- NIME 2026 측정법: throughput은 초당 생성 토큰 수의 running average를 15초 뒤에 보고하고, latency는 1000/tokens-per-second(ms/token)로 계산한다. context가 차면서 tokens/s가 떨어지므로 context window를 120 tokens로 제한했다. 이는 velocity가 없으면 40 notes(note당 3 토큰), 있으면 30 notes(note당 4 토큰)에 해당한다. 이 latency는 모델 생성만 포함하고 DAW/MIDI 인터페이스 지연은 빠져 있는데, 저자들은 실사용에서 그 지연을 지각할 수 없었다고 쓴다. [요약만] — [NIME 2026 PDF](https://nime.org/proceedings/2026/nime2026_73.pdf)
- 2026년 제3자 논문(arXiv 2606.11886)은 jam_bot 아키텍처가 "highly artist-specific"하다고 평한다. 미세조정한 스타일 데이터와 수작업 interaction strategy에 의존해 범용·다중 사용자 실시간 시나리오로 일반화하기 어렵다는 것이다. [요약만] — [arXiv 2606.11886](https://arxiv.org/html/2606.11886)
- 비교 기준: AMT 논문은 Lakh MIDI의 평균 음악 토큰 밀도를 초당 68 토큰(표준편차 51)으로 보고한다. 이것은 생성 속도가 아니라 음악 자체의 토큰율이다. [요약만] — [AMT arXiv 2306.08620](https://arxiv.org/abs/2306.08620)

### Inferences
- NIME 수치 체계(ms/token = 1000/tps, context 120 tokens 상한)를 보면 jam_bot은 짧은 슬라이딩 context로 per-token 비용을 묶어 두는 설계다. 사용자의 237–408 토큰 full re-prefill보다 context가 훨씬 짧다.
- velocity를 넣으면 note당 토큰이 3에서 4로 늘어(+33%) 같은 음악 밀도에서 필요한 tokens/s도 그만큼 오른다. NIME 논문이 ggml로 throughput을 확보한 동기가 이것이다. Tigran/bebop처럼 밀도 높은 솔로라면 이 점이 중요하다.
- "외부 MIDI 출력 지연 보상"은 FL Studio/Serum 경로에도 그대로 필요하다. 출력 장치 지연만큼 미리 보내는 스케줄링이다(§3의 Aria 구현 참고).

### Gaps
- NIME 2026의 CPU/CUDA/M3 Max ms/token 실측값과 ggml 대 PyTorch 배율: 미확인(nime.org, zenodo 차단).
- 100 ms(accompaniment) / 800 ms(melody-conditioned) 목표치: 이 세션에서 1차 출처를 찾지 못했다(미확인). 이전 세션 메모 또는 ISMIR PDF에서 온 것으로 보이지만 검증하지 못했다.
- ISMIR 2025 논문의 스레드 구조, 스케줄링 주기, 모드별 latency: 미확인.
- 공개 코드: `github.com/lancelotblanchard/jam_bot`은 clone되지 않았다(존재하지 않거나 private). 검색으로도 공개 저장소를 찾지 못했다. 다만 프로젝트 페이지만 보고 "코드 없음"이라고 단정하지는 않는다.
- 베이스 모델 명칭(Music Transformer 대 AMT): 언론 기사와 논문 스니펫이 다르다. NIME 본문의 "AMT"가 더 1차 출처에 가깝다.
- 검색 스니펫에 "multi-second latency 대 sub-200 ms target"이라는 문구가 나왔지만 이것이 jam_bot을 가리키는지는 불명확하다. 인용하지 않는다.

---

## 2. ReaLchords (ICML 2024) / ReaLJam (CHI EA 2025) / GAPT, realchords-pytorch의 MLX 백엔드

### Takeaway
ReaLJam은 서버가 "lookahead" 박자만큼 미래 코드를 미리 생성해 보낸다. 그중 "commit" 구간은 고정하고, 클라이언트는 이미 받은 예약 코드를 재생하면서 다음 응답을 기다린다. 그래서 RTT가 lookahead보다 짧기만 하면 끊김이 없다. 공개 코드(realchords-pytorch)에는 `--mlx` Apple Silicon 백엔드, ONNX 백엔드, 그리고 사용자 방향(코드→멜로디)의 online melody 모델 학습 config가 있다. ms 단위 latency 실측은 찾지 못했다.

### Cited Findings
- ReaLchords: "Adaptive Accompaniment with ReaLchords", ICML 2024, PMLR 235:53328–53345. 저자는 Wu, Cooijmans, Kastner, Roberts, Simon, Scarlatos, Donahue, Tarakajian, Omidshafiei, Courville, Castro, Jaques, Huang. MLE로 사전학습한 online 모델을 RL로 미세조정한다. reward model은 멜로디와 코드의 harmonic·temporal coherence를 본다. 미래를 볼 수 있는 teacher를 볼 수 없는 student로 distillation해 "anticipation"을 강제한다. 청취 실험에서 online MLE 기준선과 distillation-only보다 선호됐고, 나쁜 코드 예측 뒤에도 빠르게 적응하는 사례를 제시한다. [요약만] — [PMLR](https://proceedings.mlr.press/v235/wu24c.html); [arXiv 2506.14723](https://arxiv.org/abs/2506.14723)
- ReaLJam: arXiv 2502.21267, CHI EA '25(Yokohama), DOI 10.1145/3706599.3720227. 웹 인터페이스에서 사람이 멜로디를 치고 Transformer agent가 코드를 친다. agent는 앞으로의 진행을 계속 예측하고 그 계획을 시각적으로 보여 준다. [요약만] — [arXiv 2502.21267](https://arxiv.org/abs/2502.21267); [ACM DL](https://dl.acm.org/doi/10.1145/3706599.3720227)
- 프로토콜: lookahead는 사용자가 볼 수 있는 앞으로의 코드 박자 수이고, commit time은 그 이후로 agent가 예측을 바꿀 수 없는 지점이다. 논문 그림은 "2 beat round-trip latency, 4 beat lookahead, 2 beat commit"을 예시로 든다. 응답이 1 frame보다 오래 걸려도 예약된 코드가 재생되므로, lookahead frames가 대기 frames보다 많은 한 클라이언트 쪽 지연이나 끊김이 없다. 사용자는 silence 구간, lookahead, commit, 모델, temperature, 템포를 조절할 수 있다. [요약만] — [arXiv 2502.21267](https://arxiv.org/pdf/2502.21267)
- 사용자 연구: 숙련 음악가 12명(대부분 악기 경력 10년 이상). 결과는 "즐겁고 음악적으로 흥미로움"과 "인터페이스 설정의 세밀한 제어가 중요함"이다. 즉시 commit을 선호한 사람과 긴 lookahead를 선호한 사람으로 갈렸다는 서술은 2차 요약에만 있다. 관련 논문 부록에 따르면 프론트엔드는 로컬 MIDI 키보드에서, 모델은 원격 GPU 노드에서 돈다. [요약만] — [arXiv 2502.21267](https://arxiv.org/pdf/2502.21267); [alphaxiv 요약](https://www.alphaxiv.org/abs/2502.21267)
- 코드 [직접 확인] `lukewys/realchords-pytorch`(마지막 커밋 2026-05-11):
  - README: `pip install realjam`; `realjam-start-server --port 1234 --ssl --mlx`로 Apple Silicon MLX 백엔드를 쓴다. `--onnx`와 `--mlx`는 상호배타적이다. 기본 8박 silence 뒤에 반주가 시작되고, UI에서 "Lookahead Beats"와 "Commit Beats"를 조절한다. ReaLchords, ReaLJam, GAPT(arXiv 2511.17879) 세 논문의 구현이다. — [README](https://github.com/lukewys/realchords-pytorch)
  - 프론트엔드 기본값은 localhost일 때 lookahead 2박·commit 2박, 원격일 때 4박·4박이다. lookahead 입력 범위는 1–8박이고 commit은 lookahead 이하로 제한된다. `Tone.context.lookAhead = 0`. — [play_from_midi_input.js](https://github.com/lukewys/realchords-pytorch/blob/main/realchords/realjam/frontend/play_from_midi_input.js)
  - 표현: `FRAME_PER_BEAT = 4`(16분음표 frame). 토큰은 [chord, melody]가 frame마다 interleave되고 `max_frames=256`이다. `generate_live()`는 매 요청마다 prompt(BOS + 과거 chord/melody interleave)를 다시 만들고, commit 구간은 기존 chord 토큰으로 teacher-forcing하며 사용자 멜로디를 "상상"해 채운다. 그 뒤 `lookahead*2 - commit*2 - 1`개 토큰을 생성한다(마지막 멜로디 토큰은 생략해 시간 절약). commit이 lookahead 이상이면 생성 없이 캐시된 코드를 반환한다. silence_till 4 frame 전까지는 생성하지 않는다. — [agent_interface.py](https://github.com/lukewys/realchords-pytorch/blob/main/realchords/realjam/agent_interface.py); [constants.py](https://github.com/lukewys/realchords-pytorch/blob/main/realchords/constants.py)
  - online chord 모델 config: `DecoderTransformer.dim 512, depth 8, heads 8, max_seq_len 514`, bf16-mixed. — [decoder_only_online_chord.yml](https://github.com/lukewys/realchords-pytorch/blob/main/configs/gen_models/decoder_only_online_chord.yml)
  - MLX 포팅: `realchords/realjam/mlx_models/decoder.py`에 `MLXKVCache`(update/advance/reset)가 있다. `benchmark_generation_mlx.py`는 PyTorch/ONNX/MLX를 생성 토큰 8/16/24/32개로 비교해 그래프만 저장하고, 결과 수치는 저장소에 없다. ONNX KV 캐시 래퍼(`onnx_kv.py`)도 있다. — [realjam dir](https://github.com/lukewys/realchords-pytorch/tree/main/realchords/realjam)
  - 사용자 방향 모델: README "RL Training for Melody Model"에 "online model that generates melody given chords" 학습 명령과 `decoder_only_online_melody_4_datasets.yml` config가 있다(Hooktheory, POP909, Nottingham, Wikifonia, ±6 반음 transpose 증강). — [README](https://github.com/lukewys/realchords-pytorch)

### Inferences
- dim 512 × depth 8이면 대략 25M 파라미터로 추정된다(12·d²·L 근사). 사용자의 13.4M 모델과 같은 급이다. ReaLJam의 실시간성은 작은 모델 + 짧은 생성(lookahead 2박 = 8 frames = 최대 15 토큰) + 예약 버퍼에서 나온다.
- 사용자 과제(코드를 치면 솔로가 나옴)는 ReaLchords의 역방향이다. 저장소에 이미 "chords → melody" online 모델 학습 경로가 있으므로, frame 기반(16분음표) interleave 표현과 RL 보상 구조를 그대로 차용할 수 있다. 다만 데이터는 팝/포크 리드시트라 bebop/Tigran 어휘는 아니다.
- commit/lookahead 프로토콜을 사용자 시스템에 옮기면 이렇다. 128 BPM에서 1박은 468.75 ms이므로 lookahead 4박이면 1.875 s 버퍼다. 0.67–0.75 s짜리 Aria 블록 생성도 "다음 박 commit" 안에 들어간다. 대가는 코드 변경에 대한 반응이 최소 commit 길이만큼 늦어진다는 점이다.

### Gaps
- ReaLJam/ReaLchords의 ms 단위 지연, 서버 GPU 종류, MLX 백엔드 실측 tokens/s: 미확인(논문 원문 차단, 벤치 결과 미저장).
- 사용자 연구의 설정 선호 분포: 2차 요약만 있다.

---

## 3. Aria / Aria-Duet 실시간 데모 (EleutherAI, MLX)

### Takeaway
Aria-Duet의 핵심은 사용자가 치는 동안 KV-cache를 작은 청크로 계속 prefill하는 continuous prefill이다. 논문에 따르면 이 대책이 없을 때 생기는 1000–2000 ms의 takeover→첫 음 지연이 거의 사라진다. 대신 잘린 note duration을 재평가하는 단계가 약 100–200 ms를 더한다. "M-series에서 첫 음 200–300 ms"라는 주장은 README와 코드에서 확인되지 않았다(미확인). README는 응답성이 GPU 메모리 대역폭에 좌우된다고만 쓴다.

### Cited Findings
- Aria 모델은 LLaMA 3.2(1B) 아키텍처의 자기회귀 모델로, 약 6만 시간의 솔로 피아노 전사 MIDI로 학습됐다. 논문은 arXiv 2506.23869(ISMIR 2025)이고 Apache-2.0이다. CUDA와 MLX 최적화 구현이 있다. [직접 확인] — [EleutherAI/aria README](https://github.com/EleutherAI/aria)
- `config/models/medium.json`: d_model 1536, n_heads 24, n_layers 16, ff_mult 4, max_seq_len 8192. [직접 확인] — [medium.json](https://github.com/EleutherAI/aria/blob/main/config/models/medium.json)
- 데모 README: demo 전용 checkpoint(sustain pedal 제어 추가)가 필요하다. Yamaha Disklavier에서 내장 재생 모드를 끄고 key-velocity latency를 수동 보정했으며 `latency=off/realtime`가 필수다. `--midi_path --midi_through`로 MIDI 파일을 입력 삼아 흉내 낼 수 있다. "Responsiveness of the real-time demo is dependent on your system configuration, specifically GPU memory bandwidth." 측정 수치는 없다. [직접 확인] — [README Real-time demo](https://github.com/EleutherAI/aria)
- `demo/demo_mlx.py`(커밋 2026-05-12)의 상수 [직접 확인]: `DTYPE=mx.bfloat16`, `MAX_SEQ_LEN=4096`, `KV_CHUNK_SIZE=256`, `PREFILL_CHUNK_SIZE_L=128`, `PREFILL_CHUNK_SIZE=16`, `RECALC_DUR_PREFILL_CHUNK_SIZE=8`, `RECALC_DUR_BUFFER_MS=100`, `BEAM_WIDTH=3`, `FIRST_ONSET_BUFFER_MS=-150`, `MAX_STREAM_DELAY_MS=50`. — [demo_mlx.py](https://github.com/EleutherAI/aria/blob/main/demo/demo_mlx.py)
- 동작 구조 [직접 확인] — [demo_mlx.py](https://github.com/EleutherAI/aria/blob/main/demo/demo_mlx.py):
  - `continuous_prefill()`: 입력 MIDI 메시지가 10개 이상 쌓이면 전체를 다시 토큰화한다. `_find_divergence()`로 이전 context와 갈라지는 지점을 찾고, 거기서부터만 128 또는 16 토큰 청크로 prefill한다. KV는 divergence 이전까지 유지된다.
  - 시작 시 warmup에서 prefill(청크 크기별)과 decode_one의 shape를 미리 컴파일하고 ms/it을 로그로 남긴다.
  - takeover 시 순서: 남은 토큰을 강제 full prefill → "speculative-decoding inspired" duration 재계산(8 토큰 청크) → 첫 두 토큰을 3-way beam으로 선택. 이때 이미 지나간 onset은 mask하고, 벽시계 경과 시간에 맞춰 time 토큰을 삽입한다. "Time to first token"이 로그로 남는다.
  - 스레드와 큐: MIDI 캡처, continuous prefill, 토큰 생성, 토큰→MIDI 디코드, MIDI 스트림 송출이 각각 별도 스레드이고 queue로 연결된다.
- 하드웨어 보정 `demo/hardware/c4dm-disklavier.json` [직접 확인]: `HARDWARE_INPUT_LATENCY_MS 50`, `BASE_OUTPUT_LATENCY_MS 50`, `MIN_NOTE_DELTA_MS 100`, `MIN_NOTE_LENGTH_MS 100`. velocity별 추가 출력 지연은 vel ≥100에서 0 ms, 80에서 10, 60에서 60, 40에서 105, 20에서 140, ≤10에서 155 ms다. 송출 시각은 `onset_epoch_ms - BASE_OUTPUT_LATENCY_MS`로 앞당긴다. `demo/calibrate.py`는 "3-2-1-GO" 방식으로 지연을 대화식으로 보정한다. — [c4dm-disklavier.json](https://github.com/EleutherAI/aria/blob/main/demo/hardware/c4dm-disklavier.json)
- Aria-Duet 논문: "The Ghost in the Keys: A Disklavier Demo for Human-AI Musical Co-Creativity". 저자는 Bradshaw, Spangher, Biderman, Colton, NeurIPS 2025 Creative AI Track, arXiv 2511.01663. 대책 없이 prefill하면 takeover 신호와 첫 음 사이에 1000–2000 ms가 걸리고, continuous prefill(사용자가 치는 동안 작은 청크로 KV 갱신)로 거의 제거된다. truncated duration의 speculative 재평가는 약 100–200 ms를 더하며, 저자들은 이를 받아들일 만하다고 판단했다. "consumer hardware에서 low-latency"라고 하지만 칩 이름은 스니펫에 없다. [요약만] — [arXiv 2511.01663](https://arxiv.org/abs/2511.01663); [OpenReview PDF](https://openreview.net/pdf?id=yL8BrlEqHQ); [NeurIPS slides](https://neurips.cc/media/neurips-2025/Slides/129323.pdf)
- 사용자 로컬 측정(프로젝트 컨텍스트, 외부 출처 아님): Aria MLX CLI는 시퀀스당 약 62 tok/s(batch 2, 1024 tokens에 16.5 s). PyTorch MPS의 full re-prefill 블록은 중앙값 0.67 s, 최대 0.75 s.

### Inferences
- 사용자의 현재 방식(블록마다 237–408 토큰을 full re-prefill한 뒤 48 토큰 decode)은 Aria-Duet이 "1000–2000 ms"로 지적한 비용을 매 블록 다시 내는 구조다. Aria 데모처럼 divergence 지점부터만 16/128 청크로 incremental prefill하면 블록 비용은 사실상 decode(48 토큰)만 남는다. 62 tok/s 기준으로 약 0.77 s다. 따라서 추가로 decode 속도를 올리거나(양자화, §8) 생성 토큰 수를 줄여야 한다(ReaLJam식 짧은 lookahead).
- 대략적인 대역폭 상한: Aria medium(약 0.66B, bf16 약 1.3 GB). M1 Max의 공칭 메모리 대역폭(약 400 GB/s, Apple 공칭치이며 이 세션에서 확인하지 않음)으로 보면 batch-1 decode 이론 상한은 약 300 tok/s다. 측정치 62 tok/s는 그 20% 수준이라 Python과 커널 오버헤드가 크다는 신호다. 추정이며 검증이 필요하다.
- velocity 의존 출력 지연표와 "onset − output latency" 스케줄링은 Serum이나 FL Studio에서도 같은 방식으로 쓸 수 있다. 다만 소프트 신스는 velocity 의존 지연이 거의 없으므로 base offset 하나로 충분할 것이다.

### Gaps
- Aria-Duet의 실측 first-note latency(ms)와 측정 칩: 미확인. "200–300 ms" 주장은 README, 코드, 스니펫 어디서도 찾지 못했다.
- 데모의 decode ms/it 실측값: 코드가 로그로만 출력하고 저장소에는 수치가 없다.

---

## 4. Notochord (AIMC 2022, arXiv 2403.12000)

### Takeaway
Notochord는 GRU 기반 이벤트 단위 확률 모델로, 저자들은 입력에 10 ms 미만으로 반응한다고 주장한다. 하위 이벤트 수준에서 제약을 걸 수 있다. 예를 들어 다음 이벤트의 pitch whitelist나 최소·최대 지연을 줄 수 있어서, 코드톤 집합 같은 하드 제약을 이벤트마다 거는 데 바로 쓸 수 있다. 10 ms를 측정한 하드웨어는 미확인이다.

### Cited Findings
- 논문은 Shepardson, Armitage, Magnusson(Intelligent Instruments Lab)의 "Notochord: a Flexible Probabilistic Model for Real-Time MIDI Performance"다(AIMC 2022, arXiv 2403.12000, Zenodo 7088404). 구조화된 이벤트 시퀀스용 딥 확률 모델이고 Lakh MIDI로 학습했다. polyphonic·multi-track MIDI를 생성하며 "respond to inputs with latency below ten milliseconds"라고 한다. 예시 제약은 "다음 이벤트의 pitch-class를 C로" 또는 "100 ms 이전에는 발생하지 않음"이다. 학습 코드와 체크포인트가 공개돼 있다. [요약만] — [arXiv 2403.12000](https://arxiv.org/abs/2403.12000); [Zenodo 7088404](https://zenodo.org/record/7088404)
- 2025 ICMC Homunculus 논문은 "very low latency (circa 10 milliseconds) event processing"을 우선한다고 쓴다. [요약만] — [ICMC 2025 Homunculus PDF](https://iil.is/pdf/2025_icmc_notochord_homunculus.pdf)
- 코드 [직접 확인](마지막 커밋 2026-02-23): `Notochord(nn.Module)`의 기본값은 `rnn_hidden=1024`, `kind='gru'`(lstm/elman 선택 가능)다. time과 velocity는 `CensoredMixtureLogistic` 분포로 모델링한다. 추론 API는 `feed`, `sample`, 구형 `query`다. `query_tipv_onsets` 등은 `min_time/max_time`(truncation), `include_pitch`(whitelist), `include_inst`, `min_vel/max_vel` 인자를 받는다. `uvx notochord homunculus`는 텍스트 UI로 MIDI 포트에 직접 연결하는 앱이고, `simple_harmonizer.py`는 최소 인터랙티브 예제다. — [Notochord repo](https://github.com/Intelligent-Instruments-Lab/notochord); [model.py](https://github.com/Intelligent-Instruments-Lab/notochord/blob/main/src/notochord/model.py)

### Inferences
- RNN의 step 비용은 context 길이와 무관하게 O(1)이다. 이벤트 하나 처리에 KV 재계산이 필요 없으므로 10 ms급 반응이 구조적으로 가능하다. Transformer에서 같은 효과를 내려면 persistent KV + 1-토큰 decode가 필요하다.
- `include_pitch` whitelist와 `min_time` 같은 이벤트별 제약은 "코드가 바뀐 직후 첫 박의 음을 현재 코드톤으로 강제"하는 first-beat patching에 그대로 대응한다.

### Gaps
- sub-10 ms를 측정한 CPU/GPU 종류와 조건: 미확인(PDF 차단).

---

## 5. "Real-Time Language Model Jamming" (StreamMUSE, arXiv 2606.11886) 및 "Real-Time Symbolic Music Accompaniment Generation for Edge Devices" (REMIBlock)

### Takeaway
과제 설명과 달리 REMIBlock 토크나이제이션은 2606.11886이 아니라 Springer 2025 edge-device 논문(JazzyGPT2, 91M GPT-2, GGUF Q4/Q8/F16)에서 나온다. 2606.11886은 StreamMUSE를 제안한다. client–server 구조의 "frame-synchronous streaming inference"로, 추론 간격 I와 생성 길이 GL을 RTT에 맞춰 고르고 GL > I로 겹쳐 버퍼를 만든다. 두 논문 모두 ms·하드웨어 수치는 미확인이다.

### Cited Findings
- arXiv 2606.11886: "Real-Time Language Model Jamming: A Case Study for Live Music Accompaniment Generation". 2026-06-10 제출, Bowen Zheng 외 8인. 외부 신호 스트림에 맞춰 LM 생성을 하는 문제를 "frame-synchronous streaming inference"로 정의한다. 클라이언트가 최신 입력으로 고빈도 요청을 보내면 서버가 외부 클럭에 정렬된 결과를 돌려준다. 하이퍼파라미터는 inference interval I(요청 빈도)와 generation length GL(요청당 생성 길이)이다. [요약만] — [arXiv abs](https://arxiv.org/abs/2606.11886); [HTML](https://arxiv.org/html/2606.11886v1)
- StreamMUSE의 지연 처리: GL > I면 연속 생성이 겹치고, 요청이 늦으면 backup 구간이 실시간으로 스케줄돼 연속성을 유지한다. 지연 모델은 f(GL)=αGL²+βGL+γ(GL의 2차식)이다. I가 너무 짧으면 요청이 겹쳐 시스템이 정체되고, GL을 약간 길게 잡으면 지연 응답에 대한 버퍼가 된다. 템포가 빨라지면 제약이 빡빡해져 실시간 스케줄링이 불가능해지기도 한다. 로컬 설정에서 해(solution) 공간이 전체 탐색 공간의 44% 이상이라는 수치가 있다. 오픈소스이고 프로젝트 페이지는 stream-muse-webpage.vercel.app이다. [요약만] — [arXiv 2606.11886](https://arxiv.org/pdf/2606.11886)
- 같은 논문 주변 스니펫에는 "고도로 인터랙티브한 시나리오의 목표는 sub-200 ms", "30 ms를 넘는 지연은 앙상블 협응을 해칠 수 있다는 선행연구 인용" 같은 문구가 있다. 어느 문단, 어느 논문의 것인지는 불명확하다. [요약만, 귀속 불명확] — [arXiv 2606.11886 HTML](https://arxiv.org/html/2606.11886)
- Springer 논문 "Real-Time Symbolic Music Accompaniment Generation for Edge Devices"(DOI 10.1007/978-3-032-05748-8_44; DCAI 2025 프로시딩, 온라인 2026-04으로 표시): 입력 리드 멜로디(MIDI)에 반주를 실시간 생성한다. REMIBlock 토크나이제이션을 제안하고, 온디바이스 효율에 맞춘 GPT-2 기반 모델을 쓴다. test perplexity는 7.69이고 groove·scale consistency 지표에서 정답과의 차이가 최대 3.4%다. 코드는 github.com/UncleBen420/JazzyGPT2. 스니펫의 저자명(Scarlatos, Wu, Simon, Huang)은 ReaLchords 저자와 겹쳐 오류로 보인다. [요약만] — [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-032-05748-8_44)
- JazzyGPT2 코드 [직접 확인](커밋 2025-05-07):
  - `Llama.from_pretrained(repo_id="REMSLEGRAND/Jazzy_gpt2_v2", filename="JazzyGPT2_V2-91M-{F16|Q8|Q4}.gguf", n_ctx=1024)`로 llama.cpp(llama-cpp-python, ggml)를 쓴다. CUDA면 `n_gpu_layers=-1`, CPU면 0이다. 토크나이저는 miditok `REMSLEGRAND/jazzy_gpt2`다.
  - context는 마지막 9마디 이하, 256 토큰 이하로 마디 경계에서 자른다. 멜로디 마디 마커는 `BOBM{k}`/EOBM이다.
  - 스케줄링: 메트로놈 박마다 `(i+3)%4==0`, 즉 각 마디 2번째 박에서 `predict_next_bar`를 별도 스레드로 호출해 다음 마디를 생성한다. 약 3박의 계산 예산이다. 템포는 60–120 BPM만 허용하고(벗어나면 80), temperature는 0.3이다.
  - `get_latency()`는 CUDA F16/Q8/Q4와 CPU F16/Q4에서 100회 측정하는 함수지만, 저장소에 결과 수치는 없다. — [main.py](https://github.com/UncleBen420/JazzyGPT2/blob/main/main.py); [jazz_master.py](https://github.com/UncleBen420/JazzyGPT2/blob/main/src/jazz_master.py); [stream.py](https://github.com/UncleBen420/JazzyGPT2/blob/main/src/stream.py)

### Inferences
- JazzyGPT2는 사용자 구조(코드 입력 → 다음 구간 솔로)와 가장 닮은 공개 구현이다. 다음 마디를 3박 전에 생성 시작하고, 91M 모델을 Q4 GGUF로 양자화한다. 템포 상한을 120 BPM으로 둔 것은 1마디 예측 예산(120 BPM에서 3박 = 1.5 s)이 빠듯함을 시사한다. 사용자의 128 BPM에서 3박은 1.406 s다.
- StreamMUSE의 "GL > I 오버랩 + backup"은 ReaLJam의 lookahead/commit과 같은 원리다. 생성이 늦을 때를 대비해 항상 재생 가능한 미래 버퍼를 쥐고 있는 것이다. f(GL)이 2차식이므로 GL(블록 길이)을 줄이면 지연이 초선형으로 준다(추론).

### Gaps
- 2606.11886의 실험 하드웨어, RTT 분포, 템포별 실현 가능 영역 수치: 미확인.
- Edge 논문의 "edge device" 종류(Jetson? CPU?)와 Q4/Q8/F16별 latency 실측: 미확인. 코드에는 측정 함수만 있다.

---

## 6. 기타 실시간 시스템: SongDriver, BachDuet, Somax2, Design Space, LK Jam(2026)/Frontiers 2021, Shimon, AI Duet, Magenta RealTime(2), LMDM, SAGE-Music

### Takeaway
"지연을 없애는" 반복 패턴은 다섯 가지다. (a) 미리 예측해 캐시해 두고 들어온 입력에 맞춰 꺼내 쓰기(SongDriver), (b) 턴테이킹으로 침묵을 기다리기(AI Duet), (c) 고정 템포 그리드 동기화(BachDuet), (d) O(1) step 모델(GRU: LK Jam, Notochord; sliding-window attention: Magenta RT 2), (e) block-causal 캐시(LMDM). Apple Silicon에서 실측 "실시간" 표를 공개한 것은 Magenta RT 2뿐이며, 오디오 모델이다.

### Cited Findings
- SongDriver(arXiv 2209.06054, ACM MM 2022, Zihao Wang 외, ZJU): 2단계 구조다. Transformer가 멜로디가 들어오는 대로 코드를 배치하되 재생하지 않고 캐시하고, CRF가 캐시된 코드로 다가올 멜로디의 multi-track 반주를 만든다. 이렇게 "zero logical latency"를 얻고, 자기 출력 대신 캐시된 코드를 조건으로 써서 exposure bias를 피한다. 짧은 실시간 입력을 보완하려고 이전 구간에서 전역 음악 특징 4개를 뽑는다. àiSong 데이터셋을 쓰며, 물리적 지연을 크게 줄였다고 주장한다. [요약만] — [arXiv 2209.06054](https://arxiv.org/abs/2209.06054); [ZJU page](https://www.next.zju.edu.cn/publications/songdriver)
- BachDuet(NIME 2020, Benetatos, VanderStel, Duan): RNN이 사람의 단성 MIDI 키보드 연주를 처리해 기계의 단성 성부를 실시간 생성한다. 사용자 13명 연구와, 청취자 48명이 사람-기계 이중주를 사람-사람 이중주와 구별하지 못한 결과가 있다. ISMIR 2019 LBD에 따르면 4/4 박자와 조절 가능한 고정 템포로 동기화한다. 지연 수치는 미확인. [요약만] — [NIME 2020](https://nime.org/proc/nime2020_125/); [ISMIR 2019 LBD](https://labsites.rochester.edu/air/publications/benetatos19bachduet.pdf)
- Somax2(IRCAM REACH): Max와 Python 백엔드로 구현됐다. reactive machine listening, interaction policy, corpus 기반 생성 엔진(심볼릭, MIDI 파일로 학습, concatenative 유사)으로 구성된 멀티 에이전트 시스템이다. 참고문헌은 Borg, Assayag, Malt(SMC 2022)와 CMJ 2022다. 지연 수치는 미확인. [요약만] — [IRCAM REACH Somax2](https://reach.ircam.fr/index.php/2024/01/27/somax2/)
- "A Design Space for Live Music Agents"(arXiv 2602.05064, CHI 2026, Yewon Kim, Stephen Brade 외): 시스템 184개를 분석해 usage context, interaction, technology, ecosystem 4개 측면으로 정리했고, 2차 요약에 따르면 31개 차원과 165개 코드다. [요약만] — [arXiv 2602.05064](https://arxiv.org/abs/2602.05064)
  - 공개 데이터 직접 집계 [직접 확인]: `assets/paper_data.csv`(184행)와 `new_code_mapping.csv`를 집계했다. "Technical Emphasis: Latency"(코드 205)로 코딩된 시스템이 106/184, "Efficiency" 33, "Tempo adaptability" 30이다. Runtime은 "Commodity machine" 116, "Dedicated commodity hardware" 39, "High-performance compute" 1이다. — [live-music-agents repo](https://github.com/live-music-agents/live-music-agents.github.io)
- LK Jam(arXiv 2606.21018, 2026-06-19, Yakun Liu, Zhiyu Jin, Dong Liu, Hai Luan, 기술보고서): 경량 GRU다. 고정 time grid 대신 time-shift, 연속 harmonic embedding, role-aware encoding을 담은 sparse event stream으로 단일 step 추론을 한다. 3단계 점진 학습을 쓰고 C++/JUCE/RTNeural, lock-free bridge, allocation-free 설계의 VST3 플러그인으로 구현했다. 근거는 "preliminary observations and architectural analysis" 수준이고 실측 지연은 스니펫에 없다. [요약만] — [arXiv 2606.21018](https://arxiv.org/abs/2606.21018)
- Frontiers 2021(Kritsis 외, "On the Adaptability of Recurrent Neural Networks for Real-Time Jazz Improvisation Accompaniment", Frontiers in AI, 2021-02-12, DOI 10.3389/frai.2020.508727): 인간 솔로이스트에게 코드 보이싱을 제공하는 반주자다. 솔로 입력과 리드시트의 화성·박자 정보를 조건으로 하고, 단순 RNN으로 반주 예측과 솔로이스트의 "예상 의도"를 함께 모델링한다. 재즈 스탠더드 2곡에서 화성 제약 추종을 확인했다. 지연 수치는 미확인. [요약만] — [Frontiers](https://www.frontiersin.org/articles/10.3389/frai.2020.508727/full)
- Shimon(Hoffman & Weinberg, Autonomous Robots 2011): anticipatory action으로 실시간 즉흥 동기화를 하고, 재즈로 학습한 Markov 즉흥 엔진과 gesture 기반 구동을 쓴다. "without perceptible delay"는 정성적 주장이다. 컴퓨터-음악가 기준선보다 동기화와 응집 평가가 높았다. ms 수치는 미확인. [요약만] — [Springer chapter](https://www.springerprofessional.de/en/interactive-improvisation-with-a-robotic-marimba-player/3685538); [GT Shimon](https://gtcmt.gatech.edu/shimon)
- Google AI Duet 코드 [직접 확인](googlecreativelab, 마지막 커밋 2025-08-18): 서버는 Magenta Melody RNN `attention_rnn` 번들이다. 클라이언트는 모든 키가 떨어진 뒤 600 ms 침묵이 이어지면 요청을 보내고, 프레이즈가 3 s를 넘었으면 즉시 보낸다. 입력이 10 s를 넘으면 마지막 15 s로 자른다. 생성 길이는 입력 길이 + clamp(입력 길이, 1, 8) s이고, 응답 note는 `Tone.now()+0.05` s 기준으로 예약한다. 즉 call-and-response 턴테이킹이다. — [AI.js](https://github.com/googlecreativelab/aiexperiments-ai-duet/blob/master/static/src/ai/AI.js); [predict.py](https://github.com/googlecreativelab/aiexperiments-ai-duet/blob/master/server/predict.py)
- Magenta RealTime 2 [직접 확인](magenta/magenta-realtime, 커밋 2026-07-30, 오디오 모델):
  - 디코더 전용 LLM이 frame 단위(25 Hz 코덱 frame = 40 ms) 자기회귀로 frame마다 RVQ 토큰 12개를 생성한다. 레이어별 windowed attention은 base가 25 frame(1 s)×20층, small이 41 frame(약 1.6 s)×12층이고 유효 수용장은 20 s다. MIDI 조건은 frame마다 128차원 multihot(off/sustain/onset/model decides)이다. — [MODEL.md](https://github.com/magenta/magenta-realtime/blob/main/MODEL.md)
  - 실시간 스트리밍은 Apple Silicon 전용이다. `mrt2_small`(230M)은 M1 Air를 포함한 모든 Apple Silicon에서 실시간이다. `mrt2_base`(2.4B)는 M5 Max, M3 Max, M2 Max, M4 Pro에서 실시간이고 M2 Pro, M1 Pro, Air에서는 실시간이 아니다. 표에 M1 Max 행은 없다. — [README](https://github.com/magenta/magenta-realtime)
  - C++ 엔진 `magentart::core`: `.mlxfn`(MLX export) 모델, lock-free SPSC ring buffer, 오디오 콜백과 분리된 inference thread. "GPU Idle Downclocking: macOS는 작업에 공백이 생기면 GPU를 downclock하고, 그러면 frame당 추론 비용이 수 ms 영구 증가한다." 대책은 ring buffer 대기 중 아주 작은 dummy GPU 연산(`mx::array({0.0f}) + mx::array({0.0f})`)을 돌려 활성 전력 상태를 유지하는 것이다. 장시간 루프에서는 Objective-C autorelease pool을 비워야 메모리 누수가 없다. — [core/README.md](https://github.com/magenta/magenta-realtime/blob/main/core/README.md)
  - 원조 Magenta RT와 "Live Music Models"(Lyria Team)는 arXiv 2508.04651이다. [요약만] — [arXiv 2508.04651](https://arxiv.org/abs/2508.04651)
- LMDM(arXiv 2605.22717, 2026-05-22, "Live Music Diffusion Models: Efficient Fine-Tuning and Post-Training of Interactive Diffusion Music Generators"): 고정 크기 block에 block-causal attention mask를 걸어, 새 block 생성 뒤 그 block만 캐시하면 된다. ARC-Forcing은 명시적 RL이나 reward model 없이 오차 누적을 줄이는 post-training 정렬이다. 소비자용 게이밍 노트북에서 로컬로 "generative delay"로 즉흥에 썼다. 코드 공개 여부는 2차 출처끼리 엇갈린다. [요약만] — [arXiv 2605.22717](https://arxiv.org/abs/2605.22717)
- SAGE-Music(arXiv 2510.00395, 2025-10): AS-KVHS(attribute-specialized KV head sharing)로 추론을 약 30% 가속하고 객관 품질 하락은 약 0.4%다. BPE는 multi-track에서 성능이 크게 떨어진다고 분석했다. 하드웨어는 미확인. [요약만] — [arXiv 2510.00395](https://arxiv.org/abs/2510.00395)

### Inferences
- 사용자 과제에 바로 대응하는 것은 두 가지다. SongDriver의 "코드 입력 시점에 이미 예측·캐시된 다음 구간을 꺼내 쓰고, 캐시를 만든 쪽은 입력을 보고 갱신"하는 패턴과, Magenta RT 2의 "frame 단위 + sliding window로 step 비용을 일정하게"라는 패턴이다.
- Magenta RT 2는 2.4B 모델이 M2 Max/M3 Max에서 25 Hz(40 ms/frame, frame당 12 토큰)로 실시간이라고 표시한다. 다만 오디오 토큰은 depth 방향 처리 구조가 다르다(이 세션에서 세부 확인 안 함). 그래서 "M-series Max에서 수백M~수B 모델로 40 ms 주기 연속 생성이 가능하다"는 존재 증명으로만 쓰고, 심볼릭 tok/s로 직접 환산하지는 않는다.
- GPU idle downclocking 경고는 사용자의 "블록 생성 → 대기 → 블록 생성" 패턴(버스트형)에 그대로 해당할 수 있다. MPS에서 M1 Max GPU가 최저 클럭에 머문다는 포럼 보고(§7)와 같은 현상일 가능성이 있다.

### Gaps
- SongDriver, BachDuet, Somax2, Shimon, LK Jam, Frontiers 2021의 ms 단위 지연과 측정 하드웨어: 모두 미확인(원문 차단).
- Magenta RT 2의 frame당 ms 실측치: 저장소에는 벤치 실행법만 있고 수치는 없다.
- Google AI Duet 서버 응답 시간: 미확인.
- "Streaming Generation for Music Accompaniment"(arXiv 2510.22105)와 LiveBand(arXiv 2606.03803)는 검색 결과로만 존재를 확인했고 내용은 보지 못했다(검색 예산 소진).

---

## 7. Apple Silicon 추론·학습 근거 (MLX / mlx-lm / PyTorch MPS)

### Takeaway
batch-1 decode 속도는 메모리 대역폭에 묶여 있어 양자화가 곧 decode 속도 향상이다(M4 Max, Qwen3-4B 기준 bf16→q8 1.66배, →q4 2.56배). mlx-lm에는 persistent prompt cache, `trim_prompt_cache`, draft model speculative decoding, BatchGenerator, KV 양자화가 이미 있다. 커뮤니티 실측으로 M1 Max에서 1B Llama 생성은 약 150–250 tok/s다(context 의존). PyTorch MPS에는 decode 지연 비단조 스파이크와 저클럭 보고가 있어, 저지연 경로로는 MLX가 유리하다는 근거가 우세하다. M1 Max에서 지속 학습 시 열 스로틀링을 정량화한 자료는 찾지 못했다.

### Cited Findings
- mlx-lm BENCHMARKS.md [직접 확인](64GB M4 Max, mlx 0.29.2.dev, mlx-lm 0.28.2, macOS 26.1, prompt 2048 / generation 128):
  - Qwen3-4B: bf16은 prompt 1780.63 tok/s, gen 52.47 tok/s, 9.02 GB. q8은 1606.57 / 86.91 / 5.25 GB. q6은 1576.73 / 104.68. q4는 1622.27 / 134.52 / 3.35 GB.
  - Qwen3-30B-A3B q8은 1719.47 / 83.16 / 33.46 GB.
  - — [BENCHMARKS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md)
- mlx-lm LoRA [직접 확인]: Mistral-7B LoRA(`--batch-size 1 --num-layers 4`, wikisql)가 M1 Max 32 GB에서 약 250 tokens/s로 학습된다. 메모리 절감 수단은 QLoRA(양자화 모델 지정 시 자동), batch 축소(기본 4), `--grad-accumulation-steps`, `--num-layers` 축소(기본 16), 긴 예제 분할, `--grad-checkpoint`다. — [LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- mlx-lm 추론 기능 [직접 확인]:
  - rotating fixed-size KV cache `--max-kv-size n`
  - prompt caching(`mlx_lm.cache_prompt`, Python API로 요청 간 재사용)
  - `--prefill-step-size`(기본 2048)
  - `--kv-bits` KV cache 양자화(`quantized_kv_start` 기본 5000)
  - `--draft-model`/`--num-draft-tokens`(기본 2) speculative decoding
  - `trim_prompt_cache(cache, n)`로 캐시 롤백
  - `BatchGenerator`/`batch_generate`
  - 모델이 RAM에 비해 클 때 wired memory 사용(macOS 15+, `sudo sysctl iogpu.wired_limit_mb=N`)
  - — [README](https://github.com/ml-explore/mlx-lm); [generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py); [cache.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
- oMLX 커뮤니티 벤치(Llama 3.2 1B, 제3자): [요약만]
  - M1 Max 64GB 32-core GPU(macOS 15.7.3, oMLX v0.2.13): 1k context 250.0 tok/s, 4k 193.0 tok/s, batch 2 합산 약 360 tok/s.
  - M1 Max 32GB 24-core(macOS 26.3.1, v0.2.10): 1k 217.1, 4k 152.2 tok/s. 같은 칩 v0.2.9의 4k 측정은 116.8 tok/s.
  - — [omlx gdelsbgr](https://omlx.ai/benchmarks/gdelsbgr); [omlx ro7re1dx](https://omlx.ai/benchmarks/ro7re1dx); [omlx nlswyxjc](https://omlx.ai/benchmarks/nlswyxjc)
- arXiv 2511.05502(M2 Ultra 192 GB, Qwen-2.5, prompt 수백~10만 토큰): MLX가 가장 높은 지속 생성 throughput(약 230 tok/s, GPU 사용률 90% 이상)을 냈고, decode per-token latency는 중앙값 5–7 ms, P99 약 12 ms다. TTFT는 중간 길이 prompt에서 MLC-LLM이 더 낮았다. PyTorch MPS는 대형 모델과 긴 context에서 메모리 제약에 걸렸다. [요약만] — [arXiv 2511.05502](https://arxiv.org/abs/2511.05502)
- arXiv 2605.08913(2026-05, Hendria): Apple MPS 백엔드에서 GPT-2, BLOOM, OPT의 decode latency가 특정 decoding-budget 구간에서 최대 21배 튀었다가 인접 설정에서 회복된다. 원인은 주로 decode 단계이고 메모리 압박만으로는 설명되지 않으며, CPU와 CUDA에서는 같은 조건에서 나타나지 않는다. 그 구간에서는 KV cache 가속이 4.9–20.3배에서 1.9배로 무너진다. "discrete execution regime" 설명은 프로파일링 근거 없는 추론이라는 지적이 있다. [요약만] — [arXiv 2605.08913](https://arxiv.org/abs/2605.08913)
- arXiv 2605.05699 "When Quantization Is Free: An int4 KV Cache That Outruns fp16 on Apple Silicon": 제목과 존재만 확인했고 수치는 미확인. [요약만] — [arXiv 2605.05699](https://arxiv.org/abs/2605.05699)
- arXiv 2501.14925(Feng, Xu, Wang, Lin, UVA, 2025-01): LLM 학습에서 Apple Silicon(M1/M2 계열)과 NVIDIA 사이에 큰 성능 차이가 있고, 원인은 page fault, 전력 소비, kernel launch time 같은 시스템 요인이다. [요약만] — [arXiv 2501.14925](https://arxiv.org/abs/2501.14925)
- PyTorch 포럼(단일 사용자 보고): M1 Max에서 PyTorch MPS 실행 중 GPU 주파수가 약 400 MHz(최대 약 1300 MHz)에 머물렀고, MLX는 boost 클럭으로 약 3배 빨랐다. asitop으로 확인할 수 있다. 원인은 밝혀지지 않았다. [요약만] — [discuss.pytorch.org thread](https://discuss.pytorch.org/t/pytorch-on-mps-doesnt-use-m1-max-gpu-at-full-power-frequency-is-at-minimal-value/194325)
- MLX issue #1313 [직접 확인 via GitHub]: M1 계열에서 conv backward가 PyTorch MPS보다 3–6배 느렸다("unexpected performance cliff on M1"). 원인은 nearest Upsample backward의 scatter-add atomics였고 PR #1541(2024-10-31)로 수정됐다. 수정 후 M1 Max는 MLX 4.615 대 PyTorch 11.93, M3 Max는 1.938 대 10.77(단위는 스레드에 명시 없음)이다. — [mlx#1313](https://github.com/ml-explore/mlx/issues/1313)
- Magenta RT 2의 Apple Silicon 실시간 표와 GPU idle downclocking 경고는 §6 참조. [직접 확인] — [core/README.md](https://github.com/magenta/magenta-realtime/blob/main/core/README.md)
- 열 스로틀링: M1 Max 노트북에서 지속 GPU 학습 중 throughput 하락을 정량 측정한 1차 자료는 찾지 못했다. 찾은 것은 M1 Ultra(데스크톱)의 추론 전력 약 80 W 같은 일화성 보고뿐이다. [요약만] — [r/LocalLLaMA (mirror)](https://redlib.hackliberty.org/r/LocalLLaMA/comments/1jj6i4m/deepseek_v3/mjltq0a)

### Inferences
- 대역폭 한계 추정: Qwen3-4B bf16(약 8 GB 가중치)이 M4 Max(공칭 약 546 GB/s, Apple 공칭치·미검증)에서 52 tok/s다. 이는 바이트 기준 약 420 GB/s 효율로, 대역폭 한계에 근접한다. M1 Max(공칭 약 400 GB/s)와 0.66B bf16(약 1.3 GB)이면 이론상 약 250–300 tok/s, 8-bit면 그 약 1.6배다. 사용자 실측 62 tok/s(Aria MLX CLI, batch 2)는 이 상한보다 훨씬 낮으므로, 작은 모델에서는 커널 launch와 Python 오버헤드가 지배한다는 신호로 보인다. 1B Llama가 M1 Max에서 217–250 tok/s(oMLX)라는 커뮤니티 실측과 비교하면 Aria 데모와 CLI 경로에 최적화 여지가 크다(추정).
- 128 BPM에서 반 마디(937.5 ms)에 48 토큰을 내려면 51 tok/s, 한 박(468.75 ms) 안에 내려면 102 tok/s가 필요하다. 1B급 MLX의 200 tok/s대는 "한 박 이내 반응"을 계산상 허용한다. 단 persistent KV로 prefill을 없앴을 때의 이야기다.
- MPS 경로(사용자의 현재 0.67 s 블록)는 (a) 매 블록 full re-prefill, (b) 2605.08913의 decode 스파이크 위험, (c) 저클럭 보고가 겹친다. MLX와 persistent KV로 옮기는 것이 근거상 가장 큰 개선 후보다.
- LoRA 학습: 7B LoRA(4 layers, batch 1)가 M1 Max 32 GB에서 약 250 tok/s이므로, 0.5–1B 모델은 같은 설정에서 몇 배 빠를 것으로 추정된다(측정 없음). 열 문제는 asitop이나 powermetrics로 GPU 클럭과 전력을 기록하며 첫 몇 분과 정상상태를 비교해야 분리할 수 있다.

### Gaps
- M1 Max에서 0.1–0.7B Llama형 모델의 MLX 8-bit 대 bf16 decode tok/s 직접 측정: 1B(oMLX)를 빼면 미확인.
- M1 Max에서 0.5–1B 모델 LoRA·full fine-tune의 tokens/s, 메모리, 열 스로틀링 곡선: 미확인.
- speculative decoding의 실제 가속률을 소형 음악 모델에서 잰 자료: 미확인.
- MPS 대 MLX를 같은 소형 Transformer의 batch-1 decode로 비교한 1차 측정: 미확인(포럼 일화와 2605.08913 정도).

---

## 8. 한 마디 미만 chord→solo 반응을 위한 엔지니어링 패턴 (근거 연결)

### Takeaway
공개 시스템들이 반복해서 쓰는 해법은 세 가지다. (1) 생성은 항상 미래 버퍼를 앞서 채우고 commit 구간만 확정한다(ReaLJam, StreamMUSE, JazzyGPT2). (2) 입력이 오는 동안 KV를 증분 갱신해 반응 시점에는 decode만 남긴다(Aria-Duet). (3) 최근 입력에 맞춰 첫 음을 마스킹이나 제약으로 바로잡는다(Aria beam + onset mask, Notochord pitch whitelist). 출력 장치 지연 보상(Aria, jam_bot NIME)과 GPU를 계속 깨워 두는 처리(Magenta RT 2)가 이를 보완한다.

### Cited Findings
- Commit 버퍼: ReaLJam은 lookahead 안에서 commit 구간을 고정하고 예약분을 재생하며 응답을 기다린다. RTT가 lookahead보다 짧으면 무중단이다. [요약만] — [arXiv 2502.21267](https://arxiv.org/pdf/2502.21267). commit 구간은 기존 토큰을 teacher-forcing하고 나머지만 생성한다. [직접 확인] — [agent_interface.py](https://github.com/lukewys/realchords-pytorch/blob/main/realchords/realjam/agent_interface.py)
- 오버랩 생성: StreamMUSE는 GL > I로 연속 요청을 겹쳐, 지연된 요청이 있어도 backup이 재생되게 한다. 템포가 빠를수록 실현 가능 영역이 좁아진다. [요약만] — [arXiv 2606.11886](https://arxiv.org/pdf/2606.11886)
- 마디 선행 생성: JazzyGPT2는 각 마디 2번째 박에 다음 마디 생성을 시작한다(Q4 GGUF 91M, context ≤256 토큰 또는 ≤9마디). [직접 확인] — [stream.py](https://github.com/UncleBen420/JazzyGPT2/blob/main/src/stream.py)
- 예측 캐시: SongDriver는 코드를 미리 예측해 캐시하고 다음 구간 반주를 그 캐시로 만들어 "logical latency 0"을 얻는다. [요약만] — [arXiv 2209.06054](https://arxiv.org/abs/2209.06054)
- 증분 prefill과 persistent KV: Aria 데모는 메시지 10개마다 divergence 이후만 16/128 청크로 prefill한다. 대책 없는 prefill은 1000–2000 ms였다. [직접 확인 + 요약만] — [demo_mlx.py](https://github.com/EleutherAI/aria/blob/main/demo/demo_mlx.py); [arXiv 2511.01663](https://arxiv.org/abs/2511.01663). mlx-lm에도 prompt cache 재사용과 `trim_prompt_cache` 롤백이 있다. [직접 확인] — [cache.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
- 첫 음 보정: Aria는 첫 두 토큰을 3-way beam으로 고르고, 이미 지난 onset은 mask하며, 경과 시간만큼 time 토큰을 삽입한다(`FIRST_ONSET_BUFFER_MS=-150`). [직접 확인] — [demo_mlx.py](https://github.com/EleutherAI/aria/blob/main/demo/demo_mlx.py). Notochord는 이벤트별 `include_pitch` whitelist와 `min_time/max_time`을 건다. [직접 확인] — [model.py](https://github.com/Intelligent-Instruments-Lab/notochord/blob/main/src/notochord/model.py)
- 출력 지연 보상: Aria는 `send = onset - BASE_OUTPUT_LATENCY_MS`에 velocity별 표(0–155 ms)를 쓴다. [직접 확인] — [c4dm-disklavier.json](https://github.com/EleutherAI/aria/blob/main/demo/hardware/c4dm-disklavier.json). jam_bot NIME 2026은 외부 MIDI 출력 지연을 상쇄하도록 생성 note를 offset한다. [요약만] — [NIME 2026](https://nime.org/proc/nime2026_73/)
- 상수 step 비용: Magenta RT 2는 layer별 windowed attention(1–1.6 s)을 쓰고, Notochord와 LK Jam은 GRU의 O(1) step을 쓴다. jam_bot은 context를 120 토큰으로 제한한다. [직접 확인 / 요약만] — [MODEL.md](https://github.com/magenta/magenta-realtime/blob/main/MODEL.md); [arXiv 2606.21018](https://arxiv.org/abs/2606.21018); [NIME 2026 PDF](https://nime.org/proceedings/2026/nime2026_73.pdf)
- GPU warm 유지: 작업 공백이 생기면 macOS가 GPU를 downclock해 frame당 수 ms가 늘어나므로 dummy 연산을 계속 돌린다. [직접 확인] — [core/README.md](https://github.com/magenta/magenta-realtime/blob/main/core/README.md)
- 다중 가설 배치와 draft 모델: mlx-lm `BatchGenerator`, `--draft-model`. oMLX에서 1B batch 2는 합산 약 360 tok/s다(batch 1은 250). [직접 확인 / 요약만] — [generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py); [omlx gdelsbgr](https://omlx.ai/benchmarks/gdelsbgr)
- 턴테이킹 대안: AI Duet은 600 ms 침묵 뒤에 응답한다(동시 연주 대신 call-response). [직접 확인] — [AI.js](https://github.com/googlecreativelab/aiexperiments-ai-duet/blob/master/static/src/ai/AI.js)

### Inferences
128 BPM, 박 468.75 ms, 반 마디 937.5 ms, 마디 1875 ms 기준으로 사용자의 1.8 s를 1마디 미만, 더 나아가 1–2박 수준으로 줄이는 설계안이다. 모두 추정이고 각 수치는 위 근거에서 가져왔다.
1. **Persistent KV + 증분 prefill**(Aria 방식): 코드 입력 MIDI가 들어오는 즉시 그 토큰만 KV에 append한다. 블록 시작 시점에는 decode만 남으므로 현재의 237–408 토큰 re-prefill 비용이 사라진다.
2. **다중 코드 가설 선생성**(SongDriver 캐시 + BatchGenerator): 다음 마디 코드 후보 N개(예: 현재 코드 유지, ii–V 진행 다음 코드, 반음 위 tritone sub)를 batch로 미리 생성해 둔다. 실제 코드가 들어오면 맞는 가설을 즉시 꺼낸다. 틀리면 `trim_prompt_cache`로 롤백하고 첫 박만 다시 생성한다. 1B 모델 기준 batch 2 합산 throughput이 batch 1의 약 1.4배이므로 가설 2–3개는 비용 대비 효율이 좋다.
3. **commit 지평**(ReaLJam/StreamMUSE): 이미 확정한 1–2박은 그대로 재생하고, 새 코드가 들어오면 commit 경계 이후만 재생성한다. 반응 지연은 commit 길이와 생성 시간 중 큰 쪽으로 묶인다. 예컨대 commit 1박이면 약 470 ms다.
4. **첫 박 패치**(Aria beam/onset mask + Notochord whitelist): 코드가 바뀐 직후 첫 1–3 음만 작은 모델(13.4M, p99 219 ms) 또는 제약 샘플링(현재 코드톤 whitelist)으로 즉시 내보낸다. 이어지는 프레이즈는 큰 모델이 commit 이후를 채운다. 이것이 jam_bot의 "100 ms accompaniment / 800 ms melody" 이원 목표와 같은 구조라면 정합적이지만, 그 목표치 자체는 미확인이다.
5. **decode 가속**: MLX로 전환하고 8-bit 가중치를 쓴다(M4 Max에서 bf16→q8 decode 1.66배). 생성 토큰 수도 줄인다(48 → 반 마디분). 출력 지연 보상(base offset)을 넣고, 블록 사이에는 GPU warm-keeping을 둔다.
6. **FL Studio 연동**: FL Studio가 MIDI clock을 보낼 수만 있으므로 Python 쪽이 clock slave가 된다. Aria와 JazzyGPT2처럼 벽시계와 박 그리드에 note를 미리 예약해 보내고 base output latency만큼 앞당긴다. jam_bot NIME 2026의 출력 지연 보상과 같은 원리다.

### Gaps
- 위 패턴들을 0.1–0.7B 심볼릭 Transformer로 M1 Max에서 end-to-end 측정한 공개 사례: 찾지 못했다. Aria-Duet이 가장 가깝지만 칩과 ms 수치가 미확인이다.
- 다중 가설 선생성(chord hypothesis batching)을 음악 시스템에서 명시적으로 평가한 논문: 찾지 못했다(SongDriver의 캐시가 가장 가까운 사례).
- WebSearch 예산 소진과 논문 호스트 차단 때문에 jam_bot, ReaLJam, StreamMUSE, Aria-Duet 원문의 표와 그림 수치는 후속 세션에서 원문을 직접 열어 확인해야 한다(nime.org/proceedings/2026/nime2026_73.pdf, zenodo 17706584, arXiv 2502.21267, 2606.11886, 2511.01663).
