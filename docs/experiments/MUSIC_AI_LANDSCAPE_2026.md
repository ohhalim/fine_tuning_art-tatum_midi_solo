# 음악 AI 기술 지도 2026 — 이 프로젝트에 쓸 수 있는 기술 찾기 (#1764)

작성 2026-10-16, Claude. 아스트라와 논의하려고 만든 조사 문서입니다. 모델을 돌리거나 소리를 들어 보지는 않았습니다.

**목적:** 사용자가 쓸 수 있는 기술을 찾는 것입니다. 최고 성능 모델만 보지 않고, 새롭거나 특이한 방법론, 주류와 다른 방향의 기술까지 넓게 봤습니다.

**프로젝트 목표(기준):**
- 사용자가 왼손으로 코드를 치면, 모델이 오른손 피아노 솔로를 실시간 MIDI로 냅니다. 이 MIDI는 FL Studio/Serum으로 갑니다
- 스타일은 티그랑·비밥 라인이고, 스윙은 싫어합니다
- Mac 로컬에서 돌아야 합니다

**확인 수준 표기**
- **[직접]** 원문 페이지(논문 본문·초록, 공식 문서, README, 모델 카드)를 열어 읽은 것
- **[요약]** 웹 검색 결과 요약만 본 것. 틀릴 수 있습니다
- **[기록]** 이 저장소 문서에 원문 확인 기록이 있는 것(CLAUDE.md, `REALTIME_PARTNER_RESEARCH.md`)
- **[실험]** 이 저장소에서 직접 잰 것
- **[사용자 화면]** 사용자가 브라우저에서 열어 붙여 준 화면 내용

---

## 0. 한눈에 — 이 프로젝트에 바로 의미 있는 것 10개

| # | 기술 | 왜 의미 있나 | 확인 |
|---|---|---|---|
| 1 | **Aria가 사람 음을 문맥에서 "듣게" 하기**(Aria-Duet, jam_bot 방식) | 오늘 실험에서 왼손 PC 조건은 정보가 거의 없었습니다(음당 약 0.003 nats). 반면 Aria는 양손 피아노를 학습해서 옥타브·리듬·보이싱까지 쓸 수 있습니다. 학습 없이 바로 잴 수 있습니다 | [실험] + [기록] |
| 2 | **활성값 조종(MusicRFM, ICLR 2026)** | 동결된 모델의 내부 활성값에 "코드·음 방향"을 더해 학습 없이 제어합니다. MusicGen에서 목표 음 정확도가 0.23에서 0.82로 올랐습니다. Aria에도 옮겨 볼 수 있습니다(미검증) | [직접 초록] |
| 3 | **적대적·anticipatory 학습 목표**(LiveBand, GAPT, StreamMusicGen) | 2026년 실시간 반주 연구들은 공통으로 "정답만 보여 주는 기본 학습으로는 실시간 반응이 안 된다"고 말합니다. 우리 adapter가 그 기본 학습만 썼습니다 | [직접] |
| 4 | **연주자 임베딩 개인화**(Edwards 외, ISMIR 2026) | Aria에 cross-attention으로 피아니스트 12명 조건을 붙였고, 가중치가 공개돼 있습니다. "base 하나 + 교체형 개인화" 구조에 맞습니다 | [직접 모델 카드] |
| 5 | **jam_bot**(MIT, ISMIR 2025) | 사람과 실시간 MIDI 즉흥을 하고 평가까지 한 공개 시스템입니다. 스레드 구조와 역할 전략을 그대로 참고할 수 있습니다 | [기록] |
| 6 | **Somax2**(IRCAM) | 신경망 없이 녹음 말뭉치를 재조합해 실시간으로 공동 즉흥합니다. 말뭉치를 티그랑 클립이나 사용자 연주로 바꾸면 바로 개인화되는 다른 방향입니다 | [요약] |
| 7 | **ImprovNet**(IJCNN 2025) | "망가뜨리고 다시 다듬기" 학습으로 즉흥 강도를 조절합니다. 사용자 릭의 변주를 만드는 데 쓸 수 있습니다. 코드 공개 | [직접 초록] |
| 8 | **표현 렌더링**(Pianist Transformer, ICML 2026) | 박에 딱 맞춘 MIDI에 타이밍·세기·페달을 입혀 표현적인 MIDI로 바꿉니다. 단선율 화성 모델을 쓸 때 피아노 표현을 되살리는 후처리가 됩니다 | [직접 포스터] |
| 9 | **Magenta RealTime 2**(Google, 2026-06) | Mac에서 약 200 ms로 돌고, MIDI 음을 받아 따라갑니다. 출력은 오디오입니다. 나중에 소리 엔진 후보가 되고, 조건을 128음 피아노롤로 준다는 설계 참고가 됩니다 | [직접 공식] |
| 10 | **Moonbeam 조건 가중치**(약 95.9 MB) | 우리 기존 보고서는 "미공개"라고 했지만 HF에 실제로 있습니다. 코드 조건 비교 후보입니다(팝 ComMU 자료라 장르 불일치) | [직접 HF] |

---

## 1. MIDI·악보 생성

### 1.1 기반 모델

| 모델 | 내용 | 이 프로젝트와의 관계 | 확인 |
|---|---|---|---|
| **Aria-medium**(EleutherAI, ISMIR 2025) | 표현적 솔로 피아노 약 6만 시간. LLaMA 3.2 1B 구조. base·gen·embedding·demo 체크포인트. Apple Silicon MLX 실시간 데모. Apache-2.0 | 현재 기반. 2026년에도 이것보다 크면서 공개돼 있고 표현적 피아노에 맞는 모델은 못 찾았습니다 | [직접 README] |
| **Edwards 외 jazz-pianist-style**(ISMIR 2026) | Aria + gated cross-attention(뒤 8개 층) + 피아니스트 12명 임베딩. PiJAMA-12. Apache-2.0. Mehldau·티그랑은 없음 | 개인화 1순위 참고 | [직접 모델 카드] |
| **Moonbeam**(Guo·Dixon, 2025) | 81.6K시간 MIDI. 309M·839M. 상대·절대 속성 attention. ComMU 조건 생성 미세조정 가중치가 HF에 있음 | 코드 조건 비교 후보. 팝 자료, CUDA 중심 | [직접 HF 폴더] + [요약] |
| **Pianist Transformer**(ICML 2026) | MIDI 100억 토큰 자기지도 사전학습. 비대칭 Transformer + 음 단위 압축. 악보 → 표현적 연주 MIDI | 표현 렌더링 후처리 후보. 체크포인트는 프로젝트 페이지에 있다고 함 | [직접 포스터] |
| **MIDI-LLM**(Wu 외, ISMIR 2026) | Llama-3.2-1B에 AMT MIDI 토큰 55,030개를 더함. 텍스트 → 다트랙 MIDI. vLLM으로 빠른 추론 | 텍스트로 스타일을 지시하는 방향의 참고. 실시간 반응은 아님 | [요약] |
| text2midi(AAAI 2025), llama-midi | 텍스트 → MIDI 기준선 | 낮음 | [요약] |
| **NotaGen**(IJCAI 2025) | ABC 악보, 160만 악보 사전학습 + CLaMP-DPO | 클래식 악보. 방법(CLaMP-DPO)만 참고 | [요약] |
| **AMT**(Anticipatory Music Transformer, TMLR 2024) | 사건과 control을 시간 순으로 섞어 미래 조건을 미리 봄. jam_bot의 기반. 벨로시티 없음 | 사람 음을 control로 끼우는 방식의 원형 | [기록] |
| **BEAT**(ICML 2026) | 박 단위 균일 시간 토큰화, 150M. 피아노 이어 치기에서 AMT보다 나은 객관·주관 결과 | 악보 성격 자료, 멜로디 → 반주 방향 | [직접 본문] |
| MIDI-GPT(AAAI 2025) | 마디별 PC 집합 조건, macOS arm64 휠 | 마디 격자, 4/4 중심 | [기록] |
| **MIDI-RWKV**(2025) | RWKV, 적은 표본의 state tuning 개인화, 가중치 공개 | 일정 메모리 스트리밍, 개인화 대안 | [요약] |
| NanoMaestro(2026) | 13.8M·53.9M LSTM, ONNX, CC-BY-NC | 저자원 기준선(아스트라 노트) | [기록 아님, 아스트라 직접] |
| Notochord(IIL) | 실시간 MIDI 확률 모델, MIT, 직접 학습 가능, Homunculus(punch-in 다성) | 상호작용 설계 참고 | [직접 README] |

### 1.2 즉흥·재즈 전용

| 모델 | 내용 | 확인 |
|---|---|---|
| **ImprovNet**(IJCNN 2025) | corrupt-then-refine 자기지도. 장르 전환 즉흥, 즉흥 강도 조절, 멜로디 화성 붙이기. 짧은 이어 치기와 빈칸 채우기에서 AMT보다 우세. 코드 공개 | [직접 초록] |
| MINGUS(ISMIR 2021) | 현재·다음 코드 4음 + 베이스 + 박. WJazzD(관악 단선율, 스윙). HSE 재학습 체크포인트(아스트라 화성 제어 1순위) | [기록] |
| BebopNet(ISMIR 2020) | 4음 코드 임베딩, 색소폰. 화성 지표로 빔 서치를 하면 "코드음만 반복"하는 보상 해킹이 생김 | [기록] |
| Jazz Transformer(2020) | Transformer-XL, WJazzD. 약점을 정량화한 지표 | [요약] |
| CMT(2021) | 프레임 크로마 조건 단선율. 이 프로젝트에서 CPU 8마디 2.5–2.9초 측정 | [기록] |
| Chord-conditioned Melody and Bass(NeurIPS 2025 AI4Music) | 다섯 조건 방식 비교. 베이스를 먼저 만드는 방식이 코드 조건의 이득을 가장 크게 봄 | [요약] |
| Jazz piano overpainting(2024) | 재즈 피아노 변주. QMUL 리드시트 정렬 대형 데이터셋 예고 | [요약] |

### 1.3 조건·제어 방법 (MIDI)

| 방법 | 내용 | 확인 |
|---|---|---|
| **D3PIA**(2026-02) | 이산 디퓨전, 리드시트 → 피아노 반주, 이웃 attention으로 멜로디·코드 정렬. POP909. 연속 디퓨전보다 코드를 잘 지킴. 벨로시티 없음 | [요약] + [아스트라] |
| SYMPLEX | simplex 디퓨전, 외부 가이드 모델 붙이기 쉬움 | [요약] |
| Fine-Grained Guidance(ICML 2025) | 화성·리듬 제약을 학습과 샘플링 양쪽에 | [기록] + [요약] |
| Polyffusion | cross-attention·하드 제어 | [기록] |
| Function Alignment(ISMIR 2025) | 사전학습 LM + 가벼운 어댑터로 코드 → 멜로디. `prompt_lora`가 최선 | [기록] |
| MuseBarControl(2024) | 마디 제어 + 반사실 손실 | [기록] |
| **SymphonyGen**(2026-04) | 박에 맞춘 화성 골격 조건 + CLaMP 3 오디오 보상 GRPO + 불협화 회피 샘플링 | [요약] |
| Keating & Casey 코드톤 그래프(ICMC 2025) | 결정론 코드톤 라인 | [기록] |

---

## 2. 오디오 생성

### 2.1 노래 생성 — 상용 (음악계의 GPT·Opus급)

| 모델 | 최신 | 확인 |
|---|---|---|
| **Suno** | v6, v6-wild, v6-mini(2026-09-09). Studio 2.0(2026-08)에서 MIDI 가져오기·편집과 stem MIDI 내보내기. v5.5(2026-03)에 Custom Models | [직접 릴리스 노트] |
| **Mureka**(쿤룬) | V9.5(2026-08), MusiCoT(음악 생각의 사슬). O3(test-time scaling) 예고 | [요약] |
| **Google Lyria** | 3(2026-02), 3 Pro(2026-03), 3.5(2026-07, Gemini·API 9월) | [요약] |
| **ElevenLabs Music** | v2(2026-05, 곡 중간 장르 전환, 구간 재생성), v2.5(2026-09) | [요약] |
| MiniMax Music | 3.0 | [직접 벤치마크] |
| Qwen-Music(2026-07) | Melody-CoT, DPO + GSPO. 자체 A/B 주장 | [직접 초록] |
| HappyShrimp(알리바바, 2026-08 베타) | 텍스트 → 곡 | [요약] |
| Udio | 2025-03 Allegro v1.5 뒤로 새 모델 없음. 라이선스 플랫폼 미출시 | [요약] |
| Seed-Music(ByteDance, 2024) | AR + 디퓨전, 악보 → 음악, 편집, 목소리 복제 | [요약] |
| Fugatto(NVIDIA, 2024 발표) | 2.5B, 공개 안 됨 | [요약] |

- **독립 블라인드 청취 연구**(A Thousand Listeners, 청취자 1,000명, 2026-08, 중국어 프롬프트): Mureka 65.6% ≈ Suno 62.5%(맞대결 52.4%로 사실상 동률) > Happy Shrimp > MiniMax > Lyria [직접]
- **공식 Music Arena 기악 순위**(CMU G-CLef 운영, 2026-07-31까지의 투표, 30표 이상만) [사용자 화면]

| 순위 | 모델 | Arena Score | 95% CI | 표 | 속도(RTF) | 공개 |
|---|---|---|---|---|---|---|
| 1 | Lyria 3 Pro (Google) | 1394 | ±58 | 241 | 4.57 | 비공개 |
| 2 | **Stable Audio 3 Medium** (Stability) | 1333 | ±38 | 444 | 20.67 | **공개 가중치** |
| 3 | Riffusion FUZZ v1.0 (Producer.ai) | 1225 | ±66 | 220 | 9.04 | 비공개 |
| 4 | Sonauto v2.2 | 1150 | ±41 | 469 | 2.79 | 비공개 |
| 5 | Riffusion FUZZ v1.1 | 1119 | ±43 | 452 | 6.3 | 비공개 |
| 6 | ElevenLabs Music v1 | 1055 | ±40 | 330 | 2.35 | 비공개 |
| 7 | Lyria 3 Clip | 1003 | ±74 | 85 | 3.22 | 비공개 |
| 8 | Sonauto v3 Preview | 950 | ±95 | 41 | 1 | 비공개 |
| 9 | **Magenta RealTime (Large)** | 911 | ±26 | 829 | 1.06 | 공개 가중치 |
| 10 | ACE-Step 1.5 Turbo (1.7B) | 894 | ±55 | 176 | 2.43 | 공개 가중치 |
| 11 | Stable Audio Open | 842 | ±30 | 694 | 2.62 | 공개 가중치 |
| 12 | MusicGen Medium | 838 | ±31 | 569 | 0.45 | 공개 가중치 |
| 13 | **Magenta RealTime 2 (Base)** | 809 | ±69 | 200 | 0.86 | 공개 가중치 |
| 14 | MusicGen Small | 794 | ±35 | 463 | 0.87 | 공개 가중치 |
| 15 | Stable Audio Open Small | 682 | ±44 | 435 | 14.51 | 공개 가중치 |

  - 붙여 준 화면 첫 줄에 "10 ACE-Step 1.5 Turbo … 1394 ±58 241 20.67"이라는 줄이 하나 더 있었습니다. 다른 줄의 값이 섞인 화면 표시 오류로 보고 뺐습니다. 보컬 탭은 받지 못했습니다
  - RTF는 생성 시간 대비 오디오 길이로 보입니다(클수록 빠름). 공식 정의는 확인하지 않았습니다

- **Artificial Analysis 보컬 순위**(AA-Music-Vocal v1.1, 검증된 비공개 평가자의 블라인드 선호, MiniMax Music 3.0 = 1000 기준) [사용자 화면]

| 순위 | 모델 | Elo | ±95% | 표본 |
|---|---|---|---|---|
| 1 (1–2) | **Suno v6** | 1146 | ±20 | 2,458 |
| 2 (1–2) | Suno v6-mini | 1130 | ±20 | 2,585 |
| 3 | **Mureka V9.5** | 1107 | ±20 | 2,499 |
| 4 (4–6) | Mureka V9 | 1057 | ±20 | 2,536 |
| 5 (4–6) | Lyria 3.5 | 1052 | ±19 | 2,548 |
| 6 (4–6) | Lyria 3 Pro | 1051 | ±19 | 2,586 |
| 7 | Eleven Music v2.5 | 1012 | ±19 | 2,656 |
| 8 | MiniMax Music 3.0 | 1000 | 기준 | 2,692 |
| 9–11 | Eleven Music v2 · Eleven Music · MiniMax 2.6 | 999 · 999 · 998 | ±19 | 약 2,600 |
| 12 | MiniMax Music 2.5+ | 985 | ±19 | 2,730 |
| 13 | Sonauto Melodia v3 | 972 | ±20 | 2,679 |

  - 기악 탭은 받지 못했습니다

- **세 평가를 합친 읽기**
  - **보컬(노래)은 Suno v6가 1위**이고, Mureka V9.5가 바로 뒤입니다. 독립 청취 연구(Mureka ≈ Suno)와 방향이 같습니다
  - **기악은 Lyria 3 Pro가 1위**입니다. **공개 가중치 중 최고는 Stable Audio 3 Medium**(2위, 1333)입니다
  - **실시간 모델은 품질 점수가 낮습니다.** Magenta RealTime Large 911(9위), Magenta RealTime 2 Base 809(13위)입니다. 실시간으로 이어 생성할 수 있는(RTF 약 1) 대신 품질을 내준 것으로 보입니다. MRT2 Base가 MRT1 Large보다 낮은데, 표가 200개이고 구간(±69)이 넓어 확정하기 어렵습니다

### 2.2 노래·기악 생성 — 공개

| 모델 | 내용 | 확인 |
|---|---|---|
| YuE(ICLR 2026), YuE2-3B(2026-09, 편집 가능한 멜로디·코드 악보, CC-BY-NC) | 가사 → 노래. YuE2는 제3자 위키에서만 확인 | [요약] |
| ACE-Step 1.5(2026-02) | 공개 노래 생성 선두로 자주 언급. 계획 LM에 GRPO | [요약] |
| LeVo 2 / Tencent SongGeneration(2026-06) | 계층 표현, 다중 선호 DPO | [요약] |
| DiffRhythm 2 | 반자기회귀 디퓨전, Apache-2.0 | [요약] |
| HeartMuLa(2026) | 품질 지표별 DPO 후 체크포인트 병합 | [요약] |
| InspireMusic(알리바바, 2025) | 통합 음악·노래·오디오 | [요약] |
| **Stable Audio 3.0**(2026-05) | Small·Medium 공개 가중치(459M·1.4B), Large 2.7B는 API. 최대 6분 20초, 보컬 없음, LoRA 문서 | [요약] |
| JASCO(Meta) | flow matching. 코드·멜로디(기호) + 드럼(오디오) 조건. CC-BY-NC | [요약] |
| MusicGen 계열: MusiConGen, MusicGen-Chord | 오디오 생성에 코드·리듬 조건 | [요약] |

### 2.3 문맥에 맞춘 반주·stem 생성 (오프라인)

| 모델 | 내용 | 확인 |
|---|---|---|
| StemGen(2024), STAGE(2025) | 믹스를 듣고 악기 하나 stem 생성 | [요약] |
| DARC(2026-01) | 드럼 stem + 비트박스·탭 리듬 조건 | [요약] |
| Diff-A-Riff(Sony CSL, 2024) + DiT 후속 | 반주 공동 창작, consistency 학습 | [요약] |
| S2Accompanist(2026-05) | 의미·구조 유도 디퓨전 반주 | [요약] |

---

## 3. 실시간·상호작용

### 3.1 MIDI → MIDI

| 시스템 | 내용 | 확인 |
|---|---|---|
| **jam_bot**(MIT, ISMIR 2025) | AMT 약 360M, 곡마다 full FT, RTX 4090에서 반주 100 ms·멜로디 조건 800 ms, KV 캐시 + 양자화. 청취 평가에서 연주자 본인 연주와 유의차 없음. 관객은 단조롭다고 평가. 코드 공개 | [기록] |
| **Aria-Duet**(NeurIPS 2025 Creative AI) | Disklavier에서 차례 주고받기. continuous prefill로 전환 지연 감소 | [직접 README] + [요약] |
| **GAPT**(ICLR 2026, Wu·Huang 외) | 멜로디 → 코드 실시간 반주. RL + 함께 학습하는 판별자로 단조로움(보상 해킹) 해결. 전문가 12명 블라인드에서 적응 속도·주도감 유의 개선 | [직접 초록] + [요약] |
| ReaLchords / ReaLJam(Google) | 온라인 코드 반주 RL, 실시간 잼 인터페이스 | [요약] |
| StreamMUSE(RTAS 2026) | 기호, 멜로디 → 반주 스트리밍. 왕복 지연·staleness 모델링. 오픈소스 | [직접 본문] |
| **SAGE-Music**(2025) | 실시간 즉흥용 저지연 기호 생성. 속성별 KV head 공유로 약 30% 가속, 품질 손실 약 0.4%. 최신 버전은 철회 표시가 있어 확인 필요 | [요약] |
| KAIST Scott·DeepPocket(2026-10 자라섬) | 악보 추적 + 솔로 생성 + 실시간 반주. 논문·코드 미확인 | [직접 기사] |
| AIMC 2026 | 공유 주도권 피아노 즉흥 조건 전략, Pure Data 실시간 Transformer 등. 초록만 | [직접 초록] |

### 3.2 오디오 → 오디오

| 시스템 | 내용 | 확인 |
|---|---|---|
| **LiveBand**(Sony CSL Paris, 2026-06) | 인과 오디오 잠재공간의 150M 인과 Transformer + 적대적 시퀀스 학습. 악기 한 stem 반주. RTX 3090 compile 기준 평균 43.6 ms. StreamMusicGen보다 객관·청취 우세. 솔로 아님. 코드 공개는 미기재 | [직접 본문] |
| UCSD LDM(Karchkhadze·Dubnov, 2026-04) | 잠재 디퓨전 + consistency distillation. 0.3–1.4초 주기. M2 Mac 시험. Max/MSP 외부 객체, 코드·체크포인트 공개 | [직접 본문] |
| StreamMusicGen(Wu·Huang 외) | 오디오 스트리밍 반주. 미래 정보 없는 단순 MLE 학습은 부족 → anticipatory·agentic 목적 필요 | [직접 초록] |
| **LMDM**(UCSD·MIT·Adobe, 2026-05) | 양방향 오디오 디퓨전을 블록 자기회귀 스트리밍으로 바꿈. 블록 KV 캐시, ARC-Forcing(Self-Forcing + 적대적 대비 사후학습). "생성 딜레이" 잼. MIT 코드, 가중치 없음(Stable Audio Open에서 시작) | [직접 초록·저장소] |
| Live Music Models / Magenta RT 1(2025) | 오디오 2초 청크, 제어 지연 약 3초 | [기록] |
| **Magenta RealTime 2**(2026-06) | 230M·2.4B, MLX, 프레임 40 ms, 제어 지연 약 200 ms. 텍스트·오디오 스타일 + MIDI 음(128채널 피아노롤 25 Hz). Auto-Strum. 미세조정 예정 | [직접 공식] |
| Lyria RealTime(Gemini API, 실험) | 텍스트·BPM·밀도·밝기·스케일 → 오디오 스트림. 사용자 오디오·MIDI 입력은 문서에 없음 | [직접 공식 문서] |

### 3.3 실시간 음색·합성

| 기술 | 내용 | 확인 |
|---|---|---|
| RAVE / nn~(IRCAM ACIDS) | 실시간 신경 음색 변환, Max·PD, VST | [요약] |
| BRAVE(2025) | 저지연 RAVE | [요약] |
| Ravetable, AFTER, TorchBend(IRCAM 2026) | 잠재 wavetable, 멜로디 제어, 내부 표현 조작 | [요약] |

### 3.4 신경망 아닌 공동 즉흥 시스템

| 시스템 | 내용 | 확인 |
|---|---|---|
| **Somax2**(IRCAM REACH) | 말뭉치 기반 반응형 에이전트 여러 개. 오디오·MIDI 입력. Max + Python. 2025–2026 공연에 쓰임 | [요약] |
| Djazz, Dyci2, ImproteK | 같은 계열(재즈 즉흥 포함). Djazz의 최근 상태 미확인 | [요약] |

### 3.5 실시간 드럼 반주

| 시스템 | 내용 | 확인 |
|---|---|---|
| Haki 외(AIMC 2022, 박사논문 2024) | 실시간 기호 드럼 반주, VST3 | [요약] |
| Memorial Univ. 논문(2025) | 베이스에 반응하는 드럼 cGAN + 벨로시티 Transformer | [요약] |

---

## 4. 오디오와 MIDI 사이를 잇는 기술

| 분야 | 기술 | 내용 | 확인 |
|---|---|---|---|
| 전사 | **aria-amt**(EleutherAI) | seq2seq 피아노 전사, 오프라인, int8 양자화, Apache-2.0, 체크포인트 공개 | [직접 README] |
| 전사 | MuScriptor(2026-07) | 공개 다악기 전사 모델 | [요약] |
| 전사 | YourMT3+, 2025 AMT 챌린지(우승: MusicFM 인코더) | 다악기 | [요약] |
| 실시간 전사 | Wei 외(2025), Hu 외(ISMIR 2025) | 380 ms, 인과 처리. 이전 온라인 128–320 ms, 매끄러운 상호작용 최소 30 ms | [요약] |
| 악보 추적 | **Matchmaker**(ISMIR 2025) | 오픈소스 실시간 피아노 악보 추적, ACCompanion에 쓰임 | [요약] |
| 연주 렌더링 | Pianist Transformer(ICML 2026), PianoKontext(ICML 2026), RenderBox(2025), Peransformer | 무표정 → 표현적 연주(MIDI 또는 오디오) | [직접 포스터] + [요약] |
| MIDI → 오디오 | MIDI-VALLE(ISMIR 2025), Tang 외(2025), CoSaRef | 신경 피아노 합성, 참조 연주로 조건 | [요약] |

---

## 5. 이해·표현·평가

| 기술 | 내용 | 확인 |
|---|---|---|
| Music Flamingo(ICLR 2026, NVIDIA) | 음악 이해 오디오 LM. 음악이론 사고 사슬(MF-Think) + GRPO. 10개 넘는 벤치마크 최고 | [요약] |
| CLaMP 2·3 | ABC·MIDI·텍스트·오디오 검색 임베딩. NotaGen·SymphonyGen의 보상으로 쓰임 | [요약] |
| aria-medium-embedding | MIDI 임베딩(SimCSE식) | [직접 README] |
| MIDI-RAE-JEPA(2026-07), Hawley(2026-08) | JEPA식 계층 표현, piano-roll 이미지 | [직접 초록] |
| 보상·평가 | CMI-RewardBench, SongEval, MuQ-Eval, TuneJury, SongBench, ICME 2026 TTM 챌린지 | [요약] |
| 평가의 한계 | "TTM 모델은 조성·박자 지시를 정말 따르나"(2026): 반사실 평가로 의문 제기 | [요약] |

---

## 6. 새롭거나 다른 방향의 방법론

| 방법론 | 대표 | 핵심 | 이 프로젝트 적용 생각 | 확인 |
|---|---|---|---|---|
| **활성값 조종** | MusicRFM(ICLR 2026) | 학습 없이 내부 활성값에 개념 방향을 더함. 시간에 따라 바뀌는 일정, 여러 속성 동시 적용 | Aria hidden state에서 "현재 코드 PC" 방향을 probe로 찾아 실시간으로 더하기. adapter 학습 대안 | [직접 초록] |
| **적대적 사후학습** | LiveBand, GAPT, LMDM(ARC-Forcing) | teacher forcing 노출 편향·보상 해킹·단조로움을 판별자로 해결 | 학습을 다시 한다면 CE만이 아니라 판별자 신호 | [직접] |
| **Self-Forcing 계열** | LMDM, Self Forcing(영상), Mutual Forcing(2026-04) | 학습 중에 모델 자신의 rollout을 넣음 | 블록 요청 학습에 모델 자신의 이력 섞기 | [요약] + [직접 LMDM] |
| **anticipation** | AMT, jam_bot | control 사건을 앞당겨 끼워 미래 조건을 미리 봄 | 사람 음을 control로 끼우기 | [기록] |
| **선호 학습·RL** | NotaGen CLaMP-DPO, LeVo, HeartMuLa, ACE-Step GRPO, SymphonyGen GRPO, Qwen-Music GSPO, Music Flamingo GRPO | 보상 모델이나 임베딩 판정으로 정렬 | 기준 클립 지표(릭 길이·컴핑·저음)를 보상으로. 단 보상 해킹 주의(BebopNet·GAPT) | [요약] |
| **test-time scaling** | Mureka O3(예고), CMI-RewardBench best-of-N, SCORE | 후보 여러 개 중 보상으로 고르기 | 실시간 N후보 재순위(연구 문서 2안과 같음) | [요약] |
| **생각의 사슬(계획 먼저)** | MusiCoT(Mureka), Melody-CoT(Qwen-Music), MF-Think | 구조·멜로디를 먼저 계획하고 생성 | 블록마다 착지 목표를 먼저 정하는 플래너 | [요약] |
| **consistency distillation** | UCSD LDM, Diff-A-Riff 후속 | 디퓨전 단계 수 축소로 지연 감소 | 오디오 경로를 쓸 때 | [직접 UCSD] |
| **KV head 공유** | SAGE-Music | 속성별 KV head 공유로 추론 가속 | Aria 지연 개선(재학습 필요) | [요약] |
| **상태공간·RNN** | MIDI-RWKV, Mamba 음악 LM, NanoMaestro | 일정 메모리 스트리밍 | 긴 세션 문맥 문제(#1758 crop) 대안 | [요약] |
| **말뭉치 기반 재조합** | Somax2, Djazz | 신경망 없이 녹음 조각 재조합 | 티그랑·사용자 연주 말뭉치로 즉시 개인화 데모 | [요약] |
| **LLM 다중 에이전트 작곡** | ComposerX, CoComposer | GPT 계열이 ABC로 작곡, 역할 분담 | 실시간에는 느림. 기획·변주 아이디어용 | [요약] |
| **JEPA 표현학습** | MIDI-RAE-JEPA, Hawley | 라벨 없이 계층 표현, 화성은 감독 필요 | 낮음 | [직접 초록] |
| **개인화** | Edwards 임베딩, MIDI-RWKV state tuning, Suno Custom Models, MusicGen 장르 LoRA | 소수 파라미터로 스타일 | 연주자 임베딩 우선 | [직접 카드] + [요약] |

---

## 6.5 Google 음악 연구 계보와 안나 황(Cheng-Zhi Anna Huang) 교수

사용자 질문으로 따로 조사했습니다. 안나 황 교수는 Music Transformer를 만든 분이고, 사용자가 이 분야에서 가장 존경하는 연구자입니다.

### 6.5.1 Google 쪽 계보 — 이름이 비슷해서 헷갈리는 모델들

| 시기 | 이름 | 무엇인가 | 실시간 | 공개 | 확인 |
|---|---|---|---|---|---|
| 2018–2022 | Magenta(Google Brain) | Music Transformer(2018), Coconet·Bach Doodle(2019), MIDI-DDSP(2022) 등. 안나 황 교수가 이 시기 핵심 저자 | — | 논문·코드 | [직접 홈페이지] |
| 2023 | MusicLM | 텍스트 → 음악 오디오. 이후 철회됐다는 보도 | 아님 | 비공개 | [요약] |
| 2024 | **MusicFX DJ**(Google Labs) | **Lyria RealTime의 첫 공개 실험** 앱 | 예 | 앱 | [직접 Magenta 페이지] |
| 2025-04 | Lyria 2 | Music AI Sandbox, 48 kHz | 아님 | 비공개 | [요약] |
| 2025-05/06 | **Lyria RealTime API** | 텍스트 프롬프트 혼합 + 템포·키·악기군 끄기·음 밀도·밝기·"chaos" 제어. 변경이 최대 2초 안에 반영. 데모 PromptDJ, PromptDJ MIDI, PromptDJ Pad, DAW용 VST "The Infinite Crate". "Lyria team" | 예 | 실험 API | [직접 Magenta 페이지·Gemini 문서] |
| 2025-06 | **Magenta RealTime**(1) | 800M 자기회귀 Transformer, 스톡 음악 약 19만 시간(주로 기악), 가중치 CC BY 4.0, 코드 Apache 2.0. 2초 블록, 제어 지연 약 3초. **Lyria RealTime의 공개 가중치 사촌** | 예 | 공개 가중치 | [요약] + [논문 직접] |
| 2025-08 | **Live Music Models** 논문(arXiv 2508.04651) | Magenta RT(공개)와 Lyria RT(API)를 소개. 저자 35명 "Lyria Team"에 **안나 황**, Chris Donahue, Jesse Engel, Ian Simon, Adam Roberts, Douglas Eck 등 | — | 논문 | [직접 저자 목록] |
| 2026-02·03·07 | Lyria 3 → 3 Pro → 3.5 | 텍스트 → 보컬 포함 완성곡(최대 약 3분). Gemini 앱·API | 아님 | 비공개 | [요약] |
| 2026-06 | **Magenta RealTime 2** | 230M·2.4B, Apple Silicon MLX, 프레임 40 ms, 제어 지연 약 200 ms, MIDI 음(128채널 피아노롤) 입력, 출력은 오디오, 미세조정 예정 | 예 | 공개 가중치 | [직접 공식] |

- **Music Arena 기악 점수**: Lyria 3 Pro 1394(1위), Lyria 3 Clip 1003, Magenta RealTime Large 911(9위), Magenta RealTime 2 Base 809(13위) [사용자 화면]
- **정리:** "Lyria 3 Pro"는 오프라인 노래 생성기라 즉흥 모델이 아닙니다. 실시간 계열은 Lyria RealTime(API)과 Magenta RealTime(공개)입니다. 둘 다 사용자의 연주를 듣고 솔로로 받아치지는 않습니다. MRT2는 MIDI 음을 따라가지만 출력이 오디오입니다

### 6.5.2 안나 황 교수 — 현재와 연구 흐름

- **현재:** 2024년 가을부터 MIT 부교수입니다(EECS와 음악·연극 MTA 겸임). 새 "Music Technology and Computation" 대학원 프로그램을 시작했고, 첫 학생들이 2026-05-13에 연구 발표회를 했습니다. 2025–26 Radcliffe 펠로우이고, 몬트리올대 겸임 교수입니다. 그 전에 Google Brain/DeepMind Magenta에서 약 8년 있었고, Mila CIFAR AI Chair였습니다 [직접 홈페이지] + [요약]
- **연구 관심(홈페이지 원문 요지):** 상호작용 중심 생성 AI, 음악 이론·인지로 모델 해석과 제어, 음악가가 모델을 이해·조종하는 인터페이스, 사회적·다중 에이전트 RL과 음악가와 에이전트가 함께 즉흥하는 "jam space", 실시간 악기와 공동 즉흥을 위한 효율적 ML [직접]

| 논문 | 연도·학회 | 내용 | 안나 황 | 확인 |
|---|---|---|---|---|
| Music Transformer | ICLR 2019 | 상대 위치 attention으로 긴 기호 음악 생성. 이 저장소의 출발점(13.4M 시기) | 1저자 | [기록] |
| Coconet / Bach Doodle | 2017 / 2019 | 바흐 코랄 화성 붙이기, 대중용 상호작용 | 핵심 저자 | [직접 홈페이지] |
| **ReaLchords** | ICML 2024 | 멜로디에 실시간 코드 반주. MLE 사전학습 → RL(화성·시간 일관성 보상 + 미래를 보는 교사 증류) | 마지막 저자 | [직접 저자] |
| **ReaLJam** | CHI EA 2025 | ReaLchords 기반 실시간 잼 인터페이스. **anticipation**: 에이전트가 앞으로의 계획을 예측해 화면에 보여 줌. 숙련 연주자 사용자 연구 | 마지막 저자 | [직접 저자] |
| **GAPT** | ICLR 2026 | RL 사후학습의 보상 해킹(단조로운 코드)을 판별자로 해결. 다양성·화성·적응 속도·주도감 개선 | 마지막 저자 | [직접 저자] |
| Live Music Models | 2025 | Magenta RT·Lyria RT | 공동저자 | [직접 저자] |
| **StreamMusicGen** | 2025 | 오디오 스트리밍 반주. 미래 가시성(t_f)과 출력 청크 길이(k)의 절충. 단순 MLE로는 부족 | 마지막 저자 | [직접 저자] |
| jam_bot | ISMIR 2025 | MIT Media Lab, Jordan Rudess와 실시간 자유 즉흥(AMT) | 공동저자 | [기록] |
| **LMDM** | 2026 | 오디오 디퓨전 스트리밍, ARC-Forcing. UCSD(McAuley, Berg-Kirkpatrick)와 공동 | 마지막 저자 | [직접 저자] |
| H2H Music Improv | 2026 | 전문 즉흥 듀오의 소통 모델 + 6시간 시청각 데이터셋(연주자별 의도 주석) | 공동저자 | [직접 저자] |

### 6.5.3 이 프로젝트와의 연결 (Claude 판단)

- 안나 황 교수의 최근 연구 흐름("실시간으로 사람과 함께 연주하는 에이전트")은 이 프로젝트 목표와 가장 가까운 연구 계보입니다. CLAUDE.md §3.5에 적어 둔 "jam space" 구상도 같은 줄기입니다
- **바로 가져올 교훈**
  - **온라인 모델은 MLE만으로는 부족하다:** ReaLchords(RL + 미래를 보는 교사 증류), GAPT(판별자로 단조로움 방지), StreamMusicGen(단순 MLE 스트리밍 부족)이 모두 같은 방향입니다. #1760 adapter가 teacher-forced CE만 쓴 것과 바로 연결됩니다
  - **미래 가시성과 청크 길이의 절충:** StreamMusicGen의 t_f·k는 우리 블록 계약의 1박 앞섬(BEAT)·블록 길이(B)와 같은 축입니다. 우리 블록 설계를 그 틀로 다시 볼 수 있습니다
  - **계획을 보여 주는 상호작용:** ReaLJam의 anticipation 화면은 "연주자가 AI가 어디로 갈지 안다"는 설계입니다. 실시간 데모 설계 참고
  - **소통 설계:** H2H의 의도·지각 주석 체계는 상호작용 평가 설계에 쓸 수 있습니다
- **방향 (정정, 2026-10-16):** 처음에는 "ReaLchords·GAPT는 사람이 멜로디, AI가 코드라 우리와 방향이 반대"라고 적었습니다. 공식 저장소 `lukewys/realchords-pytorch`(MIT)를 열어 보니 코드를 받아 멜로디를 내는 온라인 모델의 학습 레시피("RL Training for Melody Model")와 체크포인트 `rl_melody_gail_4_datasets`가 이미 있습니다. 남은 차이는 표현입니다(16분음표 격자, 단선율, 코드 기호 어휘) [직접 저장소]. 이 사실은 #1735 조사(2026-10-11, `realtime_partner_research/realtime_systems_and_apple_silicon.md` 62행)에 이미 있었는데, 이 문서를 쓸 때 대조하지 않았습니다. 자세한 내용은 `ANNA_HUANG_RESEARCH.md`(#1766, PR #1767)
- ReaLJam 로컬 실행과 지연 측정은 #1768에서 진행 중입니다

## 6.6 2026-10 추가: 구글과 주요 회사가 공개한 프로그램·모델

사용자 질문("구글에서 최신 나온 프로그램이나 모델 또 있어?", "수노나 udio 등 잘 만드는 곳에서 뿌린 것")으로 2026-10-16에 추가 조사했습니다. 6.5와 2절 표에 없는 것, 또는 더 자세히 확인한 것만 적습니다.

### 6.6.1 Magenta RealTime 2(MRT2)는 무엇을 하나 [직접 공식 페이지 2곳]

한 줄 요약: MIDI 건반이나 글로 지시하면 그에 맞는 **오디오를 끊김 없이 계속 만드는 "AI 악기"**입니다. 완성곡을 한 번에 뽑지 않고, 연주하는 동안 계속 반응합니다.

| 항목 | 내용 |
|---|---|
| 공개 | 2026-06-04, 공개 가중치. 코드 github.com/magenta/magenta-realtime, 가중치 huggingface.co/google/magenta-realtime-2, `pip install magenta-rt`(JAX/MLX) |
| 라이선스 | 가중치 CC BY 4.0, 코드 Apache 2.0. 추가 조건에 "생성물에 대해 구글은 권리를 주장하지 않음" [직접 모델 카드] |
| 학습 데이터 | 스톡 음악 약 7만 1천 시간, 대부분 기악. 1세대(약 19만 시간)보다 적음. 장르 범위 한계와 가사 없는 목소리는 1세대와 같다고 적힘. 재즈 언급 없음 [직접 모델 카드] |
| 입력 | 스타일: 텍스트 또는 오디오 조각(MusicCoCa 임베딩). MIDI: 128채널 피아노롤(음높이마다 1채널) + 드럼 켜기/끄기 |
| 출력 | 48 kHz 스테레오 오디오(SpectroStream 코덱, 25 Hz 토큰 프레임, 프레임당 RVQ 12토큰). MIDI 출력은 없음 |
| 크기 | Base 2.4B, Small 230M |
| 속도 | 프레임 40 ms, 제어 지연 약 200 ms(1세대: 2초 청크, 약 3초). 실제 지연은 입출력 버퍼에 따라 늘어남 |
| 하드웨어 | C++ + MLX 엔진. Base 실시간은 M3 Pro·M2 Max 이상, Small은 모든 Apple Silicon. **이 Mac(M1 Max)은 Base 실시간 조건 목록에 없음** |
| 구조 | 디코더 전용, 프레임 단위 자기회귀, 슬라이딩 윈도 주의(KV 캐시 크기 고정), 학습형 attention sink, NoPE |
| 제어 | Auto-Strum(사람이 어떤 음이 켜져 있는지만 주면 모델이 시작 시점을 정함) / Auto-Strum 끔(시작 시점까지 지정, 4토큰 어휘). onset masking으로 한 모델이 두 방식을 모두 지원. 피아노롤 음역 마스킹으로 화음 추가 허용 정도 조절. 신호별 classifier-free guidance |
| 미세조정 | "예정"이라고만 되어 있음. 공개된 근거는 찾지 못함 [요약 검색] |
| 번들 | Jam(연주 앱, 스타일 프리셋), Collider(2D 면에서 프롬프트 섞기), MRT2(AU 플러그인 겸 앱: MIDI 화음 따라 합주, 텍스트로 악기 만들기, 오디오 조각 복제, 카메라·LFO 제스처 조절), Max/PD/SuperCollider 확장. 첫 실행 때 가중치 받음(Base 약 2.5 GB, Small 약 450 MB) |

- 사용자가 이 번들을 `~/Downloads/음악작업들/MRT2 Bundle`에 받아 두었습니다. macOS 권한 때문에 Claude는 폴더를 열지 못해 내용은 확인하지 못했습니다
- **이 프로젝트와의 관계 [판단]:** 사람이 코드를 누르면 그 화음에 맞는 소리를 냅니다. MIDI 솔로는 내보내지 않으므로 솔로 생성기가 아니라 우리 솔로 뒤에 깔 반주·음색 엔진 후보입니다. Small(230M)이라면 이 Mac에서도 실시간으로 돌아갑니다

### 6.6.1a MRT2는 솔로를 하나, 반주를 하나 — 사용자 경험과 분석 (2026-10-16)

사용자가 받은 MRT2 번들을 직접 써 보고 "이상하게 나온다", "재즈 솔로잉까지 되는 거냐", "솔로를 해 주는 거냐, 코드 반주를 해 주는 거냐"고 물었습니다. 사용자는 세부 동작을 잘 모른다고 해서, 아스트라가 판단할 수 있도록 근거를 나눠 적습니다.

**결론 [판단, 아래 근거]:** MRT2는 **솔로 생성기가 아니라 "사람이 정한 음을 지정한 스타일의 소리로 연주해 주는 악기"**입니다. 코드를 넣으면 그 코드를 울리는 반주·합주 소리가 나오는 것이 설계상 정상입니다. 코드 위에 즉흥 선율을 새로 만든다는 설명은 공식 문서 어디에도 없습니다.

근거 [직접 공식 페이지·모델 카드]

1. **MIDI 조건의 형식**: 프레임(40 ms)마다 128개 음높이 각각에 네 상태 중 하나를 줍니다. 꺼짐 / 지속 / 시작 / "지속인지 시작인지 모델이 결정". 네 번째 상태는 이미 켜진 음을 언제 다시 칠지(리듬)를 모델에 맡기는 것입니다. 새 음높이를 고르게 하는 상태가 아닙니다
2. **Auto-Strum**: 사람은 어떤 음이 켜져 있는지만 주고, 모델이 시작 시점을 정합니다. 이름 그대로 "스트럼(코드 긁기) 리듬을 모델이 정한다"는 기능입니다
3. **번들 MRT2 플러그인 설명**: 음이나 코드를 누르고 있으면 그 화음을 따라가는 합주(ensemble)를 만든다
4. **추론 때 마스킹**: 피아노롤 일부(음역 등)를 가리면 모델의 자유도가 바뀝니다. 공식 예시는 "화음을 더 얹게 허용"과 "친 음으로만 제한" 두 가지입니다
5. **모델 카드**: 솔로나 리드 라인 생성에 대한 언급이 없습니다. 쓰임새에는 "라이브 공연과 즉흥(improvisation)"이 있지만, 사람이 즉흥하는 악기라는 뜻으로 읽힙니다 [판단]
6. **품질**: Music Arena 기악 순위 Base 809점, 13위(2절). 실시간 모델은 전반적으로 낮습니다

사용자가 "이상하게 나왔다"의 원인 후보

| 후보 | 근거 수준 | 내용 | 가르는 방법 |
|---|---|---|---|
| H1. 기능이 다름 | 문서 근거 있음 | 코드를 치면 솔로가 아니라 그 코드의 반주가 나옴. 사용자는 솔로를 기대 | 사용자에게 "솔로 없이 코드만 울렸나" 확인 |
| H2. 사양 부족 | 추측 | 이 Mac(M1 Max)은 Base 실시간 조건(M2 Max·M3 Pro 이상)에 없음. Base로 돌렸다면 생성이 밀려 끊김·지글거림 가능 | "소리가 끊기거나 지글거렸나" 확인. Small로 바꿔 다시 들어 보기 |
| H3. 학습 데이터 | 추측 | 스톡 음악 위주라 재즈 즉흥 어법이 약할 수 있음 | H1·H2를 배제한 뒤에만 볼 수 있음 |

- 사용자에게 H1과 H2를 가르는 질문을 했습니다("끊기는 쪽인가, 코드만 울리는 쪽인가"). 답은 아직 없습니다
- 사용자가 어느 모델(Base/Small)로 돌렸는지도 모릅니다

우리 프로젝트와 비교

| | MRT2 | 우리가 만드는 것 |
|---|---|---|
| 사람 입력 | 코드나 음(MIDI), 스타일 텍스트·오디오 | 왼손 코드(MIDI) |
| AI가 하는 일 | 정해진 음을 스타일에 맞는 소리로 연주(반주·합주) | 그 코드 위에 오른손 솔로를 새로 만듦 |
| 출력 | 48 kHz 오디오 | MIDI(FL Studio/Serum으로) |
| 개인화 | 미세조정 "곧 지원"(미공개). 텍스트·오디오 예시로 분위기만 | 티그랑·사용자 기준 클립으로 학습 목표 |
| 지연 | 제어 약 200 ms + 입출력 버퍼 | 블록 계약(반 마디 937.5 ms) 기준 설계 중 |

MRT2를 쓸 수 있는 방법 (아스트라 판단 요청)

1. **소리 엔진으로 붙이기**: 우리 모델이 만든 오른손 MIDI와 사용자 왼손 MIDI를 MRT2 MIDI 조건으로 넣어, 스타일 있는 소리로 렌더링합니다. 친 음으로만 제한하면 우리 솔로의 음은 그대로 두고 음색과 연주감만 입힐 수 있습니다(미검증). 지연이 약 200 ms 더해집니다
2. **반주 담당으로 쓰기**: 사용자 왼손을 MRT2에 넣어 반주·합주를 만들고, 오른손 솔로는 우리 모델이 MIDI로 맡습니다. 역할이 겹치지 않습니다
3. **솔로가 나오는지 시험해 보기(가설)**: 왼손 코드는 주고, 높은 음역은 마스킹해 모델 자유로 두면 그 음역에 선율 비슷한 것이 생기는지 봅니다. 공식 예시는 "화음을 얹는 것"까지라 솔로가 나온다는 근거는 없습니다. 나오더라도 출력이 오디오라 MIDI로 받지 못합니다
4. **비교 기준선으로 쓰기**: 우리 시스템 데모를 만들 때 "구글 공개 실시간 모델은 이 입력에 이렇게 반응한다"는 비교 대상으로 씁니다

Claude 의견: 1·2번이 역할 구분이 분명합니다. 3번은 사용자의 "MRT2로 끝난 것 아니냐" 질문에 직접 답하는 시험이라 10분 정도의 청취 확인 가치는 있습니다. 다만 결과가 좋아도 MIDI 출력·개인화 요구는 남습니다.

### 6.6.2 구글이 2025-06 이후 공개한 것 [직접 Magenta 블로그 목록] + [요약]

| 날짜 | 이름 | 무엇 | 확인 |
|---|---|---|---|
| 2025-06-12 | Lyria RealTime API | 텍스트와 음악 파라미터로 실시간 음악 생성 실험 API | [직접 블로그 목록] |
| 2025-06-20 | Magenta RealTime 1 | 공개 가중치 실시간 모델 1세대 | [직접 블로그 목록] |
| 2025-07-09 | The Infinite Crate | Lyria RealTime을 DAW에 넣는 플러그인 시제품. **2026-03-09 Apache 2.0으로 전체 오픈소스** | [직접 블로그 목록] |
| 2025-11-03 | Space DJ | 장르 별이 있는 은하를 우주선으로 날면 음악이 바뀌는 웹앱(Lyria RealTime) | [직접 블로그 목록] |
| 2025-12-03 | Lyria Camera | Gemini 이미지 이해 + Lyria RealTime으로 주변 풍경에 맞춘 음악 | [직접 블로그 목록] |
| 2026-02-18 | Lyria 3 | Gemini 앱, 30초 곡 | [요약] |
| 2026-02-24 | ProducerAI(옛 Riffusion) 구글 랩스 합류 | | [요약 TechCrunch] |
| 2026-03 | Lyria 3 Pro | 최대 약 3분 | [요약] |
| 2026-04-18 | **Flow Music** | ProducerAI를 바꾼 구글 랩스 음악 스튜디오(Lyria 3 기반) | [요약] |
| 2026-06-04 | **MRT2** | 6.6.1 | [직접] |
| 2026-07 | Flow Music Spaces 확장 | 곡 생성·편집, stem 분리, 이미지·가사·영상 생성("vibe coding" 도구) | [요약] |
| 2026-07-24 | MRT2 커뮤니티 정리 | 해커톤·악기 사례 모음. 새 출시 아님 | [직접 블로그 목록] |
| 2026-07-29 | Lyria 3.5 | Flow Music에 먼저, Gemini API는 9월이라는 보도 | [요약] |
| 2026-10-06 | **Flow Music Spaces → VST3/AU 내보내기** | 자연어로 만든 악기·효과를 DAW 플러그인으로 내보냄 | [요약, 구글 블로그 검색 결과] |

- MusicFX·MusicFX DJ 종료(2026-07-31)와 Music AI Sandbox 신규 접수 중단은 2차 사이트 보도뿐이고 구글 공식 발표는 찾지 못했습니다 [요약]
- Lyria RealTime API의 2026년 변경은 찾지 못했습니다

### 6.6.3 다른 회사가 공개한 프로그램·모델

| 회사 | 공개한 것 | 형태 | 이 프로젝트에 쓸 수 있나 [판단] | 확인 |
|---|---|---|---|---|
| **Suno** | v6·v6-wild·v6-mini(2026-09). Studio 2.0(2026-08-13): 웹 DAW에 MIDI 녹음·편집, 웨이브테이블 신스, 대화형 도우미, 사용자 이펙트, 자동화, stem 분리 | 웹, 최상위 요금제. VST 호스팅·DAW 동기화·데스크톱 설치 없음. MIDI 내보내기는 출처마다 다름 | 실시간 솔로 아님. 참고용 | [직접 릴리스 노트(2절)] + [요약] |
| **Udio** | 2025-10 UMG, 2025-11 WMG와 합의. 라이선스 플랫폼 "2026 출시" 발표 | 실제 출시 근거 못 찾음. 2025-03 이후 새 모델 없음 | 아니오 | [요약] |
| **Stability AI** | Stable Audio 3(2026-05-20). Small Music·Small SFX(459M, CPU, 최대 2분)·Medium(1.4B) 가중치 공개, Large는 API. 보컬 없음, 학습 데이터 라이선스 주장. Mac AU/VST3 플러그인 초기 베타 | 공개 가중치 + 플러그인 | 기악 공개 가중치 최고 점수(Arena 2위). 오디오 출력이라 솔로 생성기로는 못 씀 | [요약 공식 발표 검색] |
| **ElevenLabs** | Music v2(2026-05-26, 곡 중간 장르 전환, 구간 재생성, API 가격 최대 50% 인하), v2.5(2026-09). stem 분리 API(2·4·6 stem) | API. 플러그인 없음 | 아니오 | [요약] |
| **Mureka**(쿤룬) | V9.5(2026-08). 보컬 순위 3위 | 서비스·API | 아니오 | [요약] |
| **MiniMax** | Music 3.0 | API | 아니오 | [직접 벤치마크(2절)] |
| **ACE-Step** | 1.5(2026-01-28). XL 4B까지 base·sft·turbo(8스텝) 공개. MLX·GGUF 변환판 있음. 12 GB 이상 VRAM 권장 | 공개 가중치 | 오디오 노래 생성. 아니오 | [직접 HF 목록] |
| **HeartMuLa** | 3B(2026-02), MuLaCover 커버·리믹스(2026-09) | 공개 가중치 | 아니오 | [요약, 자체 주장] |
| **YuE2** | 약 3B, 편집 가능한 멜로디·코드 악보 | 공개 가중치(CC-BY-NC) | 아니오 | [요약] |
| **Splice** | Variations·Craft·Magic Fit: 샘플 라이브러리 위 생성 기능 | 서비스 | 아니오 | [요약] |
| **MIDI Agent** | LLM으로 MIDI를 만드는 플러그인. VST3/AU/AAX/단독 앱, v1.3.0(2026-09), 49달러. 사용자가 LLM 공급자 선택 | 플러그인. 음악 전용 학습 모델 아님 | MIDI는 내지만 실시간 반응 솔로 아님 | [요약, 판매사 페이지] |
| **MIDI Lab AI** | 브라우저에서 프롬프트 → MIDI, Web MIDI로 신스·DAW 전송, 여러 명 잼 세션 | 웹 | 확인 필요 | [요약, 판매사 페이지] |
| **Hookpad Aria**(Hooktheory) | 멜로디·코드 리드시트 생성. 안나 황 교수 팀 MIDI-LLM(ISMIR 2026)이 이 제품 사용자 58명, 생성 4,002개로 실사용 평가 | 웹 | 리드시트용. 실시간 아님 | [직접 MIDI-LLM 초록] |
| NVIDIA·Meta | 2026년 새 음악 가중치 공개를 찾지 못함 | — | — | [요약 검색] |
| Apple Logic Session Players, Ableton, Yamaha | 2026 변경을 찾지 못함(한 번 검색) | — | — | [요약 검색] |

### 6.6.4 정리 [판단]

- 잘 만드는 회사들이 공개한 것은 거의 전부 **오디오를 출력하는 노래·반주 생성기**입니다
- **사람 연주를 듣고 실시간으로 MIDI 솔로를 받아치는 제품이나 공개 모델은 찾지 못했습니다.** 가장 가까운 것은 MRT2(MIDI 입력, 오디오 출력)와 ReaLJam(MIDI, 코드 반주)입니다. 우리가 노리는 자리가 비어 있다는 근거는 되지만, 이번 조사가 완전하다는 뜻은 아닙니다
- 바로 쓸 만한 것
  1. ReaLJam 구조를 솔로 방향으로 옮기기(#1766 F0~F4)
  2. MRT2 Small을 우리 MIDI 솔로 뒤의 반주·음색 엔진으로 붙여 보기(나중)
- 판매용 MIDI 플러그인(MIDI Agent 등)은 범용 LLM 기반이라 우리 기준(코드 조건 사용, 실시간)과 다릅니다

## 7. 이 프로젝트에 적용할 후보 (Claude 제안, 아스트라와 논의할 것)

순서는 비용이 낮고 정보가 큰 것부터입니다.

1. **"듣게 하기" 측정(학습 없음)**
   - 같은 블록 요청에서 오른손 window NLL을 세 문맥으로 비교합니다. (a) 모델 이력만 (b) s_k까지 관측한 사람 왼손 음을 같은 스트림에 끼움 (c) 다른 곡 왼손을 끼움
   - 주의: Aria dur 토큰은 아직 안 뗀 음의 미래 길이를 담습니다. dur를 s_k까지 자르거나 placeholder를 쓰는 규칙이 필요합니다
2. **활성값 조종 시험(MusicRFM 방식)**
   - Aria hidden state에서 조건 PC를 맞히는 probe를 학습합니다. 오늘 #1756 내보내기에 라벨이 이미 있습니다
   - 그 방향을 실시간으로 더해 생성 음의 PC가 바뀌는지 봅니다. 학습은 probe만 합니다
3. **학습을 다시 한다면 목표를 바꾼다**
   - CE만이 아니라 적대적 신호(LiveBand·GAPT)나 모델 자신의 rollout(Self-Forcing)을 씁니다
4. **개인화는 Edwards 방식**
   - 연주자 임베딩입니다. 티그랑·Mehldau 임베딩은 새로 학습해야 하고 미검증입니다
5. **다른 방향 데모: Somax2**
   - 티그랑 기준 클립과 사용자 연주를 말뭉치로 넣어 실시간 공동 즉흥을 바로 해 봅니다
   - 신경망 경로와 비교할 기준선이 되고, "3분 안에 보여 주기" 목표에도 맞습니다
6. **런타임 참고**
   - jam_bot 스레드 구조, Aria-Duet continuous prefill, StreamMUSE의 지연·staleness 측정법을 씁니다
7. **표현 후처리**
   - 단선율 화성 모델(MINGUS 등)을 쓰게 되면 Pianist Transformer로 피아노 표현을 입힙니다
8. **소리**
   - 지금처럼 MIDI → Serum입니다. Magenta RT2는 나중에 소리 엔진 후보입니다

## 8. 확인하지 못한 것과 한계

- [요약] 항목은 원문을 열지 않았습니다. 특히 SAGE-Music(철회 표시), YuE2(제3자 위키), Mureka·Lyria·ElevenLabs의 세부 수치는 원문 확인이 필요합니다
- 공식 Music Arena 순위표는 제 도구로는 열리지 않아(서버 오류, JavaScript 페이지) 사용자가 열어 붙여 준 화면으로 반영했습니다. 공식 순위표의 보컬 탭과 Artificial Analysis의 기악 탭은 받지 못했습니다
- Notochord Homunculus PDF는 글자를 추출하지 못했습니다. Somax2 원문 페이지는 405 오류였습니다
- 이 문서의 어떤 모델도 돌려 보거나 들어 보지 않았습니다. 순위는 공개 구현과 과제 적합성에 대한 판단입니다
- 아스트라 노트(`research_notes/2026 개인화 피아노 모델 비교/`)의 NanoMaestro, Magenta RT2, LK_Jam, FIGARO 조사는 아스트라가 직접 확인한 것입니다

## 출처

**MIDI·악보**
- Aria: https://github.com/EleutherAI/aria · https://arxiv.org/abs/2506.23869
- Edwards jazz-pianist-style: https://huggingface.co/drewbie/jazz-pianist-style-generator · https://arxiv.org/html/2610.02918v1
- Moonbeam: https://arxiv.org/abs/2505.15559 · https://huggingface.co/guozixunnicolas/moonbeam-midi-foundation-model/tree/main/commu_con_gen_finetune_weights
- Pianist Transformer: https://icml.cc/virtual/2026/poster/61542
- MIDI-LLM: https://arxiv.org/abs/2511.03942
- text2midi: https://ojs.aaai.org/index.php/AAAI/article/view/34516
- NotaGen: https://arxiv.org/abs/2502.18008
- AMT: https://arxiv.org/abs/2306.08620
- BEAT: https://arxiv.org/abs/2604.19532
- MIDI-RWKV: https://arxiv.org/abs/2506.13001
- Notochord: https://github.com/Intelligent-Instruments-Lab/notochord
- ImprovNet: https://arxiv.org/abs/2502.04522
- Jazz Transformer: https://arxiv.org/abs/2008.01307
- Chord-conditioned Melody and Bass: https://arxiv.org/abs/2511.08755
- Jazz piano overpainting: https://arxiv.org/abs/2412.04610
- D3PIA: https://arxiv.org/html/2602.03523v1
- SYMPLEX: https://themoonlight.io/de/review/symplex-controllable-symbolic-music-generation-using-simplex-diffusion-with-vocabulary-priors
- Fine-Grained Guidance: https://arxiv.org/abs/2410.08435
- SymphonyGen: https://arxiv.org/abs/2604.25498
- MIDI-RAE-JEPA: https://arxiv.org/abs/2607.14537
- MIDI-REPA: https://arxiv.org/abs/2610.20967

**오디오 생성**
- Suno: https://suno.com/release-notes · https://techcrunch.com/2026/09/09/suno-replaces-its-ai-models-with-a-new-one-trained-on-licensed-music-as-copyright-suits-pile-up/
- A Thousand Listeners: https://aimusicbenchmark.org/
- Music Arena: https://huggingface.co/spaces/MusicArena/MusicArenaBoard · https://arxiv.org/abs/2507.20900
- Lyria 3.5: https://blog.google/innovation-and-ai/models-and-research/google-labs/lyria-3-5/
- Mureka V9.5: https://www.creativeainews.com/blog/mureka-v9-5-ai-music-model-vocals-2026/
- ElevenLabs Music: https://techcrunch.com/2026/05/27/elevenlabss-new-music-generation-model-can-switch-genres-mid-track/
- Qwen-Music: https://arxiv.org/abs/2607.11699
- HappyShrimp: https://www.bloomberg.com/news/articles/2026-08-17/alibaba-launches-ai-music-generation-model-happyshrimp-in-beta
- Udio: https://www.musegen.ai/blog/udio-review-2026
- Seed-Music: https://arxiv.org/abs/2409.09214
- Fugatto: https://www.musicbusinessworldwide.com/nvidia-unveils-ai-audio-generator-fugatto-that-can-produce-sounds-never-heard-before/
- YuE: https://proceedings.iclr.cc/paper_files/paper/2026/hash/a6a21421022d1da16eadbba533980530-Abstract-Conference.html · YuE2: https://comfyui-wiki.com/en/models/yue2
- ACE-Step 1.5: https://arxiv.org/abs/2602.00744
- Tencent SongGeneration: https://huggingface.co/tencent/SongGeneration
- DiffRhythm 2: https://huggingface.co/ASLP-lab/DiffRhythm2
- InspireMusic: https://comfyui-wiki.com/en/news/2025-02-11-alibaba-open-source-inspiremusic
- Stable Audio 3.0: https://the-decoder.com/stability-ai-launches-stable-audio-3-0-with-up-to-six-minute-tracks-and-open-weights/
- JASCO: https://pages.cs.huji.ac.il/adiyoss-lab/JASCO/
- MusiConGen: https://arxiv.org/abs/2407.15060 · MusicGen-Chord: https://arxiv.org/abs/2412.00325
- StemGen: https://arxiv.org/abs/2312.08723 · STAGE: https://arxiv.org/abs/2504.05690 · DARC: https://arxiv.org/abs/2601.02357
- Diff-A-Riff: https://arxiv.org/abs/2406.08384 · 후속: https://arxiv.org/abs/2410.23005
- S2Accompanist: https://arxiv.org/abs/2605.17414

**Google 계보·안나 황**
- Live Music Models: https://arxiv.org/abs/2508.04651
- Magenta RealTime: https://magenta.tensorflow.org/magenta-realtime · https://simonwillison.net/2025/Jun/20/magenta-realtime
- Lyria RealTime: https://magenta.tensorflow.org/lyria-realtime · https://techcrunch.com/2025/05/20/google-brings-a-music-generating-ai-model-to-its-api-with-lyria-realtime
- Lyria 2: https://alternativeto.net/news/2025/4/google-expands-music-ai-sandbox-toolset-with-lyria-2-model-for-professional-music-creation/
- 안나 황 홈페이지: https://czhuang.github.io/ · MIT MTA: https://mta.mit.edu/node/14797 · Radcliffe: https://www.radcliffe.harvard.edu/people/cheng-zhi-anna-huang · MIT 음악기술 발표회: https://www.media.mit.edu/articles/inaugural-music-technology-research-showcase-celebrates-work-of-new-graduate-program-s-initial-students/
- ReaLchords: https://proceedings.mlr.press/v235/wu24c.html · ReaLJam: https://arxiv.org/abs/2502.21267
- GAPT: https://proceedings.iclr.cc/paper_files/paper/2026/hash/e6731a1d3eddafeb8c94669a1d82fc5d-Abstract-Conference.html
- H2H Music Improv: https://arxiv.org/abs/2608.13957

**2026-10 추가 조사(6.6)**
- MRT2: https://magenta.withgoogle.com/magenta-realtime-2 · 번들: https://magenta.withgoogle.com/mrt2 · 모델 카드: https://huggingface.co/google/magenta-realtime-2
- Magenta 블로그 목록: https://magenta.withgoogle.com/blog/
- Flow Music 플러그인 내보내기: https://blog.google/innovation-and-ai/models-and-research/google-labs/create-music-production-plugins-google-flow/
- Flow Music Spaces 확장: https://www.musicbusinessworldwide.com/googles-flow-music-adds-song-generation-editing-and-stem-splitting-to-spaces-its-vibe-coding-tool/
- ProducerAI 구글 랩스 합류: https://techcrunch.com/2026/02/24/music-generator-producerai-joins-google-labs/
- Lyria 3.5: https://www.techtimes.com/articles/322113/20260729/googles-lyria-35-sharpens-vocals-lyrics-while-rivals-fight-court.htm
- Suno Studio 2.0: https://routenote.com/radar/suno-studio-2-0-launches-with-midi-support-ai-chat-and-advanced-stem-separation/
- UMG–Udio: https://www.musicbusinessworldwide.com/universal-music-settles-udio-lawsuit-strikes-deal-for-licensed-ai-music-platform/ · WMG–Udio: https://www.musicbusinessworldwide.com/warner-music-settles-udio-lawsuit-strikes-licensing-deal-with-ai-platform/
- Stable Audio 3: https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models
- ElevenLabs Music v2 API: https://www.createwith.com/tool/elevenlabs/updates/elevenlabs-opens-music-v2-api-for-developers-to-embed-ai-music-generation
- ACE-Step 1.5: https://github.com/ace-step/ACE-Step-1.5 · https://huggingface.co/papers/2602.00744
- HeartMuLa: https://github.com/HeartMuLa/heartlib
- MIDI Agent: https://www.midiagent.com/ · MIDI Lab AI: https://www.midilab.ai/
- Splice 등: https://resources.rework.com/tools/ai-tools/best-ai-tools-for-music-production-2026

**실시간·상호작용**
- jam_bot: https://www.media.mit.edu/publications/the-jam_bot-a-real-time-system-for-collaborative-free-improvisation-with-music-language-models/
- Aria-Duet: https://arxiv.org/abs/2511.01663
- GAPT: https://arxiv.org/abs/2511.17879 · https://iclr.cc/media/iclr-2026/Slides/10010592.pdf
- ReaLchords: https://arxiv.org/abs/2506.14723
- StreamMUSE: https://arxiv.org/abs/2606.11886
- SAGE-Music: https://arxiv.org/abs/2510.00395
- KAIST Scott: https://www.koreatimes.co.kr/business/tech-science/20261016/ai-artists-perform-improvised-jazz-at-jarasum-jazz-festival
- AIMC 2026: https://aimc2026.org/proceedings
- LiveBand: https://arxiv.org/abs/2606.03803
- UCSD LDM: https://arxiv.org/abs/2604.07612
- StreamMusicGen: https://arxiv.org/abs/2510.22105
- LMDM: https://arxiv.org/abs/2605.22717 · https://github.com/ZacharyNovack/live-music-diffusion-models
- Live Music Models: https://arxiv.org/abs/2508.04651
- Magenta RealTime 2: https://magenta.withgoogle.com/magenta-realtime-2
- Lyria RealTime: https://ai.google.dev/gemini-api/docs/realtime-music-generation
- RAVE·BRAVE: https://arxiv.org/abs/2503.11562 · https://pypi.org/project/acids-rave
- Somax2: https://brahms.ircam.fr/en/media/77d9a7d56fb5e2410173
- 실시간 드럼: https://github.com/behzadhaki/MonotonicGrooveTransformer · https://research.library.mun.ca/16957

**오디오와 MIDI 사이를 잇는 기술·이해·방법론**
- aria-amt: https://github.com/EleutherAI/aria-amt
- MuScriptor: https://arxiv.org/abs/2607.08168 · 2025 AMT 챌린지: https://arxiv.org/abs/2603.27528
- 스트리밍 전사: https://arxiv.org/abs/2503.01362 · https://arxiv.org/abs/2509.07586
- Matchmaker: https://arxiv.org/abs/2510.10087
- PianoKontext: https://icml.cc/virtual/2026/77348 · RenderBox: https://arxiv.org/abs/2502.07711 · MIDI-VALLE: https://ismir2025program.ismir.net/poster_208.html
- Music Flamingo: https://arxiv.org/abs/2511.10289
- MusicRFM: https://arxiv.org/abs/2510.19127
- Self Forcing: https://arxiv.org/abs/2506.08009
- 선호 정렬 서베이: https://arxiv.org/abs/2511.15038 · CMI-RewardBench: https://arxiv.org/abs/2603.00610 · TuneJury: https://arxiv.org/abs/2606.17006
- Mamba 음악 LM: https://arxiv.org/abs/2507.06674
- ComposerX: https://arxiv.org/abs/2404.18081 · CoComposer: https://arxiv.org/abs/2509.00132
- ICME 2026 TTM 챌린지: https://arxiv.org/abs/2605.21538
