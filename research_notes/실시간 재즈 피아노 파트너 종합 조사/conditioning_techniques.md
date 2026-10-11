# 심볼릭(MIDI) 생성 모델에 조건/제어를 붙이는 기법 — 사전학습 모델 + 소량 라벨 + M1 Max 기준 비교 (2017–2026)

검증 방법과 태그 (이 문서 전체에 적용):
- **[직접 확인]**: 이 세션에서 GitHub 공개 저장소를 `git clone` 해서 코드와 README를 직접 읽은 항목이다. commit 해시를 함께 적었다.
- **[요약만·원문발췌]**: 이 세션의 egress 프록시가 arxiv.org, export.arxiv.org, ar5iv, alphaxiv, openreview.net, huggingface.co, ISMIR archives/program, Semantic Scholar, deepai, crfm.stanford.edu, 저자 개인 사이트를 모두 막았다(WebFetch는 `getaddrinfo ENOTFOUND`, curl은 `CONNECT tunnel failed, response 403`). 그래서 이 태그의 논문 내용은 웹검색 엔진이 원문 페이지에서 뽑은 발췌로만 확인했다. 수치는 PDF로 다시 대조해야 한다.
- **[요약만·2차]**: 2차 리뷰나 요약 사이트를 거친 내용이다.
- 조사 도중 세션에서 공유하는 WebSearch 한도(200회)를 다 썼다. 그 뒤로 남은 확인은 GitHub 저장소 열람으로만 했고, 못 한 확인은 각 Gaps에 적었다.

---

## 1. Anticipatory Music Transformer (AMT; Thickstun et al., TMLR 2024, arXiv 2306.08620): 메커니즘, 평가, 제어 강도, 파인튜닝, 후속 연구

### Takeaway
AMT는 제어(control) 토큰을 별도 어휘 블록으로 두고, 이벤트 스트림 안에서 "해당 시각보다 δ초 앞" 위치에 끼워 넣는다(interleave). 그래서 인과적(causal) 모델 하나로 "미래 제어를 보고 지금을 생성"할 수 있다. 반주 인필링 사람 평가에서 인간 작곡과 통계적으로 구분되지 않았고, 자기회귀(AR)·검색 기준선은 크게 졌다. 다만 원 모델은 Lakh MIDI로 처음부터 학습했고(100k step, 10배 증강), 실시간 연주처럼 미래 제어를 모르는 상황은 원 논문의 설계 범위 밖이다.

### Cited Findings
- **[직접 확인]** 설정값(commit af37397, 2024-03-18): `CONTEXT_SIZE = 1024`, `EVENT_SIZE = 3`(이벤트와 제어 모두 3토큰), `M = 341`(1024 = 1 + 3·341), `DELTA = 5`(anticipation 시간, 초), `TIME_RESOLUTION = 100`(10 ms), `MAX_DURATION_IN_SECONDS = 10`, note = pitch × instrument(128 × 129). — [anticipation/config.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/config.py)
- **[직접 확인]** 어휘는 이벤트 블록(`TIME/DUR/NOTE_OFFSET`), 별도의 제어 블록(`CONTROL_OFFSET` 아래 `ATIME/ADUR/ANOTE_OFFSET`), 특수 토큰(`SEPARATOR`, `AUTOREGRESS`, `ANTICIPATE`)으로 나뉜다. 같은 음이라도 "제어로서의 음"과 "생성된 음"이 서로 다른 토큰이다. — [anticipation/vocab.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/vocab.py)
- **[직접 확인]** 끼워 넣기 규칙은 `ops.anticipate()`에 있다. 이벤트를 시간순으로 내보내다가 `event_time >= control_time - delta`가 되면 그 제어(3토큰)를 먼저 삽입한다. 각 제어는 자기 시각보다 δ초 앞의 이벤트들 사이에 놓인다. — [anticipation/ops.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/ops.py)
- **[직접 확인]** 학습 데이터 증강(`tokenize.py`): 증강 배수 k 중 `k % 10 == 0`인 샘플은 `AUTOREGRESS`(제어 없음)이고 나머지는 `ANTICIPATE`다. 제어를 고르는 방식은 세 가지다. (a) span 증강(`lmbda = .05`, 지수분포 간격, DELTA초 구간), (b) 무작위 음 선택 증강, (c) **악기 증강**("at least one, but not all instruments": 악기 일부를 통째로 제어로 지정)이다. (c)가 "멜로디 트랙을 주고 반주 생성" 같은 트랙 조건화를 학습시키는 장치다. — [anticipation/tokenize.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/tokenize.py)
- **[직접 확인]** 샘플링 제약(`sample.py`): `safe_logits()`가 제어 블록 토큰 생성을 `-inf`로 막는다("don't generate controls"). `future_logits()`는 현재 시각 이전 시간 토큰을 막고, `instr_logits()`는 악기 수를 16개 이하로 제한한다. 이벤트 블록에서 제어와 같은 pitch·instrument를 내는 것은 막지 않는다. — [anticipation/sample.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/sample.py)
- **[직접 확인]** 모델 크기와 학습 설정(Levanter `split_lakh` 브랜치): Small 768d/12층/12헤드(LR 6e-4), Medium 1024d/24층/16헤드(LR 3e-4), Large 1280d/36층/20헤드(LR 2e-4). 셋 다 seq_len 1024, `num_train_steps: 100000`이다. 학습 코드는 저장소에 없다("does not contain code for training"). 공개 체크포인트는 `stanford-crfm/music-medium-800k`. — [train/README.md, lakh-*.yaml](https://github.com/jthickstun/anticipation/tree/main/train), [README](https://github.com/jthickstun/anticipation)
- **[직접 확인]** 사람 평가 재현 절차: Lakh 테스트 split `f`에서 멜로디가 있는 클립 50개를 고른다. anticipatory(`-a`), 자기회귀 기준선(`-b`), 무작위 검색 기준선(`-r`)으로 각각 반주를 만든다. 프롬프트 연속 생성 라운드는 FIGARO 완성곡에서 프롬프트 50개를 뽑아 쓴다. — [humaneval/README.md](https://github.com/jthickstun/anticipation/blob/main/humaneval/README.md)
- **[요약만·원문발췌]** 사람 평가 방식: Amazon Mechanical Turk 쌍대 비교, 20초 합성 오디오 2개 중 "more conventionally musical"을 고르게 했다. 작업자는 사전 자격 시험(인간 곡 vs 검색 기준선 5쌍)을 통과해야 했다. 샘플은 nucleus p = 0.95였고, 저자들은 p < 1이 더 보수적인 음악을 낸다고 밝혔다. — [arXiv 2306.08620 검색 발췌](https://arxiv.org/pdf/2306.08620)
- **[요약만·원문발췌]** 반주 평가(Table 3, 인간 작곡 대비 승/무/패): anticipatory **18 / 31 / 11 (p = 0.194, 유의하지 않음)**, 자기회귀 기준선 **5 / 10 / 45 (p ≈ 1.5e-8)**, 무작위 검색 **2 / 6 / 52 (p ≈ 1e-11)**. 반주는 Medium 모델로 생성했다. 프롬프트 연속 생성(Table 2)에서는 Medium이 인간 대비 44 / 29 / 77 (p = 0.0027)이었다. Large는 약 780M 파라미터다. — [저자 PDF 발췌](https://johnthickstun.com/assets/pdf/anticipatory-music-transformer.pdf), [ar5iv 발췌](https://ar5iv.labs.arxiv.org/html/2306.08620)
- **[요약만·2차]** δ → ∞이면 seq2seq와 같아진다. 실험에서 쓴 인필링 모델은 δ = 5초다. δ가 컨텍스트 길이에 비해 크면 경계 효과가 심해진다. — [emergentmind 요약](https://www.emergentmind.com/topics/anticipatory-music-transformer-amt)
- **[요약만·원문발췌]** AMT 실사용 사례(Für Elise 바이올린 반주 공동작곡, 2024)에서 보고된 실패 양상: 모델이 바이올린 음 생성을 멈추거나, "다운비트에 코드 근음만 치는 것"으로 퇴화했다. — [CRFM blog "Co-Composition with an AMT"](https://crfm.stanford.edu/2024/08/06/co-composition.html) (검색 발췌)
- **[요약만·원문발췌] 후속 연구 jam_bot** (Blanchard, Naseck, Brade, Lecamwasam, Rudess, C.-Z. A. Huang, Paradiso; ISMIR 2025, pp. 755–762, DOI 10.5281/zenodo.17811478): 세 가지를 다룬다. ① 컨텍스트와 조건 신호를 바꿔서 lead, 반주, call-and-response 같은 역할을 맡기는 방법, ② 연주자의 구조화된 즉흥으로 파인튜닝, ③ 듣고 프롬프트하고 생성을 스케줄링하는 저지연 멀티스레드 시스템이다. 2차 출처는 jam_bot이 AMT 기반이라고 적었다. — [MIT Media Lab 논문 페이지](https://www.media.mit.edu/publications/the-jam_bot-a-real-time-system-for-collaborative-free-improvisation-with-music-language-models/), [ISMIR 2025 poster P7-3](https://ismir2025program.ismir.net/poster_321.html)
- **[요약만·원문발췌]** jam_bot 후속(NIME 2026, "Enhancing Expressive Musical Conversation in the jam_bot"): velocity 모델링, 더 빠른 ggml 구현, 템포가 바뀌는 자유 call-and-response, 외부 MIDI 출력 지연 보상을 추가했다. — [NIME 2026 proc.](https://nime.org/proc/nime2026_73/)
- **[직접 확인] 후속 연구 MIDI-LLM** (Wu, Kim, Huang): Llama 3.2 1B에 MIDI 토큰을 덧붙인 어휘를 쓰고, MIDI 토큰화에 **Anticipation 라이브러리**를 쓴다. README에는 ISMIR 2026 논문으로 적혀 있다(commit 93c660c). — [slSeanWU/MIDI-LLM README](https://github.com/slSeanWU/MIDI-LLM)
- **[요약만·원문발췌]** MIDI-LLM은 TheoryTab 리드시트 67K개를 {Melody, Chord, Click} 3트랙으로 만들어 "AMT 기반 생성·인필링"에 맞췄다. 과제 모드는 {From Scratch, Infill Melody, Infill Chords, Infill Both}다. NeurIPS 2025 워크숍 버전도 있다. — [arXiv 2511.03942 발췌](https://arxiv.org/pdf/2511.03942), [NeurIPS 2025 virtual](https://neurips.cc/virtual/2025/123799)
- **[요약만·원문발췌]** 실시간 LM 잼 시스템 StreamMUSE (arXiv 2606.11886, 2026-06): 외부 타이밍 신호에 맞춰 클라이언트가 자주 추론을 요청하는 client-server 구조다. 핵심 조절값은 추론 간격과 생성 길이이고, 왕복 지연과 이 둘의 상호작용을 모델링했다. — [arXiv 2606.11886](https://arxiv.org/pdf/2606.11886)
- **[요약만·원문발췌]** 원조 개념인 Anticipation-RNN (Hadjeres & Nielsen 2017, arXiv 1709.06404; Neural Computing & Applications 2018): "어떤 시점에서 제약을 강제하면 이전에 생성된 항목들의 분포가 소급해서 무효가 될 수 있다". 해결책은 다가올 제약의 요약을 조건으로 넣어 네트워크가 미리 대비하게 하는 것이다. 단항(unary) 위치 제약을 다루고, 샘플링 비용은 일반 RNN과 같은 차수다. — [arXiv 1709.06404](https://arxiv.org/pdf/1709.06404), [코드](https://github.com/Ghadjeres/Anticipation-RNN)

### Inferences
- AMT에서 제어가 강하게 먹히는 구조적 이유는 세 가지로 보인다. ① 제어가 별도 토큰 블록이라 "조건 음을 그대로 다시 치는" 지름길이 어휘 수준에서 제거된다(제어 블록 생성 금지). ② 제어가 영향을 미칠 이벤트 바로 근처(δ 앞)에 놓여 긴 컨텍스트에 묻히지 않는다. ③ 학습의 90%가 제어가 있는 샘플이고, 악기 단위 증강이 "트랙 A가 주어지면 트랙 B를 생성" 과제를 직접 가르친다.
- 사용자의 "PiJAMA LH(또는 화성) → RH" 자가라벨 아이디어는 AMT의 악기 증강과 형태가 같다. 즉 LH 음을 제어 블록으로, RH 음을 이벤트로 두는 것이다. 다만 PiJAMA는 단일 트랙이라 손 분리 휴리스틱이 먼저 필요하다.
- 실시간 연주에서는 δ = 5초 앞의 LH를 알 수 없다. 현실적인 선택지는 셋이다. (a) δ ≈ 0 또는 "지금 누르고 있는 코드를 다음 반 마디까지 유지된다고 가정"한 제어로 학습·추론한다. (b) ReaLchords처럼 미래를 보는 교사 모델(δ > 0)에서 온라인 학생으로 증류한다(5절 참조). (c) 사용자가 미리 코드 진행을 입력한다. (a)가 가장 싸고, 학습과 추론 조건을 일치시켜야 한다.
- AMT의 반주 평가는 "인간과 비등"이지만 "코드 일치율" 같은 객관 제어 정확도 수치는 이번 조사에서 찾지 못했다. 제어 추종 강도의 근거는 사람 선호 평가뿐이다.

### Gaps
- AMT 논문 본문의 "prefix보다 interleave가 낫다"는 논증 문장, 그리고 Large 모델의 반주 평가 수치는 원문 차단으로 확인하지 못했다.
- 사전학습된 비(非)AMT 모델(예: Aria)에 anticipation 형식을 LoRA로 덧붙인 공개 결과는 찾지 못했다. MIDI-LLM은 Llama를 대상으로 하고 그 학습 레시피 세부는 미공개다. README는 "충분한 관심이 있으면 공개를 고려"한다고만 적었다. jam_bot의 파인튜닝 세부(스텝 수, 데이터 양, 지연 수치)도 미확인이다.
- jam_bot의 arXiv ID(다른 논문 참고문헌에 2503.02165로 언급됨)는 미확인이다.

---

## 2. 인필링·속성 제어 모델들: MMM/MIDI-GPT, Composer's Assistant 1/2, ILM, Polyffusion, Whole-song, FIGARO, MuseCoco, Musical Metadata, SymPAC

### Takeaway
대부분의 "강한 제어" 사례는 제어 신호를 학습 단계에서 데이터로부터 자동 추출해 넣고(자가라벨), 처음부터 학습하거나 대규모로 재학습했다. 이 중 사용자 상황(사전학습 AR 피아노 모델 + 소량 데이터 + 실시간)에 바로 옮길 수 있는 것은 두 가지다. 마디 단위 pitch-class set 제어 토큰(MIDI-GPT `prism`, MidiTok `BarPitchClass`)과, 규칙 기반 코드 추출로 만든 자가라벨 코드(FIGARO, Polyffusion)다. 확산 계열(Polyffusion, Whole-song)은 제어가 강하지만 AR 실시간 구조와 맞지 않는다.

### Cited Findings
- **[요약만·원문발췌] MIDI-GPT** (Pasquier et al., AAAI 2025, arXiv 2501.17011, DOI 10.1609/aaai.v39i2.32138; 전신은 MMM, arXiv 2008.06048, 2020): 트랙·마디 단위 인필링을 하고, 악기(categorical), note density(value), polyphony와 note duration(range)으로 조건화한다. 평가 항목은 학습 데이터 비복제성, StyleRank 스타일 유사도, 속성 제어 효과다. — [arXiv 2501.17011 발췌](https://arxiv.org/pdf/2501.17011), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32138)
- **[직접 확인]** MIDI-GPT 저장소(commit 89621cc, 2026-08-19): `yellow` 체크포인트는 density, polyphony min/max, duration min/max를 지원한다. `prism`·`expressive` 체크포인트는 key signature, pitch range, silence, duration, 마디 단위 density·polyphony, **마디 단위 pitch class set**, 18개 장르를 지원한다. 둘 다 "training in progress"로 표기돼 있다. 학습 기본값은 `infill_probability 0.75`, `mask_apply_probability 0.5`이고, 추론에는 하드 제약 `polyphony_hard_limit`("이 동시발음 수를 넘는 토큰을 거부")이 있다. — [README](https://github.com/Metacreation-Lab/MIDI-GPT), [docs/models.md](https://github.com/Metacreation-Lab/MIDI-GPT/blob/main/docs/models.md), [docs/api.md](https://github.com/Metacreation-Lab/MIDI-GPT/blob/main/docs/api.md)
- **[직접 확인]** MidiTok `BarPitchClass`: `ACBarPitchClass_0..11` 토큰으로 "마디 안에 존재하는 pitch class"(`pitch % 12`의 unique 값)를 표기한다. 속성 제어 토큰은 각 마디나 트랙의 시작에 놓여 causal 모델이 뒤따르는 토큰을 이에 맞춰 생성하게 한다. 다른 속성 제어로는 BarOnsetPolyphony, BarNoteDensity, BarNoteDuration, TrackRepetition 등이 있다(commit 6e23fca). — [bar_attribute_controls.py](https://github.com/Natooz/MidiTok/blob/main/src/miditok/attribute_controls/bar_attribute_controls.py), [docs/attribute_controls.rst](https://github.com/Natooz/MidiTok/blob/main/docs/attribute_controls.rst)
- **[요약만·원문발췌] Composer's Assistant** 1 (Malandro, ISMIR 2023, arXiv 2301.12525): 연속 마디 구간에서 임의 track-measure를 지운 뒤 다시 채우는 다중 트랙 인필링이다. Composer's Assistant 2 (ISMIR 2024, arXiv 2407.14700)는 T5 계열이며 두 종류의 리듬 조건화, 수평·수직 onset density, 여러 pitch 제어, rhythmic interest 제어를 추가했다. "객관 지표가 극적으로 개선"됐고, 청취 연구에서 실제 음악과 이 시스템으로 공동작곡한 음악 사이에 유의한 차이가 없었다고 보고했다. — [arXiv 2407.14700](https://arxiv.org/abs/2407.14700), [arXiv 2301.12525](https://arxiv.org/abs/2301.12525)
- **[직접 확인]** Composer's Assistant는 REAPER 스크립트와 로컬 NN 서버로 동작하고, CUDA가 없으면 CPU 버전("a bit slower")을 쓰라고 안내한다(v2.1.0, commit 5bfe5ea). — [README](https://github.com/m-malandro/composers-assistant-REAPER)
- **[요약만·원문발췌] Polyffusion** (Min, Jiang, Xia, Zhao; ISMIR 2023, arXiv 2307.10304): piano-roll 이미지를 확산 모델로 생성한다. internal control은 인페인팅이고, external control은 chord·texture 등을 cross-attention으로 넣는다. guidance scale 5(Polyffusion-S5)에서 코드 조건화와 텍스처 조건화 모두 제어력이 좋아졌다. 다만 코드 guidance를 크게 하면 일부 지표가 나빠졌고, texture 조건은 전용 기준선보다 약했다. 사전학습된 disentangled 표현을 조건으로 쓰는 쪽이 더 효과적이라고 보고했다. — [arXiv 2307.10304 발췌](https://arxiv.org/pdf/2307.10304)
- **[직접 확인]** Polyffusion 저장소(commit a996268): 변형별로 조건 방식이 다르다. `sdf_chd8bar`는 사전학습 코드 인코더 latent, `sdf_chdvnl`은 vanilla 코드 표현, `sdf_txt`·`sdf_txtvnl`은 텍스처 조건이다. cross-attention 메커니즘만 Latent Diffusion에서 빌려 왔다. CFG 기본 scale은 1이고 예시는 5다. 학습·조건용 코드는 내장된 `chord_extractor`로 MIDI에서 자동 추출한다. — [README](https://github.com/aik2mlj/polyffusion), [chord_extractor](https://github.com/aik2mlj/polyffusion/tree/main/polyffusion/chord_extractor)
- **[요약만·원문발췌] Whole-song hierarchical generation** (Wang, Min, Xia; ICLR 2024 spotlight, arXiv 2405.09901): 4단 캐스케이드 확산 구조이고, 첫 단은 최대 256마디 전체 곡을 다룬다. chord·rhythm·texture latent 같은 외부 조건을 각 단에 cross-attention으로 넣는다. — [arXiv 2405.09901](https://arxiv.org/pdf/2405.09901)
- **[요약만·원문발췌] FIGARO** (von Rütte et al., ICLR 2023, arXiv 2201.10936): description-to-sequence 방식이다. expert description(악기, 코드, 메타 등 사람이 해석 가능한 특징)과 VQ-VAE learned description으로 원곡을 재구성한다. 코드 엔트로피 H_chord는 GT 4.077, FIGARO(expert) 4.049, (learned) 4.085, (both) 4.071이고 비교 모델은 3.749, 3.949, 2.420이다. "description에 직접 들어 있는 지표에서 FIGARO(expert)가 매우 잘한다"고 했다. 코드 정확도 수치 자체는 이번에 확인하지 못했다. — [arXiv 2201.10936 발췌](https://arxiv.org/pdf/2201.10936), [ICLR 2023](https://iclr.cc/virtual/2023/poster/10959)
- **[직접 확인]** FIGARO 저장소(commit 9da3029)에는 ablation 모델 `figaro-no-chord`, `figaro-no-inst`, `figaro-no-meta`가 있다. 코드 라벨은 `chord_recognition.py`의 규칙 기반 `MIDIChord`(템플릿 + insider/outsider 점수)로 MIDI에서 자동 추출한다. — [README](https://github.com/dvruette/figaro), [src/chord_recognition.py](https://github.com/dvruette/figaro/blob/main/src/chord_recognition.py)
- **[요약만·원문발췌] MuseCoco** (arXiv 2306.00110): text → attribute → music 2단계 구조다. attribute-to-music 단계의 평균 속성 제어 정확도는 80.42%(Table 13)이고, 기준선 대비 객관 제어 정확도가 약 20% 개선됐다고 한다. ASA(average sample-wise accuracy) 지표를 쓴다. — [arXiv 2306.00110 발췌](https://arxiv.org/pdf/2306.00110)
- **[직접 확인]** MuseCoco 속성 목록은 Instrument, Rhythm Danceability/Intensity, Artist, Genre, Bar, Time Signature, Key, Tempo, Pitch Range, Emotion, Time이다. 코드(chord) 속성은 없다. "도구는 객관 속성 값만 MIDI에서 자동 추출할 수 있다"고 적혀 있다. — [microsoft/muzic/musecoco README](https://github.com/microsoft/muzic/tree/main/musecoco)
- **[요약만·원문발췌] Flexible Control via Musical Metadata** (Han et al., LG AI Research, arXiv 2409.07467): 메타데이터 토큰을 받아 4마디 다중트랙 MIDI를 생성한다. 학습 중 메타데이터 토큰을 무작위로 드롭해서, 사용자가 어떤 입력을 줄지 골라도 품질이 유지되게 했다. 규모를 키운 모델로 주관 평가에서 제어와 품질 우위를 주장했다. 2차 요약에 따르면 데이터는 LMD와 MetaMIDI, 표현은 REMI+다. — [arXiv 2409.07467](https://arxiv.org/pdf/2409.07467), [Moonlight 리뷰(2차)](https://themoonlight.io/review/flexible-control-in-symbolic-music-generation-via-musical-metadata)
- **[요약만·원문발췌] SymPAC** (Chen et al., ISMIR 2024, arXiv 2409.03055): 사전학습 MIR 모델로 오디오에서 뽑은 심볼릭 데이터만으로 학습했다(코드 검출 포함 의사라벨). 인코딩 안의 프롬프트 마디와 추론 시 유한상태기계(FSM) 제약 생성을 결합한다. 한계: 코드 검출기가 12 장조/단조만 지원하고, 1/16 양자화라 3/4·6/8에 약하며, 평균 시퀀스 길이가 약 8000이다. — [arXiv 2409.03055](https://arxiv.org/pdf/2409.03055), [ISMIR 2024 poster](https://ismir2024program.ismir.net/poster_366.html)
- **[요약만·원문발췌] GETMusic** (IJCAI 2025, arXiv 2305.10841): 트랙을 무작위로 source와 target으로 나누고 target만 마스킹하는 이산 확산 모델이다. 6개 악기 트랙의 665개 조합에서 "any-track" 생성을 했다. — [arXiv 2305.10841](https://arxiv.org/pdf/2305.10841), [IJCAI 2025](https://www.ijcai.org/proceedings/2025/1127)

### Inferences
- "pitch-class set as control"은 이미 쓰이고 있다(MIDI-GPT prism, MidiTok). 하지만 이 경우 조건은 **생성 대상 트랙 자신의** PC 집합이다. 사용자 아이디어처럼 "LH의 PC 집합으로 RH를 조건화"하면 의미가 다르다. 재즈 RH는 텐션(9, 11, 13)이나 아웃사이드 음을 써서 LH PC 집합을 벗어나므로, 타깃 자신의 PC를 조건으로 쓰면 지나치게 강한 제약이 된다. 반대로 LH PC를 조건으로 쓰면 원하는 "코드를 듣고 반응"하는 관계를 배운다.
- Composer's Assistant와 MIDI-GPT는 CPU나 소형 모델(MIDI-GPT 기본 학습 설정 512d/6층/8헤드, [README 학습표](https://github.com/Metacreation-Lab/MIDI-GPT))로도 돌아가므로 "작은 모델 + 강한 속성 토큰"이 하나의 경로가 될 수 있다. 다만 공개 모델은 팝·다중트랙 중심이라 재즈 피아노 어휘와는 거리가 있다.
- 확산 계열(Polyffusion, Whole-song)의 강한 코드 제어는 CFG와 사전학습 코드 인코더 덕분이다. 그러나 마디 단위 전체 생성이라 1.8초 지연 예산의 실시간 AR 루프에는 바로 쓰기 어렵다.

### Gaps
- ILM (Donahue, Lee, Liang, ACL 2020, arXiv 2005.05339)은 검색 한도 소진으로 원문이나 요약을 확보하지 못했다. 이 문서에서는 근거로 쓰지 않는다.
- MIDI-GPT의 pitch-class set 제어 정확도, Composer's Assistant 2의 pitch 제어 정의와 수치, SymPAC의 FSM 사용 시 코드 일치율, Whole-song의 제어 정확도 수치는 모두 미확인이다.
- Musical Metadata 논문의 정량 제어 정확도 수치는 미확인이다.

---

## 3. 제어 토큰 배치: prefix vs interleaved vs cross-attention vs adapter. 어떤 증거가 있나

### Takeaway
직접 비교 증거는 적지만 방향은 일관된다. 긴 음악 컨텍스트가 있으면 앞에 한 번 넣는 전역 prefix 제어는 묻히기 쉽다. MIDI-LLM 보고에 따르면 인필링에서는 CFG를 훨씬 키워야 제어가 먹었다. 제어를 영향받는 음 바로 근처에 시간 정렬해서 넣거나(AMT, Moonbeam의 note 단위 chord MLP), 같은 LM 안에서 입력과 출력을 이어 붙인 self-attentive 어댑터(Function Alignment의 `prompt_lora`)가 cross-attention 어댑터보다 낫다는 보고가 있다. 사용자의 "마지막 블록 late-injection Linear(29→1536)"는 이 증거들과 비교하면 주입 깊이와 용량이 가장 약한 형태다.

### Cited Findings
- **[요약만·원문발췌]** MIDI-LLM Figure 2: CFG에 따른 품질-제어 트레이드오프가 **인필링에서 훨씬 비탄력적**이다. 주변 MIDI 컨텍스트의 조건화를 이기려면 더 높은 CFG scale이 필요했다. 즉 텍스트 같은 전역 제어가 음악 컨텍스트에 밀린다. — [arXiv 2511.03942 발췌](https://arxiv.org/pdf/2511.03942)
- **[요약만·원문발췌]** "Equipping Pretrained Unconditional Music Transformers with Instrument and Genre Controls" (Xu, McAuley, Dubnov, Dong; arXiv 2311.12257, 2023-11): MuseScore 150만 곡으로 큰 무조건 트랜스포머를 사전학습한 뒤, 추가 제어 토큰을 넣고 파인튜닝했다(MMT-I/G/GI 변형). 주관 청취 평가에서 사전학습 기준선보다 coherence, harmony, arrangement, overall이 좋았다. 이 제어는 악기와 장르 같은 전역(곡 단위) 속성이다. — [arXiv 2311.12257](https://arxiv.org/pdf/2311.12257)
- **[직접 확인] Moonbeam 조건부 생성 레시피** (Guo & Dixon, arXiv 2505.15559; 브랜치 `conditional_gen_commu`):
  ① 코드를 트랜스포머 입력 앞에 prefix로 둔다. 형식은 `metadata_tokens, <SOC> chords <EOC>, <SOS> music <EOS>`이고, prefix 위치의 라벨은 dummy 처리한다.
  ② 동시에 **음표마다** `chord_condition_layer`를 둔다. 이는 [마디 one-hot, 박 one-hot, *사전학습된 pitch embedding으로 코드 구성음들을 합산한 벡터*]를 받는 2층 MLP이고, 출력 디코더 쪽에 시간 정렬로 주입된다.
  ③ LoRA 설정은 `r = 8`, `lora_alpha = 32`, `lora_dropout = 0.05`, 대상은 q/k/v/o와 디코더 모듈이다. 새 조건 모듈(`chord_condition_layer` 등)은 `modules_to_save`로 전체 학습한다.
  ④ 코드 드롭아웃(p = 0.2) 코드는 있으나 주석 처리돼 있다.
  ⑤ 파인튜닝 예시는 LR 3e-4, 300 epoch, context 848이고, 추론은 top_p 0.6, temperature 0.7이다.
  — [README](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model), [modeling_llama.py (conditional_gen_commu)](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model/tree/conditional_gen_commu), [configs/peft.py](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model/blob/conditional_gen_commu/src/llama_recipes/configs/peft.py)
- **[요약만·원문발췌]** Moonbeam은 81.6K시간, 180억 토큰으로 사전학습했다. 파인튜닝한 조건부 생성 모델이 REMI 유사 토크나이저를 쓰는 강한 트랜스포머 기준선을 이겼다고 주장한다. 수치는 미확인이다. — [arXiv 2505.15559](https://arxiv.org/abs/2505.15559)
- **[요약만·원문발췌] Function Alignment** (Jiang, Chin, Lin, Liu, Xia; ISMIR 2025, arXiv 2506.15548): 어댑터 두 가지를 비교했다. (a) 따로 사전학습한 두 LM 사이의 학습 가능한 cross-attention, (b) 입력과 출력이 같은 어휘일 때 하나의 공유 LM 안에서 입력-출력을 이어 붙인 시퀀스에 적용하는 경량 self-attentive 어댑터다. 과제는 코드→멜로디, 멜로디→코드, 드럼↔기타 트랙, few-shot 분석이다. ISMIR 리뷰어는 "핵심 기법 자체는 새롭지 않다"고 평했다. — [arXiv 2506.15548](https://arxiv.org/pdf/2506.15548), [ISMIR 2025 poster](https://ismir2025program.ismir.net/poster_25.html)
- **[직접 확인]** Function Alignment 저장소(commit d9ceb34): "For the best result, use the model with `prompt_lora` name (for our self-attentive adapters). `yinyang` refers to the cross-attention adapters." 기반 모델은 16분음표로 양자화한 MIDI로 학습한 Roformer다. Coco-Mulla 방식 기준선(`cp_transformer_cocomulla.py`)도 들어 있다. chord→melody 체크포인트는 Nottingham 데이터를 쓴다(예시 파일명 `..._nottingham_cp8_...`). — [README](https://github.com/music-x-lab/midi-function-alignment)
- **[요약만·원문발췌] Coco-Mulla** (Lin, Xia, Jiang, Zhang; arXiv 2310.17162, ISMIR 2024): MusicGen(오디오 LM)에 코드, 리듬·드럼, piano-roll 조건을 넣는 PEFT다. 원 파라미터의 4% 미만을 300곡 미만의 소량 데이터로 학습했다. 조건 어댑터는 디코더 self-attention 층에 삽입된다. — [arXiv 2310.17162](https://arxiv.org/pdf/2310.17162), [ISMIR 2023 LBD](https://ismir2023program.ismir.net/lbd_361.html)
- **[직접 확인]** Coco-Mulla 저장소는 MIDI와 코드 파일 경로를 입력으로 받는 추론 스크립트와 `r=0.2`, `r=0.4` 두 가중치를 제공한다. — [Kikyo-16/coco-mulla-repo](https://github.com/Kikyo-16/coco-mulla-repo)
- **[요약만·원문발췌]** 심볼릭 트랜스포머에 "frozen backbone + zero-init ControlNet 분기"를 적용한 사례는 검색에서 찾지 못했다. Music ControlNet (arXiv 2311.07069)과 LiLAC (arXiv 2506.11476, ISMIR 2025)은 오디오 확산 모델용이다. — [LiLAC](https://arxiv.org/pdf/2506.11476)
- **[요약만·원문발췌]** Theme Transformer (Shih et al., IEEE TMM 2022, arXiv 2111.04093): encoder-decoder에서 gated parallel attention(자기회귀 self-attention과 테마 cross-attention을 전환)과 theme-aligned positional encoding을 쓴다. prompt 기반 트랜스포머보다 테마 반복을 잘 지켰다. — [arXiv 2111.04093](https://arxiv.org/pdf/2111.04093)

### Inferences
- 배치별 기대 강도(추론): ① 시간 정렬 interleave(AMT) 또는 음표별 주입(Moonbeam MLP) > ② 같은 LM 안의 concat + LoRA(Function Alignment `prompt_lora`) > ③ cross-attention 어댑터 > ④ 전역 prefix 한 번. 사용자 측정치(코드가 NLL에 주는 효과 약 1%)는 ④가 컨텍스트에 묻힌다는 MIDI-LLM 관찰과 맞는다.
- 사용자의 late-injection 어댑터는 Linear(29→1536)를 마지막 블록 하나에만 넣었고 128 step만 학습했다. Moonbeam과 비교하면 차이가 셋이다. (a) Moonbeam은 모든 attention 층에 LoRA를 걸고 새 조건 모듈을 함께 학습한다. (b) 코드 표현을 무작위 Linear가 아니라 모델 자신의 사전학습 pitch embedding의 합으로 만든다. (c) 300 epoch 수준으로 학습한다. 사용자 어댑터의 "판별은 되는데 생성은 나빠짐"은 마지막 층 주입이 logit을 직접 흔들어 분포를 깨뜨리는 쪽으로 학습됐을 가능성으로 해석된다. 이는 가설이다.
- Aria는 어휘가 고정된 단일 스택이다. 따라서 Function Alignment의 결론(같은 LM 안에서 concat + LoRA가 cross-attn보다 낫다)을 따르면, LH/화성 토큰을 RH 이벤트와 같은 시퀀스에 시간순으로 섞고(AMT 형식) LoRA와 새 임베딩만 학습하는 방식이 가장 자연스럽다.

### Gaps
- prefix 대 interleave를 같은 모델, 같은 데이터에서 제어 정확도로 직접 비교한 수치는 찾지 못했다.
- Moonbeam 조건부 생성의 정량 결과(CommU 코드 정확도 등)와 Function Alignment의 표 수치는 원문 차단으로 미확인이다.
- "Equipping…" 논문의 게재처(학회·워크숍)는 미확인이다.

---

## 4. 트랙/채널 조건화와 "안 쓰는 악기 토큰을 제어 채널로 재활용"한 선례

### Takeaway
"한 트랙을 주고 다른 트랙을 생성"하는 학습은 표준이 됐다(AMT 악기 증강, GETMusic, MIDI-GPT 트랙 인필링, Function Alignment). 비음악 신호를 트랙이나 악기처럼 인코딩한 선례로는 MIDI-LLM/TheoryTab의 {Melody, Chord, Click} 트랙이 가장 가깝다. 하지만 "사전학습 피아노 모델에서 쓰지 않는 GM 악기 토큰을 제어 채널로 재정의"한 논문은 찾지 못했다.

### Cited Findings
- **[직접 확인]** AMT는 학습 시 악기 부분집합을 통째로 제어로 지정한다("instrument augmentation: at least one, but not all instruments"). 제어 음은 이벤트와 다른 토큰 블록에 인코딩되지만 pitch × instrument 정보는 그대로 유지한다. — [tokenize.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/tokenize.py), [vocab.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/vocab.py)
- **[직접 확인]** AMT README의 반주 예시: `extract_instruments`로 멜로디 악기(예시는 instrument 53)를 분리해 `controls`로 넘기고, 나머지 이력을 `inputs`로 넘겨 `generate`한 뒤 `ops.combine`으로 합친다. — [README](https://github.com/jthickstun/anticipation)
- **[요약만·원문발췌]** MIDI-LLM은 TheoryTab 리드시트를 {Melody, Chord, Click} 3트랙으로 만들어 AMT식 생성과 인필링 학습에 썼다. 코드와 메트로놈 클릭을 "트랙"으로 표현한 사례다. — [arXiv 2511.03942 발췌](https://arxiv.org/pdf/2511.03942)
- **[요약만·원문발췌]** GETMusic은 트랙을 source와 target으로 무작위 지정해 any-track 조건 생성을 학습했다. — [arXiv 2305.10841](https://arxiv.org/pdf/2305.10841)
- **[요약만·원문발췌]** 손 단위 피아노 모델: A-Muze-Net (2021, MMM 2022; arXiv ID 2111.12986은 검색 결과 링크로만 추정했고 미확인)은 RH와 LH를 두 네트워크로 모델링하고, LH 생성기(LSTM)가 RH의 코드 임베딩을 조건으로 받는다. Lyu et al. (2020)의 dual-track 구조는 클래식 피아노 LH·RH 상호의존을 모델링했다. 둘 다 트랜스포머 이전 RNN 계열이다. — [arXiv 2111.12986](https://arxiv.org/pdf/2111.12986v1)
- **[직접 확인]** Aria 저장소 README는 다중 트랙 토큰화·생성을 지원하지만 단일 트랙 표현 피아노가 주 학습 데이터라서 단일 트랙 입력을 권장한다(commit c2f67bc, 2026-05-12). — [EleutherAI/aria README](https://github.com/EleutherAI/aria)

### Inferences
- PiJAMA LH를 Aria의 다른 악기 트랙(예: 사용 빈도가 낮은 GM 프로그램)으로 인코딩하는 것은 어휘상 가능하다. 다만 Aria가 그 악기 토큰에 대해 가진 사전지식(다중 트랙 데이터 소량)이 간섭할 수 있다. 사용자가 이미 겪은 "duration 위치에 비피아노 악기 토큰이 생성됨"은 이 위험을 보여준다. 대안은 AMT처럼 **새 제어 토큰 블록(새 임베딩 행)**을 추가하고 출력 쪽에서는 마스킹하는 것이다. 이렇게 하면 조건 음이 "RH 이벤트"로 재생성될 경로가 어휘에서 사라진다.
- 실시간 입력(사용자가 치는 LH와 코드)은 본질적으로 "별도 트랙"이다. 학습 데이터도 같은 형식(LH 트랙 = 제어, RH = 생성)이어야 학습과 추론의 불일치가 줄어든다.

### Gaps
- 사전학습 단일 악기 모델에서 미사용 프로그램 번호를 제어 채널로 재정의한 사례(논문, 블로그, 코드)는 이번 검색 범위에서 찾지 못했다.
- PiJAMA 손 분리(LH/RH) 방법과 정확도에 대한 근거는 이 조사에서 확보하지 못했다.

---

## 5. CFG, 대조·반사실 손실, condition dropout, 보조 코드 예측 손실: 제어 추종을 올린다는 증거와 비용

### Takeaway
이 범주에서 "코드 정확도 수치"가 붙은 가장 직접적인 근거는 MuseBarControl이다(마디 단위 파인튜닝만 65.27% → 보조 사전학습과 반사실 손실 추가 78.33%). CFG는 확산 계열(Polyffusion scale 5)과 AR 계열(MIDI-LLM)에서 제어를 올린다는 보고가 있고, **Aria 저장소에는 이미 조건 드롭아웃 50% 학습 코드와 CFG 샘플러가 들어 있다**. 비용은 둘 다 대략 2배다. 반사실 손실은 학습 시 잘못된 조건으로 추가 forward를 하고, CFG는 추론 시 조건/무조건 배치를 2배로 돌린다.

### Cited Findings
- **[요약만·원문발췌] MuseBarControl** (Shu et al., arXiv 2407.04331, 2024): 코드 정확도 ablation 결과(K = 15회 샘플링 기준)는 다음과 같다.

  | 구성 | 코드 정확도 |
  |---|---|
  | BFT(마디 단위 파인튜닝) 단독 | 65.27% |
  | BFT + PA(보조 사전학습 과제) | 72.22% |
  | BFT + CF(반사실 손실) | 71.15% |
  | BFT + PA + CF | 78.33% |

  CF는 "모델이 마디 단위 prefix 프롬프트에 올바르게 반응하고, 이전 토큰만으로 다음 토큰을 생성하는 자명한 해를 피하게" 하려고 설계됐다. 초록은 마디 단위 제어가 기존 방법 대비 13.06% 개선됐다고 하는데, 이는 코드 정확도와 다른 지표다. 전역 속성 정확도에서 MuseBarControl은 BFT를 1.01%p 앞섰고, 둘 다 MuseCoco보다 크게 높았다. 게재처는 미확인이다. — [arXiv 2407.04331 발췌](https://arxiv.org/pdf/2407.04331), [HTML v1](https://arxiv.org/html/2407.04331v1)
- **[직접 확인] Aria의 내장 조건화와 CFG 경로** (commit c2f67bc):
  - `TransformerLM`에 `embedding_adapter = nn.Linear(emb_size, d_model)`가 있다. 조건 임베딩을 시퀀스 **첫 위치**에 붙인다(`hidden_states = cat([emb, hidden_states[:, :-1]])`).
  - 학습 루프는 `use_embeddings_cond = use_embeddings and (random.random() > 0.5)`, 즉 **조건 드롭아웃 50%**다.
  - 샘플러 `sample_batch_cfg`는 `num_variations = 2 * num_variations`(조건과 무조건 배치 2배)로 돌린다. `logits_cfg = γ·logits_cond + (1−γ)·logits_uncond`이고 `0 ≤ γ ≤ 10`이다. γ는 `min(250, max_new_tokens)` step에 걸쳐 선형 warm-up된다. 조건은 별도 embedding 모델로 임의 MIDI 파일에서 뽑은 512차원 전역 임베딩이다(`--embedding_midi_path`, `--cfg`).
  - 모델 설정은 `d_model 1536`, `n_layers 16`, `n_heads 24`, `max_seq_len 8192`, `emb_size 512`다.
  - MLX 구현(`sample_mlx.py`)에도 같은 CFG 함수가 있다.
  - 참고: README는 "LLaMA 3.2 (1B) architecture 기반"이라고 쓰고, 사용자는 약 658M으로 파악하고 있다.
  — [aria/model.py](https://github.com/EleutherAI/aria/blob/main/aria/model.py), [aria/training/train.py](https://github.com/EleutherAI/aria/blob/main/aria/training/train.py), [aria/inference/sample_cuda.py](https://github.com/EleutherAI/aria/blob/main/aria/inference/sample_cuda.py), [aria/run.py](https://github.com/EleutherAI/aria/blob/main/aria/run.py), [config/models/medium-emb.json](https://github.com/EleutherAI/aria/tree/main/config/models)
- **[요약만·원문발췌]** MIDI-LLM은 텍스트 조건에 CFG를 쓰고, 인필링에서는 더 높은 CFG scale이 필요했다(Figure 2). 텍스트 제어 추종은 CLAP(LAION music checkpoint) 유사도로, 품질은 VGGish FAD로 측정했고, 비교 대상은 Text2midi다. 조건 드롭아웃 비율과 기본 scale은 미확인이다. — [arXiv 2511.03942 발췌](https://arxiv.org/pdf/2511.03942)
- **[요약만·원문발췌]** Polyffusion: guidance scale 5에서 코드와 텍스처 제어력이 좋아졌고, 큰 코드 guidance는 일부 지표를 해쳤다. — [arXiv 2307.10304 발췌](https://arxiv.org/pdf/2307.10304)
- **[요약만·원문발췌]** 오디오와 확산 계열의 참고 설정: MusiConGen(오디오 MusicGen 파인튜닝, 코드·리듬 조건)은 조건 드롭아웃 0.5와 γ = 3을 쓴다. ViTex(2026, 이산 확산)는 코드·악기 조건을 각각 확률 0.5로 null 처리하고 추론 때 조건별 guidance 강도를 따로 준다. — [MusiConGen arXiv 2407.15060](https://arxiv.org/html/2407.15060v1), [ViTex arXiv 2603.01984](https://arxiv.org/pdf/2603.01984)
- **[요약만·원문발췌]** Musical Metadata 논문(2409.07467)도 학습 중 메타데이터 토큰 무작위 드롭을 썼다. 목적은 CFG가 아니라 "부분 조건 입력 허용"이었다. — [arXiv 2409.07467](https://arxiv.org/pdf/2409.07467)
- **[요약만·원문발췌] ReaLchords** (Wu, Huang et al., ICML 2024, PMLR v235; arXiv 2506.14723): 온라인 반주 모델을 MLE로 사전학습한 뒤 RL로 파인튜닝한다. 보상 모델은 멜로디와 코드의 화성·타이밍 일치를 채점하고, **미래 멜로디를 볼 수 있는 교사 모델로부터 지식을 증류하는 divergence 항**을 더한다. MLE 모델은 정답 데이터로만 학습해서 exposure bias가 있다고 지적했다. 청취자는 온라인 MLE와 증류만 한 모델보다 ReaLchords를 선호했다. — [PMLR](https://proceedings.mlr.press/v235/wu24c.html), [arXiv 2506.14723](https://arxiv.org/pdf/2506.14723)
- **[직접 확인]** ReaLchords 공식 PyTorch 구현은 대조(contrastive)와 판별(discriminative) 보상 모델, 리듬 전용 보상 모델을 각각 학습한다. RL은 "offline anchor" 스크립트로 수행한다. 평가는 note-in-chord ratio(곡별·박별), long-note·repetition penalty, chord entropy, Vendi score다. GAPT(arXiv 2511.17879)는 보상 해킹 완화용 적대적 사후학습이다. Apple Silicon용 MLX 서버 백엔드(`--mlx`)가 있다(commit 75a6324). — [lukewys/realchords-pytorch README](https://github.com/lukewys/realchords-pytorch)
- **[요약만·원문발췌]** 화성도 제어를 과하게 밀면 품질이 떨어진다. Chord-Conditioned Melody Harmonization with Controllable Harmonicity (arXiv 2202.08423)에서 제어 파라미터와 CTnCTR의 상관은 r = 0.9, PCS와는 r = 0.88, MCTD와는 r = −0.86이었다. 그러나 화성도를 너무 높이면 청취 평점이 떨어졌다. CTnCTR(chord tone 대 non-chord tone 비율)은 Yeh et al. 2021 계열의 표준 지표다. — [arXiv 2202.08423](https://arxiv.org/pdf/2202.08423), [arXiv 2001.02360](https://arxiv.org/pdf/2001.02360)

### Inferences
- 사용자의 late-injection 어댑터가 "맞는/틀린 코드 계획을 NLL로는 구분하지만 생성은 나빠진" 현상은 MuseBarControl이 겨냥한 실패와 같다. 모델은 이전 토큰만으로 다음을 예측하는 해에 머물고, 조건은 미세한 우도 차이로만 반영된다. 처방 후보는 둘이다. (a) **반사실 손실**: 같은 RH 타깃에 대해 맞는 LH/코드 조건의 NLL이 틀린(이조하거나 섞은) 조건의 NLL보다 margin 이상 낮도록 hinge·contrastive 항을 추가한다. (b) **보조 코드 예측 손실**: 은닉 상태에서 현재 반 마디의 조건 PC 집합을 예측하게 해서 표현에 화성 정보를 강제한다.
- Aria에는 "조건 드롭아웃 50% + 첫 위치 임베딩 + CFG(γ warm-up)" 파이프라인이 이미 구현돼 있다. 따라서 (i) 피아니스트 스타일(임베딩 모델로 Mehldau·Tatum 곡의 평균 임베딩)과 (ii) 화성 조건 양쪽에 CFG를 쓸 수 있다. 다만 이 경로는 **곡 전체에 대한 전역 조건 1개**다. 반 마디마다 바뀌는 코드를 이 방식으로 넣으면 전역 prefix의 약점(3절)을 그대로 물려받는다. 그래서 코드 조건은 시간 정렬 interleave로 넣고, CFG의 "무조건" 쪽은 제어 토큰을 지운(또는 null 블록으로 바꾼) 시퀀스로 만드는 변형이 필요하다(추론).
- CFG의 추론 비용: Aria 샘플러는 배치를 2배로 돈다. M1 Max에서 1.8초 지연 예산 안의 생성 토큰 수가 그만큼 줄어든다. 디코딩이 메모리 대역폭에 묶이면 2배보다는 덜 늘어날 수 있으나 실측이 필요하다.

### Gaps
- 자기회귀 심볼릭 모델에서 CFG scale에 따른 **코드 정확도** 곡선을 보고한 1차 근거는 찾지 못했다. MIDI-LLM은 텍스트·CLAP 지표, Polyffusion은 확산 모델이다.
- 보조 코드 예측 손실을 단독으로 ablation한 심볼릭 생성 논문은 찾지 못했다. MuseBarControl의 PA가 가장 가깝지만 정의 세부는 미확인이다.
- 텍스트 LM용 CFG 연구(예: "Stay on topic with CFG")는 검색 한도 소진으로 확인하지 못했다.

---

## 6. 제약 디코딩: 문법 마스크, Grammar-Aligned Decoding, SMC·particle, Markov 제약과 BP (Pachet 2026), soft harmonic constraint. 무엇이 분포를 왜곡하나

### Takeaway
하드 마스크(GCD)는 "문법은 지키지만 모델 확률을 왜곡"한다는 점이 NeurIPS 2024 GAD에서 정식화됐다. 근본 원인은 2017년 Anticipation-RNN이 이미 지적한 "미래 제약이 과거 분포를 소급해서 바꾸는 문제"다. 왜곡을 줄이는 방법은 미래 실현 가능성을 추정해 반영하는 것이다. 정확한 방법으로는 Markov와 가변차수 모델용 BP(Pachet 2015/2026)가 있고, 근사 방법으로는 ASAp, SMC, MCTS, B* 탐색이 있다. 음악에서는 하드 제약이 근음 반복 같은 퇴화를 유발한 사례(AMT 공동작곡 블로그)가 있고, 화성도를 과하게 밀면 청취 평점이 떨어진다. 실시간 재즈에는 "확률 재가중(soft bias) + 소수 후보 재순위"가 비용 대비 안전하다.

### Cited Findings
- **[요약만·원문발췌] Grammar-Aligned Decoding** (Park, Wang, Berg-Kirkpatrick, Polikarpova, D'Antoni; NeurIPS 2024, arXiv 2405.21047): 문법 제약 디코딩(GCD)과 제약 디코딩 일반이 LLM 분포를 왜곡해서, 문법적이지만 모델 하에서 우도가 낮은 출력을 낸다. ASAp(Adaptive Sampling with Approximate expected futures)는 이전 샘플로 "이 접두사가 문법적으로 완성될 확률"을 추정해 점차 문법 조건부 분포에 맞춘다. 코드 생성과 구조화 NLP 과제에서 기존 GCD보다 우도가 높은 출력을 냈다. 2차 리뷰는 수렴이 느릴 때가 있다고 적었다. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2bdc2267c3d7d01523e2e17ac0a754f3-Abstract.html), [arXiv 2405.21047](https://arxiv.org/html/2405.21047v2)
- **[직접 확인]** GAD 구현은 HF transformers `LogitsProcessor`(`GrammarAlignedOracleLogitsProcessor`)와 EBNF 문법 형태다(commit bc062c6). — [ebmoon/transformers-GAD](https://github.com/ebmoon/transformers-GAD)
- **[요약만·원문발췌]** SMC와 particle 계열 음악 선례: Walder & Kim (ACML 2017, PMLR v77)은 비마르코프 시퀀스 모델의 출력이 FSM 제약을 만족하도록 SMC 근사 샘플링을 하고 beam search와 비교했으며, 재화성화에도 적용했다. "Computer Assisted Composition in Continuous Time" (arXiv 1909.05030)은 연속시간 point process를 일부 이벤트에 조건화하는 particle filter로, 대규모 청취 실험에서 beam search 기준선보다 나았다. — [PMLR v77](https://proceedings.mlr.press/v77/walder17a.html), [arXiv 1909.05030](https://arxiv.org/pdf/1909.05030)
- **[요약만·원문발췌]** MCTS 디코딩 (Ferreira et al. 2022, arXiv 2208.05162): LM을 정책으로, 감정 분류기와 판별기를 가치함수로 쓴 PUCT가 음악 품질과 감정 지표 대부분에서 SBBS와 CS를 이겼다. — [arXiv 2208.05162](https://arxiv.org/pdf/2208.05162)
- **[요약만·원문발췌] Pachet 계열**:
  - Papadopoulos, Pachet, Roy, Sakellariou (CP 2015): regular 제약과 단항 제약이 걸린 Markov 모델을, Markov 체인과 오토마톤을 결합한 factor graph 위의 단순화된 BP로 **정확히** 샘플링한다.
  - Pachet (arXiv 2605.07839, 2026-05) "Exact Regular-Constrained Variable-Order Markov Generation via Sparse Context-State BP": 1차 제약층은 가변차수 생성기가 구분해 둔 이력을 합쳐 버린다고 지적한다. 대안으로 관측된 context state와 제약 오토마톤의 곱 상태를 쓰고, 고정된 학습 context graph에서는 시퀀스 길이에 선형인 추론을 한다. 평가는 작은 코퍼스에서 열거로 확인한 정확성 시험과 Bach 예제의 확장성 시험이다.
  - 관련: Sampling Markov Models under Constraints 복잡도 결과 (arXiv 1711.10436).
  — [arXiv 2605.07839](https://arxiv.org/pdf/2605.07839), [Papadopoulos et al. 2015 PDF](https://www.francoispachet.fr/wp-content/uploads/2021/01/papadopoulos-15b.pdf), [arXiv 1711.10436](https://arxiv.org/pdf/1711.10436)
- **[요약만·원문발췌]** B* (Kaliakatsos-Papakostas et al., arXiv 2512.07627): beam search, A*, backtracking을 결합해 사전학습 트랜스포머가 지정된 코드를 올바른 마디와 onset에서 만족하게 강제한다. 최악의 경우 지수 비용인 무차별 대입 방식이며, 휴리스틱으로 개선 가능한 첫 시도라고 저자 스스로 밝혔다. — [arXiv 2512.07627](https://arxiv.org/pdf/2512.07627)
- **[요약만·원문발췌]** Zhu et al. (ICML 2025, PMLR v267; arXiv 2410.08435): 코드 조건 제공 같은 soft 제어는 세부 음높이의 정확성을 보장하지 못할 수 있다. 그래서 샘플링 과정에 hard 제어(fine-grained harmonic·rhythmic guidance)를 넣었다. 2차 리뷰에 따르면 out-of-key 음이 크게 줄었다. 확산 모델이다. — [PMLR](https://proceedings.mlr.press/v267/zhu25m.html), [arXiv 2410.08435](https://arxiv.org/html/2410.08435v2), [Moonlight(2차)](https://themoonlight.io/review/efficient-fine-grained-guidance-for-diffusion-based-symbolic-music-generation)
- **[요약만·원문발췌]** soft constraint의 예: 멜로디→가사 생성 (arXiv 2305.07760, 2305.19228)은 규칙 위반 후보를 버리지 않고 확률에 α(0 < α < 1)를 곱하며, α = 0이면 하드 제약이다. Wang (2025, Francis Academic Press) "Theory Structured Harmonic Embeddings for Chord Conditioned Melody Generation"은 chord-tone 선호, 텐션 타당성, 비화성음 해결, 스케일 준수를 하나의 강도 파라미터로 pitch logit에 반영했다. 코퍼스 수준 pitch·rhythm 통계를 유지하면서 chord tone ratio 등이 개선됐다고 하나, 저등급 게재처이고 미검증이다. — [arXiv 2305.07760](https://arxiv.org/pdf/2305.07760), [Francis Press](https://francis-press.com/papers/20061)
- **[요약만·원문발췌]** SymPAC은 추론 시 FSM으로 어휘를 제한해 프롬프트 제약을 강제한다. — [arXiv 2409.03055](https://arxiv.org/pdf/2409.03055)
- **[직접 확인]** 실용 하드 제약 사례: MIDI-GPT `polyphony_hard_limit`(동시발음 초과 토큰 거부), AMT `safe_logits`·`future_logits`·`instr_logits`. 모두 형식이나 문법 수준 제약이고, 화성 내용 제약이 아니다. — [MIDI-GPT docs/api.md](https://github.com/Metacreation-Lab/MIDI-GPT/blob/main/docs/api.md), [AMT sample.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/sample.py)
- **[요약만·원문발췌]** AMT 공동작곡에서 관찰된 퇴화: "다운비트에 코드 근음만 침", 음 생성 중단. — [CRFM blog](https://crfm.stanford.edu/2024/08/06/co-composition.html)

### Inferences
- 사용자의 "format mask"(duration 위치에 악기 토큰 금지)는 문법 수준 마스크다. GAD가 말하는 분포 왜곡은 "문법상 막다른 길로 가는 접두사"가 드물면 작다. 토큰 형식 마스크는 거의 항상 완성 가능하므로 유지해도 안전하다.
- 반면 "코드 구성음만 허용" 같은 **내용 하드 마스크**는 재즈 RH의 텐션, 크로매틱 어프로치, 아웃사이드 음을 지우고, 남은 몇 개 음으로 확률을 몰아 반복과 근음 퇴화를 부를 위험이 크다. 이는 화성도 과다 시 평점 하락과 AMT 퇴화 사례에 근거한 추론이다. 권장 형태는 다음과 같다.
  - PC 단위 **soft bias**: 코드톤에 +β, 텐션에 +β/2, 반음 어프로치는 다음 박 해결을 조건으로 허용한다.
  - β는 note-in-chord ratio(ReaLchords 지표)를 데이터셋 기준치(PiJAMA RH 실제 비율) 근처로 맞추는 방향으로 보정한다.
  - 후보 4–8개를 병렬 생성한 뒤 판별기·보상 점수로 재순위한다.
- Pachet식 정확 BP는 Markov와 가변차수 모델 전용이다. 트랜스포머(Aria)에는 바로 적용되지 않는다. 다만 "Markov 근사 + 정확 제약"을 실시간 fallback이나 리트리벌 층으로 쓰는 선택지는 있다.

### Gaps
- 트랜스포머 음악 모델에서 하드 화성 마스크와 soft bias의 반복률·다양성 비교 수치는 찾지 못했다.
- 음악 전용 twisted-SMC나 신경 SMC 최신(2024–2026) 논문은 검색 한도 소진으로 확인하지 못했다.
- 과제 문구에 있던 "harmonic soft constraints"라는 정확한 제목의 논문은 찾지 못했다.

---

## 7. 비라벨 피아노에서 화성 조건을 스스로 뽑아 쓴 선례(LH, bass, PC 히스토그램, chroma)와 조건 음 복사(re-strike) 위험·완화

### Takeaway
"MIDI에서 코드나 chroma를 자동 추출해 조건으로 쓰기"는 표준 관행이다. FIGARO, Polyffusion, Pop909 계열, SymPAC, MuseCoco의 객관 속성, MIDI-GPT의 pitch-class set이 그렇다. 조건 표현으로는 **root, chroma, bass(PC 수준)**가 널리 쓰이고, Moonbeam은 코드를 사전학습 pitch embedding의 합으로 넣는다. 반면 "조건 음을 그대로 다시 치는(echo/re-strike) 문제"를 정면으로 다룬 논문은 찾지 못했다. 완화책은 구조에서 유추해야 한다: 별도 제어 어휘와 생성 금지(AMT), PC 수준 조건, 조건 드롭아웃, 반사실 손실.

### Cited Findings
- **[직접 확인]** FIGARO의 코드 라벨은 규칙 기반 `MIDIChord`가 마디·박 단위 템플릿 점수(구성음 매치, insider +10, outsider −1/−2)로 MIDI에서 자동 추출한다. 사람 라벨이 아니다. — [figaro/src/chord_recognition.py](https://github.com/dvruette/figaro/blob/main/src/chord_recognition.py)
- **[직접 확인]** Polyffusion의 `chord_extractor`(Junyan Jiang 작성, README는 중국어)는 다음 순서로 동작한다.
  - 코드 트랙 가중치(soft label)를 정한다.
  - 코드 트랙의 chroma와 bass chroma를 템플릿에 매칭한다.
  - 팝의 박 구조를 반영한 동적계획법으로 결과를 평활화한다.

  README는 두 가지를 경고한다. ① 대상은 LMD 다중 트랙 팝 MIDI이고, 클래식이나 **단일 트랙 음악에는 적합하지 않을 수 있다**. ② `pretty_midi`의 `get_beats()`, `get_downbeats()`가 정확해야 하므로 meter와 tempo 표기를 바로잡아야 한다. 코드 클래스 사전은 팝과 재즈 코드 라벨을 포함한다. 학습용 코드 행렬은 root, 12차원 chroma(root 기준으로 절대화), bass로 구성된다. — [polyffusion/chord_extractor/README.TXT](https://github.com/aik2mlj/polyffusion/tree/main/polyffusion/chord_extractor), [data/midi_to_data.py](https://github.com/aik2mlj/polyffusion/blob/main/polyffusion/data/midi_to_data.py)
- **[직접 확인]** 코드-텍스처 disentanglement (Wang et al., ISMIR 2020, "Learning interpretable representation for controllable polyphonic music generation") 저장소가 Polyffusion과 Whole-song의 코드·텍스처 인코더 계보다. README는 일부 파일만 공개됐다고 적었다. — [ZZWaang/polyphonic-chord-texture-disentanglement](https://github.com/ZZWaang/polyphonic-chord-texture-disentanglement)
- **[요약만·원문발췌]** SymPAC은 MIR 모델로 오디오에서 뽑은 의사 라벨(코드 포함)로 대규모 학습했다. 코드 검출 어휘가 12 장조/단조로 제한된 것이 한계로 명시돼 있다. — [arXiv 2409.03055](https://arxiv.org/pdf/2409.03055)
- **[직접 확인]** Moonbeam은 코드 심볼을 `music21`로 구성음 MIDI 번호로 풀고, **모델 자신의 사전학습 `pitch_embedding`으로 각 구성음을 임베딩해 합산**한 뒤 마디·박 one-hot과 함께 MLP에 넣는다. 코드가 없을 때("s")는 학습 가능한 placeholder 임베딩을 쓴다. — [modeling_llama.py (conditional_gen_commu)](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model/tree/conditional_gen_commu)
- **[직접 확인]** MIDI-GPT `prism`과 MidiTok `BarPitchClass`는 마디 안의 unique pitch class를 자동 계산해 제어 토큰으로 쓴다. — [MidiTok](https://github.com/Natooz/MidiTok/blob/main/src/miditok/attribute_controls/bar_attribute_controls.py), [MIDI-GPT docs/models.md](https://github.com/Metacreation-Lab/MIDI-GPT/blob/main/docs/models.md)
- **[직접 확인]** ReaLchords는 사람 라벨 없이 멜로디-코드 쌍의 "일치 대 불일치"를 대조·판별 보상 모델로 학습하고(자기지도 판별기), 이를 RL 보상과 평가에 쓴다. — [realchords-pytorch README](https://github.com/lukewys/realchords-pytorch)
- **[요약만·원문발췌]** AMT 반주 모델은 멜로디 트랙을 제어로 받아 나머지를 생성한다. 학습 데이터의 "제어 트랙"은 사람 라벨이 아니라 MIDI 악기 분리에서 자동으로 나온다. — [arXiv 2306.08620 발췌](https://arxiv.org/pdf/2306.08620), [tokenize.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/tokenize.py)
- **[요약만·원문발췌]** Theme Transformer는 조건 시퀀스(테마)를 출력에 **여러 번 나타나도록** 학습시켰다. 의도적 복사를 목표로 한 설계라서, 반대 목표인 "조건을 듣되 베끼지 않기"에는 참고 반례다. — [arXiv 2111.04093](https://arxiv.org/pdf/2111.04093)
- **[요약만·원문발췌]** 반주 생성 검색에서 "멜로디 음 복사·유니즌 중복" 완화를 직접 다룬 결과는 나오지 않았다. 보고된 퇴화는 "근음만 반복"(AMT 블로그) 같은 다른 형태였다. — [CRFM blog](https://crfm.stanford.edu/2024/08/06/co-composition.html)

### Inferences
- 사용자가 Aria 제로샷에서 본 "입력 코드 음을 다시 치기"는 조건이 **같은 어휘의 같은 트랙 음**으로 프롬프트에 들어갔을 때 나타나는 현상이다. 연속 생성 모델에게는 "직전 음형을 이어 치는 것"이 최대우도 해이기 때문이다. 완화책을 강도 순으로 정리하면 다음과 같다.
  1. 조건을 **별도 제어 토큰 블록**으로 넣고 출력에서 마스킹한다(AMT 방식). 그러면 조건 토큰이 출력 후보가 될 수 없다.
  2. 조건을 **절대 음높이가 아닌 PC 집합, 즉 12비트 chroma와 bass PC**로 준다. 복사할 정확한 음높이 정보가 사라져서 "그 PC를 다른 옥타브와 다른 리듬으로 쓰기"를 배우게 된다(Polyffusion과 Whole-song의 root/chroma/bass 관행).
  3. 학습 타깃에서 "조건과 같은 시점, 같은 음높이" 재타건을 계측해 데이터 기준치(PiJAMA RH가 LH 음을 실제로 겹쳐 치는 비율)를 정하고, 추론 출력의 echo율이 이를 넘지 않는지 감시한다.
  4. 조건 드롭아웃과 반사실 손실로 "조건 ↔ 타깃 PC 관계"를 학습시킨다.
  5. 디코딩에서 직전 0.5박 안에 조건 음과 같은 음높이가 재타건되면 soft penalty를 준다.
- PiJAMA로 자가라벨을 만들 때의 함정(추론): ① 템포와 박 정보가 없는 연주 전사 데이터라 "반 마디" 단위를 정의하려면 비트 추정이 필요하다. 시간 창(예: 0.5–1초)이나 LH onset 기준 구간으로 대체할 수 있다. ② 손 분리 오류로 RH 음이 LH 조건에 섞이면 "조건 = 타깃 일부"가 되어 복사 학습을 오히려 강화한다. 분리가 불확실하면 **모든 음의 PC 히스토그램**을 조건으로 쓰는 편이 안전하다. 다만 이 경우 조건에 RH 정보가 새므로, 조건 창을 타깃보다 한 박 이전까지로 제한하거나 하위 성부만 쓴다.

### Gaps
- "조건 음 echo/re-strike"를 측정하거나 완화한 심볼릭 생성 논문은 이번 범위에서 찾지 못했다.
- Pop909 기반 반주 조건화, AccoMontage 1/2/3 (arXiv 2108.11213, 2209.00353, 2310.16334), Q&A (IJCAI 2023, arXiv 2306.01635)의 세부 조건 표현과 결과는 검색 한도 소진으로 확인하지 못했다. 이 문서에서는 ID만 기록하고 근거로 쓰지 않는다.
- LH 또는 bass만으로 RH를 조건화한 트랜스포머 연구는 찾지 못했다. RNN 계열인 A-Muze-Net, dual-track만 있다.

---

## 8. 기법별 데이터, 연산, frozen 약 650M 모델에 LoRA/어댑터로 적용 가능한지(M1 Max 32GB), 증거 강도, 그리고 권고

### Takeaway
적은 연산과 자가라벨 피아노 데이터로 Aria 위에 강한 코드 제어를 얹을 가능성이 가장 큰 조합은 다음 세 겹이다.
- **(1) AMT식 시간 정렬 제어 interleave**: LH 또는 화성 PC를 새 제어 토큰 블록으로 RH 이벤트 사이 "조금 앞"에 넣는다.
- **(2) LoRA + 새 제어 임베딩 + 조건 드롭아웃**: 반사실(틀린 코드) 손실을 함께 쓴다.
- **(3) 추론 시 약한 CFG 또는 PC soft bias, 그리고 소수 후보 재순위.**

확산 계열, 텍스트 조건 LLM, 처음부터 학습하는 seq2seq(FIGARO, MuseCoco)는 이 제약 조건에서 우선순위가 낮다.

### Cited Findings
- **[직접 확인]** 처음부터 학습한 AMT의 학습 규모는 100k step, seq_len 1024, 증강 배수 10의 배수이고, Lakh 기준 전처리 데이터는 증강 없이 약 10GB다. — [train/README.md](https://github.com/jthickstun/anticipation/blob/main/train/README.md), [lakh-medium.yaml](https://github.com/jthickstun/anticipation/blob/main/train/lakh-medium.yaml)
- **[직접 확인]** Moonbeam 조건부 생성은 LoRA r = 8과 새 조건 모듈만 학습하는 PEFT 레시피다(300 epoch, CommU 데이터). — [Moonbeam peft.py](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model/blob/conditional_gen_commu/src/llama_recipes/configs/peft.py)
- **[요약만·원문발췌]** Coco-Mulla는 원 파라미터의 4% 미만, 300곡 미만 데이터로 대형 음악 LM(MusicGen)에 코드와 리듬 제어를 붙였다. — [arXiv 2310.17162](https://arxiv.org/pdf/2310.17162)
- **[직접 확인]** Aria의 조건 임베딩 + CFG 학습·추론 코드가 PyTorch와 MLX로 존재하고, CFG는 배치 2배로 동작한다. — [aria/inference/sample_mlx.py](https://github.com/EleutherAI/aria/blob/main/aria/inference/sample_mlx.py), [sample_cuda.py](https://github.com/EleutherAI/aria/blob/main/aria/inference/sample_cuda.py)
- **[요약만·원문발췌]** MuseBarControl은 기존 모델에 반사실 손실과 보조 과제를 추가해 코드 정확도를 65.27%에서 78.33%로 올렸다. — [arXiv 2407.04331](https://arxiv.org/pdf/2407.04331)
- **[직접 확인]** ReaLchords와 ReaLJam은 Apple Silicon MLX 백엔드로 실시간 서버를 돌린다. 보상 모델 기반 RL은 별도 학습 단계가 필요하다. — [realchords-pytorch](https://github.com/lukewys/realchords-pytorch)
- **[요약만·원문발췌]** Function Alignment은 공유 LM concat + LoRA 어댑터로 여러 조건부 과제(코드→멜로디 포함)를 다뤘다(파라미터 효율). — [arXiv 2506.15548](https://arxiv.org/pdf/2506.15548), [repo README](https://github.com/music-x-lab/midi-function-alignment)
- **[직접 확인]** Aria 모델 설정은 1536d, 16층, 24헤드, 최대 시퀀스 8192다. — [config/models](https://github.com/EleutherAI/aria/tree/main/config/models)

### Inferences
**기법 비교표** (증거 강도: A = 동료심사와 정량 제어 지표 있음, B = 동료심사 또는 정량 중 하나, C = 코드나 주장 수준. M1 Max 적합도는 이 문서의 추정이다.)

| 기법 | 필요한 데이터 | 학습 연산 | frozen 약 650M + LoRA로 M1 Max 적용 | 실시간 추론 비용 | 증거 강도 | 핵심 근거 |
|---|---|---|---|---|---|---|
| AMT식 시간 정렬 제어 interleave (LH/화성 → RH) | 손 분리한 PiJAMA(자가라벨), 반사실용 이조 사본 | 중간(LoRA + 새 임베딩, 수천 step 이상 필요 추정) | 가능(새 토큰 행과 LoRA만 학습) | 시퀀스에 제어 토큰이 추가돼 길이 증가 | A(TMLR, 사람 평가 18/31/11 대 AR 5/10/45) | §1 |
| Moonbeam식 음표별 코드 MLP 주입 + LoRA | 코드나 PC 라벨(자가추출 가능) | 중간 | 가능(Aria 출력 head 앞 주입으로 변형 필요) | 거의 0 | B(코드 공개, 수치 미확인) | §3 |
| 전역 prefix 제어 토큰 | 라벨 | 낮음 | 가능 | 0 | B(전역 속성에는 유효, 시간가변 코드에는 약함) | §3 MIDI-LLM, Equipping |
| cross-attention 어댑터 | 쌍 데이터 | 중간 | 가능하나 구조 변경 필요 | 소량 | B(Function Alignment에서 self-attentive보다 열세라는 README 언급) | §3 |
| 조건 드롭아웃 + CFG | 위와 동일 | 거의 0(학습 시 드롭만) | 가능(Aria에 코드 존재) | 배치 2배 | B(확산·텍스트에서 유효, AR 코드 정확도 곡선은 미확인) | §5 |
| 반사실·대조 손실(틀린 코드 대비 margin) | 조건 섞기·이조로 자동 생성 | 약 2배 forward | 가능 | 0 | B(MuseBarControl +5.88%p 단독, 동료심사 미확인) | §5 |
| 보조 코드(PC) 예측 손실 | 자가라벨 PC | 소량 | 가능(작은 head) | 0 | C–B(PA 조합에서 +6.95%p) | §5 |
| RL + 보상 모델(ReaLchords) | 쌍 데이터로 판별기 학습 | 높음(RL 루프) | 어려움(긴 GPU 런, 발열) | 0 | A(ICML 2024) | §5 |
| 하드 화성 마스크 | 코드 사전 | 0 | 해당 없음(추론) | 0 | B(음정 정확성 보장, 분포 왜곡과 퇴화 위험) | §6 |
| soft PC bias + 후보 재순위 | 판별기(선택) | 0–소량 | 해당 없음 | 후보 N배(배치) | C(음악 AR에서 직접 근거 약함) | §6 |
| GAD, SMC, BP 정확 제약 | 문법·오토마톤 | 0 | 해당 없음 | 샘플 다수 필요(실시간 부담) | A(GAD NeurIPS, Pachet CP 2015; 음악 트랜스포머 적용은 미확인) | §6 |
| 확산(Polyffusion, Whole-song) | 코드 라벨 피아노롤 | 높음(새 모델) | 부적합 | 반복 denoise | A(ISMIR, ICLR) | §2 |

**권고용 분석** (전부 추론):
1. **1순위: "AMT 형식의 LH/PC 제어 + LoRA + 반사실 손실" on Aria.**
   - 학습 쌍: PiJAMA에서 (LH 음 또는 반 마디 PC 집합 + bass PC)를 제어로, RH 이벤트를 타깃으로 만든다.
   - 제어 표현: 새 제어 토큰 블록을 만들고 출력에서는 마스킹한다. 음높이 그대로가 아니라 PC와 bass로 넣으면 re-strike 경로를 막을 수 있다.
   - 배치: 영향받는 RH 음보다 Δ(예: 1박) 앞에 interleave한다. 실시간 추론에서는 "현재 누른 코드 + 유지 가정"으로 Δ를 채운다.
   - 손실: 같은 RH에 대해 이조하거나 다른 곡에서 가져온 LH 조건의 NLL이 정답 조건보다 margin 이상 높도록 hinge를 추가한다. MuseBarControl CF와 같은 원리다.
   - 학습 비용은 LoRA(r 8–16)와 새 임베딩 수천 행이다. 사용자의 128 step 실험보다 훨씬 긴 학습(추정 수천 step)이 필요하지만, 발열을 피하려면 짧은 세션 여러 번(체크포인트 재개)으로 나누는 게 현실적이다.
2. **2순위: Aria에 이미 있는 "조건 드롭아웃 50% + CFG"를 코드 조건에도 쓰기.** 1순위 학습 때 제어 토큰 드롭아웃(예: 10–50%)을 함께 넣으면, 추론에서 γ를 1.0–2.0 범위로 조절하는 노브가 생긴다. 배치가 2배가 되므로 1.8초 지연 예산 안에서 실측이 필요하다. 같은 메커니즘을 Mehldau·Tatum 스타일 임베딩 조건에도 쓸 수 있다(LoRA 스왑의 대안이나 보완).
3. **3순위(안전망): 추론 시 PC soft bias와 판별기 재순위.**
   - 하드 마스크는 형식 마스크까지만 쓴다.
   - 화성은 텐션을 허용하는 soft bias로 넣는다.
   - 4–8개 후보를 배치로 뽑은 뒤 note-in-chord ratio, echo율, 반복률과 판별기 점수(사용자의 late-injection 어댑터도 판별기로 재활용 가능)로 고른다.
4. **보류:**
   - ReaLchords식 RL: 효과 근거는 강하지만 긴 학습 루프가 필요해 M1 Max 열 제약에 맞지 않는다. 대신 "미래를 보는 교사에서 증류"하는 아이디어만 오프라인 소규모로 차용할 수 있다.
   - 확산과 처음부터 학습하는 seq2seq: 연산과 실시간 제약 때문에 보류한다.
5. **측정 지표 권고:** 코드 효과를 NLL 차이(약 1%)로만 보지 말고, 생성 출력에서 다음을 같은 프롬프트로 A/B 비교한다.
   - note-in-chord ratio(박별), CTnCTR
   - 조건 PC 대비 RH PC 분포의 KL
   - echo율(조건 음과 같은 시점·음높이 재타건 비율)
   - 반복률과 Vendi 다양성

   그리고 "맞는 코드 대 틀린 코드" 조건에서 이 지표들의 차이(제어 감도)를 핵심 KPI로 삼는다.

### Gaps
- 이 표의 M1 Max 학습 step 수와 지연 수치는 모두 추정이다. MPS에서 Aria + LoRA의 실측 처리량 근거는 이번 조사에 없다.
- AMT형 제어를 사전학습 피아노 모델에 PEFT로 이식했을 때 제어 정확도가 얼마나 나오는지(처음부터 학습한 AMT 대비) 보고한 연구는 찾지 못했다.
- 재즈 피아노(텐션과 아웃사이드가 많은 장르)에서 note-in-chord 지표의 적정 목표치에 대한 근거가 없다. PiJAMA 자체 통계로 기준선을 정해야 한다.
- 원문 사이트 차단과 WebSearch 한도 소진 때문에 MuseBarControl, MIDI-LLM, Moonbeam, Function Alignment, Polyffusion의 표 수치는 PDF로 재확인하지 못했다. 위 수치는 모두 검색 발췌 기반이다.
