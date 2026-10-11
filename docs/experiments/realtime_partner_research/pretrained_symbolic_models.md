# 사전학습 심볼릭 음악(특히 피아노) 파운데이션 모델 비교 — 코드 조건 피아노 솔로 + 아티스트 어댑터용 베이스 선정 자료

> 조사일: 2026-10-11. 태그 규칙: **[직접 확인]** = 이번 세션에서 GitHub 저장소(README, LICENSE, config, 소스코드, 저장소에 포함된 HF 모델카드 사본)를 직접 clone/열람해 확인. **[요약만]** = 웹 검색 결과 스니펫/요약만 확인(원문 미열람).
> 접근 제약(이번 세션): `huggingface.co`, `arxiv.org`, `export.arxiv.org`, `alphaxiv.org`, `api.semanticscholar.org`, `openreview.net`, `ismir*.ismir.net`, `*.github.io`(WebFetch DNS 실패)는 조직 egress 정책으로 403/ENOTFOUND. **GitHub(git clone, github.com 페이지)와 PyPI만 접근 가능**했다. 따라서 HF 모델카드·arXiv 본문 수치는 GitHub에 사본이 있을 때만 [직접 확인], 나머지는 [요약만]이다. GitHub Pages 프로젝트 페이지는 해당 gh-pages 저장소를 clone해서 HTML을 읽는 방식으로 우회했다.

## Q1. 종합 비교표 — 어떤 사전학습 심볼릭 모델이 있고, 이 프로젝트(실시간·코드 조건·피아노 솔로·M1 Max) 기준으로 어떤가?

### Takeaway
피아노 표현(10 ms 타이밍·벨로시티·페달)과 공개 가중치·허용적 라이선스·Apple Silicon(MLX) 추론 코드를 모두 갖춘 모델은 사실상 **Aria(EleutherAI, ~650M, Apache-2.0)** 하나이고, 2026-10에는 Aria 위에 **피아니스트 조건(게이트 cross-attention, PiJAMA-12, Apache-2.0 가중치 공개)** 을 얹은 Edwards et al.(ISMIR 2026)까지 나와 "베이스 + 아티스트 조건" 구조의 직접 선례가 생겼다. 반면 **코드(화성) 조건을 기본 탑재하고 가중치가 공개된 피아노 모델은 확인되지 않았다**. 코드/반주 조건이 있는 공개 모델은 대부분 팝/Lakh 멀티트랙(GETMusic, REMI-chord, FIGARO, MIDI-GPT의 pitch-class-set 속성, AMT의 반주 조건)이거나 체크포인트가 미공개(Moonbeam ComMU)라서, 코드 조건은 Aria 위에 직접 학습해야 할 가능성이 높다.

### Cited Findings

**비교표 (수치는 출처가 있는 것만; 없으면 "미확인")**

| 모델 (연도, venue, ID) | 파라미터 / 구조 | 토큰화 (시간해상도·폴리포니·벨로시티·페달·악기) | 학습 데이터 (+데이터 라이선스) | 가중치 위치 / 라이선스 | 내장 조건화 | 재즈 근거 | 사람 평가 | 속도·하드웨어 | 파인튜닝/LoRA | M1 Max 32GB 현실성 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Aria medium** (2025, ISMIR 2025, arXiv 2506.23869) | ~650M(사용자 측정 658M, Edwards 카드 659M), LLaMA-3.2식 디코더 d=1536, 16층, 24헤드, ff×4, max_seq 8192 [직접 확인] | 절대시간 10 ms, 5 s마다 절대시간 토큰, 최대 길이 5 s, 벨로시티 10단계 양자화, 기본은 페달을 길이에 접어 넣음(include_pedal=false), 데모 토크나이저는 페달 포함, 멀티악기 prefix 지원 [직접 확인] | Aria-MIDI 계열 ~60k h 피아노 전사 [직접 확인 README]; Aria-MIDI는 CC-BY-NC-SA 4.0 [요약만] | HF `loubb/aria-medium-base`/`-gen`/`-embedding`, `model-demo.safetensors`; 모델·툴링 Apache-2.0 [직접 확인 README] | prefix 토큰(instrument/composer/form/genre: "jazz","classical") 토크나이저 지원 [직접 확인]; **임베딩 조건 + CFG** 생성 경로(embedding_adapter, `--cfg`) [직접 확인 코드]; 코드 조건 없음 | genre prefix "jazz"; 재즈 파인튜닝은 Edwards(아래) | 5 prompt×8 continuation=모델당 40개, AMT large·MusicGen large·Suno 3.5와 비교, 일관성에서 AMT·MusicGen보다 선호 [요약만] | CUDA·MLX 백엔드, 둘 다 KV cache 구현 [직접 확인]; 실시간 데모는 MLX [직접 확인] | 공식 코드에 LoRA 없음(전체 FT·분류·대조학습 스크립트) [직접 확인]; 사용자 측정 LoRA r16 4.1 GiB | **가능**(추론 MLX+KV cache, LoRA 학습 단시간) |
| **Edwards et al. jazz-pianist-style** (2026, ISMIR 2026, arXiv 2610.02918) | Aria-medium(659M) + 마지막 8/16층에 게이트 cross-attn, 12명 × 4 컨텍스트 벡터 아티스트 임베딩 [직접 확인 카드] | Aria 토크나이저 + 아티스트 토큰 확장 [직접 확인] | PiJAMA-12(PiJAMA는 CC BY-NC 3.0) [직접 확인] | HF `drewbie/jazz-pianist-style-generator` 등 3개, 코드·체크포인트 Apache-2.0 [직접 확인] | **피아니스트 조건**(12명); 코드 조건 없음 | 재즈 피아노 전용 | 사람 평가 대신 분류기 agreement 70% vs 37%(FT baseline) vs 25%(사전학습) [직접 확인] | 학습 ~18 h/24GB GPU; 평가 스크립트는 Apple-silicon 노트북 기준 시간 표기 [직접 확인] | 베이스까지 전체 파라미터 학습(동결 아님) [직접 확인 코드] | 추론 가능, 동일 레시피 전체 재학습은 비현실적(어댑터만 학습으로 변형 필요) |
| **Anticipatory Music Transformer** (2023 arXiv 2306.08620, TMLR 2024) | small 128M [요약만], medium 360M(CRFM 블로그) [요약만], large 존재(Aria 논문 baseline) [요약만]; HF GPT-2식 causal LM [직접 확인 README] | 3토큰/이벤트(arrival time, duration, note=악기×피치), 10 ms, 최대 길이 10 s, 129 악기, **벨로시티 없음**, context 1024, anticipation 5 s [직접 확인 config] | Lakh MIDI [요약만+직접 확인 train 안내] | HF `stanford-crfm/music-{small,medium,large}-800k`; 코드 Apache-2.0 [직접 확인] | **controls(멜로디 등 다른 트랙) 조건 반주/인필링** [직접 확인 README] | 미확인(재즈 특화 없음) | 반주 과제 human eval 절차 공개 [직접 확인 humaneval] | 미확인 | 표준 HF 모델 → PEFT 적용 가능(공식 레시피 없음) | 추론 가능, 피아노 표현력은 낮음(벨로시티 없음) |
| **Moonbeam** (2025, arXiv 2505.15559; venue 미확인) | M: d=1920, 15층, 12헤드/6 KV, ff 6720 + GRU 디코더(4층, 1536); S: d=1536, 9층 [직접 확인 config]; 백본 추정 746M/287M(내 계산) → 사용자 제시 839M/309M과 정합 | onset/duration/octave/pitch-class/instrument/velocity 6속성 + 속성별 RoPE(MRA) [직접 확인 config] | 81.6K h·18B 토큰(README) vs 85k h(데모 페이지) 불일치 [직접 확인] | HF `guozixunnicolas/moonbeam-midi-foundation-model`; Apache-2.0 [직접 확인] | ComMU 코드+메타데이터 조건 파인튜닝 브랜치(`conditional_gen_commu`, chord_dict에 maj7/m7/7/dim/m7b5/sus4/+) [직접 확인]; **ComMU 체크포인트 링크는 README에 "TODO"** [직접 확인] | PiJAMA 프롬프트 연속 데모, PiJAMA30 분류 Acc 0.679 [직접 확인 데모 페이지] | ComMU에서 전문가 평가로 baseline보다 코드 적합·일관성·음악성 우수 [요약만] | 미확인 | llama-recipes + PEFT LoRA, torchrun/DDP/bf16 [직접 확인] | 추론·LoRA는 메모리상 가능 추정, CUDA 중심 코드 포팅 필요 |
| **MIDI-GPT** (2025, AAAI 2025, arXiv 2501.17011) / MMM (2020, arXiv 2008.06048) | GPT-2: medium d=768·12층·12헤드(본체 ~85M 추정), small d=512·6층(~19M 추정) [직접 확인 config] | 마디 단위 트랙별 시퀀스; yellow/prism 벨로시티 32 bin, expressive는 128 bin + 마이크로타이밍 delta 토큰 [직접 확인] | 미확인(GigaMIDI 여부 미확인) | HF `Metacreation/MIDI-GPT`; 코드 MIT(2026) [직접 확인]; 가중치 라이선스 미확인 | 밀도·폴리포니·길이, prism/expressive는 키·음역·침묵·**마디별 pitch-class set**·장르 18그룹 [직접 확인] | expressive를 "jazz, solo piano"용으로 권장 [직접 확인 docs] | 미확인 | macOS arm64 휠, OSC 실시간 서버(마디 단위 생성) [직접 확인] | train extra(Lightning) [직접 확인] | **매우 가능**(소형) — 단 prism/expressive는 "training in progress" 스냅샷 [직접 확인] |
| **MIDI-RWKV** (2025; v2 2026-01, arXiv 2506.13001; venue 미확인) | RWKV-7(x070), 12층, d=384, vocab 16000, ctx 2048 [직접 확인]; 가중치 파일 70 MB → ~35M 추정 | miditok 기반(BarAttributeControl) [직접 확인] | 사전학습 GigaMIDI, 파인튜닝 POP909 [직접 확인] | `midi_rwkv.pth` 저장소 동봉, MIT [직접 확인] | 인필링 + 마디 속성 제어; 코드 조건 미확인 | 없음 | 28명 청취 테스트, state-tuned가 LoRA r4보다 유의하게 선호 [요약만] | rwkv.cpp(GGML) 추론 [직접 확인] | state tuning(~294k 파라미터) vs LoRA r4(~331k)/r32(~2.7M), 4–6분 학습 [요약만] | **매우 가능**(엣지 지향) |
| **Notochord** (2022 AIMC; 확장판 arXiv 2403.12000) | GRU 백본 기본 emb 512, hidden 1024, 2층 [직접 확인 코드] | 이벤트 단위(악기·피치·시간·벨로시티), time_res 10 ms, 320 악기 [직접 확인] | Lakh(`notochord_lakh_50G_deep.pt`) [직접 확인] | GitHub releases; MIT; PyPI 0.8.0 [직접 확인] | 서브이벤트 단위 개입(피치/시간/벨로시티 범위 제약), 하모나이저 [직접 확인 README]; 코드 조건 없음 | 없음 | 미확인 | 응답 지연 10 ms 미만(저자 주장) [요약만] | train CLI [직접 확인] | **매우 가능**(CPU 실시간) — 피아노 솔로 품질은 미확인 |
| **MIDI-LLM** (2025 arXiv 2511.03942; ISMIR 2026) | Llama-3.2-1B + AMT MIDI 토큰 어휘 확장(55,030 토큰) [직접 확인 README / 요약만 수치] | AMT 토큰(10 ms, 벨로시티 없음) | MusicPile·GigaMIDI 연속학습 + Lakh×MidiCaps SFT [요약만] | HF `slseanwu/MIDI-LLM_Llama-3.2-1B`; Llama 3.2 Community License [직접 확인] | 텍스트(코드 진행 서술 포함) [직접 확인 예시] | 없음 | Text2midi 대비 품질·속도 우수 주장 [요약만] | vLLM(fp8) / transformers, 16GB+ CUDA 권장 [직접 확인] | 학습 파이프라인 미공개 [직접 확인] | 추론 가능성 높음(MPS 미검증), 실시간 코드 추종은 부적합 |
| **Text2midi** (2024 arXiv 2412.16526, AAAI 2025) | FLAN-T5-base 인코더 + 디코더(d=768, 18층, 8헤드) [직접 확인 코드 인자] | REMI(miditok) [직접 확인] | SymphonyNet 사전학습 + MidiCaps 168k [직접 확인] | HF `amaai-lab/text2midi`; MIT [직접 확인] | 텍스트(코드·템포·스타일) | 없음 | 미확인 | MPS 지원 명시 [직접 확인] | 미확인 | 가능, 실시간·코드 추종 부적합 |
| **NotaGen / NotaGen-X** (2025, arXiv 2502.18008; venue 미확인) | 110M/244M/516M, patch+char 디코더 [직접 확인] | ABC 악보(표현 타이밍·벨로시티 없음) | 사전학습 1.6M 곡, ~9k 클래식 파인튜닝 [직접 확인] | HF `ElectricAlexis/NotaGen`; 코드 MIT [직접 확인] | period-composer-instrumentation 프롬프트 [직접 확인] | 없음(클래식) | 미확인 | 로컬 추론 ~8GB GPU [직접 확인] | CLaMP-DPO RL [직접 확인] | 추론 가능, 목적 부적합 |
| **ChatMusician** (2024, arXiv 2402.16153) | LLaMA2-7B [직접 확인] | ABC | MusicPile [직접 확인] | HF `m-a-p/ChatMusician`; 저장소 LICENSE 파일 없음(라이선스 미확인) | 텍스트 지시, 코드 진행 포함 ABC | 없음 | 미확인 | 미확인 | LoRA로 학습됨 [직접 확인] | 학습 비현실적, 목적 부적합 |
| **MelodyT5** (2024, ISMIR 2024, arXiv 2407.02277) | 113M [요약만] | ABC 멜로디 | 7개 멜로디 과제 100만+ 인스턴스 [직접 확인] | HF `sander-wood/melodyt5`; MIT [직접 확인] | 하모나이제이션 등 score-to-score | 없음 | 미확인 | 미확인 | 미확인 | 가능, 목적 부적합 |
| **MuseCoco** (2023, Microsoft muzic) | attribute2music 1B급("linear_mask-1billion"), 200M 학습 스크립트 [직접 확인] | REMI류 | 미확인 | HF `XinXuNLPer/MuseCoco_attribute2music`; muzic MIT [직접 확인] | 텍스트→속성→음악 | 없음 | 미확인 | 미확인 | 미확인 | 부적합(실시간 X) |
| **GETMusic** (2023, arXiv 2305.10841) | 미확인(이산 확산) | GETScore 트랙 표현 | 크롤링한 팝 [직접 확인] | muzic(MIT), 체크포인트 경로 README 미기재 [직접 확인] | **chord 트랙 조건 → lead/piano 생성** [직접 확인] | 없음 | 미확인 | 미확인 | 미확인 | 오프라인 생성만 |
| **Pop Music Transformer(REMI)** (2020) / **CP Transformer** (2021, arXiv 2101.02402) | 미확인 | REMI/compound word, Chord 토큰 | 팝 피아노 | Google Drive; **GPL-3.0** [직접 확인] | REMI-tempo-**chord** 체크포인트 [직접 확인] | chorder(규칙 기반 코드 인식) [직접 확인] | 미확인 | 미확인 | TF1/구형 | 참고용 |
| **FIGARO** (2022) | 미확인 | REMI+description | Lakh [직접 확인] | polybox; MIT [직접 확인] | 바 단위 description(코드 포함; `figaro-no-chord` 변형 존재) [직접 확인] | 없음 | 미확인 | 미확인 | 미확인 | 오프라인 |
| **Giant / Orpheus Music Transformer** (asigalov61) | Giant 786M-8k-44L-32H [직접 확인]; Orpheus 479M→2026 748M [요약만] | 3토큰/노트(Orpheus) [요약만] | Godzilla MIDI [요약만] | GitHub/HF Apache-2.0 [직접 확인] | 인페인팅 데모 | 없음 | 미확인 | 미확인 | 미확인 | 추론 가능 |
| **ImprovNet** (2025, arXiv 2502.04522; IJCNN 2025 [요약만]) | 미확인 | 5초 세그먼트 corruption-refinement | 클래식+재즈(세부 미확인) | Google Drive artifacts; 저장소 LICENSE 없음 [직접 확인] | 장르 토큰(클래식→재즈), 하모나이제이션 [직접 확인] | 재즈 스타일 변환 79% 식별 [직접 확인 README] | 청취 실험 있음 [직접 확인] | 미확인 | 미확인 | 오프라인 반복 정제 → 실시간 부적합 |
| **Pianist Transformer** (2025-12 arXiv 2512.02652, ICML 2026) | 135M [요약만] | 통합 MIDI 표현 [요약만] | 10B 토큰 자기지도 [요약만] | 코드·체크포인트 공개 주장, 라이선스 미확인 [요약만] | 악보→연주 표현 렌더링 | 미확인 | 블라인드 청취에서 인간 수준 주장 [요약만] | 미확인 | 미확인 | 미확인 |
| **PianoBART** (2024, ICME 2024) / **MusicBERT** (2021) | 미확인 | OctupleMIDI류 | POP1K7·ASAP·POP909·Pianist8·EMOPIA(PianoBART) [직접 확인] | HF `RS2002/PianoBART`(LICENSE 없음); MusicBERT OneDrive/Google Drive, MIT [직접 확인] | 이해/분류 위주 | 없음 | 미확인 | 미확인 | 미확인 | 참고용 |
| **HeartMuLa** (2026-01, arXiv 2601.10547) | 3B 공개, 7B 내부 [요약만] | **오디오 코덱(12.5 Hz)** — 심볼릭 아님 [직접 확인 README+요약만] | 미확인 | Apache-2.0 [요약만] | 가사/텍스트; MuLaCover는 멜로디·코드 MIDI 입력 사용 [직접 확인] | 없음 | 미확인 | RTF≈1.0 [요약만] | — | 제외(오디오) |
| **StreamMUSE / Real-Time LM Jamming** (2026-06, arXiv 2606.11886) | 시스템 논문(모델 아님) [요약만] | 미확인 | 미확인 | 코드 공개 주장, 저장소 미확인 [요약만] | 라이브 반주(프레임 동기 스트리밍 추론) | 없음 | 실시간성과 음악 품질 상관 보고 [요약만] | RTT별 분석 [요약만] | — | 추론 서빙 설계 참고용 |
| **ReaLchords / ReaLJam** (Google, 2024–2025; ReaLJam arXiv 2502.21267, CHI EA '25) | 미확인 | 1/16 프레임, 멜로디·코드 토큰 [요약만] | Hooktheory ~30k 스니펫 [요약만] | 가중치 공개 미확인 | 멜로디→**코드 반주**(우리와 반대 방향), RL 튜닝 [요약만] | 없음 | 미확인 | 실시간 설계 [요약만] | RL | 방향 반대, 설계 참고용 |

### Inferences
- Aria는 (1) 10 ms·벨로시티·페달을 다루는 피아노 표현력, (2) Apache-2.0 가중치, (3) MLX KV-cache 추론과 실시간 데모 코드, (4) 재즈 피아니스트 조건화 선례(Edwards)를 모두 가진 유일한 후보라서, 베이스 모델은 **Aria 유지가 타당**하다고 판단된다.
- "코드 조건 + 피아노 솔로 + 공개 가중치" 조합은 공백이다. 따라서 코드 조건은 (a) Aria에 코드 토큰/코드 임베딩 cross-attention을 추가하는 자체 학습, (b) MIDI-GPT prism의 마디별 pitch-class-set 속성 같은 "화성 제약 속성"으로 우회, (c) 디코딩 단계 제약(constrained decoding) 중 하나로 풀어야 한다.
- Moonbeam은 구조적으로 코드 조건 파인튜닝 레시피가 공개된 유일한 대형 피아노/범용 모델이지만 ComMU 체크포인트가 README상 미공개("TODO")이고, ComMU 코드 사전이 팝식(7th까지)이라 재즈 텐션 표현은 별도 확장이 필요할 것으로 보인다.

### Gaps
- HF 모델카드(라이선스 필드, 파일 크기), arXiv 본문(정확한 표 수치)은 egress 차단으로 직접 확인하지 못했다. 특히 Aria 논문의 청취 테스트 정확 수치, AMT medium/large 파라미터, Moonbeam 839M/309M 공식 수치, MIDI-GPT HF 가중치 라이선스는 미확인.
- SymPAC(ISMIR 2024, arXiv 2409.03055), Amadeus(arXiv 2508.20665, ACL 2026 [요약만]), XMusic(검색 요약에 XMIDI 데이터셋 언급만 있고 출처 URL을 특정하지 못함), MuPT(저장소는 Megatron-LM 포크만 확인)의 가중치 공개·파라미터·라이선스는 미확인.

## Q2. Aria 계열 상세 — base/gen/embedding/demo 체크포인트, Aria-MIDI, Aria-Duet, aria-amt, Edwards ISMIR 2026, 재즈 파인튜닝

### Takeaway
Aria 저장소(최종 커밋 2026-05-12)는 base·gen·embedding·demo(페달) 4개 체크포인트, CUDA/MLX 추론(KV cache 포함), 임베딩 조건 + CFG 생성, MLX 실시간 데모를 제공한다. 사용자가 겪은 "KV cache 없는 39분 생성"은 공식 `--backend mlx` 경로로 해결될 가능성이 크다. Edwards et al.(ISMIR 2026)은 Aria-medium에 게이트 cross-attention + 12명 아티스트 임베딩을 넣어 피아니스트 조건 생성기를 Apache-2.0으로 공개했으며, **Art Tatum은 포함, Brad Mehldau는 PiJAMA-12에 없음(PiJAMA-30에는 있음)**.

### Cited Findings
- Aria는 "LLaMA 3.2 (1B) 아키텍처 기반", ~60k시간 솔로 피아노 MIDI 전사로 학습; base 외에 생성 품질 개선용 파인튜닝 체크포인트와 SimCSE식 임베딩 체크포인트 공개 [직접 확인] — [EleutherAI/aria README](https://github.com/EleutherAI/aria)
- 체크포인트: `aria-medium-base`, `aria-medium-gen`, `aria-medium-embedding`, 실시간 데모용 `model-demo.safetensors`(서스테인 페달 제어 추가) [직접 확인] — [EleutherAI/aria README](https://github.com/EleutherAI/aria)
- 라이선스: "Our models and MIDI tooling are released under the Apache-2.0 license"; LICENSE 파일 Apache 2.0 [직접 확인] — [EleutherAI/aria](https://github.com/EleutherAI/aria)
- 모델 설정 `medium.json`: d_model 1536, n_heads 24, n_layers 16, ff_mult 4, max_seq_len 8192, grad_checkpoint true; 분류 헤드 설정 파일(composer/emotion/form/genre/music_period/pianist) 존재 [직접 확인] — [aria/config/models](https://github.com/EleutherAI/aria/tree/main/config/models)
- 토크나이저(AbsTokenizer) 설정: time_step_ms 10, abs_time_step_ms 5000, max_dur_ms 5000, velocity_quantization_step 10, include_pedal false(페달 효과를 노트 길이에 반영), 드럼 포함 가능, prefix 토큰 instrument/composer/form/genre(genre_names = "jazz","classical") [직접 확인] — [EleutherAI/aria-utils config](https://github.com/EleutherAI/aria-utils)
- 데모 토크나이저 설정은 `include_pedal: true` [직접 확인] — [aria/demo/demo-tokenizer-config.json](https://github.com/EleutherAI/aria/tree/main/demo)
- 추론: CLI `--backend {torch_cuda, mlx}`; MLX 모델에 `KVCache` 클래스와 `setup_cache` 구현; 기본 샘플링 temp 0.98, min_p 0.035, length 2048 [직접 확인] — [aria/aria/run.py, inference/model_mlx.py](https://github.com/EleutherAI/aria/tree/main/aria/inference)
- 생성 CLI는 `--embedding_model_checkpoint_path`와 `--embedding_midi_path`("path to MIDI file used for conditioning")를 필수로 받고 `--cfg`(CFG gamma, 기본 1.0)로 classifier-free guidance; 모델에 `embedding_adapter`(Linear)로 전역 임베딩 조건을 주입 [직접 확인] — [aria/aria/run.py, aria/aria/model.py](https://github.com/EleutherAI/aria/blob/main/aria/model.py)
- 공식 학습 코드에는 LoRA/PEFT 구현이 없음(grep 결과 없음; train/classifier_finetune/contrastive_finetune만 존재) [직접 확인] — [aria/aria/training](https://github.com/EleutherAI/aria/tree/main/aria/training)
- 사용 제약: 지시튜닝/RLHF 없음 → 입력 품질에 민감, 처음부터 생성보다 "기존 피아노 MIDI 이어치기"에 최적, 쇼팽 등 유명 클래식 암기 가능성 경고 [직접 확인] — [EleutherAI/aria README](https://github.com/EleutherAI/aria)
- 실시간 데모: `demo/demo_mlx.py`(Apple Silicon), Disklavier 사용, 제어 신호 CC 67(takeover)/66(reset), "응답성은 GPU 메모리 대역폭에 의존"; 코드 상수 KV_CHUNK_SIZE 256, PREFILL_CHUNK_SIZE 16/128, MAX_STREAM_DELAY_MS 50 [직접 확인] — [aria/demo](https://github.com/EleutherAI/aria/tree/main/demo)
- Aria 논문 청취 평가: 5개 프롬프트 × 8개 이어치기 = 모델당 40개, 비교 대상 AMT(music-large-800k), MusicGen(large), Suno 3.5, 사람 원곡; 참가자들이 AMT·MusicGen보다 Aria의 일관성을 일관되게 선호 [요약만] (검색 추출 표 수치가 깨져 정확한 승/무/패는 미확인) — [arXiv 2506.23869](https://arxiv.org/html/2506.23869); ISMIR 2025 포스터 P4-10, 저자 Bradshaw, Spangher, Fan, Biderman, Colton [요약만] — [ISMIR 2025 program](https://ismir2025program.ismir.net/poster_111.html)
- Aria-MIDI: ICLR 2025, arXiv 2504.15071, Bradshaw & Colton; 전체 1,186,253 파일·100,629시간, 작곡 중복 제거 후 800,973 파일·66,650시간; HF 서브셋 Full/Pruned(820,944)/Deduped(371,053)/Unique(32,522); **CC-BY-NC-SA 4.0** [요약만] — [arXiv 2504.15071](https://arxiv.org/abs/2504.15071), [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f7f5f501282771c96bb3fedcc96bedfe-Abstract-Conference.html)
- aria-amt: seq2seq 피아노 전사, Apache-2.0, 가중치 `piano-medium-double-1.0.safetensors`는 HF `loubb/aria-midi` 데이터셋 저장소에 위치 [직접 확인, GitHub README] — [EleutherAI/aria-amt](https://github.com/EleutherAI/aria-amt)
- Aria-Duet("The Ghost in the Keys", arXiv 2511.01663, NeurIPS 2025 Creative AI 트랙 데모): 연주자가 치는 동안 모델이 듣고 takeover 신호 시 같은 건반으로 이어 연주; 사용자 연주 중 KV cache를 미리 채워 전환 지연을 줄임; 소비자 하드웨어 저지연 주장 [요약만]; 정확한 지연 수치 미확인 — [arXiv 2511.01663](https://arxiv.org/html/2511.01663v1), [NeurIPS 2025](https://neurips.cc/virtual/2025/129323)
- Edwards, Maezawa, Dixon, "Learning Jazz Pianist Style with Cross-Attention Conditioning", ISMIR 2026, arXiv 2610.02918(2026-10-02 제출) [요약만] — [arXiv 2610.02918](https://arxiv.org/abs/2610.02918), [QMUL PDF](https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2026/EdwardsEtAl-ISMIR2026.pdf)
- Edwards 생성기: "Aria-medium (659M) with cross-attention over 12 learned pianist embeddings inserted in the last 8 of 16 transformer layers, fine-tuned on PiJAMA-12"; 파일 `model.safetensors`(base + cross-attn adapter), `artist_embeddings.safetensors`(12명 × 4 컨텍스트 벡터); 모델카드 license: apache-2.0 [직접 확인, 저장소 내 HF 카드 사본] — [jazz-pianist-style release/huggingface/cards/generator.md](https://github.com/almostimplemented/jazz-pianist-style)
- Edwards 학습 레시피: cross-attn 마지막 8/16층, 아티스트 컨텍스트 4벡터, 게이트 초기값 0.1, 임베딩 dropout 0.3, 15 epoch, 유효 배치 32, peak LR 5e-6, weight decay 0.02, fp16, **24GB GPU 1장에서 약 18시간**; `--no-cross-attention`은 FT baseline [직접 확인] — [jazz-pianist-style README](https://github.com/almostimplemented/jazz-pianist-style)
- Edwards 학습은 `params = list(model.parameters()) + artist_embeddings` 전체를 AdamW로 최적화 → **베이스 동결이 아닌 전체 파인튜닝** [직접 확인 코드] — [scripts/train_cross_attention.py](https://github.com/almostimplemented/jazz-pianist-style/blob/main/scripts/train_cross_attention.py)
- Edwards 결과(README 재현 표): PiJAMA-12 분류기 95.8% chunk / 98.8% track; 합성 데이터만으로 학습한 분류기 87.1 / 95.0; perplexity 6.82(cross-attn) / 6.96(FT baseline) / 11.41(사전학습); agreement 70% / 37% / 25%, 95% CI 예 70 [64, 75]; 피아니스트별 agreement 96%(Hank Jones)–29%(Cedar Walton); 암기 Jaccard 0.20(생성) vs 0.32(실제) [직접 확인] — [jazz-pianist-style README](https://github.com/almostimplemented/jazz-pianist-style)
- Edwards 추론: KV cache 지원 cross-attention 추론 구현; "Runs on CUDA, Apple silicon (MPS), or CPU. Training needs a GPU; evaluation of the released checkpoints runs on a laptop"; agreement 평가는 프롬프트당 4,096 토큰 175개 생성이라 "slow on a laptop" [직접 확인] — [jazz-pianist-style](https://github.com/almostimplemented/jazz-pianist-style)
- PiJAMA-12 피아니스트: Dick Hyman(187), Art Tatum(118), Abdullah Ibrahim(84), Dave McKenna(76), Tete Montoliu(52), Erroll Garner(45), Hank Jones(44), Andy Laverne(42), Oscar Peterson(41), George Cables(31), Cedar Walton(31), Horace Parlan(28) (괄호=CSV 행 수) [직접 확인] — [data/pijama12.csv](https://github.com/almostimplemented/jazz-pianist-style/tree/main/data)
- Brad Mehldau는 PiJAMA-12에는 없고 PiJAMA-30 CSV에 72곡(8.87시간), Art Tatum 122곡(8.03시간); PiJAMA 전체 2,777곡, 120명, 223.6시간(CSV duration 합) [직접 확인, CSV 집계] — [almostimplemented/PiJAMA](https://github.com/almostimplemented/PiJAMA), [pijama_30.csv](https://github.com/almostimplemented/jazz-pianist-style/tree/main/data)
- PiJAMA 데이터 라이선스: Creative Commons Attribution-NonCommercial 3.0 Unported [직접 확인 LICENSE] — [almostimplemented/PiJAMA](https://github.com/almostimplemented/PiJAMA)
- Yamaha 발표에서 이 논문을 ISMIR 2026 채택 3편 중 하나로 소개 [요약만] — [Yamaha news](https://www.yamaha.com/en/tech-design/research/news/26_ismir/)
- Aria 기반 2026 후속(제목·스니펫만): 표현 피아노 연주 평가에 문맥 임베딩 활용(arXiv 2607.27909, 29곡 7점 척도 청취), ARIMA 표현학습(arXiv 2607.10003), PianoCoRe 데이터셋(arXiv 2605.06627, TISMIR) [요약만] — [2607.27909](https://arxiv.org/html/2607.27909v1), [2607.10003](https://arxiv.org/html/2607.10003), [PianoCoRe](https://transactions.ismir.net/articles/10.5334/tismir.333)

### Inferences
- 사용자가 원하는 "베이스 1개 + 아티스트별 교체형 어댑터"와 가장 가까운 공개 선례는 Edwards의 게이트 cross-attention + 아티스트 임베딩이다. 다만 공개 생성기는 베이스까지 함께 학습된 전체 FT이므로, M1 Max에서는 "Aria 동결 + cross-attn/LoRA만 학습" 형태로 변형해야 하며 이 변형의 품질은 논문이 검증하지 않았다.
- Edwards 생성기에는 Art Tatum 임베딩이 이미 있으므로 **Tatum 쪽은 공개 가중치로 바로 실험 가능**하고, Brad Mehldau는 PiJAMA-30 데이터(8.87시간)로 추가 임베딩/어댑터를 학습해야 한다. Tigran Hamasyan은 전체 PiJAMA에 13곡(앨범 "A Fable", 스튜디오, 합계 0.89시간)만 있고 PiJAMA-30에는 없다 [직접 확인, CSV 집계 — [PiJAMA pijama.csv](https://github.com/almostimplemented/PiJAMA)] → 독립 어댑터를 만들기엔 데이터가 매우 적다.
- Aria의 embedding_adapter + CFG 경로는 "참조 MIDI 한 곡의 전역 임베딩으로 스타일을 끌어당기는" 방식이라, 아티스트 스타일 조건의 저비용 대안(학습 없이 레퍼런스 MIDI만 교체)으로 먼저 시험해볼 가치가 있다. 어느 체크포인트(gen 추정)가 이 어댑터로 학습됐는지는 미확인.
- 사용자의 "KV cache 없는 생성 39분" 문제는 공식 MLX 백엔드(KV cache 구현)로 전환하면 크게 줄어들 것으로 예상된다(이번 세션에서 측정하지는 않음).

### Gaps
- Aria 논문의 정확한 청취 테스트 표·추론 속도 수치, Aria-Duet의 측정 지연(ms), aria-medium-gen이 어떤 데이터로 파인튜닝됐는지, embedding_adapter가 붙은 체크포인트가 어느 것인지 — HF/arXiv 차단으로 미확인.
- Aria 재즈 파인튜닝 체크포인트는 Edwards 외에는 찾지 못했다(검색 범위 내 "없음"이 아니라 "발견 못함").

## Q3. Anticipatory Music Transformer — 체크포인트와 데이터, 조건화

### Takeaway
AMT(TMLR 2024)는 Apache-2.0이고 HF에 small/medium/large 800k 체크포인트가 있으며 모두 Lakh MIDI 학습으로 보고된다. "다른 트랙(멜로디 등)을 control로 주고 반주를 생성"하는 조건화가 내장돼 있지만 벨로시티가 없고(3토큰 이벤트) 피아노 표현력이 낮다.

### Cited Findings
- 저장소는 데이터셋 구성·샘플링 코드 제공, 학습은 Levanter 사용; 사전학습 모델은 CRFM이 HF Hub에 호스팅; Apache-2.0 [직접 확인] — [jthickstun/anticipation](https://github.com/jthickstun/anticipation)
- 반주 생성 예: 멜로디를 `controls`로 주고 `generate(model, 5, 20, inputs=history, controls=melody)` [직접 확인] — [anticipation README](https://github.com/jthickstun/anticipation)
- 토큰화 상수: CONTEXT_SIZE 1024, EVENT_SIZE 3, DELTA(anticipation) 5 s, TIME_RESOLUTION 100(10 ms), MAX_DURATION 10 s, MAX_PITCH 128, MAX_INSTR 129 [직접 확인] — [anticipation/config.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/config.py)
- human eval 재현 디렉터리: Lakh 테스트 split에서 멜로디 클립을 골라 반주 생성 비교 [직접 확인] — [anticipation/humaneval](https://github.com/jthickstun/anticipation/tree/main/humaneval)
- music-small-800k: "Small (128M parameter) Transformer trained for 800k steps on arrival-time encoded music from the Lakh MIDI dataset" [요약만, HF 카드 스니펫] — [HF stanford-crfm/music-small-800k](https://huggingface.co/stanford-crfm/music-small-800k)
- CRFM 블로그: 예제 생성에 쓴 360M 모델 공개, 800k step 학습 [요약만] — [CRFM blog](https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html)
- TMLR 04/2024 게재 [요약만] — [mlanthology](https://mlanthology.org/tmlr/2024/thickstun2024tmlr-anticipatory)
- music-large-800k는 Aria 논문의 비교 baseline으로 사용됨 [요약만] — [arXiv 2506.23869](https://arxiv.org/html/2506.23869)
- MIDI-LLM은 AMT 어휘를 Llama 3.2 1B에 이식 [직접 확인 README] — [slSeanWU/MIDI-LLM](https://github.com/slSeanWU/MIDI-LLM)

### Inferences
- "Lakh 전용 여부": 세 크기 모두 Lakh 학습으로 보는 것이 자료와 일치하지만, medium/large 카드 원문은 확인하지 못했다.
- 코드 진행을 "control 트랙"(예: 블록 코드 MIDI)으로 넣어 반주가 아닌 솔로를 생성하는 응용은 구조상 가능하나, 피아노 솔로 표현력(벨로시티 부재)과 재즈 데이터 부족 때문에 Aria 대비 우선순위는 낮다.

### Gaps
- medium/large 정확 파라미터 수, 각 체크포인트 카드의 데이터 기술 원문 — HF 차단으로 미확인.

## Q4. Moonbeam — 크기, 토큰화, ComMU 코드 조건 LoRA, PiJAMA 실험

### Takeaway
Moonbeam(Guo & Dixon, QMUL)은 Apache-2.0, 6속성 토큰 + 속성별 상대 어텐션(MRA), LoRA 파인튜닝 레시피(llama-recipes/PEFT)까지 공개된 "코드 조건 파인튜닝 경로가 있는" 대형 모델이다. 하지만 README상 ComMU 조건 체크포인트 링크는 "TODO"이고, 재즈 증거는 PiJAMA 프롬프트 연속 데모와 PiJAMA30 피아니스트 분류(Acc 0.679) 수준이다.

### Cited Findings
- README: 81.6K(시간)·18B 토큰 사전학습, 도메인 지식 기반 토큰화 + 새로운 어텐션; 사전학습 체크포인트 HF `guozixunnicolas/moonbeam-midi-foundation-model`; "Unconditional Generation Checkpoint (ATEPP-Bach)"와 "Conditional Generation Checkpoint (Commu)" 링크는 `https://TODO` [직접 확인, 최종 커밋 2025-06-12] — [Moonbeam README](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model)
- 데모 페이지는 "85k hours"로 표기 — README(81.6K)와 불일치 [직접 확인, gh-pages 저장소 HTML] — [aim-qmul/moonbeam-midi-foundation-model](https://github.com/aim-qmul/moonbeam-midi-foundation-model)
- `model_config.json`(M): hidden 1920, intermediate 6720, 12 heads, 6 KV heads, 15 layers, max_len 1024, 속성별 rope_theta(onset 199999, dur 1031, octave 19, pitch 20, velocity 131), 어휘 onset 4099/dur 4099/octave 13/pitch-class 14/instrument 131/velocity 130, GRU 디코더 hidden 1536·4층; `model_config_small.json`: hidden 1536, intermediate 5376, 9 layers [직접 확인] — [src/llama_recipes/configs](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model/tree/main/src/llama_recipes/configs)
- 파인튜닝: `--use_peft True --peft_method lora`, `--pure_bf16 True`, `torchrun`+DDP; 비조건 연속(context 2048), 조건 생성(context 848, batch 15, lr 3e-4) [직접 확인] — [Moonbeam README](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model)
- `conditional_gen_commu` 브랜치: `if_add_chords_in_transformer`, `if_add_metadata_in_transformer`; `chord_dict`에 A, A#7, A#maj7, Adim, Am7, Asus4, Bm7b5, C#+ 등 트라이어드·7th·dim·m7b5·sus4·aug [직접 확인, 브랜치 fetch] — [conditional_gen_commu](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model/tree/conditional_gen_commu)
- 분류 결과(데모 페이지 표): PiJAMA30 Acc/F1 — Moonbeam(M) 0.679/0.638, Moonbeam(S) 0.649/0.596, MusicBERT 0.55/0.452, CLaMP2 0.44/0.219, M3 0.262/0.077; Pianist8 M 0.946 [직접 확인] — [Moonbeam demo HTML](https://github.com/aim-qmul/moonbeam-midi-foundation-model)
- 비조건 연속 파인튜닝 데이터: ATEPP(클래식 피아노), GAPS(클래식 기타), Pijama(재즈 피아노); 샘플은 cherry-pick [직접 확인] — [Moonbeam demo HTML](https://github.com/aim-qmul/moonbeam-midi-foundation-model)
- ComMU: 객관 지표는 baseline이 약간 높았으나 전문가 주관 평가에서 Moonbeam이 코드/메타데이터 적합·일관성·음악성에서 유의하게 우수 [요약만] — [alphaxiv 2505.15559](https://www.alphaxiv.org/abs/2505.15559), [themoonlight review](https://www.themoonlight.io/en/review/moonbeam-a-midi-foundation-model-using-both-absolute-and-relative-music-attributes)
- 리뷰는 PiJAMA가 사전학습 코퍼스와 겹칠 가능성(제거 실험 없음)을 지적 [요약만] — [themoonlight review](https://www.themoonlight.io/en/review/moonbeam-a-midi-foundation-model-using-both-absolute-and-relative-music-attributes)
- 제3자 포크 `Apisferum/moonbeam-codebase`가 2026-09 최신 transformers/hub 호환 수정 [직접 확인] — [Apisferum/moonbeam-codebase](https://github.com/Apisferum/moonbeam-codebase)

### Inferences
- 사용자 지시문의 "ComMU chord-conditioned LoRA adapter on HF"는 GitHub README와 충돌한다(README는 TODO). HF 저장소 파일 목록을 직접 볼 수 없어 결론은 보류 — HF에서 adapter 파일 존재를 먼저 확인할 것.
- 코드 조건 아키텍처(코드 토큰을 트랜스포머에 주입 + LoRA)는 Aria에 이식할 설계 참고로 유용하다. 단, Moonbeam 자체를 베이스로 바꾸면 피아노 표현 연속 품질의 근거(청취 비교)가 Aria보다 약하고, CUDA/DDP 중심 코드라 MPS 포팅 비용이 든다.

### Gaps
- 공식 839M/309M 수치(내 계산 백본 746M/287M + GRU 디코더·임베딩으로 정합 추정), ComMU 정량 수치, 추론 속도 — 미확인.

## Q5. MIDI-GPT / MMM, MIDI-RWKV, Notochord — 소형·실시간 지향 모델

### Takeaway
세 모델 모두 M1 Max에서 실시간 추론이 현실적인 소형 모델이다. MIDI-GPT(2026 MIT 패키지)는 마디별 pitch-class set·장르 등 속성 제어와 OSC 실시간 서버를 제공하고, MIDI-RWKV(MIT, ~35M 추정)는 state tuning 개인화가 LoRA보다 낫다는 결과를 냈으며, Notochord(MIT)는 10 ms 미만 응답을 주장하는 이벤트 단위 모델이다. 다만 피아노 재즈 솔로 품질 증거는 셋 다 없다.

### Cited Findings
- MIDI-GPT 패키지: `pip install "midigpt[inference]"`, CPython 3.10–3.12 macOS arm64 휠; 모델 표 — yellow_medium/yellow_small(4·8마디, 밀도·폴리포니·길이), prism_medium(4–16마디, 키·음역·침묵·길이·마디 밀도·마디 폴리포니·**마디 pitch class set**·장르, "training in progress"), expressive_medium(+nomml, 마이크로타이밍 delta, 128 벨로시티 bin, "training in progress") [직접 확인] — [Metacreation-Lab/MIDI-GPT README](https://github.com/Metacreation-Lab/MIDI-GPT), [docs/models.md](https://github.com/Metacreation-Lab/MIDI-GPT/blob/main/docs/models.md)
- expressive 권장 용도: "e.g. jazz, solo piano, or any music where expressive timing is essential"; prism은 18개 장르 그룹 [직접 확인] — [docs/models.md](https://github.com/Metacreation-Lab/MIDI-GPT/blob/main/docs/models.md)
- 아키텍처 설정: yellow/expressive medium n_embd 768·n_layer 12·n_head 12(expressive max_seq 4096), yellow_small 512·6·8; 체크포인트 메타데이터 arch "gpt2" [직접 확인] — [MIDI-GPT models/](https://github.com/Metacreation-Lab/MIDI-GPT/tree/main/models)
- 실시간: `midigpt[realtime]` OSC 서버가 `/midigpt/bar/end`로 마디 단위 생성 트리거, 백그라운드 스레드 [직접 확인] — [MIDI-GPT README](https://github.com/Metacreation-Lab/MIDI-GPT)
- 라이선스: MIT(2026 Metacreation Lab), CHANGELOG 0.2.0(2026-05-29)에 "MIT LICENSE" 추가 [직접 확인]; HF 가중치 라이선스는 검색으로도 확인 불가, CC BY-NC 언급도 찾지 못함 [요약만] — [MIDI-GPT](https://github.com/Metacreation-Lab/MIDI-GPT), [search: metacreation.net](https://www.metacreation.net/projects/midi-gpt)
- MIDI-GPT 논문: AAAI 2025, DOI 10.1609/aaai.v39i2.32138, 인필링·악기·스타일·밀도·폴리포니·길이 조건 [요약만] — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32138), [arXiv 2501.17011](https://arxiv.org/abs/2501.17011v2)
- MIDI-RWKV: 저장소에 사전학습 가중치 `midi_rwkv.pth`(70,224,852 bytes) 포함, 사전학습 데이터 GigaMIDI(HF gated), 파인튜닝 POP909, state tuning·LoRA 스크립트, rwkv.cpp 변환/평가, LoRA r=alpha=4 및 32 실험, MIT [직접 확인] — [christianazinn/MIDI-RWKV](https://github.com/christianazinn/MIDI-RWKV)
- MIDI-RWKV 설정: MODEL_TYPE x070(RWKV-7), N_LAYER 12, N_EMBD 384, CTX_LEN 2048, vocab_size 16000, miditok BarAttributeControl [직접 확인] — [train/train.sh](https://github.com/christianazinn/MIDI-RWKV/blob/main/train/train.sh)
- MIDI-RWKV 결과: state tuning ~294k 파라미터 vs LoRA r4 ~331k / r32 ~2.7M, 학습 4–6분; 28명 청취 테스트에서 state-tuned가 LoRA r4보다 유의하게 선호, base와 LoRA r4는 차이 없음 [요약만, 제3자 리뷰 기반] — [arXiv 2506.13001](https://arxiv.org/pdf/2506.13001), [themoonlight review](https://www.themoonlight.io/en/review/personalizable-long-context-symbolic-music-infilling-with-midirwkv)
- MIDI-RWKV v2(2026-01) 제목 "Adaptable Symbolic Music Infilling with MIDI-RWKV" [요약만] — [arXiv 2506.13001v2](https://arxiv.org/html/2506.13001v2)
- Notochord: MIT, PyPI 0.8.0, homunculus 텍스트 UI가 MIDI in/out 포트에 직접 연결, `--midi-prompt`, OSC 서버; 기본 체크포인트 `notochord_lakh_50G_deep.pt`; 기본 하이퍼파라미터 emb 512, rnn_hidden 1024, rnn_layers 2(GRU), num_instruments 320, time_res 1e-2 s [직접 확인] — [Intelligent-Instruments-Lab/notochord](https://github.com/Intelligent-Instruments-Lab/notochord), [PyPI](https://pypi.org/project/notochord/)
- Notochord: 폴리포닉·멀티트랙 생성, 입력 응답 지연 10 ms 미만, Lakh 학습(AIMC 2022) [요약만] — [AIMC 2022 PDF](https://iil.is/pdf/2022_aimc_shepardson_et_al_notochord.pdf), [arXiv 2403.12000](https://arxiv.org/pdf/2403.12000)

### Inferences
- MIDI-GPT prism의 "마디별 pitch class set" 속성은 사실상 화성 제약 입력이라, 사용자가 친 코드의 pitch-class 집합을 마디마다 넘겨 솔로 트랙을 인필하는 방식으로 "코드 추종"을 흉내낼 수 있다. 다만 prism/expressive는 학습 진행 중 스냅샷이고 마디 단위 생성이라 "즉시" 반응보다는 마디 지연 구조다.
- MIDI-RWKV의 state tuning 결과는 "아티스트별 교체형 어댑터를 아주 작게"라는 요구에 직접 맞는 증거지만, 모델 자체가 ~35M급 팝/멀티트랙(GigaMIDI) 모델이라 Aria 수준의 피아노 표현을 기대하기 어렵다.

### Gaps
- MIDI-GPT HF 가중치 라이선스(사용자 메모의 MIT vs CC BY-NC 충돌)는 확인 불가. MMM(arXiv 2008.06048) 가중치 라이선스도 미확인.
- Notochord·MIDI-RWKV의 M1 Max 실측 속도는 없음.

## Q6. 기타 모델(2017–2026): ABC/텍스트 LLM, 속성 조건, 반주/코드 조건, 재즈·피아노 특화, 오디오 계열 제외 확인

### Takeaway
ABC·텍스트 계열(NotaGen, ChatMusician, MelodyT5, MuPT, Text2midi, MIDI-LLM, MuseCoco)은 악보 수준이거나 텍스트 조건이라 실시간 피아노 솔로에는 부적합하다. 코드 조건이 명시된 공개 체크포인트는 팝/Lakh 계열(REMI-chord, GETMusic, FIGARO)뿐이며 라이선스(GPL-3.0 등)나 구형 프레임워크 문제가 있다. HeartMuLa와 stream-music-gen은 오디오 모델이고, StreamMUSE는 모델이 아니라 서빙 시스템이다.

### Cited Findings
- NotaGen: 사전학습 110M/244M/516M(patch 12/16/20층, hidden 768/1024/1280), 사전학습 가중치는 period-composer-instrumentation 조건 생성 불가, NotaGen-large ~9k 클래식 파인튜닝 + CLaMP-DPO 3회, NotaGen-X 개선판, 로컬 추론 ~8GB GPU, MIT [직접 확인] — [ElectricAlexis/NotaGen](https://github.com/ElectricAlexis/NotaGen)
- ChatMusician: LLaMA2-7B에서 ABC로 연속사전학습+파인튜닝, LoRA를 어텐션·MLP에 통합, MusicPile 사전학습, 1.1M SFT [직접 확인]; 저장소 루트 LICENSE 파일 없음 [직접 확인] — [hf-lin/ChatMusician](https://github.com/hf-lin/ChatMusician)
- MelodyT5: 인코더-디코더 score-to-score(생성·하모나이제이션·분절), 가중치 HF `sander-wood/melodyt5`, MIT [직접 확인]; 113M [요약만] — [sanderwood/melodyt5](https://github.com/sanderwood/melodyt5), [arXiv 2407.02277](https://arxiv.org/pdf/2407.02277)
- MuPT: 공식 저장소는 Megatron-LM 포크(최종 커밋 2024-05-02)만 확인, 파라미터·가중치 미확인 [직접 확인] — [multimodal-art-projection/MuPT](https://github.com/multimodal-art-projection/MuPT)
- Text2midi: "첫 end-to-end 텍스트→MIDI", AAAI 채택, FLAN-T5 토크나이저/인코더, `Transformer(vocab, 768, 8, 2048, 18, 1024, ...)`, MPS 장치 지원, SymphonyNet 사전학습 + MidiCaps(168k), MIT [직접 확인] — [AMAAI-Lab/Text2midi](https://github.com/AMAAI-Lab/Text2midi)
- MIDI-LLM: Llama 3.2(1B) 어휘 확장, ISMIR 2026, vLLM/transformers 백엔드, 16GB+ CUDA 권장, Llama 3.2 Community License [직접 확인]; 4B+ 토큰 2단계 학습, Text2midi 대비 품질·텍스트 제어·속도 우수 [요약만] — [slSeanWU/MIDI-LLM](https://github.com/slSeanWU/MIDI-LLM), [arXiv 2511.03942](https://arxiv.org/pdf/2511.03942)
- MIDI-LLaMA(arXiv 2601.21740)는 이해용(MusicBERT+Llama-3-8B), 생성 아님 [요약만] — [arXiv 2601.21740](https://arxiv.org/pdf/2601.21740)
- MuseCoco: 2023-06-30 체크포인트 공개, attribute2music 체크포인트 HF `XinXuNLPer/MuseCoco_attribute2music`(`linear_mask-1billion`), ~200M 학습 스크립트, muzic MIT [직접 확인] — [microsoft/muzic/musecoco](https://github.com/microsoft/muzic/tree/main/musecoco)
- MusicBERT: small/base 체크포인트(OneDrive), 2026-08-05 Google Drive 갱신 [직접 확인] — [microsoft/muzic/musicbert](https://github.com/microsoft/muzic/tree/main/musicbert)
- GETMusic: lead+chord 트랙 조건으로 drum/guitar/piano 생성 예, chord guidance 상시 권장, 팝 크롤링 데이터라 스타일 갭 경고, 사용자 지정 코드 진행 입력 기능은 미공개 [직접 확인] — [microsoft/muzic/getmusic](https://github.com/microsoft/muzic/tree/main/getmusic)
- Pop Music Transformer(REMI): `REMI-tempo-checkpoint`(428 MB), `REMI-tempo-chord-checkpoint`(429 MB), GPL-3.0 [직접 확인] — [YatingMusic/remi](https://github.com/YatingMusic/remi)
- Compound Word Transformer: GPL-3.0, chorder(규칙 기반 코드 인식) 사용 [직접 확인] — [YatingMusic/compound-word-transformer](https://github.com/YatingMusic/compound-word-transformer)
- FIGARO: Lakh 학습, 사전학습 모델 polybox 배포, `figaro-no-chord` 변형(=기본은 코드 포함 description), MIT [직접 확인] — [dvruette/figaro](https://github.com/dvruette/figaro)
- Giant Music Transformer: "786M-8k-44L-32H multi-instrumental", Apache-2.0 [직접 확인]; Orpheus Music Transformer: Apache-2.0 저장소 [직접 확인], 479M → 2026 업데이트 748M, Godzilla MIDI, 8k 컨텍스트 [요약만] — [asigalov61/Giant-Music-Transformer](https://github.com/asigalov61/Giant-Music-Transformer), [Orpheus Space](https://asigalov61-orpheus-music-transformer.hf.space)
- PianoBART: ICME 2024, HF `RS2002/PianoBART`, 사전학습 POP1K7·ASAP·POP909·Pianist8·EMOPIA, 저장소 LICENSE 없음 [직접 확인] — [RS2002/PianoBart](https://github.com/RS2002/PianoBart)
- ImprovNet: 클래식→재즈 장르 토큰 기반 반복 corruption-refinement, 하모나이제이션·짧은 연속·인필링, AMT보다 짧은 연속/인필링 우수, 재즈 스타일 79% 식별, artifacts Google Drive 배포, LICENSE 없음 [직접 확인] — [keshavbhandari/improvnet](https://github.com/keshavbhandari/improvnet)
- Jazz Transformer(ISMIR 2020): Transformer-XL, Weimar Jazz Database 리드시트+구조 이벤트, 청취 테스트에서 실제 곡과 뚜렷한 격차 [요약만] — [arXiv 2008.01307](https://arxiv.org/pdf/2008.01307)
- Pianist Transformer(arXiv 2512.02652, ICML 2026): 135M, 10B 토큰 자기지도, 표현 렌더링, 블라인드 청취 인간 수준 주장, 코드·체크포인트 공개 주장 [요약만] — [arXiv 2512.02652](https://arxiv.org/abs/2512.02652v1), [ICML 2026](https://icml.cc/virtual/2026/poster/61542)
- SymPAC(ByteDance, ISMIR 2024, arXiv 2409.03055): 오디오에서 MIR로 추출한 심볼릭 이벤트로 학습, prompt bars + FSM 제약 생성 [요약만]; 가중치 미확인 — [arXiv 2409.03055](https://arxiv.org/pdf/2409.03055)
- Amadeus(arXiv 2508.20665; ACL 2026 long [요약만]): AMD 데이터셋 공개 주장, 가중치 미확인 — [arXiv 2508.20665](https://arxiv.org/pdf/2508.20665), [ACL Anthology](https://aclanthology.org/2026.acl-long.1898.pdf)
- HeartMuLa: 오디오 노래 생성(HeartCodec 12.5 Hz 등) — 심볼릭 아님; MuLaCover는 멜로디/코드 MIDI를 입력으로 사용 [직접 확인 README 일부]; Apache-2.0, RTF≈1.0 [요약만] — [HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib), [arXiv 2601.10547](https://arxiv.org/abs/2601.10547)
- StreamMUSE(arXiv 2606.11886): 프레임 동기 스트리밍 추론 클라이언트-서버 시스템, RTT 환경별 실시간성과 음악 품질 상관 [요약만]; 저장소 미확인 — [arXiv 2606.11886](https://arxiv.org/pdf/2606.11886)
- Streaming Generation for Music Accompaniment(arXiv 2510.22105)의 `lukewys/stream-music-gen`은 DAC 오디오 토큰·Slakh2100 기반 → 오디오 도메인, MIT [직접 확인] — [lukewys/stream-music-gen](https://github.com/lukewys/stream-music-gen)
- ReaLJam/ReaLchords(Google): RL 튜닝 온라인 코드 반주, 1/16 프레임, Hooktheory ~30k, CHI EA '25 [요약만] — [arXiv 2502.21267](https://arxiv.org/pdf/2502.21267)
- Composer's Assistant 2(arXiv 2407.14700, REAPER 멀티트랙 인필링·세밀 제어), 1편은 허용 라이선스 MIDI만으로 학습·모델 공개 [요약만]; 저장소 `Malandro/composers-assistant-REAPER` clone 실패 → 라이선스 미확인 — [dblp](https://dblp.uni-trier.de/pid/137/8467)
- 2025–2026 기타(제목 수준): Versatile Symbolic Music-for-Music Modeling via Function Alignment(arXiv 2506.15548, ISMIR 2025, 가중치 공개 주장), MIDI-RAE-JEPA(arXiv 2607.14537) [요약만] — [arXiv 2506.15548](https://arxiv.org/pdf/2506.15548), [arXiv 2607.14537](https://arxiv.org/pdf/2607.14537)

### Inferences
- GigaMIDI로 학습된 공개 모델 중 확인된 것은 MIDI-RWKV(직접 확인)와 MIDI-LLM(요약만)이며, 둘 다 팝/멀티트랙 성향이다.
- PiJAMA/재즈 피아노로 학습·파인튜닝된 공개 생성 가중치는 Edwards(PiJAMA-12)가 사실상 유일하게 확인되었고, Moonbeam의 PiJAMA 파인튜닝 체크포인트는 공개 여부 미확인이다.
- 대기업(구글·바이트댄스·MS·야마하)의 2025–2026 피아노 특화 심볼릭 파운데이션 모델 공개 가중치는 찾지 못했다(Yamaha는 Edwards 공저, Google은 코드 반주 방향, ByteDance SymPAC은 가중치 미확인, MS muzic은 2023 위주). Sony·Meta의 심볼릭 피아노 모델은 검색 범위에서 발견하지 못했다.

### Gaps
- MuPT, SymPAC, XMusic, Amadeus, Pianist Transformer, Orpheus의 정확한 라이선스·가중치 위치; Kong et al.(GiantMIDI-Piano 계열)의 생성 모델 존재 여부 — 미확인.

## Q7. 코드/화성/반주 조건이 내장된 공개 가중치 모델과, 피아니스트/스타일 조건 모델은 무엇인가?

### Takeaway
**코드/반주 조건 + 공개 가중치**: AMT(트랙 control 반주), GETMusic(chord 트랙), REMI-tempo-chord(코드 토큰, GPL-3.0), FIGARO(코드 포함 description), MIDI-GPT prism/expressive(마디 pitch-class set 속성) 정도이며 **피아노 솔로·재즈 대상 코드 조건 공개 가중치는 없음**. Moonbeam ComMU는 레시피만 공개(체크포인트 TODO). **피아니스트/스타일 조건 + 공개 가중치**: Edwards(12 재즈 피아니스트, Apache-2.0)가 핵심이고, Aria 자체에도 genre/composer prefix와 참조 MIDI 임베딩+CFG 경로가 있다.

### Cited Findings
- AMT: 멜로디 control을 주고 반주 생성 [직접 확인] — [jthickstun/anticipation](https://github.com/jthickstun/anticipation)
- GETMusic: lead·chord 조건으로 다른 트랙 생성, chord guidance [직접 확인] — [muzic/getmusic](https://github.com/microsoft/muzic/tree/main/getmusic)
- REMI-tempo-chord 체크포인트 [직접 확인] — [YatingMusic/remi](https://github.com/YatingMusic/remi)
- FIGARO 기본 모델은 코드 포함 description(`figaro-no-chord`가 제외 변형) [직접 확인] — [dvruette/figaro](https://github.com/dvruette/figaro)
- MIDI-GPT prism/expressive: 마디 pitch class set·키·장르 속성 [직접 확인] — [MIDI-GPT docs](https://github.com/Metacreation-Lab/MIDI-GPT/blob/main/docs/models.md)
- Moonbeam ComMU 코드+메타데이터 조건 파인튜닝 코드 공개, 체크포인트 링크 TODO [직접 확인] — [Moonbeam README](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model)
- Text2midi/MIDI-LLM은 텍스트 안에 코드 진행 서술 가능(실시간 제어 아님) [직접 확인] — [Text2midi](https://github.com/AMAAI-Lab/Text2midi), [MIDI-LLM](https://github.com/slSeanWU/MIDI-LLM)
- ReaLchords/ReaLJam은 멜로디→코드(반대 방향) [요약만] — [arXiv 2502.21267](https://arxiv.org/pdf/2502.21267)
- Edwards: 12명 피아니스트 임베딩 cross-attention, agreement 70% vs 37%(프롬프트만) [직접 확인] — [jazz-pianist-style](https://github.com/almostimplemented/jazz-pianist-style)
- Aria: genre/composer/form prefix 토큰, 임베딩 조건 + CFG [직접 확인] — [aria-utils](https://github.com/EleutherAI/aria-utils), [aria/run.py](https://github.com/EleutherAI/aria/blob/main/aria/run.py)
- NotaGen: period-composer-instrumentation 프롬프트(클래식 작곡가 스타일) [직접 확인] — [NotaGen](https://github.com/ElectricAlexis/NotaGen)
- ImprovNet: 장르 토큰(클래식→재즈) [직접 확인] — [improvnet](https://github.com/keshavbhandari/improvnet)
- MIDI-RWKV: state tuning 스타일 적응 [직접 확인 스크립트 / 요약만 결과] — [MIDI-RWKV](https://github.com/christianazinn/MIDI-RWKV)

### Inferences
- 사용자 목표(코드 추종 + 아티스트 스타일)에 맞는 구성은 "Aria(또는 Edwards 생성기) + 코드 조건 모듈(Edwards식 게이트 cross-attention을 코드 임베딩으로 확장하거나 코드 prefix/인터리브 토큰) + 아티스트 임베딩/LoRA"로 보이며, 코드 조건 부분은 공개 선례 없이 직접 학습·검증해야 한다.
- Edwards의 "프롬프트 prefix 조건은 긴 생성에서 감쇠하지만 cross-attention 조건은 유지된다"는 결과(agreement 70 vs 37)는, 실시간에서 계속 바뀌는 코드 조건도 prefix보다 cross-attention/연속 주입 방식이 유리할 수 있다는 간접 근거다(코드에 대한 직접 검증은 아님).

### Gaps
- 재즈 코드 진행(텐션·대리코드) 조건을 학습한 공개 피아노 모델은 찾지 못했다.

## Q8. M1 Max 32GB(MPS/MLX)에서 파인튜닝·추론 현실성

### Takeaway
추론은 Aria(MLX + KV cache)·Edwards 생성기(MPS 명시)·MIDI-GPT·MIDI-RWKV·Notochord 모두 현실적이다. 학습은 Aria급(650M)에서 LoRA/어댑터만 짧게 학습하는 것이 상한이며(사용자 실측: LoRA r16 + gradient checkpointing 4.1 GiB, 128 step ≈120 s), Edwards식 전체 FT(24GB GPU 18시간)나 Moonbeam 전체 FT, ChatMusician 7B 학습은 열 스로틀링을 고려하면 비현실적이다.

### Cited Findings
- 사용자 실측(과제 지시문 제공): Aria LoRA r16, 1024 토큰, gradient checkpointing → ~4.1 GiB MPS; 128 업데이트 ~120 s; KV cache 없는 생성 16×512 토큰 39분; 배치 1 블록(prefill 237–408 + decode 48) ≈ 0.67 s — 사용자 제공(이번 세션 재측정 안 함)
- Aria 공식 MLX 추론 경로와 KV cache 구현, 실시간 데모 MLX 전용, 응답성은 메모리 대역폭 의존 [직접 확인] — [EleutherAI/aria](https://github.com/EleutherAI/aria)
- Edwards: "Runs on CUDA, Apple silicon (MPS), or CPU"; 재현 표의 시간은 Apple-silicon 노트북 기준(분류기 평가 ~3분, perplexity ~20분/모드), 생성기 학습 ~18시간/24GB GPU [직접 확인] — [jazz-pianist-style README](https://github.com/almostimplemented/jazz-pianist-style)
- Moonbeam: torchrun + DDP + pure bf16 + PEFT LoRA(CUDA 지향 설정) [직접 확인] — [Moonbeam README](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model)
- MIDI-GPT: macOS arm64 휠, 소형 GPT-2 [직접 확인] — [MIDI-GPT](https://github.com/Metacreation-Lab/MIDI-GPT)
- MIDI-RWKV: rwkv.cpp(GGML) 추론, state tuning 수분 [직접 확인 / 요약만] — [MIDI-RWKV](https://github.com/christianazinn/MIDI-RWKV)
- Text2midi: MPS 경로 명시, `requirements-mac.txt` [직접 확인] — [Text2midi](https://github.com/AMAAI-Lab/Text2midi)
- MIDI-LLM: 16GB+ CUDA 권장, fp8은 vLLM 전용 [직접 확인] — [MIDI-LLM](https://github.com/slSeanWU/MIDI-LLM)
- NotaGen-X 로컬 추론 ~8GB GPU [직접 확인] — [NotaGen](https://github.com/ElectricAlexis/NotaGen)

### Inferences
- 실시간 파이프라인 관점에서 Aria 배치 1 블록 0.67 s(사용자 실측)는 "즉시" 반응에는 길다. Aria-Duet식으로 사용자가 연주하는 동안 KV cache를 미리 채우고(공식 데모 코드의 chunked prefill), 짧은 decode 블록을 스트리밍하는 구조가 필요해 보인다.
- 아티스트 개인화는 (1) Edwards 생성기의 Tatum 임베딩을 그대로 사용, (2) Mehldau는 PiJAMA-30의 8.87시간으로 "Aria 동결 + 아티스트 임베딩/cross-attn 또는 LoRA만" 학습하는 경량 변형이 M1 Max에서 현실적인 범위로 판단된다(검증 필요).
- MIDI-RWKV state tuning은 "교체형 초소형 어댑터"의 가장 저렴한 선례이나, 베이스 품질 문제로 Aria 대체재라기보다 설계 아이디어 참고용이다.

### Gaps
- 각 모델의 M1 Max 실측 tokens/s(특히 Aria MLX KV-cache 경로, Edwards cross-attn 추론, MIDI-GPT 마디 생성 지연)는 공개 자료에서 찾지 못했다 — 로컬 벤치마크 필요.
- PiJAMA(CC BY-NC 3.0)·Aria-MIDI(CC-BY-NC-SA 4.0 [요약만]) 데이터 라이선스가 비상업이므로, 이 데이터로 학습한 어댑터의 상업적 배포 가능 여부는 법적 검토가 필요하다(가중치 자체는 Apache-2.0).
