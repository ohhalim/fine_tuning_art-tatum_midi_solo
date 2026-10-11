# 코드(화성) 조건 심볼릭 재즈 즉흥·솔로·멜로디 생성 2017–2026 카탈로그와 코드 추종(chord-following) 평가법

**조사 방법과 검증 태그 (2026-10-11).** 이 세션에서는 egress 프록시가 arxiv.org, export.arxiv.org, ar5iv, alphaxiv, archives.ismir.net, transactions.ismir.net, zenodo, huggingface.co, semanticscholar, openreview, mdpi, ieeexplore, dl.acm, github.io 등을 모두 403으로 막았다(`/__agentproxy/status`의 `connect_rejected` 기록). WebFetch도 이들 호스트에서 DNS 실패였다. 열 수 있었던 1차 출처는 **github.com뿐**이다(`git clone --depth 1`과 WebFetch). 그래서 저장소에 들어 있는 논문·보충자료 PDF(MINGUS 본문·보충자료, BebopNet 보충자료, Edwards 외 ISMIR 2026 본문), README, LICENSE, 소스 코드는 직접 열었다. 나머지 논문 본문은 검색 스니펫만 봤다. WebSearch는 세션 공유 예산(200회)이 이 작업 도중 소진됐다. 그 뒤로는 우회 검색을 하지 않고 이미 아는 GitHub URL만 열었다.

태그는 다음과 같다.
- **[직접 확인]**: 이번 세션에 1차 출처(저장소 PDF·코드·README·LICENSE)를 직접 열었다.
- **[요약만]**: 이번 세션의 검색 스니펫만 봤다.
- **[요약만(이전 세션)]**: 같은 저장소의 이전 조사 노트(`docs/experiments/model_selection_research/chord_conditioned_jazz_models_and_datasets.md`, 2026-10-03)가 스니펫으로 확인한 내용을 재인용했다.
- **[내부 문서]**: 사용자 저장소 문서(`docs/experiments/*.md`)에 기록된 사용자 측 검증을 재인용했다. 이번 세션에서는 HF 차단으로 재확인하지 못했다.

## Q1. 2017–2026 코드/화성 조건 심볼릭 재즈·즉흥·멜로디 생성기 전수 카탈로그 (연도, venue, 조건 표현, 단성/다성, 데이터, 크기, 코드·가중치·라이선스, 청취 테스트, 화성 지표, 실시간 주장)

### Takeaway
2017–2026 사이 코드 조건 재즈 솔로 생성기는 거의 모두 **단성·혼(색소폰/트럼펫) 데이터·소형**이다. Improv RNN, JazzGAN, BebopNet, MINGUS, CMT(재즈 재학습), Frieler & Zaddach가 여기에 든다. 코드는 매 음표나 매 프레임에 붙는 특징(4음 임베딩, 12차원 크로마, 48-트라이어드 원-핫)으로 주입한다.
- **재즈 피아노(다성) 계열**(ImprovNet, JAZZVAR/Overpainting, Edwards 외 2026의 Aria 교차어텐션)은 코드 기호를 조건으로 받지 않는다.
- **다성 피아노 + 명시적 코드 조건**(Poly-dis, Polyffusion, Whole-song, FGG, GETMusic, MIDI-GPT pitch-class-set)은 모두 팝(POP909/Lakh 등) 데이터다.

"코드 조건 + 재즈 피아노 다성 + 실시간 + 전문가 블라인드 검증"을 모두 갖춘 공개 작업은 찾지 못했다.

### Cited Findings

#### A. 단성 재즈 솔로 생성기 (코드 조건) — 핵심 계열

**1. Improv RNN (Magenta)** — [직접 확인]
- 출처: 오픈소스 공개. 논문은 없고 페이지에 연도 표기가 없다(과제 지정 연도 2017) — [magenta/improv_rnn](https://github.com/magenta/magenta/tree/main/magenta/models/improv_rnn)
- 구조: Melody RNN 계열 LSTM이다. 매 스텝 현재 코드를 입력 벡터로 받는다.
- 조건 표현 (세 가지 config):
  - `basic_improv`, `attention_improv`: 48개 트라이어드(12근음 × maj/min/aug/dim) 원-핫
  - `chord_pitches_improv`: 근음 원-핫(12) + 구성 피치클래스 이진(12) + 베이스 원-핫(12)을 이어 붙인 36차원
- 단성. 학습 데이터는 MusicXML 리드시트다.
- 사전학습 번들은 `chord_pitches_improv` 하나다. 페이지는 "thousands of MIDI files"로 학습했다고 적어 리드시트 설명과 어긋난다.
- 생성 인자: `--steps_per_chord` 기본 16스텝(1마디).
- 저장소는 2026-01-06에 아카이브(read-only)됐다.
- 라이선스는 해당 페이지에 없음(미확인). 청취 테스트·화성 지표·지연은 미확인.

**2. Johnson, Keller, Weintraut, "Learning to Create Jazz Melodies Using a Product of Experts" (ICCC 2017)** — [요약만(이전 세션)]
- LSTM 두 개(product of experts)로 과거 멜로디와 코드에 조건화된 다음 음 분포를 예측한다. Impro-Visor 9.0(2017)에 LSTM이 탑재됐다 — [ResearchGate](https://www.researchgate.net/publication/318311055_Learning_to_Create_Jazz_Melodies_Using_a_Product_of_Experts); [Kritsis et al. 2021 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7907589/)
- 크기·청취 테스트는 미확인.

**3. JazzGAN (Trieu & Keller, MUME 2018)** — [요약만]
- RNN(LSTM) 기반 GAN으로 코드 진행 위에 단성 재즈 멜로디를 생성한다 — [Frontiers AI 리뷰](https://www.frontiersin.org/articles/10.3389/frai.2020.508727/full); [arXiv 2211.09124 리뷰](https://arxiv.org/pdf/2211.09124)
- 학습 데이터는 리드시트 44곡(약 1,700마디)이다 — [Jazz Transformer 논문 스니펫](https://program.ismir2020.net/static/final_papers/339.pdf)
- [요약만(이전 세션)] Impro-Visor 음 범주(chord/color/approach/other)와 harmonic "bricks"를 쓰고, 자체 지표에서 ImprovRNN보다 낫다고 보고 — [MUME 2018 PDF](https://musicalmetacreation.org/mume2018/proceedings/Trieu.pdf)
- 코드·가중치 공개는 미확인. 이번 세션 검색 결과에는 ImprovRNN 비교 문장이 나오지 않았다.

**4. Genchel, Pati, Lerch, "Explicitly Conditioned Melody Generation: A Case Study with Inter-dependent RNNs" (MUME 2019)** — [직접 확인(참고문헌만)] / [요약만]
- 코드를 명시적으로 조건화한 상호의존 RNN이다. MINGUS 논문 참고문헌 [9]로 직접 확인 — [MINGUS PDF](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf); [GT 페이지(스니펫)](https://musicinformatics.gatech.edu/publication/explicitly-conditioned-melody-generation)
- 세부 수치는 미확인.

**5. Hung, Wang, Yang, Wang, "Improving Automatic Jazz Melody Generation by Transfer Learning Techniques" (APSIPA ASC 2019, arXiv 1908.09484)** — [요약만]
- 순환 VAE다. 장르 미지정 source 데이터와 Jazz target 데이터를 쓰고, source:target 비율 6단계에서 전이학습 두 방법을 비교한다 — [arXiv](https://arxiv.org/pdf/1908.09484)
  - (1) 사전학습 후 Jazz 미세조정
  - (2) 동시학습 + 잠재벡터 장르 라벨 + 장르 분류기
- 출처 충돌: arXiv 초록은 (2)가 전반적으로 낫다고 하고, NTU 저장소 초록은 (1)이 낫다고 한다 — [NTU repository](https://scholars.lib.ntu.edu.tw/handle/123456789/636065)
- 초록에 코드 조건 언급이 없어 **코드 조건 여부는 미확인**이다.

**6. BebopNet (Haviv Hakimi, Bhonker, El-Yaniv; ISMIR 2020)** — [직접 확인] (코드·보충자료)
- 코드 표현: 각 음표 토큰에 코드 4음의 피치 임베딩을 이어 붙인다 — [mem_transformer.py](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/B_next_note_prediction/transformer/mem_transformer.py)
  - 구현 위치: `mem_transformer.py:538`의 `torch.cat((word_emb, chord_emb), 2)`
  - 데이터 빌더가 `sum(chord_notes) == 4`를 assert한다(`gather_data_from_xml.py:264`) — [gather_data_from_xml.py](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/A_data_prep/gather_data_from_xml.py)
  - `ensure_4_notes()`는 `chord_kind == 'major'`를 **dominant(7)로** 바꾸고, 9·11화음은 7화음으로 축약한다 — [vectorXmlConverter.py](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/utils/music/vectorXmlConverter.py)
- 모델: Transformer-XL. n_layer=4, n_head=8, d_model=400, d_inner=1028, mem_len=64, tgt_len=64, `chord_bias: False` — [training_results/transformer/model/args.json](https://github.com/shunithaviv/bebopnet-code/tree/master/training_results/transformer/model)
  - `model.pt`는 26,297,862 B다(fp32 기준 약 6.5M 파라미터로 추정 — 추정치).
- 라이선스와 의존성: MIT, 마지막 커밋 2020-11-09, `torch==1.1.0` 고정 — [bebopnet-code](https://github.com/shunithaviv/bebopnet-code)
- 데이터: [요약만(이전 세션)] saxsolos.com에서 구매한 색소폰 위주 비밥 전사 284곡(4/4, 코드 포함). 저장소에는 없다.
- 평가 (보충자료) — [BebopNet Supplementary](https://github.com/shunithaviv/bebopnet/blob/master/BebopNet_SupplementaryMaterial.pdf):
  - 사용자 4명의 CRDI 연속 선호 라벨, 표절(LCS) 분석, harmony-guided 생성을 다룬다.
  - **화성 coherence(스케일) 지표를 beam search 목적함수로 쓰면 지표는 최대화되지만, 솔로가 "코드에 맞는 음을 반복하는" 쪽으로 편향돼 성능이 나빠졌다.** 저자들은 이를 RL의 reward hacking에 비유했다.
- 실시간·지연: 미확인.

**7. Jazz Transformer (Wu & Yang, ISMIR 2020, arXiv 2008.01307)** — [직접 확인] (코드) / [요약만(이전 세션)] (청취)
- 코드 조건 모델이 **아니다**. Transformer-XL 리드시트 모델로 코드와 멜로디를 스스로 생성한다.
  - 추론 인자는 `--temp`(기본 1.2), `--n_bars`(기본 32)뿐이다 — [inference.py](https://github.com/slSeanWU/jazz_transformer/blob/master/inference.py)
  - 설정: n_layer=12, d_model=512, n_head=8, mem_len=512 — [model_aug.py](https://github.com/slSeanWU/jazz_transformer/blob/master/transformer_xl/model_aug.py)
- 라이선스: MIT, 마지막 커밋 2020-10-29, TensorFlow, Python 3.6.
- 데이터: WJazzD(단성 솔로 + 코드 라벨).
- 블라인드 청취: 모델 약 2.7–2.9/5, 실제 곡 약 3.0–3.4/5. 참가자 수는 미확인 — [arXiv(스니펫)](https://arxiv.org/pdf/2008.01307)
- 지표는 MusDr로 공개됐다(Q3 참고).

**8. SeqAttn (Jiang, Xia, Berg-Kirkpatrick, "Discovering Music Relations with Sequential Attention", NLP4MusA 2020)** — [직접 확인(MINGUS 논문 내 기술)]
- 코드 조건 BiLSTM이다. sustain 토큰을 쓰는 프레임 표현이다.
- MINGUS 논문은 WJazzD에서 이 모델을 재학습해 비교했다 — [MINGUS PDF](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf)

**9. MINGUS (Madaghiele, Lisena, Troncy; ISMIR 2021)** — [직접 확인] (본문 PDF·보충자료·코드)
- 구조: pitch 모델과 duration 모델이 따로 있는 Transformer encoder(causal mask)다.
- 조건 표현: 음 단위 특징을 concat한다 — [MINGUS_model.py](https://github.com/vincenzomadaghiele/MINGUS/blob/master/B_train/MINGUS_model.py)
  - 현재 코드 C와 다음 코드 NC: 각각 4음 피치 임베딩 → `nn.Linear(4*pitch_embed_dim, ·)`
  - 베이스 B, 박 BE, 마디 내 오프셋 O
  - 5음 이상 코드는 4음으로 자른다.
- 크기 (보충자료 Table 1): hidden 200, encoder layers 4, heads 4, seq len 35, epochs 100 — [Supplementary](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/Supplementary_material_MINGUS_ISMIR21.pdf)
- 데이터: WJazzD(28곡 제외 — BebopNet이 내부 사전 밖의 코드를 인식하지 못해서)와 Nottingham(1,034곡). 분할 70/10/20.
- 결과 (WJazzD):
  - perplexity: MINGUS pitch 11.01 / duration 4.14, BebopNet 44.70(합산), SeqAttn 3.71(sustain 토큰 포함이라 직접 비교 불가)
  - **harmonic coherence**: 생성음 중 현재 코드음 % / 코드에 대응하는 스케일음 %

    | | 코드음 % | 스케일음 % |
    |---|---|---|
    | 원곡 | 49.17 | 72.16 |
    | MINGUS | 51.81 | 77.49 |
    | BebopNet | 40.66 | 64.55 |
    | SeqAttn | 35.92 | 60.26 |

  - MINGUS 문서 링크: [MINGUS PDF](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf)
- 결과 (Nottingham): 원곡 75.63/95.34, MINGUS 70.32/97.53, SeqAttn 74.06/95.95 — [Supplementary](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/Supplementary_material_MINGUS_ISMIR21.pdf)
- **조건 ablation (보충자료 Table 4, WJazzD pitch 모델, 15 epoch)** — [Supplementary](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/Supplementary_material_MINGUS_ISMIR21.pdf)
  - 무조건: ppl 12.30, 정확도 13.57%
  - 최적 조합: ppl 11.71(D-C-B) ~ 11.96(D-C-B-BE-O), 정확도 최대 14.99%
  - 즉 **코드 조건의 우도 이득은 수 % 수준**이다.
  - 본문은 pitch 모델의 최적 특징 조합이 D, C, B, BE, O라고 썼다. NC(다음 코드)는 최적 조합에 없다.
- 블라인드 퀴즈: 28명(음악 애호가 8, 음악 전공생 9, 전문 연주자 11)이 15개 멜로디(평균 20초)를 1–5점 선호로 평가했다. 원곡 5, MINGUS 5, BebopNet 5개이고 셔플 드럼과 코드 반주를 붙였다.
  - MINGUS가 BebopNet보다 선호됐다.
  - 전문가일수록 원곡을 더 잘 구분했다.
- 공개: 사전학습 가중치(`B_train/models/{pitchModel,durationModel}/MINGUS COND I-C-NC-B-BE-O Epochs {10,100}.pt`), LGPL-3.0, 마지막 커밋 2024-07-14 — [repo](https://github.com/vincenzomadaghiele/MINGUS)
- 실시간: 미확인.

**10. CMT — Chord Conditioned Melody Transformer (Choi et al., "Chord Conditioned Melody Generation with Transformer Based Decoders", IEEE Access 2021)** — [직접 확인] (코드·README) / [요약만] (논문)
- **venue 정정: ISMIR 2021이 아니라 IEEE Access 2021이다** — [IEEE Xplore 9376975 (README 링크)](https://ieeexplore.ieee.org/abstract/document/9376975); [DOAJ(스니펫)](https://doaj.org/article/fecda55c17fe41d38bb1628f14f57fb4)
- 조건 표현: 16분음표 프레임마다 12차원 크로마(멀티핫) × 학습 행렬(12×128)을 **양방향 LSTM**에 통과시킨다. 그 결과를 rhythm decoder와 pitch decoder 입력에 프레임별로 concat한다 — [model.py](https://github.com/ckycky3/CMT-pytorch/blob/master/model.py)
  - 구현 위치: `bidirectional=True`, `chord_hidden[0]`, `chord_hidden[1]`
  - **8마디 창 전체의 미래 코드를 본다.**
- 설정: num_bars 8, frame_per_bar 16, num_layers 8, num_heads 16, hidden 512, num_pitch 50(48음역 + 2), topk 5 — [hparams.yaml](https://github.com/ckycky3/CMT-pytorch/blob/master/hparams.yaml)
- 학습은 2단계다 — [README](https://github.com/ckycky3/CMT-pytorch)
  - (1) 12조 전조한 데이터로 rhythm decoder
  - (2) 단일 조 데이터로 pitch decoder
- 공개: `generated samples.zip`에 청취 테스트에 쓴 15개 샘플이 있다. **LICENSE 파일이 없다**(마지막 커밋 2021-06-27).
- 논문 평가 [요약만]: 피치·리듬 특징(chord tone ratio, 리듬 분포)이 데이터셋 특성을 반영했다고 보고했고 청취 테스트도 했다(결과 수치 미확인) — [DOAJ](https://doaj.org/article/fecda55c17fe41d38bb1628f14f57fb4)
  - 원 학습 데이터는 EWLD로 추정된다. 근거는 Wang 2025가 EWLD에서 "CMT-style baseline"을 썼다는 스니펫뿐이라 미확인이다.
- **HSE 2026 WJazzD 재학습** [내부 문서]: HF `maxkudryashov/{cmt-1, mingus-1, bebopnet-1}`(MIT)
  - CMT `paper/8bars`를 사용자 T1에서 실행했다.
  - 비교 파이프라인 저장소는 있으나 라이선스 표기가 없다.
  - "CMT가 특화 모델과 경쟁 가능"이라는 결론은 결과표를 못 찾아 미확인이다.
  - MINGUS 재현 pitch ppl은 13.08(논문 11.01)이다.
  - 출처: `docs/experiments/MODEL_SELECTION_ROUND2_3.md`, `docs/experiments/T1_PUBLIC_CHECKPOINTS.md`
  - 이번 세션 검색으로는 HF 페이지나 학위논문 원문을 찾지 못했다(러시아어 검색 포함) — [검색 결과 예](https://transactions.ismir.net/articles/10.5334/tismir.87)

**11. Frieler & Zaddach, "Evaluating an Analysis-by-Synthesis Model for Jazz Improvisation" (TISMIR 5(1), 2022)** — [요약만(이전 세션)]
- 모델: WJazzD 통계 + Weimar Bebop Alphabet(9종 "atom")의 계층 Markov 모델. chord-scale 이론으로 피치를 고른다. 단성이다.
- Turing 유사 테스트: 41명(재즈 전문가 29명). 컴퓨터 솔로 탐지 정확도는 전문가 64.4%, 비전문가 41.7%였다.
- 손으로 고르고 편집한 표현적 렌디션 1개는 사람 연주로 더 자주 판정됐다.
- 코드 공개는 미확인.
- 출처: [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.87)

**12. Raposo & Soares, "Phrase-Oriented Generative Rhythmic Patterns for Jazz Solos" (Applied Sciences 15(20):11058, 2025-10-15)** — [요약만]
- 저자·게재 정보 — [IPCB 저장소](https://repositorio.ipcb.pt/entities/publication/c17a877f-1374-4704-afda-8ba56a5f522b); [DOI](https://doi.org/10.3390/app152011058)
- [요약만(이전 세션)] WJazzD 통계 기반 Markov 체인으로 프레이즈 단위 **리듬만** 생성한다. 평가는 통계적이다.
- 코드 조건 피치 생성은 없다.

**13. Keating & Casey, "A Graph Engine for Guitar Chord-Tone Soloing Education" (ICMC 2025 데모, arXiv 2510.19666)** — [직접 확인] (README·LICENSE)
- 방법: 코드마다 chord-tone 아르페지오 노드를 만든다. 연속 코드 간 "optimal transition tones" 가중 간선을 잇고 최단경로로 chord-tone 솔로 라인을 만든다.
- 학습 없는 결정론적 방법이다.
- MIT, 마지막 커밋 2025-12-11 — [repo](https://github.com/mbkeating/chord-tone-generator)
- 청취 테스트: 미확인.

**14. Impro-Visor (Keller, Harvey Mudd)** — [직접 확인] (저장소)
- 연주자별 학습 grammar 파일(`grammars/`)이 들어 있다. 피아니스트로는 BillEvans, BudPowell, CedarWalton 등이 있고 Tatum/Mehldau는 없다.
- LSTM용 `connectomes`(CliffordBrown.ctome, JohnColtrane.ctome 등)도 있다.
- 버전 10.2(2019-06-12), 마지막 커밋 2019-06-19, GPL v2(LICENSE.txt) — [repo](https://github.com/Impro-Visor/Impro-Visor)
- grammar 학습 원 논문(Gillick, Tang, Keller "Machine Learning of Jazz Grammars")은 검색 예산 소진으로 이번 세션에 미확인이다 — [ResearchGate(제목만)](https://www.researchgate.net/publication/247932286_Machine_Learning_of_Jazz_Grammars)

#### B. 일반(팝·클래식·포크) chord→melody / melodization / 제어 생성 — 재즈 아님

**15. MelodyT5 (Wu, Wang, Li, Yu, Sun; ISMIR 2024, arXiv 2407.02277)** — [요약만] + [내부 문서]
- ABC 표기 encoder-decoder다. MelodyHub(261K+ 멜로디, 7과제)로 사전학습했다. melodization(= 코드 → 멜로디)이 과제 중 하나다 — [ISMIR 2024 poster](https://ismir2024program.ismir.net/poster_90.html); [HF card(스니펫)](https://huggingface.co/sander-wood/melodyt5)
- [내부 문서] 113M, MIT, HF 가중치 공개(`MODEL_SELECTION_ROUND2_3.md`)
- `github.com/sander-wood/melodyt5`는 이번 세션 WebFetch에서 404였다.

**16. Jiang, Chin, Lin, Liu, Xia, "Versatile Symbolic Music-for-Music Modeling via Function Alignment" (ISMIR 2025, arXiv 2506.15548)** — [요약만] + [내부 문서]
- 사전학습 LM 두 개(참조·목표 시퀀스)를 경량 adapter로 연결한다.
- chord→melody에서 **MelodyT5와 주관평가가 비슷**하다고 보고했다. melody→chord에서는 perplexity와 chord consistency로 평가했다 — [ISMIR 2025 program](https://ismir2025program.ismir.net/poster_25.html); [arXiv](https://arxiv.org/pdf/2506.15548)
- 코드·가중치 공개를 주장한다. [내부 문서] 저장소 실재와 chord→melody 과제 포함을 확인했고, 학습 데이터와 격자는 미확인이다.

**17. Salem, Shokri, Devaney, "Chord-conditioned Melody and Bass Generation" (NeurIPS 2025 AI4Music workshop, arXiv 2511.08755)** — [요약만]
- Transformer 5설정을 비교한다: 무조건 / 독립 / bass-first / melody-first / co-generation.
- 데이터: TAVERN(고전 변주곡).
- 지표: 피치 내용, 음정 크기, 라인 전체의 chord tone vs non-chord tone 비율(Yeh et al. 2021 인용).
- 결과: 코드 조건이 양식적 피치 내용과 코드톤 사용 재현을 개선했고, bass-first가 가장 이득이었다 — [arXiv](https://arxiv.org/pdf/2511.08755)

**18. Junnuo Wang, "Theory Structured Harmonic Embeddings for Chord Conditioned Melody Generation" (Academic Journal of Computing & Information Science, Francis Academic Press, 2025)** — [요약만]
- 코드를 Root + Quality + Extension + Tension 가산 임베딩으로 분해한다. 추론 시 이론 prior(코드톤 선호, 텐션 유효성, 비화성음 해결, 스케일)로 피치를 재가중하며, 강도 파라미터는 1개다.
- 데이터: EWLD. 기준: CMT-style, EC²-VAE.
- 지표: chord tone ratio, tension correctness, non-chord-tone resolution — [Francis Press](https://francis-press.com/papers/20061)

**19. "Chord-Conditioned Melody Harmonization with Controllable Harmonicity" (arXiv 2202.08423)** — [요약만]
- 지표: CTnCTR(코드톤/비코드톤 비), PCS(Pitch Consonance Score), MCTD(Melody-Chord Tonal Distance).
- harmonicity 조절값이 CTnCTR·PCS와 강한 양의 상관, MCTD와 음의 상관을 보였다 — [arXiv](https://arxiv.org/pdf/2202.08423)
- venue: 미확인.

**20. Kaliakatsos-Papakostas 외, "Incorporating Structure and Chord Constraints in Symbolic Transformer-based Melodic Harmonization" (arXiv 2512.07627, 2025-12)** — [요약만]
- B* 탐색(beam search + A* + backtracking)으로 사전학습 AR transformer가 특정 마디·onset에 고정한 코드를 지키게 한다. 최악의 경우 지수 복잡도다 — [arXiv](https://arxiv.org/pdf/2512.07627)
- 방향은 멜로디 → 화성이다.
- 같은 그룹의 encoder-only 화성화 논문이 PMLR v303(2026)에 있다 — [PMLR](https://proceedings.mlr.press/v303/kaliakatsos-papakostas26a.html)
- venue: 미확인.

**21. MuseBarControl (Shu et al., arXiv 2407.04331, 2024)** — [요약만]
- 마디 단위 제어를 위해 두 가지를 쓴다. 제어 신호와 토큰을 잇는 사전학습 과제, 그리고 **counterfactual loss**다.
  - counterfactual loss: 마디 속성을 다른 값으로 바꿨을 때 그 마디 NLL이 margin η 이상 올라가지 않으면 hinge로 벌한다.
- 기존 방법 대비 13.06% 개선을 보고했다 — [arXiv](https://arxiv.org/pdf/2407.04331)

**22. Xu, McAuley, Dubnov, Dong, "Equipping Pretrained Unconditional Music Transformers with Instrument and Genre Controls" (arXiv 2311.12257, 2023)** — [요약만]
- MuseScore 150만 곡으로 무조건 사전학습한 뒤 제어 토큰(악기·장르)으로 미세조정했다. 변형은 MMT-I / MMT-G / MMT-GI다.
- 청취 테스트에서 coherence, harmony, arrangement, overall quality가 기준보다 높았다 — [arXiv](https://arxiv.org/pdf/2311.12257)
- 재즈 장르 포함 여부와 venue는 미확인. 코드 조건은 없다.

**23. FIGARO (von Rütte et al.; ICLR 2023 — README는 OpenReview 링크만 제공하므로 venue·연도는 이번 세션 미확인)** — [직접 확인] (코드)
- 마디별 "description"(악기, **코드**, 박자, 음 밀도, 피치·벨로시티·길이 통계)으로 조건화한다.
- 코드는 학습 MIDI 자체에서 규칙 기반 `MIDIChord` 인식기로 추출한다. **자기 라벨링 선례다** — [chord_recognition.py](https://github.com/dvruette/figaro/blob/main/src/chord_recognition.py); [input_representation.py](https://github.com/dvruette/figaro/blob/main/src/input_representation.py)
- 평가 지표: `chord_prec/rec/f1`, `inst_*`, `time_sig_acc`, `note_dens_oa`, `pitch_oa`, `chroma_*`, `groove_*` — [evaluate.py](https://github.com/dvruette/figaro/blob/main/src/evaluate.py)
- 데이터: Lakh MIDI. MIT, 마지막 커밋 2024-10-15 — [repo](https://github.com/dvruette/figaro)

**24. MIDI-GPT (Metacreation Lab, arXiv 2501.17011)** — [직접 확인] (README)
- 다트랙 infill 생성기다. MIT, PyPI 휠(macOS arm64 포함), 마지막 커밋 2026-08-19 — [repo](https://github.com/Metacreation-Lab/MIDI-GPT)
- 모델별 제어:

  | 모델 | 제어 속성 | 상태 |
  |---|---|---|
  | `yellow_*` | 밀도, 폴리포니, 음 길이 | 공개 |
  | `prism_medium` | 키, 음역, 침묵, 마디별 밀도·폴리포니, **마디별 pitch class set**, 장르 | "training in progress" |
  | `expressive_medium` | `prism_medium` 속성 + nomml | "training in progress" |

- venue(AAAI 등)와 속성 제어 정확도 수치는 이번 세션 미확인이다.

**25. GETMusic (Microsoft Muzic)** — [직접 확인] (README)
- 조건 트랙을 고를 수 있다(`c` = chords, `l` = lead 등). 예: 리드 + 코드 트랙으로 드럼·기타·피아노 생성, 코드 트랙 조건 등 — [getmusic README](https://github.com/microsoft/muzic/tree/main/getmusic)
- Muzic 저장소는 MIT다. 팝 데이터이고 재즈 아님.

**26. Moonbeam conditional (Guo 외, 2025)** — [내부 문서] + [직접 확인] (LICENSE)
- [내부 문서] HF에 309M/839M 체크포인트와 ComMU 코드 조건부 LoRA 어댑터(r8)가 있다.
- GitHub 저장소 LICENSE는 **MIT**다(마지막 커밋 2025-05-19) — [repo](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model)
- 내부 문서는 Apache-2.0으로 기록해 **라이선스 표기가 충돌**한다(HF 카드는 미확인).

**27. 그 밖의 확산 기반 코드 조건 멜로디/피아노롤 (모두 비재즈)** — [요약만]
- MelodyDiffusion(Mathematics 11(8):1915, 2023): transformer 확산 + 코드 인코더 — [RePEc](https://ideas.repec.org/a/gam/jmathe/v11y2023i8p1915-d1126684.html)
- Mamba-Diffusion(QMUL, arXiv 2505.03314, 2025): classifier-free guidance로 목표 코드 피아노롤 — [arXiv](https://arxiv.org/pdf/2505.03314)
- D3PIA(KAIST, 2026): 리드시트 → 피아노 반주 이산 확산, POP909 — [arXiv(스니펫상 2602.03523으로 연결됐으나 ID 대응 미확인)](https://arxiv.org/pdf/2602.03523)

#### C. 실시간/상호작용 시스템

**28. SongDriver (Wang 외, ACM MM 2022, arXiv 2209.06054)** — [요약만]
- 두 단계 구조다. (1) Transformer가 들어오는 멜로디에 코드를 배치해 캐시하고, (2) CRF가 캐시된 코드로 다음 멜로디용 다트랙 반주를 낸다.
- "zero logical latency"와 노출편향 회피를 주장한다. 장기 특징 4종을 쓴다. 데이터는 공개셋 + 자체 중국 팝 "àiSong"이다 — [arXiv](https://arxiv.org/pdf/2209.06054); [ZJU](https://www.next.zju.edu.cn/publications/songdriver)
- 방향은 **멜로디 → 반주**(사용자 과제의 역방향)다. 수치는 미확인.

**29. ReaLchords / ReaLJam / GAPT (Wu, Huang, Jaques 외)** — [직접 확인] (코드 README·평가 코드) + [요약만] (논문)
- ReaLchords: ICML 2024, arXiv 2506.14723 [요약만]
  - 온라인 MLE 모델을 RL로 미세조정한다.
  - 보상은 멜로디-코드의 화성·시간 적합도 보상모델 + 미래를 보는 teacher의 distillation이다.
  - 청취 테스트에서 online MLE와 distillation-only보다 선호됐다 — [arXiv](https://arxiv.org/pdf/2506.14723); [ICML](https://icml.cc/virtual/2024/poster/33172)
- ReaLJam: CHI EA 2025, arXiv 2502.21267 [요약만(이전 세션)]
  - 사용자 멜로디 → 모델 코드의 실시간 잼이다. lookahead와 commit 시간을 설정할 수 있다 — [arXiv](https://arxiv.org/abs/2502.21267)
- GAPT: ICLR 2026, arXiv 2511.17879 [요약만]
  - 코히런스 보상 RL이 다양성 붕괴(reward hacking)를 일으켜, 공진화 판별기 보상(적대적 사후학습)을 더했다.
  - 슬라이드 기준 전문가 12명 블라인드 테스트에서 적응 속도·통제감 유의(p<0.05) 개선, 다양성은 데이터 수준에 근접 — [arXiv](https://arxiv.org/pdf/2511.17879); [ICLR](https://iclr.cc/virtual/2026/poster/10010592)
- **코드 [직접 확인]** — [realchords-pytorch](https://github.com/lukewys/realchords-pytorch)
  - MIT, 마지막 커밋 2026-05-11, `pip install realjam`.
  - **Apple Silicon용 MLX 백엔드**(`--mlx`)와 ONNX 가속이 있다.
  - **"online model that generates melody given chords"**(사용자 과제 방향) 레시피가 있다: MLE 사전학습 → contrastive/discriminative/rhythm 보상모델 → GAIL RL.
  - 데이터는 Hooktheory 19,086 / POP909 909 / Nottingham 1,019 / Wikifonia 약 6,000곡이고, ±6 반음 전조 증강을 한다.
  - 멜로디 모델 설정: DecoderTransformer dim 512, depth 8, heads 8, max_seq_len 514, 30k steps, batch 64 — [config](https://github.com/lukewys/realchords-pytorch/blob/main/configs/gen_models/decoder_only_online_melody_4_datasets.yml)
  - 토큰은 프레임마다 멜로디/코드가 교대로 놓인다 — [eval_utils.py](https://github.com/lukewys/realchords-pytorch/blob/main/realchords/utils/eval_utils.py)

**30. jam_bot (Blanchard, Naseck 외 + Jordan Rudess; ISMIR 2025, pp. 755–762, DOI 10.5281/zenodo.17811478)** — [요약만]
- 실시간으로 최적화한 음악 LM을 저지연 멀티스레드 시스템(listen–prompt–schedule)에 넣었다.
- 맥락과 조건 신호를 바꿔 lead / accompaniment / call-and-response 역할을 맡긴다.
- **초록에 지연 수치가 없다(미확인)** — [MIT Media Lab](https://www.media.mit.edu/publications/the-jam_bot-a-real-time-system-for-collaborative-free-improvisation-with-music-language-models/); [ISMIR 2025 poster](https://ismir2025program.ismir.net/poster_321.html)

**31. Anticipatory Music Transformer (Thickstun, Hall, Donahue, Liang; TMLR 2024, arXiv 2306.08620)** — [직접 확인] (README) / [요약만] (평가)
- control 이벤트 infill(예: 멜로디 → 반주)을 한다. Lakh MIDI로 학습했다.
- Apache-2.0, 학습 코드 미포함, HF `stanford-crfm/music-medium-800k` 등, 마지막 커밋 2024-03-18 — [repo](https://github.com/jthickstun/anticipation)
- 평가 [요약만]: MTurk 쌍대비교(20초 클립, "더 관습적으로 음악적인 쪽")에서 반주의 음악성이 인간 작곡과 비슷했다. nucleus p=0.95라 보수적이라고 저자들이 스스로 인정했다 — [arXiv](https://arxiv.org/pdf/2306.08620)

**32. Notochord (Intelligent Instruments Lab, AIMC 2022)** — [직접 확인] (README·LICENSE)
- 실시간 확률적 MIDI 이벤트 모델이다. MIT — [repo](https://github.com/Intelligent-Instruments-Lab/notochord); [AIMC 2022 paper(zenodo)](https://zenodo.org/record/7088404)
- **코드 조건은 없다.** [내부 문서] 이벤트당 10 ms 미만이라는 기록이 있으나 이번 세션에서는 README에서 수치를 찾지 못했다.

**33. Aria-Duet / "The Ghost in the Keys" (NeurIPS 2025 Creative AI, arXiv 2511.01663)** — [요약만(이전 세션)]
- Aria 기반 Disklavier 턴테이킹 듀엣이다. 코드 조건은 없다 — [arXiv](https://arxiv.org/abs/2511.01663)

#### D. 재즈 피아노(다성) 생성 — 코드 기호 조건 없음

**34. ImprovNet (Bhandari, Chang, Lu, Enus, Bradshaw, Herremans, Colton; IJCNN 2025 [요약만: venue는 검색 스니펫과 C4DM 뉴스 기준], arXiv 2502.04522)** — [직접 확인] (README·config·평가 코드·데모 페이지 소스)
- corruption–refinement 자기지도 학습이다. 5초 세그먼트를 9종 손상(pitch/velocity mask, onset/duration mask, whole mask, permute, fragmentation, incorrect transposition ±5, skyline, note modification) 후 복원한다.
- 장르 토큰(classical/jazz)과 손상 종류 토큰을 encoder에 넣는다.
- 지원 과제: cross-genre(Classical→Jazz) / intra-genre 즉흥, 장르 스타일 화성화(단성 멜로디 → 피아노), 5–20초 continuation, infill.
- encoder 12층/8헤드/512, decoder 12층/8헤드/512 — [config_style_transfer.yaml](https://github.com/keshavbhandari/improvnet/blob/master/configs/config_style_transfer.yaml)
- 입력은 피아노 단일 트랙 MIDI다. 구조 보존용 SSMNet에는 WAV가 필요하다.
- **LICENSE 파일 없음.** 가중치는 Google Drive 아카이브. 마지막 커밋 2025-04-18 — [repo](https://github.com/keshavbhandari/improvnet)
- 평가:
  - AMT보다 short continuation/infill이 낫다고 보고했다.
  - 청취자 79%가 재즈 스타일 변환을 올바로 식별했다 — [README](https://github.com/keshavbhandari/improvnet); [데모 페이지 소스](https://github.com/keshavbhandari/keshavbhandari.github.io/blob/master/_portfolio/improvnet.html)
  - 평가 코드는 tonal tension과 MusPy(pitch_in_scale_rate, groove_consistency 등)를 원곡 대 생성으로 비교하고, **random_chords 기준선** 결과 파일도 들어 있다 — [eval/](https://github.com/keshavbhandari/improvnet/tree/master/eval)
- 참가자 수, 전문성, 블라인드 여부, 실시간: 미확인.

**35. JAZZVAR (Row, Tang, Fazekas; arXiv 2307.09670, 2023)와 "Exploring Transformer-Based Music Overpainting for Jazz Piano Variations" (Row, Shanin, Fazekas; ISMIR 2024 LBD, arXiv 2412.04610)** — [요약만]
- JAZZVAR: 원곡/변주 MIDI 502쌍 + Music Transformer 기준선.
- Overpainting: 리드시트 구간과 연주 head 구간을 반자동 정렬한 VAR4000(4,352쌍)을 만들었다. 두 transformer 설정이 일반화에서 "promising" 개선을 보였다 — [arXiv 2412.04610](https://arxiv.org/pdf/2412.04610); [ISMIR 2024 LBD](https://ismir2024program.ismir.net/lbd_479.html); [JAZZVAR](https://arxiv.org/pdf/2307.09670)
- 과제는 멜로디·화성 구조를 보존하는 재즈 피아노 변주다. 코드 기호 조건은 아니고 원곡 MIDI 조건이다.
- 크기·청취 테스트·코드 공개: 미확인.

**36. Edwards, Maezawa, Dixon, "Learning Jazz Pianist Style with Cross-Attention Conditioning" (ISMIR 2026)** — [직접 확인] (본문 PDF·README·LICENSE·데이터 CSV)
- 백본: Aria-medium(659M, 16층, d=1536). 마지막 8층 self-attention 뒤에 **gated cross-attention**(x′ = x + g·CrossAttn(x, E_artist), g 초기값 0.1, Flamingo식)을 넣었다. E_artist ∈ R^{4×d}, 12명이다. 추가 파라미터는 78M(12%)이다.
- 학습: 전체 + adapter, 15 epoch, 4096토큰, lr 5e-6, effective batch 32, fp16, **RTX 6000 1장 약 18시간**.
- PiJAMA-12: 779곡, 약 57시간 — [paper.pdf](https://github.com/almostimplemented/jazz-pianist-style/blob/main/docs/paper.pdf)
  - 곡 수: Dick Hyman 187, **Art Tatum 118**, Abdullah Ibrahim 84, Dave McKenna 76, … Horace Parlan 28. **Brad Mehldau는 없다** — [pijama12.csv](https://github.com/almostimplemented/jazz-pianist-style/blob/main/data/pijama12.csv)
- **핵심 관찰: 우도는 조건 품질을 거의 못 본다** (Table 2) — [paper.pdf](https://github.com/almostimplemented/jazz-pianist-style/blob/main/docs/paper.pdf)

  | 모델 | 테스트 CE | 자유생성 분류기 일치율 |
  |---|---|---|
  | 미세조정(no CA) | 1.94 | 37% |
  | cross-attention | 1.92 | 70% |
  | pretrained Aria | — | 25% |

  - 일치율은 P=256 기준이다. 조건 효과는 자유생성에서만 드러난다.
  - 합성만으로 학습한 분류기가 실제 피아니스트를 87.1%(청크)/95.0%(곡) 식별했다.
- **코드/리드시트 결합 조건과 인간 청취 연구는 향후 과제로 남겼다.**
- 공개: Apache-2.0. HF `drewbie/jazz-pianist-style-{generator,classifier,synthetic-classifier}`. MPS/CPU 평가 가능. 마지막 커밋 2026-10-01 — [README](https://github.com/almostimplemented/jazz-pianist-style)

**37. Aria (Bradshaw 외, "Scaling Self-Supervised Representation Learning for Symbolic Piano Performance", ISMIR 2025, arXiv 2506.23869)** — [직접 확인] (README·LICENSE)
- Apache-2.0, 마지막 커밋 2026-05-12 — [EleutherAI/aria](https://github.com/EleutherAI/aria)
- 코드 조건은 없다(사용자가 이미 사용 중).

#### E. 다성 피아노 + 명시적 코드 조건 (팝 데이터) — Q2에서 상세

**38–40. Poly-dis(ISMIR 2020), Polyffusion(ISMIR 2023, arXiv 2307.10304), Whole-song(ICLR 2024)** — [직접 확인] (README·LICENSE). 세부는 Q2에 있다.

**41. FGG (Zhu, Liu, Wang, Jiang, Zheng; ICML 2025, PMLR 267:79938–79962, arXiv 2410.08435)** — [요약만]. 세부는 Q2에 있다.

#### F. 미식별·미발견 항목
- **"EAR Drummer"**: 검색으로 해당 이름의 모델을 찾지 못했다. JukeDrummer(오디오), "Language Models are Drummers" 등만 나왔다 — [검색 결과 예: JukeDrummer](https://arxiv.org/pdf/2210.06007)
- **"JazzDDPM"/재즈 확산 솔로**: 재즈 솔로 전용 확산 모델은 찾지 못했다. 확산 계열은 모두 팝·일반이다(위 27, Q2).
- **2025 LLM의 WJazzD 코드 조건 솔로 미세조정 논문**: 찾지 못했다 — [검색 결과 예](https://github.com/slSeanWU/jazz_transformer)
- **기타 제목만 확인 [요약만(이전 세션)]**:
  - "MMT-BERT: Chord-Aware Symbolic Music Generation…"(ISMIR 2024) — [accepted papers](https://ismir2024.ismir.net/accepted-papers/)
  - "Comparative Assessment of Markov Models and RNNs for Jazz Music Generation"(arXiv 2309.08027) — [arXiv](https://arxiv.org/pdf/2309.08027)
  - Kritsis et al. 2021(실시간 재즈 반주 RNN, 역방향) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7907589/)

#### 요약 표 (재즈·즉흥 관련 핵심)

| 시스템 (연도, venue) | 조건 표현 | 단/다성 | 데이터 | 크기 | 공개·라이선스 | 청취 테스트 | 화성 지표 | 태그 |
|---|---|---|---|---|---|---|---|---|
| Improv RNN (2017?, OSS) | 48-트라이어드 원-핫 / 근음+PC 이진+베이스 (스텝별) | 단 | MusicXML 리드시트 | 미확인 | 번들 1개, 2026-01 아카이브, 라이선스 페이지 미표기 | 없음(논문 없음) | 없음 | 직접 |
| JazzGAN (2018, MUME) | 코드 진행 입력 | 단 | 44 리드시트 | 미확인 | 미확인 | 미확인 | Impro-Visor 음 범주 | 요약 |
| BebopNet (2020, ISMIR) | 4음 피치 임베딩 concat (음표별) | 단 | 색소폰 284곡 | 4층 d400 (~6.5M 추정) | MIT, 가중치 | 사용자 4명 CRDI(비블라인드) | 화성 coherence, 표절 | 직접 |
| Jazz Transformer (2020, ISMIR) | 없음(스스로 생성) | 단 | WJazzD | 12층 d512 | MIT | 블라인드 2.7–2.9 vs 3.0–3.4 | MusDr H/GS/CPI/SI | 직접/요약 |
| MINGUS (2021, ISMIR) | 현재·다음 코드 4음 + 베이스·박·오프셋 | 단 | WJazzD + Nottingham | 4층 h200 ×2 | LGPL-3.0, 가중치 | 블라인드 28명 | 코드/스케일 % | 직접 |
| CMT (2021, IEEE Access) | 12-크로마 프레임 → BiLSTM | 단 | (원) 미확인 / HSE WJazzD | 8층 h512 | 코드 LICENSE 없음, HSE 가중치 MIT[내부] | 15샘플 청취(결과 미확인) | chord tone ratio | 직접/요약 |
| Frieler & Zaddach (2022, TISMIR) | chord-scale 규칙 + Markov | 단 | WJazzD | — | 미확인 | Turing 41명(전문가 29) | — | 요약 |
| ImprovNet (2025, IJCNN) | 코드 기호 없음(원곡+장르 토큰) | 다 | 클래식+재즈 피아노 | enc/dec 12+12 d512 | LICENSE 없음, GDrive | 79% 장르 식별 | MusPy pitch-in-scale, tonal tension | 직접 |
| Edwards 외 (2026, ISMIR) | 피아니스트 cross-attn (코드 없음) | 다 | PiJAMA-12 | 659M+78M | Apache-2.0, HF | 없음(향후 과제) | 분류기 일치율 | 직접 |
| ReaLchords/GAPT (2024/2026) | 프레임 교대 토큰 (온라인) | 단(멜로디)·코드 | Hooktheory 등 팝 | 8층 d512 | MIT, HF, MLX | 전문가 12명(GAPT) | note-in-chord, 반복·장음 벌점, Vendi | 직접/요약 |

### Inferences
- **코드 주입 방식.** 2017–2022 재즈 솔로 모델은 예외 없이 **매 스텝/음표에 코드 특징을 붙였다**(concat 또는 BiLSTM 인코딩). 프롬프트로 코드를 준 사례는 없다. 2024–2026의 강한 조건 방식은 **사전학습 LM + adapter**다(Function Alignment의 adapter, Edwards의 gated cross-attention, Moonbeam의 LoRA 조건). "Aria 위 코드 cross-attention adapter"는 Edwards가 향후 과제로 명시한 빈칸이다.
- **우도로는 조건 효과를 보기 어렵다.** MINGUS ablation(무조건 → 최적 조건 ppl 12.30 → 11.71, 약 5%)과 Edwards(CE 1.94 → 1.92인데 일치율 37% → 70%)가 같은 방향을 가리킨다. 사용자가 관측한 "코드 효과 NLL 약 1%"는 실패 신호라기보다 **우도가 원래 조건 효과에 둔감하다**는 문헌적 패턴과 일치한다. 판정은 자유생성 기반 지표로 해야 한다.
- **실시간 사용 시 lookahead가 필요하다.** CMT는 양방향 LSTM으로 8마디 미래 코드를 보고, MINGUS는 다음 코드를 입력으로 쓴다. 키보드에서 코드를 "지금" 치는 실시간 설정에서는 lookahead가 없거나 짧다. 사용자 CMT 실험의 "블록마다 진행 반응"(15/16)이 lookahead 가정과 맞물려 있는지 확인할 가치가 있다.
- **코드 단순화의 위험.** BebopNet은 major 트라이어드를 dominant 7로 바꾼다. 모달·Tigran식 sus/quartal 보이싱이나 트라이어드 입력에서 b7 충돌을 낼 수 있다.

### Gaps
- 열지 못한 논문 본문: CMT(IEEE Access), JazzGAN, 1908.09484, SongDriver, MelodyT5, Function Alignment, FGG, 2512.07627, MuseBarControl, 2311.12257, Frieler & Zaddach, Jazz Transformer. 각 논문의 모델 크기, 청취 테스트 참가자·블라인드 여부, 화성 지표 수치 대부분이 미확인이다.
- HSE 2026 학위연구: 원문·결과표·저자명 정식 표기를 찾지 못했다. HF 카드도 이번 세션에서 열지 못했다.
- 검색 예산 소진으로 확인하지 못한 항목:
  - MidiNet(ISMIR 2017, 코드 조건 CNN-GAN), JamBot(2017, 코드 기반 다성 LSTM), MusicVAE 코드 조건 모델, Bretan 외 unit-selection(Shimon), GenJam(Biles)
  - 위 항목은 출처 없이 사실을 적지 않는다.
- Improv RNN, MIDI-GPT, MelodyT5의 정식 연도·venue·라이선스 일부가 미확인이다.

## Q2. 다성(양손) 피아노 솔로를 코드 조건으로 생성한 작업, 그리고 Tigran Hamasyan·모달·변박·non-swing 재즈 생성

### Takeaway
코드 조건 **다성 피아노** 생성은 존재하지만 전부 **팝 반주/텍스처**(POP909, Lakh)다. Poly-dis, Polyffusion, Whole-song, FGG, GETMusic, MIDI-GPT pitch-class-set, AMT infill이 여기에 든다. 다성 **재즈** 피아노 계열(ImprovNet, Overpainting, Edwards 2026)은 코드 기호가 아니라 원곡이나 피아니스트 ID로 조건화한다. Tigran Hamasyan·변박·모달 재즈 **생성 모델은 찾지 못했다**. 분석 문헌만 있다.

### Cited Findings
- **Poly-dis** (Wang 외, "Learning Interpretable Representation for Controllable Polyphonic Music Generation", ISMIR 2020) — [직접 확인]
  - 다성 음악을 chord 표현과 texture 표현으로 분리해 코드를 바꾸면서 텍스처를 유지하는 생성을 노린다 — [repo README](https://github.com/ZZWaang/polyphonic-chord-texture-disentanglement); [icm-deep-music-generation](https://github.com/ZZWaang/icm-deep-music-generation)
  - 학습 코드는 제거된 정리본이다. 사전학습 파라미터는 Google Drive 링크로 제공한다.
  - MIT, 마지막 커밋 2022-12-25.
  - 데이터(POP909로 추정)와 청취 테스트 수치는 미확인.
- **Polyffusion** (Min, Jiang, Xia, Zhao; ISMIR 2023, arXiv 2307.10304) — [직접 확인]
  - 피아노롤 이미지 확산 모델이다. Latent Diffusion의 **cross-attention 조건 메커니즘**만 차용했다.
  - 조건 설정: `sdf_chd8bar`(사전학습 코드 인코더 latent), `sdf_chdvnl`(vanilla 코드 표현), `sdf_txt`(텍스처) 등.
  - 데이터는 POP909 추출 특징이다. 저장소에 `chord_extractor`가 포함되어 코드를 데이터에서 추출한다.
  - MIT, 마지막 커밋 2024-07-16 — [repo](https://github.com/aik2mlj/polyffusion)
- **Whole-song hierarchical generation** (Wang, Min, Xia; ICLR 2024) — [직접 확인]
  - Form → counterpoint → lead sheet → accompaniment 4단계 cascaded diffusion이다. `--external`로 외부 조건 학습을 한다.
  - 프롬프트·외부 제어 생성 기능은 미공개이고 체크포인트는 일부만 공개했다. MIT — [repo](https://github.com/ZZWaang/whole-song-gen)
- **FGG** (ICML 2025) — [요약만]
  - 코드 같은 soft 조건만으로는 세부 피치 정확성을 보장하지 못한다고 보고, 샘플링 중 **hard control**(코드 행렬 M_cond: 피치가 코드에 속하면 1)로 조성 정확성을 보장한다.
  - "실시간 상호작용 생성이 가능"하다는 코드 공개 주장이 있다 — [arXiv](https://arxiv.org/html/2410.08435v3); [PMLR](https://proceedings.mlr.press/v267/zhu25m.html)
  - [내부 문서] 주 과제는 "멜로디 + 코드 → 반주", 데이터는 POP909이고, 공식 저장소는 찾지 못했다.
- **GETMusic** — [직접 확인] 코드 트랙을 조건으로 피아노 등 다른 트랙을 생성한다 — [README](https://github.com/microsoft/muzic/tree/main/getmusic)
- **MIDI-GPT `prism_medium`/`expressive_medium`** — [직접 확인] 마디별 **pitch class set** 속성으로 다성 트랙을 조건화한다. 체크포인트는 "training in progress" 표기다 — [README](https://github.com/Metacreation-Lab/MIDI-GPT)
- **AMT** — [직접 확인] 멜로디를 control로 주고 반주를 infill한다. Lakh 학습이다 — [repo](https://github.com/jthickstun/anticipation)
- **다성 재즈 피아노지만 코드 기호 조건이 아닌 것** — [직접 확인]/[요약만]
  - ImprovNet: 원곡 + 장르 토큰. 단성 멜로디의 재즈 스타일 화성화를 포함한다 — [repo](https://github.com/keshavbhandari/improvnet)
  - Overpainting/JAZZVAR: 원곡 head 조건 변주 — [arXiv](https://arxiv.org/pdf/2412.04610)
  - Edwards 2026: 피아니스트 ID — [paper.pdf](https://github.com/almostimplemented/jazz-pianist-style/blob/main/docs/paper.pdf)
- **Tigran Hamasyan / 변박** — [요약만]
  - 생성 모델·데이터셋·토큰화는 검색되지 않았다.
  - 분석 문헌으로 Schumann(2021, *Music Theory Online*) "Asymmetrical Meter, Ostinati, and Cycles in the Music of Tigran Hamasyan"이 있다. 저자 전사로 13개 앨범 137트랙(2006–2020)에서 39개 metric cycle을 다뤘고, complete/incomplete cycle을 정의한다 — [MTO](https://www.mtosmt.org/ojs/index.php/mto/article/view/688)
  - Berklee 강좌(Brubeck–Tigran 복합 리듬)는 계산 자료가 아니다 — [Berklee](https://college.berklee.edu/node/2524061)
- **non-swing / straight 격자** — [직접 확인]
  - CMT는 16분음표 straight 격자를 쓴다(frame_per_bar 16) — [hparams.yaml](https://github.com/ckycky3/CMT-pytorch/blob/master/hparams.yaml)
  - MINGUS, BebopNet은 4/4 재즈 솔로 데이터다. BebopNet은 4/4 솔로만 쓴다 [요약만(이전 세션)].
  - [내부 문서] 사용자 T1은 CMT straight 출력에 2:1 스윙을 렌더 단계에서 덧씌웠다(`T1_PUBLIC_CHECKPOINTS.md`). straight 그대로 쓰면 non-swing 요구에 맞는다.

### Inferences
- **"양손 피아노 솔로 + 코드 추종"에 가장 가까운 구조 선례**는 두 가지다.
  - (a) Polyffusion/Poly-dis/FGG식 코드 → 피아노롤 확산(팝). 실시간 반마디 블록에는 확산 스텝 수가 부담이다. FGG의 실시간 주장은 미확인이다.
  - (b) MIDI-GPT식 마디별 pitch-class-set 속성(자기 라벨링 가능, 다성).
- 재즈 피아노 데이터에 이 조건을 붙인 사례는 없다. 사용자의 PiJAMA 왼손 PC-set 자기 라벨링은 **FIGARO(규칙 코드 인식), Polyffusion(chord_extractor), MIDI-GPT(pitch class set 속성)의 "데이터에서 조건 추출" 방식과 같은 계열**이다. 다만 아래 Q3의 echo 위험이 크다.
- 변박·모달 재즈는 **데이터 표현(마디 길이·박 그룹)부터 직접 설계해야** 하는 미개척 영역이다. 기존 재즈 솔로 모델들이 4/4·16분 격자를 전제한다는 점(CMT frame_per_bar 16, BebopNet 4/4 전용)이 그 이유다.

### Gaps
- FGG·Function Alignment의 공식 저장소 URL과 실제 지연(ms), D3PIA의 arXiv ID 대응을 확인하지 못했다.
- 다성 코드 조건 모델들(Poly-dis, Polyffusion)의 청취 테스트·코드 정확도 수치는 논문 본문 차단으로 미확인이다.
- 모달·쿼탈 보이싱(sus, 4도 쌓기)을 코드 조건으로 다룬 생성 연구는 발견하지 못했다.

## Q3. 코드 추종(chord-following) 평가법과 스타일 평가 — 특히 "입력 코드음 재타건/복사"와 "진짜 선율 반응"의 구분

### Takeaway
표준 코드 추종 지표는 **코드톤·스케일 비율 계열**이다. MINGUS harmonic coherence, CMT chord tone ratio, CTnCTR/PCS/MCTD, ReaLchords note-in-chord, MusPy pitch-in-scale, FIGARO chord F1이 여기에 든다. 이 계열은 **입력 코드음을 반복·재타건하면 점수가 오르는 구조**다. BebopNet 보충자료는 이 지표를 최적화하면 "코드에 맞는 음을 반복"하는 퇴화가 생긴다고 직접 보고했고, GAPT는 같은 reward hacking을 다양성 붕괴로 다룬다.

입력 복사와 진짜 반응을 **명시적으로 분리하는 공개 지표는 찾지 못했다.** 가장 가까운 장치는 네 가지다.
- 반복·장음 벌점과 Vendi 다양성(ReaLchords/GAPT)
- 전조·삼전음 섭동 적응 테스트(ReaLchords)
- counterfactual 속성 교체(MuseBarControl)
- 분류기 기반 자유생성 평가(Edwards 2026)

### Cited Findings

#### 1) 코드톤·스케일 비율 계열
- **MINGUS harmonic coherence** — [직접 확인]: "생성음 중 현재 코드에 속한 음 %, 코드에 대응하는 스케일 음 %"다. **WJazzD 원곡이 49.17%/72.16%인데 MINGUS가 51.81%/77.49%로 원곡보다 높다** — [MINGUS PDF](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf)
- **BebopNet harmony coherence(스케일)** — [직접 확인]: beam search로 최대화하면 지표는 오르지만 "solos were biased to prefer repeating notes that match the chord"였다. 저자는 "reward hacking"에 비유했다 — [Supplementary](https://github.com/shunithaviv/bebopnet/blob/master/BebopNet_SupplementaryMaterial.pdf)
  - [요약만(이전 세션) code] 구현은 (PC ∈ 4 코드음) ÷ 음길이의 beam 평균이다 — [music_utils.py](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/utils/music_utils.py)
- **CMT chord tone ratio** — [요약만]: 데이터셋 특성 재현으로 보고(수치 미확인) — [DOAJ](https://doaj.org/article/fecda55c17fe41d38bb1628f14f57fb4)
- **CTnCTR / PCS / MCTD** — [요약만]: harmonicity 조절값과 상관을 보였다(2202.08423). Salem 2025도 라인 전체 CTnCTR을 썼다(Yeh 2021 인용) — [2202.08423](https://arxiv.org/pdf/2202.08423); [2511.08755](https://arxiv.org/pdf/2511.08755)
- **Wang 2025** — [요약만]: Chord Tone Ratio, Tension Correctness(코드 기호가 허용하는 텐션), Non-Chord-Tone Resolution(비화성음의 순차 해결) — [Francis Press](https://francis-press.com/papers/20061)
- **ReaLchords note-in-chord ratio** — [직접 확인]: 멜로디·코드 토큰이 교대하는 시퀀스에서 각 프레임의 (pitch mod 12) ∈ 코드 구성음 여부를 평균한다. PAD/BOS/EOS/SILENCE 프레임은 제외한다. 즉 **시간 가중**이다 — [eval_utils.py](https://github.com/lukewys/realchords-pytorch/blob/main/realchords/utils/eval_utils.py)
  - `per_beat_note_in_chord_ratio`(박별)
  - `penalties.pt`(long-note·repetition 벌점, 이벤트 수)
  - 코드 엔트로피, **Vendi score**(다양성)
  - 출처 — [README §Evaluate Harmony and Diversity](https://github.com/lukewys/realchords-pytorch)
- **MusPy pitch_in_scale_rate, groove_consistency + tonal tension** (ImprovNet 평가) — [직접 확인]: 원곡 대 생성, key 지정 유무로 비교하고 **random_chords 기준선**을 함께 산출한다 — [metrics.py](https://github.com/keshavbhandari/improvnet/blob/master/eval/tonal_tension_muspy/metrics.py); [eval_results/random](https://github.com/keshavbhandari/improvnet/tree/master/eval/eval_results/random)
- **FIGARO chord precision/recall/F1** — [직접 확인]: 생성 MIDI에 규칙 기반 코드 인식을 다시 돌려 마디별 목표 코드 집합과 비교한다. chroma cross-entropy/KL/유사도와 groove 유사도도 함께 본다 — [evaluate.py](https://github.com/dvruette/figaro/blob/main/src/evaluate.py)

#### 2) 분포·구조 지표
- **Jazz Transformer MusDr** — [직접 확인]
  - H: 1·4마디 pitch-class histogram entropy, 피치 사용의 산만함
  - GS: grooving pattern similarity, 리듬 일관성
  - CPI: chord progression irregularity, 화성 일관성
  - SI: 오디오 scape plot 기반 structureness indicator
  - MIT — [MusDr](https://github.com/slSeanWU/MusDr)
- **MGEval** (MINGUS) — [직접 확인]: 총 사용 피치, IOI, pitch shift, 음길이 히스토그램, PC 히스토그램, PC·음길이 전이행렬, 음역의 KL과 overlap area. "뚜렷한 승자 없음" — [MINGUS PDF](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf)

#### 3) 조건 의존성 검증 (counterfactual·섭동·ablation)
- **MuseBarControl counterfactual loss** — [요약만]: 마디 제어 속성을 다른 값으로 바꾸면 그 마디 토큰의 NLL이 margin η 이상 올라야 한다(hinge). 학습 손실이자 그대로 **평가 프로브**가 된다 — [arXiv 2407.04331](https://arxiv.org/pdf/2407.04331)
- **ReaLchords 섭동 적응 테스트** — [직접 확인]: 조건 데이터에 `multiple_transpose`(곡 중 여러 번 전조) 또는 `single_transpose_6`(삼전음 한 번 전조)를 가한 뒤 생성·평가한다. OOD 적응 측정이다 — [experiment_utils_data_perturbation.py](https://github.com/lukewys/realchords-pytorch/blob/main/realchords/utils/experiment_utils_data_perturbation.py)
  - GAPT는 "adaptation speed"를 사용자 연구 지표로 보고했다 [요약만] — [arXiv 2511.17879](https://arxiv.org/pdf/2511.17879)
- **조건 ablation 대비** — [직접 확인]
  - MINGUS 무조건 대비(ppl 12.30 → 11.71) — [Supplementary](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/Supplementary_material_MINGUS_ISMIR21.pdf)
  - Salem 2025 무조건 대비 [요약만]
  - Edwards 2026의 "같은 구조 + 조건 0" ablation이 미세조정 기준선과 거의 일치해, 효과가 파라미터 증가가 아니라 조건 신호 때문임을 보였다 — [paper.pdf](https://github.com/almostimplemented/jazz-pianist-style/blob/main/docs/paper.pdf)
- **사용자 자체 T1-c 진단** [내부 문서]: 근음은 같고 품질만 다른 진행 P(Gm7–C7–Fmaj7)와 Q(Gm7b5–C7–Fm7)를 같은 seed로 넣는다. 두 진행이 갈리는 음(D↔Db, A·E↔Ab·Eb)만 세는 2×2 표로 판정한다(`docs/experiments/T1_PUBLIC_CHECKPOINTS.md`). 문헌의 counterfactual 교체를 생성 출력에 적용한 형태다.

#### 4) 스타일·전체 품질 평가
- **분류기 기반 자유생성 평가** — [직접 확인]: 슬라이딩 1024토큰 창(stride 128)의 top-1이 목표 피아니스트와 일치하는 비율을 본다. 합성 데이터로만 학습한 분류기의 실제 데이터 식별률도 쓴다. 테스트 CE는 조건 품질에 거의 둔감했다 — [paper.pdf](https://github.com/almostimplemented/jazz-pianist-style/blob/main/docs/paper.pdf)
- **표절·암기** — [요약만(이전 세션)]: BebopNet 평균 LCS 4.4음, 표절 AUC 0.713(유명 연주자 0.680–0.746). Edwards 2026은 memorization/diversity 분석 스크립트를 저장소에 둔다 [직접 확인 README].
- **청취 테스트 사례**

  | 연구 | 방식 | 참가자 | 결과 |
  |---|---|---|---|
  | MINGUS [직접] | 블라인드, 1–5 선호, 15개 × 20초 | 28명(애호가 8 / 전공생 9 / 전문 11) | MINGUS가 BebopNet보다 선호, 전문가일수록 원곡 식별 |
  | Frieler & Zaddach [요약만(이전 세션)] | Turing 유사 | 41명(전문가 29) | 탐지 64.4% / 41.7% |
  | Jazz Transformer [요약만(이전 세션)] | 블라인드 | 미확인 | 모델 2.7–2.9 vs 실제 3.0–3.4 |
  | AMT [요약만] | MTurk 쌍대, 20초 | 사전 자격 시험 통과자 | 반주 음악성이 인간과 비슷 |
  | ImprovNet [직접 README] | 장르 식별 | 미확인 | 79% |
  | GAPT [요약만] | 블라인드 | 전문가 12명 | 적응·통제감 p<0.05 |
  | BebopNet [직접 supp] | 실시간 CRDI 선호, 비블라인드 | 아마추어 4명 | 선호 모델 학습용 |

  - 출처: [MINGUS](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf), [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.87), [arXiv 2008.01307](https://arxiv.org/pdf/2008.01307), [AMT](https://arxiv.org/pdf/2306.08620), [ImprovNet](https://github.com/keshavbhandari/improvnet), [GAPT](https://arxiv.org/pdf/2511.17879), [BebopNet supp](https://github.com/shunithaviv/bebopnet/blob/master/BebopNet_SupplementaryMaterial.pdf)
  - Frieler는 렌디션(음색, 아티큘레이션, 마이크로타이밍)이 음 내용만큼 중요하다고 보고했다 [요약만(이전 세션)].

#### 5) "입력 복사(echo)" 분리
- 찾은 문헌 중 **조건 입력음(코드 보이싱 음)을 다시 치는 비율을 따로 측정**한 코드 → 멜로디 연구는 없었다.
- 가장 가까운 공개 장치는 세 가지다 — [BebopNet supp](https://github.com/shunithaviv/bebopnet/blob/master/BebopNet_SupplementaryMaterial.pdf); [realchords README](https://github.com/lukewys/realchords-pytorch)
  - (a) BebopNet의 "코드 일치 음 반복" 정성 보고
  - (b) ReaLchords/GAPT의 repetition·long-note 벌점과 Vendi 다양성
  - (c) GAPT의 판별기 보상
- [요약만(이전 세션)] 재즈 솔로 모델이 컴핑 대비 avoid-note 비율이나 반음 충돌 비율을 보고한 사례도 없었다.

### Inferences
- **코드톤 비율은 "높을수록 좋다"가 아니라 "원곡 분포에 가까울수록 좋다"로 읽어야 한다.** WJazzD 원곡도 코드톤은 약 49%에 불과하다(MINGUS). 재즈 솔로의 절반은 텐션·경과음·접근음이다. Tigran/비밥 목표에서는 Wang 2025식 tension correctness와 NCT resolution이 더 적합한 축이다.
- **사용자의 "Aria가 입력 코드음을 재타건" 증상을 측정하는 지표 설계안** (문헌에 없으므로 추론). 입력 보이싱의 PC 집합을 V, 해당 코드 기호의 PC 집합을 C, 생성 오른손 음을 Y라 한다.
  - (i) **echo율** = Y 중 (PC ∈ V이고 입력 onset ±Δ 이내 또는 같은 옥타브 ±12) 비율. 동시·근접 재타건만 따로 센다.
  - (ii) **비입력 코드톤율** = PC ∈ C \ V 비율. 예: 쉘 보이싱에서 빠진 5음, 9·13 텐션이 해당한다. 입력에 없는 화성음을 고른다는 것은 복사로는 설명되지 않는 반응이다.
  - (iii) **counterfactual 차등** = 같은 seed로 P/Q 품질 교체(T1-c) 또는 MuseBarControl식 마디 교체를 했을 때, 갈리는 음 중 자기 조건 쪽 비율. 단, V를 같은 옥타브로 고정하고 V와 겹치지 않는 판별음만 센다.
  - (iv) **섭동 적응 지연** = ReaLchords식 곡 중 전조나 삼전음 대치 후 note-in-chord가 회복되는 박 수.
  - (v) **기준선 대비** = random_chords(ImprovNet), 결정론적 chord-tone 그래프 라인(Keating & Casey), 입력 아르페지오 복사 봇과 비교. 아르페지오 복사 봇은 echo율 100%이면서 코드톤율도 높으므로, 이 봇과 구별되는지가 핵심이다.
  - (vi) **다양성·반복 벌점** = ReaLchords penalties와 Vendi.
- **조건 효과는 teacher-forced NLL이 아니라 자유생성에서 재야 한다**(Edwards 2026, MINGUS ablation). 사용자의 "chord adapter가 맞는/틀린 코드를 판별은 하지만 더 나은 생성기는 아님" 관찰은 이 문헌 패턴과 정확히 겹친다. 판별(teacher-forced) 능력과 자유생성 추종 능력은 별개로 측정해야 한다.

### Gaps
- 확인하지 못한 수치·정의 (논문 본문 차단 + 검색 예산 소진):
  - Yeh et al. 2021(JNMR) 원 정의의 CTnCTR/PCS/MCTD 수식
  - MuseBarControl 평가 표
  - MIDI-GPT 속성 제어 정확도
  - Polyffusion/FGG 코드 정확도 수치
- "re-striking conditioning notes"를 명시적으로 다룬 반주(AMT 등) 평가가 있는지 본문 수준에서 확인하지 못했다.
- 블라인드 전문가 청취로 **코드 추종 자체**(선호가 아니라 "코드를 따르는가")를 판정한 재즈 연구는 찾지 못했다.

## Q4. 이 사용자 목표(실시간 코드 → 피아노 솔로 MIDI, Tigran + 비밥 어휘, non-swing, M1 Max 학습·추론, 아티스트 adapter)에 쓸 만한 항목과 이미 시도한 것과의 차이

### Takeaway
이미 시도한 것과 **실질적으로 다른** 후보는 네 가지다.
- (1) **ReaLchords 코드베이스의 "chords → melody 온라인 모델 + 화성 보상 RL + GAIL(GAPT) 반-해킹"** 레시피. MIT, MLX 백엔드, 8층 d512.
- (2) **Edwards 2026식 gated cross-attention adapter를 코드 스트림에 적용**. 문헌상 미시도이며 저자들의 향후 과제다.
- (3) **counterfactual loss(MuseBarControl)로 조건 무시를 직접 벌하기**.
- (4) **MIDI-GPT·FIGARO·Polyffusion식 데이터 자기 라벨링 선례**를 PiJAMA 왼손 PC-set에 적용하되 echo 지표로 감시하기.

반면 재즈 전용 코드 조건 모델(BebopNet/MINGUS/CMT)은 모두 혼 단성이고 이미 시험했다.

### Cited Findings
- **이미 시도한 것과 겹치는 항목**: CMT·BebopNet·MINGUS·Aria·WJazzD 검색은 사용자가 시험했다(과제 설명).
  - 문헌상 이들은 단성·혼 데이터다 — [MINGUS PDF](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf); [bebopnet-code](https://github.com/shunithaviv/bebopnet-code)
  - CMT는 8마디 미래 코드를 양방향으로 본다 — [model.py](https://github.com/ckycky3/CMT-pytorch/blob/master/model.py)
- **ReaLchords 코드베이스** — [직접 확인]
  - 코드 → 멜로디 온라인 모델 학습 레시피, 보상모델(contrastive/discriminative/rhythm), GAIL RL, MLX 백엔드, ±6 전조 증강이 있다. MIT다.
  - 데이터는 팝·포크 4종이다 — [README](https://github.com/lukewys/realchords-pytorch); [melody config](https://github.com/lukewys/realchords-pytorch/blob/main/configs/gen_models/decoder_only_online_melody_4_datasets.yml)
- **Edwards 2026 adapter** — [직접 확인]
  - gated cross-attention(g 초기 0.1), 78M 추가, Aria-medium 659M.
  - **학습은 RTX 6000에서 약 18시간(전체 미세조정)**이다.
  - 평가 코드는 MPS/CPU에서 돈다.
  - 코드·리드시트 결합 조건이 향후 과제로 명시돼 있다.
  - PiJAMA-12에 Art Tatum 118곡이 있고 Brad Mehldau는 없다 — [paper.pdf](https://github.com/almostimplemented/jazz-pianist-style/blob/main/docs/paper.pdf); [pijama12.csv](https://github.com/almostimplemented/jazz-pianist-style/blob/main/data/pijama12.csv); [README](https://github.com/almostimplemented/jazz-pianist-style)
- **자기 라벨링 선례** — [직접 확인]
  - FIGARO: 학습 MIDI에서 규칙 코드 인식 → description — [chord_recognition.py](https://github.com/dvruette/figaro/blob/main/src/chord_recognition.py)
  - Polyffusion: `chord_extractor` — [repo](https://github.com/aik2mlj/polyffusion)
  - MIDI-GPT: 마디별 pitch class set 속성 — [README](https://github.com/Metacreation-Lab/MIDI-GPT)
- **조건 무시를 직접 벌하는 손실**: MuseBarControl counterfactual hinge [요약만] — [arXiv](https://arxiv.org/pdf/2407.04331)
- **지표 해킹 경고**:
  - BebopNet: 화성 지표 최적화 → 코드음 반복 [직접 확인] — [Supplementary](https://github.com/shunithaviv/bebopnet/blob/master/BebopNet_SupplementaryMaterial.pdf)
  - GAPT: 코히런스 보상 RL → 다양성 붕괴 [요약만] — [arXiv](https://arxiv.org/pdf/2511.17879)
- **결정론적 chord-tone 라인 생성**: Keating & Casey 그래프 엔진(MIT)은 학습 없이 코드 간 최적 전이음을 잇는다 [직접 확인] — [repo](https://github.com/mbkeating/chord-tone-generator)
- **연주자별 문법**: Impro-Visor 85개 grammar 관련 파일(Bill Evans·Bud Powell·Cedar Walton 포함, GPL v2) [직접 확인] — [repo](https://github.com/Impro-Visor/Impro-Visor)

### Inferences
- **(1) ReaLchords식 RL은 "프롬프트/어댑터 지도학습"과 학습 신호가 다르다.** 자유생성 출력에 보상을 준다. 그래서 "판별은 하나 생성은 안 좋아짐"이라는 갭을 직접 겨냥한다. 다만 BebopNet·GAPT가 보여주듯 note-in-chord 보상만 쓰면 재타건·반복으로 해킹된다.
  - 보상에는 반드시 Q3의 echo율과 비입력 코드톤율, 반복 벌점, 판별기(GAIL)를 함께 넣어야 한다.
  - 8층 d512 규모는 M1 Max MPS에서 단기 학습이 현실적인 크기로 보인다. 단 실측이 없으므로 추정이다.
- **(2) Edwards식 adapter를 코드 스트림에 쓰면 프롬프트 방식과 달리 "지속 채널"이 생긴다.** 이것이 사용자가 겪은 "코드 효과 약 1%" 문제의 구조적 대안이다.
  - 그러나 원 레시피는 전체 미세조정 18 GPU시간이라 M1 Max 열 제약에서는 비현실적이다.
  - backbone을 동결하고 adapter만 학습하는 축소안은 문헌상 검증되지 않았다.
- **(3) PiJAMA 왼손 PC-set 자기 라벨링은 copy shortcut 위험이 크다.** 같은 시간대 같은 곡의 음에서 조건을 뽑으면, 모델이 조건 PC를 그대로 재사용하는 것만으로 손실을 낮출 수 있다. 사용자가 본 Aria 재타건 증상과 같은 메커니즘이다.
  - 완화책 (추론):
    - 조건은 왼손에서만, 목표는 오른손에서만 뽑는다.
    - 조건을 박 시작 시점 이전 정보로 제한한다.
    - 조건 PC를 일부 dropout하거나 C \ V 확장(코드 기호 추정 후 텐션 추가)을 쓴다.
    - MuseBarControl식 counterfactual loss를 쓴다.
  - 평가에는 echo율과 비입력 코드톤율을 필수로 넣는다.
- **(4) "Tigran + 비밥, non-swing" 스타일을 직접 지원하는 모델이나 데이터는 없다.** 현실적인 경로는 straight 격자 모델(CMT류 16분 격자)이나 표현 단계에서 변박을 허용하는 토큰 설계다. 스타일은 분류기 기반 평가(Edwards식)로 측정하는 조합이 문헌과 맞는다.

### Gaps
- ReaLchords 멜로디 모델과 GAPT의 M1 Max(MLX) 실측 지연, 학습 시간은 미확인이다. README에 수치가 없다.
- adapter-only(동결 backbone) cross-attention 코드 조건의 효과를 보고한 심볼릭 음악 논문은 찾지 못했다.
- 자기 라벨링(코드를 같은 MIDI에서 추출)으로 학습한 모델의 echo/copy 정도를 정량 보고한 연구는 찾지 못했다.
