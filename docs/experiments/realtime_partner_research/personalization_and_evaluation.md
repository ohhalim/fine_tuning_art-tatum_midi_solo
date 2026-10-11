# 소량 데이터 연주자 스타일 개인화 + 스타일/코드추종 평가 (symbolic piano, 2017–2026)

> 검증 방법 메모 (작성일 2026-10-11): 이 세션의 egress 정책상 arxiv.org / export.arxiv.org / huggingface.co / transactions.ismir.net / openreview.net / semanticscholar / alphaxiv / webspace.eecs.qmul.ac.uk / zenodo / *.github.io 는 모두 CONNECT 403(조직 정책 차단), WebFetch는 DNS 실패였다. **raw.githubusercontent.com 은 열려 있어서 공식 코드 저장소의 README·소스·메타데이터 CSV를 직접 읽었다.** 따라서 태그 의미는 다음과 같다.
> - **[직접 확인]** = 공식 GitHub 저장소의 README/코드/CSV 원문을 이 세션에서 직접 읽음 (논문 PDF 본문은 아님).
> - **[요약만]** = 검색엔진 요약/스니펫(논문 초록·HTML 일부, 2차 리뷰 사이트 포함)만 확인. 표 수치는 원문 대조 필요.
> - **미확인** = 수치/사실을 찾지 못함.

## (a) 소량 데이터·저연산으로 "연주자 고유" 스타일을 어댑터에 담는 방법

### Takeaway
가장 직접적인 선행연구는 Edwards·Maezawa·Dixon (ISMIR 2026)의 pianist-conditioned Aria로, **같은 데이터로 무조건부 fine-tune만 해도 perplexity가 11.41→6.96으로 크게 떨어지고, 연주자 조건부를 추가해도 6.82로 거의 안 변하지만 분류기 일치율(agreement)은 37%→70%로 두 배가 된다** — 즉 사용자가 관찰한 "CE 하락의 대부분은 generic(PiJAMA 전사 포맷/도메인) 적응"이라는 현상이 이 논문에서도 정량적으로 재현되며, 연주자 특이성은 CE가 아니라 분류기 기반·대조(counterfactual) 지표로 봐야 한다. 소량 데이터 개인화에서는 "공유 도메인 적응 + 아주 작은 per-artist 파라미터(임베딩/상태/residual LoRA)" 구조가 근거가 가장 많다.

### Cited Findings
**Edwards, Maezawa, Dixon — "Learning Jazz Pianist Style with Cross-Attention Conditioning", ISMIR 2026, arXiv 2610.02918**
- [직접 확인] 구조: 사전학습 Aria-medium(16층)의 **마지막 8개 층(8–15)** self-attention 뒤에 gated cross-attention 삽입, query=음악 시퀀스, key/value=학습되는 **연주자당 4-벡터 컨텍스트**, gate 초기값 0.1, embedding dropout 0.3, CA dropout 0.1. 코드 `CrossAttentionLayer`는 `return self.gate * output`(스칼라 학습 gate). — [GitHub README](https://github.com/almostimplemented/jazz-pianist-style), [cross_attention_model.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/llama_pijama/models/cross_attention_model.py), [train_cross_attention.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/scripts/train_cross_attention.py)
- [직접 확인] **어댑터만 학습하는 것이 아님**: 학습 스크립트가 `params = list(model.parameters()) + artist_embeddings` 로 Aria 전체 + CA + 임베딩을 함께 AdamW로 학습(full fine-tune). 15 epoch, effective batch 32, peak LR 5e-6, 0.5 epoch warmup + cosine, weight decay 0.02, fp16, seq len 4096. "classifier-free" 용 conditioning dropout 옵션은 있으나 "논문 모델은 이것 없이 학습". `--no-cross-attention`이 같은 데이터의 무조건부 fine-tune baseline. — [train_cross_attention.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/scripts/train_cross_attention.py)
- [직접 확인] 학습 비용 표기가 **서로 다름**: README는 "about 18 hours on one 24 GB GPU", 학습 스크립트 docstring은 "A full paper-scale run needs a >=40GB GPU for ~18h". 평가(분류기·perplexity)는 Apple-silicon 노트북에서 가능(분류기 평가 ~3분, perplexity 모델당 ~20분), agreement 평가는 "175 continuations × 4,096 tokens"라 "GPU hours, slow on a laptop". — [README](https://github.com/almostimplemented/jazz-pianist-style)
- [직접 확인] 데이터: PiJAMA-12 = 12명 779개 solo 연주(앙상블 19곡 제외), **곡(song) 단위 split 623/76/80곡**; 4096-token 청크 기준 train/val/test 1,354/166/177(논문 원 split은 1,383/175/177, test 동일). 12명 곡수: Dick Hyman 187, **Art Tatum 118**, Abdullah Ibrahim 84, Dave McKenna 76, Tete Montoliu 52, Erroll Garner 45, Hank Jones 44, Andy LaVerne 42, Oscar Peterson 41, George Cables 31, Cedar Walton 31, Horace Parlan 28. **Brad Mehldau·Tigran Hamasyan은 PiJAMA-12에 없음.** — [data/README.md](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/data/README.md), [pijama12.csv](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/data/pijama12.csv)
- [직접 확인] 결과(README "Reproducing the paper" 표): Table 2 perplexity **6.82(조건부 CA) / 6.96(무조건부 fine-tune) / 11.41(사전학습 Aria)**; agreement(프롬프트 256 토큰, 1024-token 창을 stride 128로 슬라이딩하며 분류기가 프롬프트 연주자로 판정한 비율) **70% / 37% / 25%**, CI 예: 70 [64, 75]; 연주자별 agreement **96%(Hank Jones) … 29%(Cedar Walton)**. 12-class 우연 수준은 8.3%(계산). — [README](https://github.com/almostimplemented/jazz-pianist-style), [agreement_eval.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/scripts/agreement_eval.py)
- [직접 확인] agreement 평가에는 **conditioned / ablation(컨텍스트 0) / mismatch(다른 연주자로 조건부, 프롬프트 연주자·조건 연주자 양쪽 agreement 측정)** 모드가 있다 — 스타일판 counterfactual swap 테스트. — [agreement_eval.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/scripts/agreement_eval.py)
- [직접 확인] 가중치·라이선스: HF `drewbie/jazz-pianist-style-generator`(Aria-medium + gated CA + 12 연주자 임베딩), `drewbie/jazz-pianist-style-classifier`(PiJAMA-12 분류기), `drewbie/jazz-pianist-style-synthetic-classifier`(생성물로만 학습한 분류기). **코드·체크포인트 모두 Apache-2.0**(Aria와 동일), 동봉된 Cheston ResNet-50 baseline은 MIT. PiJAMA-30 분류기와 DPI-20 데이터는 미공개(같은 스크립트로 재학습 가능). — [README](https://github.com/almostimplemented/jazz-pianist-style)
- [요약만] 논문 요지: "standard perplexity barely distinguishes conditioned from unconditioned models"; HF 모델 카드 요약에 "conditioned continuations are attributed to the intended pianist about 33 points more often than unconditioned baselines", 긴 생성에서도 조건 효과 유지. — [arXiv abs 2610.02918](https://arxiv.org/abs/2610.02918), [HF generator card](https://huggingface.co/drewbie/jazz-pianist-style-generator), [프로젝트 페이지](https://almostimplemented.github.io/jazz-pianist-style/)

**PiJAMA 데이터 메타(사용자 데이터와 직접 관련)**
- [직접 확인] `pijama_30.csv`(30명, 1,558행): **Brad Mehldau 72곡(live 54 / studio 18, 합계 약 8.87시간), Art Tatum 122곡(전부 studio, 약 8.03시간)**, Alan Pasqua 36(studio). Mehldau의 song_split = train 57 / val 8 / test 7. CSV에 `recording_condition`, `song_split`, `album_split_0/1/2` 열이 있어 **녹음조건 층화·앨범 단위 split**이 가능. — [pijama_30.csv](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/data/pijama_30.csv) (시간 합계는 duration_sec 합으로 직접 계산)
- [요약만] PiJAMA: Edwards, Dixon, Benetos, TISMIR 6(1):89–102 (2023), 200시간+ solo jazz piano, 2,777 연주, 120명, 244 앨범, 자동 전사 MIDI; 벤치마크로 performer identification과 generative modeling 제시. — [TISMIR 10.5334/tismir.162](https://transactions.ismir.net/articles/10.5334/tismir.162), [Zenodo 8354955](https://zenodo.org/records/8354955)

**MIDI-RWKV — state tuning vs LoRA (few-shot 개인화)**
- [직접 확인] Zhou-Zheng & Pasquier, arXiv 2506.13001 (2025; PDF 제목은 "Adaptable Symbolic Music Infilling with MIDI-RWKV", venue 미확인). 저장소에 base 가중치, state tuning·LoRA 스크립트(RWKV-PEFT 기반), 평가·통계검정 코드 공개; LoRA는 rank=alpha=4 와 32 두 설정; 미세조정 데이터 POP909; 미세조정 체크포인트는 미공개. — [GitHub christianazinn/MIDI-RWKV](https://github.com/christianazinn/MIDI-RWKV)
- [요약만] state tuning(초기 hidden state만 최적화, 294k 파라미터) vs LoRA r=4(331k) / r=32(2.7M), 학습시간 동일; 저샘플 regime에서 state tuning이 CP·GS·PCHE 객관지표에서 대체로 우세, F1은 비슷; 28명 청취 테스트에서 state-tuned가 base·LoRA r4보다 유의하게 선호, base와 LoRA r4는 유의차 없음. 후속 논문(S0 Tuning, arXiv 2604.01168)은 "state tuning은 저랭크에서 경쟁적이나 seed 간 분산이 더 큼"으로 요약. — [arXiv 2506.13001](https://arxiv.org/abs/2506.13001), [themoonlight 리뷰](https://www.themoonlight.io/en/review/personalizable-long-context-symbolic-music-infilling-with-midi-rwkv), [S0 Tuning](https://arxiv.org/pdf/2604.01168)

**Moonbeam — PiJAMA30 실험**
- [직접 확인] Guo & Dixon, arXiv 2505.15559 (2025, venue 미확인), 81.6K MIDI·18B 토큰 사전학습, 하류 미세조정은 PEFT LoRA(`--use_peft True --peft_method lora`, lr 3e-4, context 2048) 예시. — [GitHub Moonbeam](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model)
- [요약만] PiJAMA30 연주자 분류 정확도 Moonbeam(M) 67.9%, CLaMP2 0.440; 소형 모델도 PiJAMA30에서 대형 모델들보다 우수. 독립 리뷰(Pith)는 **PiJAMA30 test 곡이 사전학습 코퍼스에 포함됐는지(overlap) 미해결**, 모든 표가 단일 run·오차막대 없음을 지적. — [alphaxiv 2505.15559](https://www.alphaxiv.org/abs/2505.15559), [Pith 리뷰](https://pith.science/paper/2505.15559)

**Aria (기반 모델)**
- [직접 확인] Bradshaw et al., "Scaling Self-Supervised Representation Learning for Symbolic Piano Performance", ISMIR 2025, arXiv 2506.23869; LLaMA 3.2(1B) 아키텍처 기반, ~60k시간 solo piano 전사로 학습; aria-medium-base / -gen / -embedding(SimCSE식 대조학습 임베딩) 공개; **PyTorch(CUDA)와 MLX(Apple Silicon) 추론 구현 제공**; README가 "instruction tuning/RLHF 안 됨 → 프롬프트 품질에 민감", 유명 클래식 곡 기억·재현 가능성 경고. — [GitHub EleutherAI/aria](https://github.com/EleutherAI/aria)
- [요약만] Aria-MIDI(arXiv 2504.15071): 100만+ MIDI, 약 10만 시간 전사. — [arXiv 2504.15071](https://arxiv.org/abs/2504.15071)

**generic vs 연주자 특이 분리 기법 (일반 ML)**
- [요약만] GenKnowSub (ACL 2025 short, arXiv 2505.10939): task별 LoRA 라이브러리 + 일반 도메인 LoRA 하나를 학습한 뒤 **task LoRA에서 일반 LoRA를 빼서 residual 모듈**을 만들고 Arrow 라우팅으로 조합; Phi-3/Phi-2에서 단일·교차언어 성능 향상. — [arXiv 2505.10939](https://arxiv.org/html/2505.10939v1), [ACL Anthology](https://aclanthology.org/2025.acl-short.54)
- [요약만] Task arithmetic(Ilharco et al. 2023)의 "fine-tuned − base = task vector" 개념이 LoRA 병합/부정(negation)으로 확장됨; PaLM 2 연구에서 언어 LoRA를 빼면 zero-shot 교차언어 요약이 개선(QA는 효과 작음). LoRA 부정은 계수를 두 인수에 나눠야 함(c²로 스케일되는 문제) — 도구 문서 수준 근거. — [Task-Aware LoRA composition 2602.21222](https://arxiv.org/html/2602.21222v1), [ACL MRL 2024 language/task arithmetic](https://aclanthology.org/2024.mrl-1.7.pdf), [Soup 문서](https://trysoup.dev/docs/adapter-arithmetic)
- [요약만] "LoRA Learns Less and Forgets Less" (Biderman et al., TMLR 2024, arXiv 2405.09673): LoRA는 full FT보다 target 성능은 낮지만 도메인 밖 성능 보존이 좋고, weight decay·dropout보다 망각 억제 효과가 크며 생성 다양성 유지; full FT의 섭동 rank가 일반 LoRA보다 10–100× 큼. replay 관련 수치는 미확인. — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673v2), [TMLR](https://mlanthology.org/tmlr/2024/biderman2024tmlr-lora)

**기타 스타일 조건부/모방 (2019–2025)**
- [직접 확인] BebopNet (Haviv Hakimi, Bhonker, El-Yaniv, ISMIR 2020): Bebop 거장 데이터로 next-note 모델 학습 → 사용자 청취 라벨 수집(online tagger) → **사용자 선호 지표(regression) 학습** → 그 지표로 beam search(생성기 재학습 없이 개인화); 별도로 "harmony" score_model(현재 코드 스케일과의 일관성 선호) 옵션. — [GitHub bebopnet-code](https://github.com/shunithaviv/bebopnet-code)
- [요약만] 표현적 연주 렌더링 쪽 스타일 조건: ScorePerformer(ISMIR 2023, global/bar/beat/onset 수준 스타일 인코딩), VirtuosoNet(ISMIR 2019, CVAE), "Disentangling Score Content and Performance Style"(arXiv 2509.23878, 연주자별로 군집되는 style embedding t-SNE), Pianist Transformer(arXiv 2512.02652). 특정 연주자 정체성을 재현했는지 직접 검증한 결과는 미확인. — [ScorePerformer ISMIR 2023](https://ismir2023program.ismir.net/poster_183.html), [2509.23878](https://arxiv.org/pdf/2509.23878), [2512.02652](https://arxiv.org/abs/2512.02652)
- [요약만] Composer style-specific VQ discrete diffusion (arXiv 2310.14044): 작곡가 스타일 조건 생성, 청취평가(humanness/harmony/rhythm/richness/선호). — [arXiv 2310.14044](https://arxiv.org/pdf/2310.14044)

### Inferences
- 사용자 관측(Mehldau LoRA: Mehldau held-out CE −0.30, 타 연주자 −0.28)은 Edwards의 "pretrained 11.41 → 무조건부 FT 6.96 → 조건부 6.82"와 같은 구조다. **연주자 특이 효과 = (target 연주자 ΔCE) − (비-target 연주자 ΔCE) ≈ 0.02 nats** 수준이라 사실상 미검출로 봐야 한다. Edwards처럼 "같은 다연주자 풀에서 학습한 무조건부(=도메인) 모델"을 baseline으로 두고 그 위의 증분만 연주자 효과로 계상하는 3-way 설계(사전학습 / 도메인-FT / 도메인-FT+artist)가 필요.
- "베이스 하나 + 교체형 per-artist 어댑터" 목표에는 Edwards 방식이 가장 직접적이다: 공유 파라미터(CA 층 + 도메인 적응)는 1회 학습, **연주자별 파라미터는 4×d_model 벡터 하나**. 새 연주자(Tigran 53분)는 공유 부분을 동결하고 새 임베딩만 학습하는 textual-inversion식 확장이 저비용 후보다 — 단, 논문은 이 "새 연주자 추가" 시나리오를 실험하지 않았다(추론).
- 대안으로 2단계: (1) PiJAMA 전체(타깃 제외 포함 다연주자)로 "도메인 LoRA" 학습 → 병합 또는 동결, (2) 그 위에 연주자 LoRA 학습, 또는 GenKnowSub식 `artist_LoRA − general_LoRA` residual. 학습 시 replay(도메인 데이터 일부 혼합)를 섞으면 연주자 LoRA가 다시 generic 성분을 흡수하는 것을 억제할 수 있다(추론; 음악 도메인 실증 미확인).
- MIDI-RWKV의 state tuning 결과는 transformer에서 **prefix/prompt tuning(소수 벡터만 학습)** 이 저샘플 개인화에서 LoRA보다 나을 수 있다는 간접 근거다(아키텍처가 달라 직접 이전은 추론).
- 규모 비교(계산): Edwards 학습량 ≈ 1,383 seq × 4,096 tok × 15 epoch ≈ 8.5천만 token-pass, 최적화 step ≈ 650(eff. batch 32). 사용자 128-step LoRA는 이보다 훨씬 작으므로 "연주자 효과 없음"이 방법의 한계인지 학습량 부족인지 아직 구분 불가.
- 데이터 혼동 위험: PiJAMA의 Mehldau는 75%가 live, Tatum은 100% studio → 연주자 분류·CE 차이가 녹음조건/전사 아티팩트를 탈 수 있다. `recording_condition`, `album_split_*` 열로 층화·앨범 분리 평가를 해야 한다. 또한 Aria 사전학습(Aria-MIDI, YouTube 전사)에 PiJAMA 연주가 겹칠 가능성이 있다(Moonbeam에 대한 Pith의 overlap 지적과 같은 종류의 위험; Aria 측 overlap 여부는 미확인).
- 사용자 Tatum 어댑터의 경우, 공개된 Edwards generator가 이미 **Art Tatum 임베딩을 포함**(Apache-2.0)하므로 baseline/상한 비교 대상으로 바로 쓸 수 있다. Mehldau는 PiJAMA-30 분류기를 공개 스크립트로 재학습해야 측정 가능.

### Gaps
- Edwards 논문 PDF 본문(Table 1–3 전체, LoRA/prefix 등 다른 PEFT와의 비교 유무, 청취 평가 유무, 새 연주자 추가 실험 유무)은 차단으로 직접 열지 못함 — README 표 수치만 [직접 확인].
- PiJAMA 원논문(TISMIR 2023)의 performer identification 정확도 수치: 미확인.
- MIDI-RWKV 표의 정확한 수치(CP/GS/PCHE 값, 유의확률): 미확인(요약만).
- 음악 도메인에서 "LoRA 빼기/difference adapter/task arithmetic"으로 generic과 artist 성분을 분리한 실증 논문은 찾지 못함.
- "Learning to Generate Music in the Style of a Composer"라는 제목의 논문은 찾지 못함(유사 주제는 2310.14044 등).
- Music Transformer/Aria에 대한 체계적 LoRA rank·step 연구(심볼릭 피아노)는 찾지 못함.

## (b) 소예산 symbolic music 선호학습/RL post-training

### Takeaway
작은 예산에서 실증된 경로는 (1) **사람 대신 임베딩 유사도(CLaMP 2)로 chosen/rejected를 자동 구성하는 DPO(NotaGen CLaMP-DPO)**, (2) **소수 사용자 라벨로 선호 점수기를 학습해 재순위/beam search만 하는 방식(BebopNet)** 이다. 화성·미적 보상으로 RL을 하면 거의 항상 반복·다양성 붕괴로 reward hacking이 나타났고(GAPT, SMART, ReaLchords 후속), 판별자 보상·반복 페널티·KL이 완화책으로 보고됐다.

### Cited Findings
- [직접 확인] NotaGen (Wang et al., IJCAI 2025, arXiv 2502.18008) CLaMP-DPO 구현: 생성 → CLaMP 2로 각 생성물과 fine-tune 데이터(ground truth) 특징 비교한 "CLaMP 2 score"로 정렬 → `data_select_portion=0.1`(상·하위 10%)로 chosen/rejected 구성; chosen에서 제외 규칙 3개(마디 길이 정렬 오류, **ground truth 표절 의심 ld_sim>0.95**, 같은 악기 보표 비그룹). DPO 설정 `BETA=0.1`, DPOP `LAMBDA=10`, `LR=1e-6`, `OPTIMIZATION_STEPS=10000`, 반복(iteration)마다 새 모델로 재생성; 논문 모델은 NotaGen-large(516M)에 3회 반복. 코드가 정책 모델과 **참조 모델(model_ref)을 둘 다 메모리에 로드**. NotaGen-large 학습/추론은 "at least 24GB GPU memory" 필요, small(110M)/medium(244M) 대안 제시. 가중치는 HF 공개(라이선스는 README 발췌에서 미확인). — [GitHub NotaGen README](https://github.com/ElectricAlexis/NotaGen), [RL/config.py](https://raw.githubusercontent.com/ElectricAlexis/NotaGen/main/RL/config.py), [RL/train.py](https://raw.githubusercontent.com/ElectricAlexis/NotaGen/main/RL/train.py)
- [요약만] CLaMP-DPO 결과: MuPT의 CLaMP 2 score 0.515 → 3회 후 0.674; 적용한 모든 모델에서 청취 음악성 향상 보고; chosen 출력의 likelihood가 떨어져 품질 저하가 생기는 현상 때문에 DPO-Positive(DPOP) 채택. — [arXiv 2502.18008](https://arxiv.org/html/2502.18008v4), [IJCAI 2025](https://www.ijcai.org/proceedings/2025/1134)
- [직접 확인] BebopNet (ISMIR 2020): 사용자 청취 라벨 → 사용자 선호 지표 회귀모델 → 그 지표 기반 beam search(note/measure 단위 depth·width)로 개인화 생성; 생성기 자체는 재학습하지 않음. — [GitHub bebopnet-code](https://github.com/shunithaviv/bebopnet-code)
- [요약만] MusicRL (Cideron et al., ICML 2024, arXiv 2402.04229): MusicLM(오디오)을 RL로 미세조정; 텍스트 충실도·오디오 품질 보상(MusicRL-R), 실사용자 **30만 쌍** 선호로 RLHF(MusicRL-U), 결합(MusicRL-RU)이 최고; 텍스트 충실도·품질만으로는 선호의 일부만 설명. 대규모 데이터라 소예산 재현은 비현실적. — [PMLR](https://proceedings.mlr.press/v235/cideron24a.html), [arXiv 2402.04229](https://arxiv.org/abs/2402.04229v1)
- [요약만] ReaLchords (Wu et al., ICML 2024, PMLR 235:53328–53345; arXiv 2506.14723): 온라인 MLE 사전학습 → RL 미세조정; 보상 = 데이터의 양/음 멜로디–코드 쌍으로 학습한 **대조(contrastive) 보상모델(조화)** + **판별(discriminative) 보상모델(시간적 동기화)**, 미래를 보는 teacher로부터 distillation 항; 보상이 사람 평가와 강하게 상관; 보상모델이 금방 해소되는 짧은 실수에 과도하게 가혹 → multi-scale 버전(ReaLchords-M). — [PMLR](https://proceedings.mlr.press/v235/wu24c.html), [arXiv 2506.14723](https://arxiv.org/html/2506.14723)
- [요약만] GAPT — "Generative Adversarial Post-Training Mitigates Reward Hacking in Live Human-AI Music Interaction" (ICLR 2026, arXiv 2511.17879): 실시간 멜로디→코드 반주에서 RL 정책이 **coherence 보상을 단순·고득점 코드 반복으로 착취**; ReaLchords와 적대보상 제거 ablation은 화성은 강하지만 다양성 감소. 해결: 정책 롤아웃 vs 실데이터 판별자의 realism 점수를 보상에 추가(판별자는 최근 보상 이동평균이 임계 초과 시에만 업데이트), **같은 코드 4회 초과 연속 시 규칙 기반 반복 페널티**; 고정 멜로디·공동적응 시뮬레이션·뮤지션 라이브 세션에서 다양성 붕괴 방지 + 화성 유지, 체감 속도·agency 향상. 구체 수치는 미확인. — [arXiv 2511.17879](https://arxiv.org/pdf/2511.17879), [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/a6a21421022d1da16eadbba533980530-Paper-Conference.pdf)
- [요약만] SMART (Jonason, Casini, Sturm; arXiv 2504.16839, AIMC 2025): 피아노 MIDI 생성기를 **GRPO**로, 렌더링 오디오의 Meta Audiobox Aesthetics 점수를 보상으로 미세조정; 저수준 특징 여럿이 변하고 14명 예비 청취에서 평균 평점 상승; **과최적화 시 출력 다양성 급감**. — [arXiv 2504.16839](https://arxiv.org/abs/2504.16839)
- [요약만] ReaLJam (arXiv 2502.21267): RL-tuned transformer로 실시간 human-AI 잼(venue 미확인). — [arXiv 2502.21267](https://arxiv.org/pdf/2502.21267)
- [요약만] RL-Duet (arXiv 2002.03082): 온라인 반주 생성에 deep RL 적용(2020). — [arXiv 2002.03082](https://arxiv.org/pdf/2002.03082)
- [요약만] 검증가능 보상(verifiable rewards) 기반 다중제약 symbolic 생성(arXiv 2609.23665, "ICLR 2027" 헤더 = 심사중 추정): 학습된 보상모델 없이 GRPO + 규칙 검증기. — [arXiv 2609.23665](https://arxiv.org/html/2609.23665)
- [요약만] "How Far Can Pretrained LLMs Go in Symbolic Music? Controlled Comparisons of Supervised and Preference-based Adaptation" (Kumar, Karystinaios, Widmer, Schedl; NLP4MusA 2026, arXiv 2601.22764): SFT vs 선호 기반 최적화를 통제 비교, 도메인 향상 vs 일반능력 유지 trade-off 질문. DPO 결과 수치는 미확인. — [ACL Anthology](https://aclanthology.org/2026.nlp4musa-1.5/), [arXiv 2601.22764](https://arxiv.org/html/2601.22764)
- [요약만] 서베이 "Aligning Generative Music AI with Human Preferences: Methods and Challenges" (arXiv 2511.15038). — [arXiv 2511.15038](https://arxiv.org/pdf/2511.15038)

### Inferences
- M1 Max 32GB에서 Aria-medium(약 6.6억) LoRA-DPO는 참조모델을 "어댑터 비활성 base"로 두면 가중치 사본이 필요 없고, chosen/rejected의 참조 log-prob을 **오프라인 사전계산**하면 학습 중 forward가 정책 1회로 줄어 메모리·발열 부담이 작다(일반 DPO 구현 지식 기반 추론; NotaGen 코드는 모델 2개를 올리는 방식이라 그대로는 비효율).
- 사용자가 만들 수 있는 소량 A/B 판단은 (i) BebopNet식 **선호 점수기/재순위(best-of-N)** 에 먼저 쓰고, (ii) 쌍이 수백 개 쌓이면 DPO/DPOP로 가는 순서가 위험이 낮다. 사람 라벨 대신 CLaMP-DPO처럼 "타깃 연주자 실연주 임베딩과의 유사도"나 Edwards 분류기 확률로 자동 쌍을 만들 수 있으나, 이 경우 보상=평가지표가 되므로 **평가용 분류기와 보상용 분류기를 분리**해야 한다(추론).
- 코드추종 보상(chord-tone 비율 등)으로 RL하면 GAPT가 보인 "고득점 코드 반복"의 솔로 버전, 즉 **입력 코드음 재타건/에코·같은 릭 반복**으로 붕괴할 가능성이 높다. 반복 페널티, 실데이터 판별자/분류기 realism, KL(또는 DPOP 항)을 함께 둬야 한다(추론, GAPT·SMART 근거).
- NotaGen의 "ld_sim>0.95 표절 제외"는 소량 연주자 데이터(Tatum 122곡, Tigran 13곡)에서 특히 중요 — chosen이 학습곡 복제일 때 DPO가 암기를 강화한다(추론).

### Gaps
- 심볼릭 피아노(연주 MIDI)에서 소규모 사람 A/B 쌍(수십~수백)으로 DPO를 한 실증과 필요한 쌍 수: 찾지 못함.
- GAPT·SMART·ReaLchords의 정량 수치(다양성 지표 값, 보상 곡선): 원문 차단으로 미확인.
- "RLHF for melody" 전용 소규모 연구, Apple Silicon/MPS에서 DPO 실측 사례: 찾지 못함.

## (c) 객관적 스타일 평가(연주자 분류기·임베딩·분포거리)와 신뢰도

### Takeaway
"pianist X처럼 들리는가"의 현재 최선의 자동 계측은 **실연주로 학습한 연주자 분류기의 생성물 판정률(agreement)을 조건 일치/불일치/무조건 baseline과 대조**하는 것이며(Edwards 2026), 분포 거리(FMD, MGEval 히스토그램)는 "도메인 일반성"을 잡는 보조 지표다. 단, 분류기는 녹음조건·전사 아티팩트·데이터 불균형("Hyman sink")에 취약하고, 객관 지표와 사람 청취의 상관은 symbolic 영역에서 체계적으로 검증된 바가 거의 없다.

### Cited Findings
- [직접 확인] Edwards 2026 분류기(Aria 표현 기반): PiJAMA-12 test **95.8%(1024-token 청크) / 98.8%(트랙, 청크 다수결·동률은 평균 logit)**; **생성물로만 학습한 분류기가 실연주 test에서 87.1% / 95.0%**; 생성물로 처음부터 학습한 ResNet-50은 70.2%(30초 클립) / 91.3%(트랙). — [README](https://github.com/almostimplemented/jazz-pianist-style)
- [요약만] 같은 분류기가 외부 DPI(Deep Pianist Identification, Cheston 데이터 20명) 벤치마크에서 트랙 96.9%, 공개 ResNet-50 baseline 94.4%보다 높음. — [검색요약: arXiv 2610.02918 / QMUL PDF](https://arxiv.org/abs/2610.02918)
- [직접 확인] Edwards의 신뢰도 보조 분석: 암기 검사(8초 창·50ms bin piano-roll Jaccard를 **같은 연주자의 학습 트랙별**로 계산해 top-1/top-2 비율 확인, held-out 실연주를 null로 보정) — 생성물 0.20 vs 실연주 0.32; 다양성(pitch/interval 4-gram Jaccard) — 같은 연주자 내 유사도가 연주자 간보다 1.5–3.5배; characteristic regions(분류기로 연주 내 서명 구간 탐지) median peak z 0.90, max 1.99. README 표에 "Hyman sink"(Table 2 CI 항목) 언급. — [README](https://github.com/almostimplemented/jazz-pianist-style), [jaccard_top2.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/scripts/analysis/jaccard_top2.py), [measure_diversity.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/scripts/analysis/measure_diversity.py), [memorization.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/llama_pijama/analysis/memorization.py)
- [직접 확인] `measure_diversity.py` 해석 규칙: within≈between → 연주자 스타일 미학습; within>between이고 within≪1 → 건강; within≈1 → 붕괴(near-duplicate). — [measure_diversity.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/scripts/analysis/measure_diversity.py)
- [요약만] Cheston, Bance, Harrison, "Machine learning of artistic fingerprints in jazz", **Nature Machine Intelligence 2026** (preprint arXiv 2504.05009, 구 제목 "Deconstructing Jazz Piano Style Using Machine Learning"): JTD+PiJAMA에서 20명, 84시간, 1,629 연주; 최고 모델 **94%(일부 보도 94.4%)**; 해석가능 bag-of-features 로지스틱 회귀 top-1 0.767 / top-5 0.939(다른 출처는 91.3%로 상충); multi-input 모델에서 단일 개념만 쓸 때 **화성 74.4%, 리듬 61.9%, 멜로디 57.5%, 다이내믹 26.3%**. — [Nature MI](https://www.nature.com/articles/s42256-026-01279-9), [arXiv 2504.05009](https://arxiv.org/html/2504.05009v1), [MusicRadar 보도](https://www.musicradar.com/music-tech/ai-models-can-identify-iconic-jazz-pianists-based-on-their-recordings-while-also-revealing-their-musical-fingerprints)
- [직접 확인] Cheston 코드(MIT): 30초 클립, CRNN/ResNet50 baseline, "disentangle" multi-input(ResNet18, 개념 마스킹) 체크포인트와 데이터는 Zenodo 14774191; 추론 4GB+ GPU. — [GitHub deep-pianist-identification](https://github.com/huwcheston/deep-pianist-identification)
- [요약만] Moonbeam PiJAMA30 67.9% vs CLaMP2 44.0% (사전학습 overlap 의문 제기됨) — (a) 참조. — [Pith 리뷰](https://pith.science/paper/2505.15559)
- [직접 확인] Frechet Music Distance (Retkowski et al., arXiv 2412.07948; venue 미확인): `pip install frechet-music-distance`, MIDI/ABC 지원, **CLaMP 2 / CLaMP 1 임베딩**, 공분산 추정기 MLE·Ledoit-Wolf·Shrinkage·OAS·Bootstrap, 표본수 보정 FMD-Inf, **곡별 FMD로 이상치 탐지**, MIT, Linux/macOS 테스트. — [GitHub frechet-music-distance](https://github.com/jryban/frechet-music-distance)
- [직접 확인] CLaMP 3 (arXiv 2502.10362, venue 미확인): text/sheet/audio/MIDI/image 공유 임베딩, MIDI는 MTF(512 메시지) 입력, 글로벌(768-d 평균) 또는 시간축 임베딩, 선형분류·검색 스크립트 제공. — [GitHub clamp3](https://github.com/sanderwood/clamp3)
- [직접 확인] MGEval (Yang & Lerch, "On the evaluation of generative models in music", Neural Computing and Applications, doi 10.1007/s00521-018-3849-7): pitch count, **pitch-class histogram**, PC transition matrix, pitch range, 평균 pitch interval, note count, **평균 IOI**, note-length histogram/transition; 절대지표 + 두 집합 비교(상대지표). — [GitHub mgeval](https://github.com/RichardYang40148/mgeval)
- [요약만] 신뢰도: 2025 서베이(arXiv 2509.00051)는 객관 지표와 인간 지각의 **상관이 약함**, 표준화 부재를 주요 한계로 정리. 대규모 인간선호 벤치마크(6,000곡, 12모델, 15,600 쌍비교, 2,500+명)는 오디오 대상이며 지표별 사람 일치도가 다름을 보고(venue 미확인). symbolic 전용 지표–청취 상관 연구는 찾지 못함. — [arXiv 2509.00051](https://arxiv.org/pdf/2509.00051), [Benchmarking … Human Preference Studies](https://openreview.net/pdf?id=105yqGIpVW)
- [요약만] 암기 지표: PhraseVAE/PhraseLDM(arXiv 2512.11348)의 top-2 distance ratio(T2R), melody memorization rate; Edwards 코드는 "top-2 distance ratio of arXiv 2511.07268"과 Yin et al. 2022 originality report를 근거로 인용. — [arXiv 2512.11348](https://arxiv.org/pdf/2512.11348), [jaccard_top2.py docstring](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/scripts/analysis/jaccard_top2.py)

### Inferences
- 사용자 평가 프로토콜 제안(추론): (1) 분류기 = PiJAMA-30(Mehldau 포함)로 재학습한 Aria 분류기 + 앨범 분리(`album_split_*`) + 녹음조건 층화 검증; (2) 지표 = 연주자 agreement를 **[target 어댑터 / 도메인-only 어댑터 / base / 다른 연주자 어댑터(mismatch)]** 4조건으로 같은 프롬프트·seed에서 비교, 차이(target − domain-only)만 "연주자 특이 효과"로 보고; (3) CE도 같은 4조건 × 여러 연주자 행렬로 보고 대각 우세(diagonal dominance)를 확인.
- Cheston 결과상 화성 개념 단독이 74.4%로 가장 판별력이 높았고 다이내믹은 26.3% — 사용자 생성기가 velocity만 바꿔서 분류기를 속일 위험은 작지만, 반대로 **화성/보이싱 특징이 스타일 판정을 지배**하므로 "코드 추종"과 "스타일" 지표가 서로 얽힐 수 있다(코드 진행을 고정한 상태에서 스타일 판정 필요).
- 분류기는 학습 데이터(전사 스타일, 실황 노이즈)를 그대로 학습하므로 "생성물이 PiJAMA 전사처럼 보이는가"를 "연주자처럼 들리는가"로 오인할 수 있다. 사용자의 듀레이션 토큰 지배 현상(PiJAMA 전사 포맷)과 같은 함정 — 실시간 키보드 입력(사람이 친 MIDI)은 전사 MIDI와 통계가 다르므로 별도 도메인 갭 점검 필요.
- FMD는 CLaMP 2 임베딩 기반이라 장르·편성 수준 차이에는 민감하나 동일 장르 내 연주자 미세 차이에 대한 민감도는 미검증(추론; Moonbeam 표에서 CLaMP2 PiJAMA30 분류 44.0%는 그 한계를 시사).

### Gaps
- Edwards 분류기 판정과 사람 청취 판단의 상관(청취 실험 유무): 미확인.
- "Hyman sink"의 정확한 정의(최다 클래스 Dick Hyman으로 오분류 흡수 추정): 원문 미확인.
- FMD/CLaMP 기반 지표의 symbolic 연주자 수준 민감도 및 사람 판단 상관: 찾지 못함.
- Cheston 해석가능 모델 수치 상충(0.767 vs 91.3%): 원문 대조 필요.

## (d) 입력 재타건/에코를 구분하는 코드추종(chord-following) 지표

### Takeaway
표준 화성 적합 지표(CTnCTR·PCS·MCTD, pitch-class 엔트로피 등)는 존재하지만, **"입력(조건) 음을 그대로 다시 치거나 따라 치는 음을 제외한" 코드추종 지표의 선행례는 찾지 못했다**. 조건 민감도는 최근 "부분 코드 손상(corruption) 스트레스 테스트"와 "minimal-pair counterfactual"로 평가하는 흐름이 있으나, 정답 코드 vs 틀린 코드 NLL gap을 표준으로 쓰는 symbolic 논문도 찾지 못했다 — 사용자 프로젝트는 이 부분을 자체 설계해야 한다.

### Cited Findings
- [요약만] Yeh et al. 2021 (JNMR, arXiv 2001.02360) 화성 지표: **CTnCTR** = (코드톤 + "적절한" 비코드톤[다음 음과 2반음 거리]) / (코드톤 + 전체 비코드톤), 반마디 단위 현재 코드 기준; **PCS** = 16분음표 창마다 멜로디–코드 음정의 협화 점수 평균; **MCTD** = 멜로디 음–코드 간 tonal distance를 음 길이로 가중 평균. CTnCTR·PCS 높을수록, MCTD 낮을수록 좋음; 이후 화성화 논문들이 표준처럼 재사용. — [arXiv 2001.02360](https://arxiv.org/pdf/2001.02360), [Chord-conditioned harmonization 2202.08423](https://arxiv.org/pdf/2202.08423), [서베이 2509.00051](https://arxiv.org/html/2509.00051v1)
- [직접 확인] BebopNet의 "harmony" 점수 모델: beam search에서 **현재 연주 중 코드의 스케일과 일관된 음**을 선호(코드–스케일 일관성 기반). — [GitHub bebopnet-code](https://github.com/shunithaviv/bebopnet-code)
- [요약만] ReaLchords/GAPT: 조화(contrastive)와 시간 동기(discriminative) 보상을 분리, 화성 보상만 쓰면 반복으로 붕괴 → 코드추종 "점수"만으로는 반응성의 질을 담보 못함. — [arXiv 2506.14723](https://arxiv.org/html/2506.14723), [arXiv 2511.17879](https://arxiv.org/pdf/2511.17879)
- [요약만] 코드 조건 전파 스트레스 테스트: MIDI-SAG 관련 2026 preprint(arXiv 2609.03584)는 합성 코드 손상에서 **relative-root 치환이 same-root quality 변경보다 출력 변화(CENS)가 2.88배**, "국소 손상은 메커니즘 검사, 전체 replay는 배치된 조건 전파 평가"라고 구분. — [arXiv 2609.03584](https://arxiv.org/html/2609.03584v1)
- [요약만] Minimal-pair counterfactual 평가(텍스트→음악, key·박자 그룹): 조건만 바꾼 쌍으로 응답 변화를 귀속, condition→generation→re-extraction 패러다임. — [arXiv 2608.11899](https://arxiv.org/pdf/2608.11899)
- [요약만] 반주 생성 평가의 일반 관행(SongDriver 등)은 멜로디–반주 조화의 주관평가 + 코드 진행 객관지표 위주이며 "입력 복사" 측정은 없음; AccoMontage2는 반복 패턴이 있으나 보이싱·코드톤이 변해 기계적 반복과 다르다고 기술 → 복사 지표는 구조적 반복과 에코를 구분해야 함. — [SongDriver 2209.06054](https://arxiv.org/pdf/2209.06054), [AccoMontage2 2209.00353](https://arxiv.org/pdf/2209.00353)
- [직접 확인] 릭 반복 측정에 바로 쓸 수 있는 구현: Edwards `measure_diversity.py`(pitch/interval n-gram Jaccard, n=4, 민감도 n=6/8), `memorization.py`(pitch n-gram·interval n-gram 학습데이터 겹침률의 n-감쇠 곡선 + piano-roll Jaccard; "암기 모델은 큰 n에서도 겹침 유지"). — [measure_diversity.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/scripts/analysis/measure_diversity.py), [memorization.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/llama_pijama/analysis/memorization.py)
- [요약만] Weimar Jazz Database 패턴 마이닝: WJD 456개 수작업 전사 솔로; melpat 모듈로 653개 interval 패턴·11,630 instance; pitch/interval n-gram(최대 20) 약 400만/350만 instance 유사도 검색; 4/4 솔로 435곡에서 3–10 interval n-gram으로 릭의 박 위치 의존성 연구. — [ISMIR 2018 "Two web applications for exploring melodic patterns in jazz solos"](http://ismir2018.ircam.fr/doc/pdfs/177_Paper.pdf), [Frieler "Constructing Jazz Lines" 2019](https://jazzforschung.hfm-weimar.de/wp-content/uploads/2019/06/JazzforschungHeute2019_Frieler-Constructing-Jazz-Lines.pdf), [DH2019 abstract](https://dev.clariah.nl/files/dh2019/boa/0723.html)

### Inferences (지표 설계 제안 — 선행례 없음, 추론)
- **Echo 제외 코드톤 비율**: 생성 음 중, 직전 Δ(예: 1박 또는 500ms) 안에 입력(왼손/코드)에 같은 pitch(또는 같은 pitch-class) onset이 있는 음을 "echo/re-strike"로 태깅 → (i) echo rate 자체 보고, (ii) **non-echo 음만으로 CTnCTR·박 위치별 코드톤 비율** 계산. 강박(1·3박)/약박, 온셋/지속 가중을 분리.
- **페어드 코드 스왑(same seed)**: 같은 프롬프트·seed·샘플링 난수에서 코드만 (a) 반음 이동, (b) 같은 근음 다른 quality, (c) 무작위 진행으로 바꿔 생성 → "새 코드의 코드톤 비율 증가량"과 "원 코드 코드톤 비율 감소량"을 차분. 진짜 추종이면 둘 다 커야 하고, 에코만 하면 echo rate만 바뀐다. Edwards의 mismatch 모드가 스타일판 동일 설계.
- **NLL gap**: 실제 솔로(held-out)에 대해 정답 코드 vs 이동/셔플 코드 조건에서 teacher-forced NLL 차이(echo 후보 토큰 제외 버전 포함). 모델이 코드를 쓰지 않으면 gap≈0.
- **코드 변경 후 착지**: 코드 변경 시점 t로부터 반마디(또는 2박) 내 첫 강박/장음(예: ≥8분음표) 음이 새 코드톤(가이드톤 3·7도 별도 집계)인 비율; 무작위 코드 null과 비교.
- **릭 반복**: 한 솔로 내 interval n-gram(n=5–8) 자기반복률, 생성물 간 공통 n-gram 비율, 학습데이터 대비 겹침률의 n-감쇠 곡선을 **타깃 연주자 실연주의 같은 통계(null)** 와 비교(Tatum은 원래 반복 릭이 많을 수 있으므로 절대 임계값 대신 상대비교).

### Gaps
- "입력 음 재타건/에코를 제외한 코드추종" 지표, "코드 변경 후 반마디 내 착지" 지표, "정답 vs 틀린 코드 NLL gap"의 symbolic 선행 논문: 찾지 못함.
- ReaLchords 논문의 note-in-chord류 지표 정의: 원문 미확인.
- WJD 기반 "같은 릭 반복" 정량 지표를 생성모델 평가에 적용한 사례: 찾지 못함.

## (e) "재즈다움"과 "pianist X처럼 들림"을 다른 연구는 어떻게 평가했나

### Takeaway
재즈 생성 연구는 (1) 실데이터 대비 통계 지표(Jazz Transformer: pitch-class 엔트로피·grooving 유사도·코드진행 불규칙성·구조성), (2) 장르/연주자 분류기 판정(ImprovNet 장르 분류기, Edwards 연주자 분류기), (3) 청취 판별·선호 테스트(ImprovNet 28명, BebopNet 사용자 라벨)를 조합했다. 결정적 교훈은 Jazz Transformer가 보인 "그럴듯한 국소 통계에도 불구한 구조·일관성 결여"와 Edwards가 보인 "perplexity로는 연주자성 측정 불가"다.

### Cited Findings
- [요약만] The Jazz Transformer on the Front Line (Wu & Yang, ISMIR 2020, arXiv 2008.01307): WJazzD로 학습, 지표 = 마디별 **pitch-class histogram entropy**(조성 불안정 시 높음), **grooving pattern similarity**(마디 내 onset 위치 64-d 이진벡터 간 유사도), **chord progression irregularity**, **structureness indicator**(fitness scape plot); 모델 출력은 실데이터보다 엔트로피 높고 grooving 유사도 낮으며 장기 반복 약함. — [ISMIR 2020 PDF](https://archives.ismir.net/ismir2020/paper/000339.pdf), [arXiv 2008.01307](https://arxiv.org/pdf/2008.01307)
- [요약만] ImprovNet (Bhandari et al., arXiv 2502.04522, 2025; venue 미확인): 클래식 사전학습 + 클래식/재즈 미세조정, 반복 corruption-refinement; 청취 테스트에서 **참가자 79%가 클래식곡의 재즈 스타일 변환을 재즈로 식별**(28명, p=0.0037은 alphaXiv 요약 출처); 장르 분류기로 변환 확인(분류기 구조·정확도 미확인); AMT 대비 단기 continuation/infilling 우세(pitch-class KL, transition similarity, note density, IOI). — [arXiv 2502.04522](https://arxiv.org/pdf/2502.04522), [GitHub improvnet](https://github.com/keshavbhandari/improvnet)
- [직접 확인] Edwards 2026: "sounds like pianist X"의 계측기 = 실연주 학습 분류기, 생성 연속에서 슬라이딩 창 agreement, 조건/ablation/mismatch/사전학습/무조건부 FT 대조, 암기·다양성 동시 보고. — [README](https://github.com/almostimplemented/jazz-pianist-style), [agreement_eval.py](https://raw.githubusercontent.com/almostimplemented/jazz-pianist-style/main/scripts/agreement_eval.py)
- [직접 확인] BebopNet: "개인화"를 단일 사용자의 청취 라벨로 정의하고 그 선호 지표로 생성 선택. — [GitHub bebopnet-code](https://github.com/shunithaviv/bebopnet-code)
- [요약만] MIDI-RWKV: CP·GS·PCHE(=Wu & Yang 지표 계열) + 28명 청취 테스트 병행. — [arXiv 2506.13001](https://arxiv.org/abs/2506.13001)
- [요약만] Cheston 2026: 연주자 판별 근거를 화성·리듬·멜로디·다이내믹 개념별로 분해하고 웹앱(ImprovID)으로 공개 — "X다움"의 설명가능 분해 사례. — [Nature MI](https://www.nature.com/articles/s42256-026-01279-9), [GitHub](https://github.com/huwcheston/deep-pianist-identification)

### Inferences
- 사용자 목표(Tigran + bebop 어휘, swing 아님)는 기존 분류기 클래스에 없는 혼합 스타일이므로 "X 분류기 agreement 최대화"로 정의하면 안 된다. (i) bebop 어휘 = WJD 패턴 사전 대비 interval n-gram 적중률, (ii) Tigran성 = 소규모(13곡) 참조와의 임베딩 거리 + 박자/리듬 통계(아래 f), (iii) 최종 = 사용자 A/B 청취 — 의 다축 평가가 현실적(추론).
- Jazz Transformer 지표는 마디 그리드가 전제다. PiJAMA/Aria 같은 연주(performance) MIDI는 박 그리드가 없어 grooving similarity 등 계산 전 beat tracking/정렬이 필요(추론).

### Gaps
- 재즈 생성물에 대해 "전문가 vs 비전문가" 청취 판정과 분류기 판정의 일치도를 측정한 연구: 찾지 못함.
- ImprovNet 장르 분류기 상세, BebopNet 사용자 라벨 수·선호 지표 정확도 수치: 미확인(README에는 수치 없음).

## (f) Tigran Hamasyan 특화 자료와 "non-swing" 타이밍 측정

### Takeaway
Tigran Hamasyan에 대한 MIR/생성 연구나 공개 전사 데이터셋은 찾지 못했고, 학술 분석은 Schumann (MTO 2021)의 비대칭 박자·오스티나토·사이클 분석이 사실상 유일한 1차 근거다. "swing 아님"은 Corcoran & Frieler (2021)의 beat-upbeat ratio(BUR)로 측정할 수 있는데, 재즈 솔로이스트의 실제 BUR도 평균 약 1.3:1로 이미 2:1보다 훨씬 곧으므로 목표치는 "≈1.0 근처 + 비대칭 박자 그룹핑"으로 정의해야 한다.

### Cited Findings
- [요약만] Scott C. Schumann, "Asymmetrical Meter, Ostinati, and Cycles in the Music of Tigran Hamasyan", Music Theory Online 27.2 (2021): 2006–2020 스튜디오 앨범 13장 전반에서 비대칭 박자·오스티나토·사이클이 핵심 어법; 사이클은 *Red Hail*부터 상용; 비대칭 박자를 아르메니아 민속 춤곡과 연결; Meshuggah 리듬 분석(Pieslak)과 비교; "그의 음악에 대한 학술적 관심은 적다". — [MTO 27.2](https://mtosmt.org/issues/mto.21.27.2/mto.21.27.2.schumann.html)
- [요약만] 대중 출처: 아르메니아 민속 리듬은 2·3박 기반이나 2+2+3 등 혼합 박자가 흔함; 화성 언어는 아르메니아 민속 선법에 빚짐; Hamasyan 본인은 bebop을 넘어선 뒤 홀수 박자에 몰두했다고 언급. — [New East Digital Archive](https://www.new-east-archive.org/articles/show/12547/introducing-tigran-hamasyan-armenian-folk-contemporary-jazz), [Ithaca.com](https://www.ithaca.com/entertainment/music/tigran-hamasyan-s-genre-defying-music-comes-to-bailey-hall/article_c4eef3d6-e216-430a-b63e-ab67cf826e0b.html)
- [요약만] Scribd에 26개 오스티나토/사이클 예시 전사 문서가 있으나 사용자 업로드로 품질 미검증. — [Scribd](https://www.scribd.com/document/561389480/Rhythms-of-Tigran-Hamasyan-examples)
- [요약만] Corcoran & Frieler, "Playing It Straight: Analyzing Jazz Soloists' Swing Eighth-note Distributions with the Weimar Jazz Database", Music Perception 38(4):372 (2021): WJD 456 솔로, 대부분 **BUR ≈ 1.3:1**의 약한 불균등, 2:1은 드묾 → "안정적 2:1 swing은 개념적 신화"; 느린·중간 템포와 postbop/hardbop에서 높은 비율. — [Music Perception](https://online.ucpress.edu/mp/article-abstract/38/4/372/116701/Playing-It-StraightAnalyzing-Jazz-Soloists-Swing)
- [요약만] Downbeat delay가 swing의 핵심 요소라는 Communications Physics(2022) 연구는 다른 방법으로 유사 결과(160bpm 미만에서 증가 경향 예외). — [Comms Physics 2022](https://www.nature.com/articles/s42005-022-00995-z)
- [요약만] 오디오 swing ratio 자동 추정: Dittmar, Pfleiderer, Müller (ISMIR 2015) 라이드 심벌 swing ratio; DAFx-15 swing ratio estimation; Swingogram 표현. — [ISMIR 2015 PDF](https://audiolabs-erlangen.de/content/05_fau/professor/00_mueller/03_publications/2015_DittmarPM_SwingRatio_ISMIR.pdf), [DAFx-15](https://www.ntnu.edu/documents/1001201110/1266017954/DAFx-15_submission_59.pdf), [Swingogram](https://www.researchgate.net/publication/319392963_A_Swingogram_Representation_for_Tracking_Micro-Rhythmic_Variation_in_Jazz_Performances)

### Inferences
- **Non-swing 지표(추론)**: (1) 박 내 연속 8분음 쌍의 BUR 분포(중앙값·IQR), 목표 ≈1.0–1.1, 비교군 = Tatum/Mehldau 실연주의 BUR; (2) 셋잇단 그리드 vs 16분 그리드 onset 정렬 오차 비교(어느 그리드에 더 잘 맞는가); (3) 박자 그룹핑 = IOI 자기상관/마디 내 강세 패턴에서 2+2+3 등 비대칭 주기 검출. 연주 MIDI에는 박 그리드가 없으므로 사용자 입력 코드 타이밍(왼손)에서 템포·박을 추정하거나 클릭과 함께 연주하도록 설계해야 측정 가능.
- Tigran 13곡(약 53분)은 단독 학습에는 극소량 — 스타일 "어댑터"보다 (i) 리듬/선법 규칙을 디코딩 제약·재순위 특징으로 주입, (ii) 공유 모델 위 소수 벡터(임베딩/prefix)만 학습, (iii) 사용자 A/B로 재순위하는 조합이 데이터 규모에 맞다. 또한 bebop 어휘(WJD 패턴)와 Tigran 리듬은 서로 다른 축이므로 하나의 어댑터에 섞기보다 축별 제어가 평가·디버깅에 유리(추론).
- Aria-MIDI(YouTube 전사)에 Hamasyan 연주가 포함됐을 가능성은 있으나 미확인 — 포함 시 사전학습 overlap으로 "Tigran성" 평가가 오염될 수 있다.

### Gaps
- Tigran Hamasyan 관련 MIR·생성모델 논문, 공개 MIDI 전사 데이터셋: 찾지 못함.
- 아르메니아 선법의 계산적 정의(음계 집합)·Hamasyan 사용 빈도 정량 분석: 찾지 못함(MTO 본문 미열람).
- 연주(비양자화) MIDI에서 BUR를 박 정보 없이 안정 추정한 검증된 방법: 찾지 못함.
