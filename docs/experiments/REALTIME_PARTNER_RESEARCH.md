# 시간정렬 화성 제어로 Aria를 재즈 파트너로 바꾼다

> **리뷰 반영 상태 (2026-10-12, 아스트라 + Claude, PR #1736 댓글).** 이 보고서의 **자료 목록과 가설 지도는 참고 자료로 채택**한다. 다만 아래 실행 계획을 그대로 확정 계획으로 쓰지 않는다. 본문은 원문 그대로 두고, 리뷰에서 고칠 점으로 나온 것을 여기 모은다.
> - **HIGH — 재타격 차단 주장(1안):** AMT식 제어 토큰 생성 금지는 제어 어휘의 출력만 막는다. 같은 음높이의 일반 piano 토큰으로는 여전히 복사할 수 있다. "재타격 경로를 어휘에서 지운다"는 과장이다. 왼손 PC와 타깃 PC의 겹침은 정상 화성에서도 높아서 누수율이 아니다. 따로 감사할 것: 같은 원 음표가 조건과 타깃 양쪽에 쓰이는 비율, 미래 onset 접근, 50 ms 묶음의 관측 완료 시각. 조건은 실시간으로 입력 가능한 시각 이전 정보만 준다. "데이터 병목 해소, 남은 것은 사용자 기준뿐"이라는 결론은 철회한다
> - **HIGH — 지속 KV + 제어 삽입 + 첫 박 패치(2안·런타임):** 단순한 append 조합이 아니다. 확정 송출 경계, 입력 삽입 뒤 캐시 무효화, 어댑터 교체 시 재prefill, note-off 소유권 계약이 필요하다. 이미 확정한 음을 바꾸며 즉시 반응한다고 주장할 수 없다. "반영된 첫 음 0.1~0.25초"는 실측 전 가정치다. 이후 실측과 계약: `ARIA_MLX_KV_SMOKE.md`(#1737), `ARIA_MLX_ROLLBACK.md`(#1739·#1741·#1745), `ARIA_MLX_STREAM_LOOP.md`(#1743)
> - **MEDIUM — 원인 단정:** "기존 실패는 가장 약한 주입 탓", "LoRA는 연주자가 아니라 전사 형식을 학습"은 가설이다. 서로 다른 논문·과제의 성능으로 주입법 우열을 확정할 수 없다. 대조 어댑터 실험(`CONTROL_ADAPTER_DESIGN.md`, #1733)은 Mehldau val에서 M이 C보다 0.02~0.03 더 내려가는 작지만 일관된 차이를 보였다(등록 기준은 미충족). "유일한 경로", "표준 실패 패턴"은 가설로 읽는다
> - **MEDIUM — Edwards 연주자 벡터:** Edwards 생성기는 Aria + 학습된 gated cross-attention + 연주자 임베딩으로 이뤄진 전체 생성기다. 벡터만 기존 model-gen/MLX에 붙여 Tatum을 바로 쓸 수 없다. 우리 조건에서 LoRA보다 낫다는 비교도 없다
> - **MEDIUM — E1 평가:** PC 집합으로 학습한 모델을 임의의 maj7/m7로 시험하면 분포 밖 제어가 섞인다. 지원 조건과 invalid/empty/coverage 분모를 미리 정의한다. 16짝은 8 prompt × 2 seed라 독립 16이 아니다. C1~C9는 임시 운영 기준이고, 사용자 최종 목표가 아니다
> - **기타:** 학습(PyTorch MPS)과 실시간(MLX)이 달라서 LoRA·새 임베딩의 MLX 이식과 수치 동등성 검사가 필요하다. base 가중치의 MPS↔MLX 동등성은 #1737에서 확인했다. `bebop_rh`는 스윙 연주자 위주라 비스윙 목표와 충돌한다. PiJAMA 라이선스는 배포처별로 적는다(GitHub LICENSE는 CC BY-NC 3.0, Zenodo 배포 메타데이터는 4.0)
> - **기준 클립:** 사용자가 고를 후보 22개(티그랑 13, Mehldau 4, 비밥 3, Tatum 2)를 `scripts/make_reference_candidates.py`로 만들었다. 오디오는 로컬에만 둔다

작성 2026-10-11. 이슈 #1735. 선행: `MODEL_SELECTION_RESEARCH.md`(#1672), `MODEL_SELECTION_ROUND2_3.md`. 근거 노트: `realtime_partner_research/`(6편). 대상: 실시간 코드→피아노 솔로 MIDI 파트너(M1 Max 32GB, 유료 GPU 없음). `musical_quality_verified: false`.

이 조건에서 가장 맞는 방법론은 **Aria-medium(약 6.6억 파라미터, Apache-2.0)을 동결한 채, 사용자가 치는 왼손 화성을 AMT식 '별도 제어 토큰'으로 시간에 맞춰 끼워 넣고, 전층 LoRA와 반사실(틀린 코드) 손실로 학습시키는 1안**이다. 2위는 학습 없이 Aria를 MLX 지속 KV 캐시로 스트리밍하면서 코드가 바뀐 직후의 착지음만 부드럽게 유도하고 후보를 재순위하는 2안, 3위는 ReaLchords 저장소의 '코드→선율 온라인 모델' 레시피를 피아노 오른손 자가 라벨로 다시 학습하는 소형 3안이다. 1안이 1위인 이유는 코드 따름, 피아노 관용구, 연주자 교체 구조를 한 모델에서 모두 노릴 수 있는 유일한 경로이고, 지금까지의 실패(프롬프트 코드, 마지막 블록 `Linear(29→1536)`, 관악 24곡 128 update)가 문헌상 가장 약한 주입 방식이었다는 진단이 서기 때문이다. 다만 사전학습 피아노 모델에 AMT식 제어를 PEFT로 이식해 성공한 공개 사례는 찾지 못했으므로 1안은 검증되지 않은 베팅이고, 2안은 1안과 런타임·지표를 공유하므로 먼저 만들 기반이자 1안이 실패할 때의 대안이다. 연주자 개인화는 LoRA보다 Edwards 외(ISMIR 2026)의 **연주자 벡터(4×1536)**가 '베이스 1개 + 교체형 어댑터' 구조에 맞고, Art Tatum 벡터는 공개 가중치에 이미 들어 있다. 1차 보고서(2026-10-03)의 1순위였던 릭 검색 하이브리드는 코드 변경 창 coverage 2/4와 스타일 변경(티그랑, 비스윙)으로 탈락했고, 그 착지음 플래너만 2안에 남았다. 아래의 모든 숫자 기준은 사용자가 기준 클립과 합격 문장을 정하기 전까지 임시이며, 이번 세션은 프록시가 arXiv·Hugging Face·ISMIR 사이트를 막아 GitHub 위주로만 원문을 직접 확인했다.

> **검증 표기.** 각 사실 뒤에 근거 수준을 붙였다. **[직접 확인]** 이번 세션 조사자가 GitHub 저장소의 코드·README·LICENSE·CSV·저장소에 올라온 PDF를 직접 열었다. **[요약만]** 검색 스니펫이나 2차 요약만 봤다(이전 세션의 요약은 "(이전 세션)"을 붙였다). **[프로젝트 실측]** 이 저장소의 실험 기록(`docs/experiments/`, `docs/PERSONALIZATION_STATUS.md`)에 있는 사용자 측 측정과 코드 확인이다. **[내부 기록]** 이 저장소 문서가 과거에 확인했다고 적었지만 이번 세션에서 재확인하지 못한 사실이다. **[추론]** 이 보고서의 판단이고, **[추정치]**는 공개 측정이 없어 계산한 값이다. **미확인**은 출처 있는 수치를 찾지 못했다는 뜻이다.
>
> **조사 범위의 한계.** 이번 세션의 egress 프록시는 arxiv.org, export.arxiv.org, ar5iv, alphaxiv, huggingface.co, ISMIR archives·program·TISMIR, zenodo, semanticscholar, openreview, nime.org, ieeexplore, dl.acm, *.github.io를 모두 막았다. 열린 곳은 github.com(raw 포함)과 pypi.org뿐이었고, 세션 공용 웹 검색 예산(200회)도 조사 중에 소진됐다. 그래서 논문 본문의 표 수치는 저장소에 PDF가 있을 때만 [직접 확인]이다.

## 1. 결론: 화성을 '듣게' 학습시키는 1안이 1위, 무학습 착지 제약과 소형 온라인 모델이 뒤를 받친다

### 공개 모델 지형에는 빈칸이 하나 있고, 세 안은 그 빈칸을 다르게 메운다

2017–2026년 문헌에서 **"코드 조건 + 재즈 피아노 다성 + 공개 가중치 + 실시간"을 모두 갖춘 공개 작업은 찾지 못했다** [직접 확인: 부재 확인 범위는 GitHub 저장소와 검색 스니펫]. 코드 조건 재즈 솔로 생성기는 Improv RNN, JazzGAN, BebopNet, MINGUS, CMT처럼 거의 모두 단성 관악 데이터의 소형 모델이고, 다성 재즈 피아노 계열(ImprovNet, JAZZVAR/Overpainting, Edwards 외 2026)은 코드 기호가 아니라 원곡이나 피아니스트 ID로 조건화한다. 다성 피아노에 코드를 조건으로 거는 모델(Polyffusion, Whole-song, FGG, GETMusic, MIDI-GPT `prism`)은 모두 팝 데이터다 ([MINGUS PDF](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf), [Edwards 외 paper.pdf](https://github.com/almostimplemented/jazz-pianist-style/blob/main/docs/paper.pdf), [Polyffusion](https://github.com/aik2mlj/polyffusion)). 피아노 표현(10 ms 타이밍, 벨로시티, 페달), 허용적 라이선스, Apple Silicon(MLX) 추론 코드, 재즈 피아니스트 조건 선례를 모두 가진 베이스는 **Aria 하나**다 ([EleutherAI/aria](https://github.com/EleutherAI/aria)) [직접 확인]. 그래서 세 안 모두 "Aria에 화성 제어를 어떻게 얹느냐" 또는 "Aria를 포기하고 무엇을 얻느냐"의 선택이다.

**1안(1위): 시간정렬 화성 제어 토큰 학습.** 사용자가 겪은 "코드 효과 NLL 약 1%"와 "맞는/틀린 코드는 구별하지만 생성기로는 base보다 못함"은 실패 신호라기보다 문헌의 표준 패턴이다. MINGUS 절제 실험에서 모든 조건 특징을 넣어도 퍼플렉서티는 12.30에서 11.71로 약 5%만 내려갔고 ([MINGUS 보충자료](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/Supplementary_material_MINGUS_ISMIR21.pdf)) [직접 확인], Edwards 외는 테스트 CE가 1.94에서 1.92로 거의 그대로인데 자유생성의 피아니스트 분류기 일치율은 **37%에서 70%로 두 배**가 됐다 ([paper.pdf](https://github.com/almostimplemented/jazz-pianist-style/blob/main/docs/paper.pdf)) [직접 확인]. 즉 조건 효과는 teacher-forced 우도가 아니라 자유생성에서 드러난다. 또 제어 배치에 대한 증거는 일관된 방향을 가리킨다. 시간 정렬 interleave(AMT)나 음표별 주입(Moonbeam의 코드 MLP + 전층 LoRA)이 강하고, 같은 LM 안의 concat + LoRA가 cross-attention 어댑터보다 낫다는 저장소 기록(Function Alignment의 `prompt_lora`)이 있으며, 전역 prefix는 긴 음악 문맥에 묻힌다(MIDI-LLM은 인필링에서 CFG를 훨씬 키워야 했다) ([AMT](https://github.com/jthickstun/anticipation), [Moonbeam conditional_gen_commu](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model/tree/conditional_gen_commu), [Function Alignment README](https://github.com/music-x-lab/midi-function-alignment)) [직접 확인], ([MIDI-LLM arXiv 2511.03942](https://arxiv.org/abs/2511.03942)) [요약만]. 사용자의 마지막 블록 `Linear(29→1536)` 128 update는 이 스펙트럼에서 가장 약한 형태다 [추론]. 여기에 보조 사전학습 과제와 반사실 손실을 함께 써서 마디 단위 코드 정확도를 **65.27%에서 78.33%로**(반사실 손실만 더하면 71.15%) 올렸다는 MuseBarControl 결과가 더해지고 ([arXiv 2407.04331](https://arxiv.org/abs/2407.04331)) [요약만], AMT 샘플러는 제어 블록 토큰의 생성을 `-inf`로 막아 "조건 음을 그대로 다시 치는" 경로를 어휘에서 지운다 ([sample.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/sample.py)) [직접 확인]. 이 장치는 T4A에서 확인된 **재타격 문제를 정면으로 겨냥한다**. T4A에서 입력 코드 음의 재타격을 빼자 m7 쪽 반응(Q음 13 대 P음 11)이 거의 사라졌다 ([T4A_CHORD_PROBE.md](T4A_CHORD_PROBE.md)) [프로젝트 실측].

**2안(2위): 무학습 착지 제약 스트리밍.** 학습이 없어 열 압력 문제를 피하고, 사용자가 처음으로 "노래 같다"고 판정한 Aria를 그대로 쓴다. Aria 저장소에는 이미 MLX 지속 KV 캐시와 continuous prefill이 구현돼 있고 ([demo_mlx.py](https://github.com/EleutherAI/aria/blob/main/demo/demo_mlx.py)) [직접 확인], Aria-Duet 논문은 이 대책 없이 prefill하면 첫 음까지 1000–2000 ms가 걸리고 continuous prefill로 거의 사라진다고 보고했다 ([arXiv 2511.01663](https://arxiv.org/abs/2511.01663)) [요약만]. 사용자의 0.67초 블록은 매 블록 237–408 토큰을 처음부터 다시 prefill한 PyTorch MPS 경로의 값이다 [프로젝트 실측]. 2안은 합격 문장("코드가 바뀌면 다음 반 마디 안에 코드톤으로 착지")을 구성으로 직접 시험한다. 순위가 2위인 이유는 화성 위반을 디코딩에서 막는 방식이 분포를 왜곡한다는 경고가 강하기 때문이다. Grammar-Aligned Decoding은 하드 마스크의 분포 왜곡을 정식화했고 ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2bdc2267c3d7d01523e2e17ac0a754f3-Abstract.html)) [요약만], BebopNet 보충자료는 화성 일관성 지표로 빔 서치를 하자 "코드에 맞는 음을 반복"하는 reward hacking이 생겼다고 적었으며 ([BebopNet Supplementary](https://github.com/shunithaviv/bebopnet/blob/master/BebopNet_SupplementaryMaterial.pdf)) [직접 확인], 이 프로젝트의 13.4M 스케일 밖 하드 마스크는 같은 음 반복을 20%로 늘렸다(실제 5.5%) [프로젝트 실측, 1차 보고서 기록]. 그래서 2안은 착지 지점에만 소프트 바이어스를 쓰고, 사이 구간은 Aria에 맡긴다. 화성을 '들은' 선율이 아니라 착지만 맞춘 선율이 될 수 있다는 것이 2안의 상한이다 [추론].

**3안(3위): 소형 온라인 코드→선율 모델.** 프레임마다 코드를 명시 입력하는 소형 모델은 코드 반응이 가장 확실하다. HSE의 WJazzD 재학습 CMT는 블록 절단 없는 8마디 생성에서 Fmaj7/Fm7 마디 **16개 중 15개**에서 코드 쪽 구별 음으로 움직였다 ([CMT_CONTINUOUS.md](CMT_CONTINUOUS.md)) [프로젝트 실측]. 그러나 CMT는 양방향 LSTM으로 8마디 미래 코드를 보는 구조라 실시간 입력과 맞지 않고 ([CMT model.py](https://github.com/ckycky3/CMT-pytorch/blob/master/model.py)) [직접 확인], 티그랑 연주를 CMT 입력으로 바꾸면 곡별 23–60%의 음이 사라진다 ([REP_AUDIT.md](REP_AUDIT.md)) [프로젝트 실측]. 3안은 같은 '프레임별 명시 조건' 원리를 미래를 보지 않는 온라인 모델로 다시 만든다. realchords-pytorch(MIT)에는 사용자 방향 그대로인 "online model that generates melody given chords" 학습 config와 MLX 백엔드가 있다 ([realchords-pytorch](https://github.com/lukewys/realchords-pytorch)) [직접 확인]. 3위인 이유는 단성 출력이라 피아노 관용구를 포기하고, 사용자가 소형 모델들(13.4M, BebopNet, CMT)의 어휘에 이미 "무작위적", "그나마"라고 판정했기 때문이다.

| | 코드 따름 | 피아노 관용구(다성) | 교체형 개인화 | M1 Max 실시간 | 티그랑·비스윙 | 이 Mac 학습 비용 |
|---|---|---|---|---|---|---|
| **1안** 시간정렬 제어 학습 | 학습으로 노림(미검증) | 유지(Aria) | 연주자 벡터와 결합 가능(조합 미검증) | 2안 런타임 공유, 제어 토큰 몇 개 추가 | 데이터 선택으로만 대응, 변박은 포기 | 높음(수 시간, 세션 분할) [추정치] |
| **2안** 무학습 착지 제약 | 착지점에서만 확보, 사이 구간은 포기 | 유지 | Tatum 즉시(공개 벡터) | 가장 유리 | 1안과 같음 | 0 |
| **3안** 소형 온라인 모델 | 가장 강함(프레임별 명시) | **포기**(단성) | 약함(데이터 부족) | 매우 유리 | 16분 격자·4/4 가정, 변박 포기 | 중간(수 시간) [추정치] |

### 1차 결론(2026-10-03)에서 달라진 것

| 항목 | 1차 보고서 | 이번 결론 | 바뀐 이유 |
|---|---|---|---|
| 1순위 | 착지음 플래너 + 실제 비밥 릭 조각 검색 + 박 인지 검증기 + 13M 채점기 | 1안(Aria 시간정렬 제어 학습) | 원연주 검색 R2에서 코드가 바뀌는 2마디 목표 coverage 2/4, 돌려받은 두 목표도 출처가 한두 곡 ([WJAZZD_RETRIEVAL_R2.md](WJAZZD_RETRIEVAL_R2.md)) [프로젝트 실측]. 릭 메모리가 관악 단선율(WJazzD 피아노 약 6곡)인데 사용자 목표가 티그랑·비스윙으로 바뀌었다 |
| 2순위 | MINGUS식 명시 코드 소형 모델 + 대조 손실 | 2안(무학습 착지 제약) | Aria가 처음으로 "노래 같다"는 판정을 받았다 ([T3_ARIA.md](T3_ARIA.md)) [프로젝트 실측]. 소형 명시 모델은 코드 반응(15/16)은 확인했지만 어휘 문제가 남았다 |
| 3순위 | AMT-Small, 왼손 = 컨트롤 | 3안(소형 온라인 모델) | AMT-Small은 Lakh 팝 사전분포이고 벨로시티가 없다. 왼손을 컨트롤로 쓰는 **메커니즘은 1안으로 옮겨 Aria에 이식**한다 |
| 살아남은 부품 | 착지음 플래너, 반사실 손실, 소프트 검증기, MLX 런타임, 왼손 PC 컨트롤 | 그대로 쓴다 | 플래너는 2안, 반사실 손실과 왼손 PC는 1안, 소형 명시 조건은 3안 |
| 버린 부품 | 13M 채점기, WJazzD 1차 학습, 릭 메모리, 2:1 스윙 렌더 | 채점기 역할 폐기, 스윙 렌더 금지 | 사용자 비스윙 선호(T3c), 13M 이득 미검증 |
| 런타임 판단 | 13M 경로가 예산 안이므로 포팅 보류 | MLX 지속 KV 이식이 필수 | Aria 블록 0.67초(full re-prefill)는 반응 지연을 키운다 [프로젝트 실측] |
| Aria-Duet 첫 음 200–300 ms | [요약만(이전 세션)]으로 인용 | **미확인으로 강등** | 이번 세션은 README·코드·스니펫 어디에서도 찾지 못했다 |
| Moonbeam | Apache-2.0, HF에 ComMU 조건 LoRA 공개(정정) | 라이선스와 체크포인트 공개 모두 **충돌**로 표시 | 7절 참고 |

### 목표 사이의 네 가지 충돌과 각 안의 선택

**코드 제어 대 개인화.** 연주자 스타일은 상당 부분 화성 선택 그 자체다. Cheston 외는 연주자 식별에서 화성 개념만 쓴 모델이 74.4%, 리듬 61.9%, 선율 57.5%, 다이내믹 26.3%였다고 보고했다 ([Nature Machine Intelligence 2026](https://www.nature.com/articles/s42256-026-01279-9), [arXiv 2504.05009](https://arxiv.org/abs/2504.05009)) [요약만]. 그래서 착지음을 강하게 강제할수록 Mehldau·Tatum의 화성적 개성이 지워지고, 연주자 어댑터는 화성 분포를 옮겨 코드 준수를 떨어뜨릴 수 있다. 스타일 LoRA가 코드 준수를 얼마나 잊게 하는지 잰 음악 연구는 찾지 못했다(NLP에서는 LoRA가 full FT보다 덜 배우고 덜 잊는다는 결과가 있다 ([TMLR 2024, arXiv 2405.09673](https://arxiv.org/abs/2405.09673)) [요약만]). 이 프로젝트의 Mehldau LoRA는 val과 다른 피아니스트에서 CE가 거의 같게 내려갔고(−0.297 대 −0.283), 가장 큰 몫이 duration(−0.556)이었으며, 생성 유효성이 1/8로 떨어졌다 ([MEHLDAU_ARIA_LORA.md](MEHLDAU_ARIA_LORA.md)) [프로젝트 실측]. 1안은 코드 제어를 다연주자 데이터에서 먼저 학습하고 연주자 성분은 작은 벡터로 분리해 이 충돌을 줄인다. 2안은 착지점 이외에서 연주자 모델을 건드리지 않는다. 3안은 개인화를 사실상 포기한다.

**코드 제어 대 실시간.** 강한 코드 추종 모델은 미래 코드를 본다. CMT는 8마디 미래를 양방향으로 보고, MINGUS는 다음 코드를 입력으로 받으며, AMT는 제어를 5초 앞(δ = 5 s)에 둔다 ([AMT config.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/config.py)) [직접 확인]. 키보드에서 "지금" 치는 코드에는 lookahead가 없다. 실시간 시스템들은 commit 구간을 고정하고 그 뒤만 갱신하는 방식으로 이를 메운다. ReaLJam 프론트엔드의 기본값은 로컬 lookahead 2박·commit 2박이다 ([play_from_midi_input.js](https://github.com/lukewys/realchords-pytorch/blob/main/realchords/realjam/frontend/play_from_midi_input.js)) [직접 확인]. 대가는 코드 변경에 대한 반응이 commit 길이만큼 늦어진다는 점이다. CFG는 배치를 2배로, N후보 재순위는 N배로 늘린다 ([Aria sample_mlx.py](https://github.com/EleutherAI/aria/blob/main/aria/inference/sample_mlx.py)) [직접 확인]. 1·2안은 "첫 박 패치"(코드 직후 1–3음만 다시 생성)로 반응을 당기고 나머지는 commit 뒤로 미룬다. 3안은 모델이 작아 이 충돌이 가장 작다.

**개인화 대 실시간.** 개인화 베이스가 클수록 느리다. Edwards 생성기는 Aria-medium 659M에 cross-attention 78M(12%)을 더했다 ([paper.pdf](https://github.com/almostimplemented/jazz-pianist-style/blob/main/docs/paper.pdf)) [직접 확인]. 교체 단위도 문제다. 13.4M 런타임은 같은 base에서 어댑터 교체가 0.8 ms였지만([PERSONALIZATION_STATUS.md](../PERSONALIZATION_STATUS.md)) [프로젝트 실측], Aria에서 연주자를 바꾸면 이전 연주자로 계산한 KV 캐시가 엄밀히는 낡는다. 연주자 교체는 프레이즈 경계에서 하고 문맥을 다시 prefill하는 운용이 안전하다 [추론].

**비밥 어휘 대 비스윙, 티그랑 박자 대 4/4 격자.** 비밥 어휘는 스윙에서 태어났고 PiJAMA의 대부분은 스윙 감각의 연주다. Aria는 원본 미세 타이밍을 보존하므로 학습 데이터의 스윙이 그대로 나온다 [추론]. 사용자는 인위적 스윙 추가도, 스트레이트 양자화도 원하지 않으므로([T3C_ARIA_TIGRAN.md](T3C_ARIA_TIGRAN.md)) [프로젝트 실측], 비스윙은 데이터 선택과 측정으로만 다룰 수 있다. 측정 도구로는 박 안 연속 8분음표 쌍의 길이비(beat-upbeat ratio, BUR)가 있고, WJazzD 솔로이스트의 실제 BUR은 대부분 약 1.3:1이었다 ([Corcoran & Frieler, Music Perception 38(4), 2021](https://online.ucpress.edu/mp/article-abstract/38/4/372/116701/Playing-It-StraightAnalyzing-Jazz-Soloists-Swing)) [요약만]. Tatum은 스트라이드·루바토라 개인화 목표 자체가 비스윙과 부딪친다. 티그랑의 비대칭 박자·오스티나토·사이클은 분석 문헌(Schumann, [MTO 27.2, 2021](https://mtosmt.org/issues/mto.21.27.2/mto.21.27.2.schumann.html)) [요약만]뿐이고 생성 모델·데이터셋은 찾지 못했다. JTD 박자 라벨은 3 또는 4 quarter-note만 있다 ([mirdata jtd.py](https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/jtd.py)) [직접 확인]. 세 안 모두 변박은 포기하고, 비스윙은 측정 기준(2절 C5)과 데이터 가중으로만 대응한다.

## 2. 아홉 개의 임시 합격 기준 (모두 사용자 확정 필요)

사용자가 기준 클립과 합격 문장을 비워 뒀으므로, 아래 기준은 측정 가능하게 만든 **임시안**이다. 절대값의 출처가 없는 항목은 기준 연주(사용자가 고를 클립, 그 전까지는 로컬 `data/bebop_rh` val 28곡)의 같은 통계에 대한 상대값으로 정했다. 128 BPM에서 1박 468.75 ms, 반 마디 937.5 ms, 1마디 1,875 ms다.

| ID | 무엇을 | 측정법 | 임시 기준 | 근거 | 사용자가 확정할 것 |
|---|---|---|---|---|---|
| C1 착지 | 코드가 바뀌면 다음 반 마디 안에 코드톤으로 착지 | 코드 변경 시각 t부터 937.5 ms 창에서 첫 '착지음'(길이 ≥ 234 ms인 첫 음, 없으면 창 안 최장음)의 PC가 새 코드의 코드톤(가이드톤 3·7도 별도 집계) 또는 허용 텐션인 비율. **입력 보이싱과 같은 pitch(echo)는 제외한 값**을 주 지표로 쓴다 | echo 제외 착지율 ≥ 기준 연주의 0.8배, 그리고 무작위 코드 null보다 0.2 이상 높음 | 사용자 예시 문장, 착지 지표 설계 [추론] | 창 길이, 착지음 정의, 허용 텐션 목록 |
| C2 진짜 반응 | 재타격이 아니라 선율이 코드를 따름 | 같은 seed·같은 문맥에서 코드 품질만 바꾼 짝(maj7↔m7 등)의 d = [(P−Q)/n]_P − [(P−Q)/n]_Q, 재타격 제외 | d > 0인 짝 ≥ 12/16 | Aria base T4A 8/16, CMT 15/16 사이 [프로젝트 실측] | 짝 수, 코드 쌍 종류 |
| C3 echo | 입력 화음 복사 억제 | 생성 음 중 입력 보이싱과 같은 pitch를 입력 onset ±1박 안에 친 비율 | T4A Aria base 수준(10–12%) 이하, 그리고 기준 연주에서 오른손이 왼손 음을 겹치는 비율 이하 | T4A 창 음 140개 중 14개, 136개 중 16개 [프로젝트 실측] | 허용 수준 |
| C4 릭 반복 금지 | 같은 릭을 반복하지 않음 | (a) 4초 루프 감지기 0회, (b) 한 출력(30초) 안 음정 5-gram이 3회 이상 반복되는 비율이 기준 연주 분포 이하, (c) 학습 데이터 16-gram 일치 0 | 세 조건 모두 | MINP_AB 감지기 [프로젝트 실측], Edwards 외 `measure_diversity.py`·`memorization.py` [직접 확인] | n, 반복 허용 횟수 |
| C5 비스윙 | 스윙 느낌 없음 | 박 안 연속 8분음표 쌍의 BUR 중앙값(박은 사용자 코드 입력 또는 DAW 클록) | 0.95–1.15 | WJazzD 솔로이스트 약 1.3:1과 구별 [요약만] | 목표 범위, 셋잇단 허용 여부 |
| C6 지연 | 바로 반응 | 코드 keydown → 그 코드를 반영한 첫 생성 음의 p95 | ≤ 1박(468.75 ms). 블록 생성 p99 < commit 길이 | 현 13.4M 런타임 PC 입력 평균 0.9 s [프로젝트 실측], jam_bot 목표는 미확인 | 목표치 |
| C7 형식·음역 | 연주 가능한 MIDI | 비피아노 토큰 0, 형식 유효 100%, 출력 최저음이 사용자 왼손 최고음보다 위 | 100% | 형식 마스크 16/16 ([MEHLDAU_ARIA_MASKED_GEN.md](MEHLDAU_ARIA_MASKED_GEN.md)) [프로젝트 실측] | 음역 경계 |
| C8 개인화 | 그 연주자처럼 | 실연주로 학습한 분류기의 일치율: 목표 연주자 조건 − 도메인만 적응한 모델, mismatch 조건 대조 | +15%p 이상, mismatch에서 조건 쪽으로 이동 | Edwards 외 70% 대 37% [직접 확인] | 분류기, 연주자 목록 |
| C9 청취 | 실제로 낫다 | 블라인드 쌍대 선호, 사용자 기준 진행 4–8개 | 현 최선(Aria 이어 치기, min_p 0) 대비 비열등(선호 ≥ 50%) | — | 기준 클립, 진행, 템포 |

사용자가 정해야 할 것은 표의 마지막 열 외에 다섯 가지다. 기준 클립 3–5개와 구간(티그랑에서 원하는 솔로 구간, 비밥 예시), 기준 코드 진행 4–8개와 템포 범위, 박자 범위(4/4만인지, 7/8 등을 넣을지), 코드톤 재진술을 어디까지 허용할지(echo 정책), 그리고 Mehldau·Tatum에서 기준으로 삼을 녹음이다. 문헌에는 "입력 코드음을 다시 친 것"과 "진짜 선율 반응"을 분리하는 공개 지표가 없으므로([BebopNet Supplementary](https://github.com/shunithaviv/bebopnet/blob/master/BebopNet_SupplementaryMaterial.pdf), [realchords eval_utils.py](https://github.com/lukewys/realchords-pytorch/blob/main/realchords/utils/eval_utils.py)) [직접 확인], C1–C3은 이 프로젝트가 직접 정의해야 한다. 판별용 기준선으로 무작위 코드(ImprovNet의 random_chords 결과 파일 ([improvnet eval](https://github.com/keshavbhandari/improvnet/tree/master/eval)) [직접 확인]), 결정론적 코드톤 라인(Keating & Casey, [chord-tone-generator](https://github.com/mbkeating/chord-tone-generator)) [직접 확인], 그리고 입력 화음을 아르페지오로 복사하는 봇을 함께 돌린다. 복사 봇은 echo율 100%이면서 코드톤율도 높으므로, 이 봇과 구별되는지가 C2·C3의 핵심이다 [추론].

## 3. 평가 기준 비교표

### 추천 3안

| 안 | 코드 조건 기본 지원 | MIDI 출력 | 피아노 다성 | 이 Mac 실시간(근거) | 유료 GPU 없이 미세조정 | 필요한 데이터 | 라이선스 | 코드·가중치 공개 | 근거 수준 | 위험 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1안 Aria + AMT식 제어 토큰 + 전층 LoRA + 반사실 손실 | 없음 → 학습으로 추가 | 예 | 예 | 가능 추정: 2안 런타임 + 제어 토큰 수 개 [추정치] | 가능: Aria LoRA r16, 128 update 120초, MPS 5.17 GiB [프로젝트 실측] | 로컬 `bebop_rh` 원본 235곡 + 왼손 PC 자가 라벨 | Aria Apache-2.0, PiJAMA CC BY-NC 3.0 | 구성요소 모두 공개 | 구성요소는 동료심사(AMT TMLR 2024, Function Alignment ISMIR 2025), 조합은 미검증 | 사전학습 피아노 모델로의 PEFT 이식 선례 없음, 손 분리 누수로 복사 학습, 열 |
| 2안 Aria 무학습 착지 제약 | 없음 → 디코딩에서 착지점만 | 예 | 예 | 가능 추정: MLX KV + continuous prefill 구현 존재 [직접 확인], 반 마디 블록 0.07–0.5 s [추정치] | 해당 없음 | 보정용 PiJAMA 통계(로컬) | 위와 같음 | 공개 | 장치별 근거(Aria-Duet NeurIPS 2025 데모, ReaLJam CHI EA 2025), 조합 미검증 | 착지만 맞고 사이는 화성을 못 들음, 소프트 바이어스도 반복 유발 가능 |
| 3안 소형 온라인 코드→선율 모델 | 예(프레임별) | 예 | **아니오**(단성) | 가능 추정: dim 512·depth 8, MLX 백엔드 [직접 확인] | 가능 추정: 수 시간 [추정치] | 박 격자가 붙은 피아노 오른손 + 코드 프레임(자가 라벨), 선택적으로 팝 리드시트 | 코드 MIT, Hooktheory CC BY-NC-SA 3.0 | 코드 공개, 가중치 없음(새 학습) | ReaLchords ICML 2024, GAPT ICLR 2026 | 단성, 소형 어휘에 대한 사용자 불만 재현, 박 추정 오류 |

### 구성 후보 모델과 기법

| 후보 (연도, venue, ID) | 코드 조건 기본 지원 | MIDI | 피아노 다성 | 이 Mac 실시간 (근거) | 유료 GPU 없이 미세조정 | 필요한 데이터 | 라이선스 | 코드·가중치 | 근거 수준 | 위험 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Aria-medium** (2025, ISMIR 2025, arXiv 2506.23869) | 없음. 곡 단위 임베딩 + CFG 경로만 [직접 확인] | 예 | 예(10 ms, 벨로시티 10단계, 페달) [직접 확인] | 조건부 예: MLX KV 캐시·실시간 데모 [직접 확인]. MPS full re-prefill 블록 0.674 s, MLX CLI 시퀀스당 약 62 tok/s [프로젝트 실측] | LoRA 가능 [프로젝트 실측]. 공식 학습 코드는 CUDA assert [프로젝트 실측] | 코드 조건에는 자가 라벨 필요 | 가중치 Apache-2.0, 학습 데이터 Aria-MIDI CC BY-NC-SA 4.0 [직접 확인] | 공개 [직접 확인] | 동료심사, 청취 비교 [요약만]. 로컬 재현(T3) | 루프(min_p 0.035에서 3/10), 재타격, 비피아노 토큰, 사전학습 겹침 |
| **Edwards 외 jazz-pianist-style** (2026, ISMIR 2026, arXiv 2610.02918) | 없음. 피아니스트 12명 조건 [직접 확인] | 예 | 예 | 추론 MPS 명시 [직접 확인], 지연 미확인 | 원 레시피는 전체 FT 약 18시간 GPU [직접 확인] → 불가. 벡터만 학습은 미검증 [추론] | PiJAMA-12(Mehldau·티그랑 없음) | Apache-2.0(저장소 내 카드 사본) [직접 확인], HF 원 카드 미확인 | 공개 | 동료심사, 분류기 평가만, 청취 연구 없음 [직접 확인] | 스윙·스트라이드 연주자 위주 |
| **AMT** (2023 arXiv, TMLR 2024, arXiv 2306.08620) | 트랙 control interleave(반주) [직접 확인] | 예 | 예, **벨로시티 없음** [직접 확인] | 미확인(jam_bot ggml M3 Max 수치 미확인) | 학습 코드 없음, HF 모델에 PEFT 가능 추정 [추론] | Lakh(팝) | Apache-2.0 [직접 확인] | 공개 | 동료심사, 반주 사람 평가 18승 31무 11패 [요약만] | 팝 사전분포 |
| **Moonbeam** (2025, arXiv 2505.15559, venue 미확인) | ComMU 코드+메타 조건 레시피 [직접 확인] | 예 | 예 | 미확인 | LoRA 레시피, CUDA/DDP 중심 [직접 확인] | ComMU(팝, 7화음까지) | **충돌**: MIT 대 Apache-2.0 | ComMU 체크포인트 **충돌**(README TODO 대 HF 실재) | arXiv, 전문가 평가 [요약만] | 포팅, 코드 사전 확장 필요 |
| **MIDI-GPT** (AAAI 2025, arXiv 2501.17011) | `prism`: 마디별 pitch class set(대상 트랙 자신) [직접 확인] | 예 | 예 | macOS arm64 휠, OSC 마디 단위 서버 [직접 확인], 지연 미확인 | 소형(768·12층) 가능 추정 | 다트랙 | 코드 MIT, 가중치 **충돌** | `prism`·`expressive`는 "training in progress" [직접 확인] | 동료심사 | 마디 지연 구조 |
| **ReaLchords / realjam** (ICML 2024; ReaLJam CHI EA 2025 arXiv 2502.21267; GAPT ICLR 2026 arXiv 2511.17879) | 예(코드→선율 온라인 레시피) [직접 확인] | 예 | 아니오 | `--mlx` 백엔드 [직접 확인], ms 수치 미확인 | MLE 가능 추정, RL 단계는 무거움 | 팝 리드시트 4종 | 코드 MIT [직접 확인] | 코드 공개 | 동료심사, 전문가 12명(GAPT) [요약만] | 화성 보상 해킹 |
| **CMT** (IEEE Access 2021) | 예(프레임 크로마, 미래 8마디) [직접 확인] | 예 | 아니오 | CPU 8마디 2.49–2.93 s [프로젝트 실측] | 이어 학습 코드 있음 [프로젝트 실측] | 2트랙(선율+코드) | 원 코드 LICENSE 없음, HSE 가중치 MIT [내부 기록] | 공개 | 15/16 [프로젝트 실측] | lookahead, 티그랑 표상 손실 23–60% |
| **MINGUS** (ISMIR 2021) | 예(현재·다음 코드 4음, 베이스, 박) [직접 확인] | 예 | 아니오 | 미확인 | 소형 | WJazzD | LGPL-3.0 [직접 확인] | 가중치 포함 | 블라인드 28명 [직접 확인] | 관악 관용구 |
| **BebopNet** (ISMIR 2020) | 예(4음 임베딩) [직접 확인] | 예 | 아니오 | 미확인 | torch 1.1.0 고정 [직접 확인] | 색소폰 284곡(구매) | MIT [직접 확인] | 가중치 | 사용자 4명 비블라인드 | major 3화음을 7화음으로 바꿈 [직접 확인] |
| **Notochord** (AIMC 2022, arXiv 2403.12000) | 없음. 이벤트별 pitch whitelist·시간 제약 [직접 확인] | 예 | 예 | 10 ms 미만 저자 주장 [요약만] | 학습 CLI [직접 확인] | Lakh | MIT [직접 확인] | 공개 | 동료심사 | 재즈 피아노 품질 근거 없음 |
| **MIDI-RWKV** (2025, arXiv 2506.13001, venue 미확인) | 마디 속성 제어, 코드 미확인 | 예 | 예 | rwkv.cpp [직접 확인] | state tuning 4–6분 [요약만] | GigaMIDI, POP909 | MIT [직접 확인] | 가중치 70 MB [직접 확인] | 28명 청취 [요약만] | 약 35M 팝 모델 [추론] |
| **Function Alignment** (ISMIR 2025, arXiv 2506.15548) | chord→melody 어댑터, `prompt_lora`가 최선 [직접 확인] | 예 | 단성(멜로디) | 미확인 | 어댑터만 학습 | Nottingham, 16분 격자 [직접 확인] | 미확인 | 공개 [직접 확인] | 동료심사, 수치 미확인 | 피아노 연주 아님 |
| **MuseBarControl** (2024, arXiv 2407.04331, venue 미확인) — 기법 | 마디 제어 + 반사실 손실 | — | — | 0(학습 기법) | 약 2배 forward [추론] | 오염 조건 자동 생성 | 미확인 | 코드 미확인 | 65.27%→78.33%(보조 과제 + 반사실), 반사실만 71.15% [요약만] | 거짓 음성(리하모니제이션) |
| **JazzyGPT2 / REMIBlock** (Springer DCAI 2025, DOI 10.1007/978-3-032-05748-8_44) | 멜로디→반주(반대 방향) | 예 | 예 | Q4/Q8/F16 GGUF, 다음 마디 선행 생성 [직접 확인], 수치 미확인 | 91M | 미확인 | 미확인 | 코드 공개 [직접 확인] | 학회 proceedings, perplexity 7.69 [요약만] | 60–120 BPM 제한 [직접 확인] |
| **Keating & Casey 코드톤 그래프** (ICMC 2025 데모, arXiv 2510.19666) | 예(결정론) | 예 | 아니오 | 학습 없음 | 해당 없음 | 코드 진행 | MIT [직접 확인] | 공개 | 데모, 청취 미확인 | 기계적 라인 |
| **Polyffusion / FGG** (ISMIR 2023 arXiv 2307.10304 / ICML 2025 arXiv 2410.08435) | 예(cross-attn·하드 제어) | 예 | 예 | 반복 denoise로 부적합 [추론] | 새 모델 학습 | POP909 | MIT / 미확인 | 공개 / 저장소 못 찾음 | 동료심사 | 팝, 마디 단위 오프라인 |

## 4. 후보별 적용 계획

### 4.1 1안: 별도 제어 어휘로 왼손 화성을 끼워 넣고, 틀린 코드를 벌한다

**베이스.** 로컬 Aria `model-gen.safetensors`(파라미터 658,538,496개, sha256 앞 16자리 `5e6b07ee2680f004`)를 동결한다 ([ARIA_COND_SMOKE.md](ARIA_COND_SMOKE.md)) [프로젝트 실측]. 지금까지의 T3·T4 결과와 비교하기 위해서다. 개인화를 결합하는 단계에서 Edwards 생성기로 옮길지 정한다(4.4).

**이미 해 본 것과 무엇이 다른가.** 이전 경로가 끝날 때 기록된 재개 조건은 "새 구조나 학습 목적의 근거, 제한된 대조 설계, 피아노 보존을 평가할 자료와 조건"이었다 ([ARIA_COND_CONTRACT_V2.md](ARIA_COND_CONTRACT_V2.md)) [프로젝트 실측]. 1안은 아래 다섯 가지 차이로 이 조건을 채운다.

| 이전 시도 | 1안 | 근거 |
|---|---|---|
| 코드 음을 **피아노 음 토큰으로** 프롬프트에 넣음(T4A·T4B) → 재타격 | 화성을 **새 제어 토큰 블록**으로 넣고 샘플링 때 생성 금지 | AMT `safe_logits` [직접 확인] |
| 마지막 블록 입력에 `Linear(29→1536)` 하나, 0 초기화 | q·k·v·o 전층 LoRA r16 + 새 제어 임베딩. 각 PC 토큰은 Aria의 해당 pitch 임베딩 평균으로 초기화 | Moonbeam이 코드 구성음을 사전학습 pitch 임베딩 합으로 넣음 [직접 확인] |
| 코드 기호 29차원(리드시트) | **연주 화성**: 왼손 PC 집합 12비트 + 최저음 PC. 런타임 입력과 같은 함수로 만든다 | 학습·실행 분포 일치 [추론], Cheston HarmonyExtractor 선례 [직접 확인] |
| CE(+ KL λ=1), 128 update, WJazzD 관악 24곡 | CE + 반사실 hinge(정답 제어 pitch NLL < 틀린 제어 pitch NLL − m) + 제어 드롭아웃 20%, 1,000 update부터, PiJAMA 피아노 | MuseBarControl [요약만] |
| teacher-forced NLL 중심 평가 | 자유생성 반사실 짝 + echo 제외 착지율(C1–C3) | Edwards 외, MINGUS 절제 [직접 확인] |

**데이터와 라벨.** 새 다운로드 없이 로컬 `data/bebop_rh`의 원본 235곡(비밥 피아니스트 13명, train 189 / val 28 / test 18)을 쓴다 ([BEBOP_RH_DATA.md](BEBOP_RH_DATA.md)) [프로젝트 실측]. 출처는 PiJAMA(2,777곡, 120명, 223.6시간, CC BY-NC 3.0)다 ([PiJAMA pijama.csv](https://github.com/almostimplemented/PiJAMA/blob/main/pijama.csv), [LICENSE](https://github.com/almostimplemented/PiJAMA/blob/main/LICENSE)) [직접 확인]. 라벨 함수는 이렇다 [추론]. 50 ms 동시 타건 묶음의 최고음과 G3 이상 음을 이벤트(생성 대상)로 두고, 나머지 가운데 G3 미만 음이나 동시음 3개 이상 묶음(Cheston: 0.3초 격자, 최소 3음, 최고음 제거 ([extractors.py](https://github.com/HuwCheston/deep-pianist-identification/blob/master/deep_pianist_identification/extractors.py)) [직접 확인])을 왼손 화성으로 둔다. 왼손 화성 묶음이 시작될 때마다 `<H>` + PC 토큰들 + 베이스 PC 토큰을 그 시각 이후 첫 이벤트 앞에 넣는다(δ ≈ 0). 박 격자는 쓰지 않는다. PiJAMA MIDI에는 120 BPM 기본 템포 하나만 있고 박·마디·손 정보가 없다 ([REP_AUDIT.md](REP_AUDIT.md)) [프로젝트 실측, A Fable 13곡에서 확인]. 반사실 조건은 같은 구간의 PC 집합을 반음~삼전음 이동하거나 3도만 장↔단으로 바꿔 만든다. 리하모니제이션으로 유효한 '틀린' 코드가 섞이는 거짓 음성은 따로 센다. 왼손 PC로 조건을 만들 때의 최대 위험은 손 분리 오류로 오른손 음이 조건에 새어 들어가 "조건 = 타깃 일부"가 되는 것이다 [추론]. 학습 전에 데이터 기준치로 "이벤트 음의 PC가 같은 시각 제어 PC와 겹치는 비율"을 기록한다. rootless 보이싱(3-7-9-13)은 근음 라벨로는 모호하지만 "실제로 울린 PC 집합"으로만 쓰면 문제가 줄어든다 [추론].

**학습 비용 [추정치].** 프로젝트 실측 기준점은 Aria LoRA r16, 1,024 토큰, gradient checkpoint, MPS에서 128 update 120초, 관측 최대 5.17 GiB다(update당 약 0.94초) ([MEHLDAU_ARIA_LORA.md](MEHLDAU_ARIA_LORA.md)) [프로젝트 실측]. 반사실 항은 틀린 조건으로 forward·backward를 한 번 더 하므로 update당 약 1.9초로 잡는다. 첫 실험 세 팔(도메인만 LoRA, 제어 CE, 제어 CE + 반사실) × 1,000 update는 약 16 + 16 + 31 = **63분**이다. 열 압력 Heavy로 실행을 멈춘 기록이 있고 같은 조건 재실행은 955초 동안 GPU 80–82°C, 열 압력 Nominal이었으므로([MEHLDAU_ARIA_MASKED_GEN.md](MEHLDAU_ARIA_MASKED_GEN.md)) [프로젝트 실측], 한 세션을 20분 이하로 끊고 체크포인트로 이어 4–5세션에 나눈다. 1,000 update는 약 100만 토큰이고, `bebop_rh` 원본 235곡(곡당 평균 약 4.8분, 초당 약 40토큰 가정)의 약 0.4 epoch다. 본 학습 10,000 update를 반사실 항과 함께 하면 약 5.3시간으로 여러 날에 나눠야 한다.

**예상 지연 [추정치].** 2안 런타임을 그대로 쓰고, 코드가 바뀔 때 제어 토큰 5–15개를 KV 캐시에 덧붙이는 비용만 더해진다. 키다운 뒤 첫 박 패치(다음 1–3음, 3–9토큰 재생성)는 초당 60–100토큰 기준 0.05–0.15초이므로, 출력 지연 보정을 더해 **반영된 첫 음 0.1–0.25초**로 추정한다. CFG를 쓰면 배치가 2배가 되어 decode가 1.5–2배 느려진다.

**첫 실험 E1: 제어 감도 3팔 대조.** 세 팔을 같은 데이터·seed·1,000 update로 학습한다. (0) 제어 없이 같은 데이터로 학습한 도메인 LoRA, (1) 제어 + CE + 드롭아웃, (2) 제어 + CE + 반사실. 도메인 팔은 Mehldau LoRA에서 본 "공통 적응이 CE 하락의 대부분"이라는 혼동을 분리하는 대조군이다. 평가는 val 28곡 중 앞 8곡의 8초 프라이머 끝에 T4A와 같은 규칙(근음 = 프라이머 마지막 1초 최저음 PC)으로 maj7 대 m7 제어를 넣고 seed 1–2로 다음 두 반 마디를 생성한다(팔마다 16짝). 모든 팔과 base에 같은 형식 마스크를 건다.

| 합격 조건(팔 2 기준) | 기준 | 비교 대상 |
|---|---|---|
| C2 반사실 짝 d(재타격 제외) > 0 | ≥ 12/16 | Aria base T4A 8/16 |
| C1 echo 제외 착지율 | 도메인 팔보다 높고, 무작위 제어 null + 0.2 이상 | 팔 0 |
| C7 형식 유효 | 16/16, dur 자리 비피아노 질량이 도메인 팔의 2배 이하 | Mehldau adapter dur 자리 비 중앙값 66배 [프로젝트 실측] |
| 피아노 보존 | null 제어 pitch CE ≤ 도메인 팔 + 0.02 nats | 팔 0 |
| C4(a) 4초 루프 | 첫 16초에서 0 | min_p 0 |

팔 2가 teacher-forced에서만 맞음 < 틀림이고 C2를 통과하지 못하면 "판별만 되고 생성은 안 됨"의 재현이므로 1안을 멈춘다(6절 전환 조건).

### 4.2 2안: Aria를 끊지 않고 흘려보내며 착지점만 이끈다

**베이스.** 같은 `model-gen.safetensors`를 MLX bf16로 돌린다. Tatum 개인화가 필요하면 Edwards 생성기의 Tatum 벡터를 쓴다(4.4).

**이미 해 본 것과 무엇이 다른가.** 하드 스케일 마스크, harmony bias, 박 인지 realizer는 13.4M 모델에서 시험해 청취로 기각됐다 ([MODEL_SELECTION_ROUND2_3.md](MODEL_SELECTION_ROUND2_3.md)) [프로젝트 실측]. Aria에서는 형식 마스크와 순수 샘플링만 시험했다. 2안은 네 가지가 다르다. 베이스가 Aria다. 제약을 모든 음이 아니라 **코드 변경 직후의 착지 창에만** 건다. 단일 샘플이 아니라 N후보를 echo 제외 지표로 재순위한다. T4B처럼 반 마디마다 독립 블록을 잘라 생성하지 않고 지속 KV로 이어 생성한다. T4B에서는 채택 음 247개 중 86개가 블록 끝에서 잘렸고 블록 연결부 음정 중앙값이 11반음이었다 ([T4B_SEQUENTIAL.md](T4B_SEQUENTIAL.md)) [프로젝트 실측].

**구성.** 런타임은 Aria 데모의 continuous prefill(입력 메시지 10개마다 갈라지는 지점부터 16/128 토큰 청크로만 prefill, `KV_CHUNK_SIZE` 256, 첫 두 토큰 3-way beam, 지난 onset 마스크, `MAX_STREAM_DELAY_MS` 50)을 이식한다 ([demo_mlx.py](https://github.com/EleutherAI/aria/blob/main/demo/demo_mlx.py)) [직접 확인]. 생성은 commit 길이(기본 1박)만큼 앞서 채운다. 코드 키다운 때 플래너가 착지 목표 집합 T를 정한다. T는 새 코드의 가이드톤(3·7도)과 허용 텐션이고, 가능하면 입력 보이싱에 없는 PC(C\V)를 우선한다. 입력에 없는 화성음을 고르면 복사로는 설명되지 않는 반응이기 때문이다 [추론]. 연결음 참고로 Keating & Casey의 코드톤 그래프(코드 간 최적 전이음 최단 경로, MIT)를 쓴다 [직접 확인]. 착지 창 안의 pitch 토큰에만 T에 +β, 텐션에 +β/2를 더하고, 입력 보이싱과 같은 pitch에는 1박 동안 −γ를 준다. β는 echo 제외 착지율이 기준 연주 수준에 오도록 PiJAMA 통계로 보정한다. 화성도를 과하게 밀면 청취 평점이 떨어진다는 보고(제어 파라미터와 CTnCTR 상관 r = 0.9) ([arXiv 2202.08423](https://arxiv.org/pdf/2202.08423)) [요약만]와 AMT 공동작곡에서 "다운비트에 근음만 침"으로 퇴화한 사례 ([CRFM blog](https://crfm.stanford.edu/2024/08/06/co-composition.html)) [요약만]가 상한을 정한다. N = 2–4 후보를 배치로 뽑아 C1·C3·C4와 Aria 로그우도로 고르고, min_p는 0으로 둔다(min_p 0.035에서 3/10 루프, 0에서 0/10 ([MINP_AB.md](MINP_AB.md)) [프로젝트 실측]). 사용자 왼손 최고음 아래로는 소프트 벌점을 준다. 사용자의 코드 음을 문맥에 피아노 음으로 넣는 변형(Aria가 화성을 듣지만 재타격 위험)과 빼는 변형(바이어스만 화성을 앎)을 둘 다 잰다.

**학습 비용.** 0이다. β 보정용 통계는 CPU에서 분 단위로 끝난다 [추정치].

**예상 지연 [추정치].** `bebop_rh` 오른손 밀도(곡별 중앙값 초당 3.3–4.0음 [프로젝트 실측])와 이중음을 감안해 반 마디에 15–30토큰을 잡는다. 프로젝트의 MLX CLI 시퀀스당 약 62 tok/s [프로젝트 실측]면 0.25–0.5초, M1 Max에서 1B Llama가 MLX로 1k 문맥 217 tok/s라는 커뮤니티 측정 ([oMLX](https://omlx.ai/benchmarks/ro7re1dx)) [요약만]에 가까워지면 0.07–0.15초다. 후보 2개 배치는 단일 대비 처리량이 약 1.44배(1B Llama 250 → 합산 약 360 tok/s) [요약만]이므로 블록 시간이 약 1.4배 늘어난다. 반영된 첫 음은 첫 박 패치로 0.1–0.25초다.

**첫 실험 E2: T4B 재연.** T4B와 같은 4개 문맥 × seed 2개 × 반 마디 4블록(maj7↔m7 교대), 같은 체크포인트로, 지속 KV 스트림 + 착지 제약 + 재순위와 순수 샘플링을 비교한다.

| 합격 조건 | 기준 | T4B 기준값 |
|---|---|---|
| 재타격 제외 코드 따름 | 판단 가능 블록의 ≥ 20/26 | 14/26 [프로젝트 실측] |
| C1 착지 | 무작위 코드 null + 0.2 이상 | — |
| C3 echo | 순수 샘플링 이하 | — |
| 윗성부 인접 음정 중앙값 | ≤ 기준 연주 중앙값 | 블록 연결부 11반음 |
| C4(a) 루프 | 0 | — |
| 블록 생성 p95 | < commit 길이(468.75 ms), M1 Max MLX | 비용 중앙값 0.700 s, 마감 초과 4/32 |
| C9 청취 | 순수 Aria(min_p 0) 대비 블라인드 비열등 | — |

### 4.3 3안: 코드를 프레임마다 듣는 소형 온라인 모델을 피아노 오른손으로 학습한다

**베이스.** realchords-pytorch의 온라인 선율 config(DecoderTransformer dim 512, depth 8, heads 8, `max_seq_len` 514, bf16-mixed, 원 설정 30k step·batch 64)를 처음부터 학습한다 ([config](https://github.com/lukewys/realchords-pytorch/blob/main/configs/gen_models/decoder_only_online_melody_4_datasets.yml)) [직접 확인]. 파라미터는 약 25M으로 추정한다(12·d²·L 근사) [추정치]. 토큰은 16분음표 프레임마다 [chord, melody]가 교대한다(`FRAME_PER_BEAT = 4`) [직접 확인].

**이미 해 본 것과 무엇이 다른가.** CMT·BebopNet·MINGUS는 관악 데이터의 소형 모델이고 CMT는 미래 8마디를 본다. 3안은 미래를 보지 않는 온라인 인과 모델이고, 피아노 오른손으로 학습하며, 필요하면 대조·판별 보상과 GAIL 기반 RL(GAPT의 반-해킹 판별기 포함)을 붙일 수 있는 코드가 이미 있다 [직접 확인]. 13.4M 모델은 코드를 프롬프트로 받았지만, 3안은 코드가 프레임마다 입력에 있다.

**데이터와 라벨.** `bebop_rh` 오른손 단선율(이미 추출됨) + 왼손 PC 집합 코드 프레임. 프레임 표현에는 박 격자가 필요하므로 PiJAMA MIDI를 로컬에서 피아노 음원으로 렌더해 Beat This!(코드·가중치 MIT, JAAH·Filosax·ASAP를 학습)로 박을 얻는다 ([beat_this](https://github.com/CPJKU/beat_this), [beat_this_annotations](https://github.com/CPJKU/beat_this_annotations)) [직접 확인]. 렌더라서 MIDI와 시간축이 정확히 맞고 YouTube 다운로드가 필요 없다 [추론]. 재즈 솔로 피아노(드럼 없음, 루바토)에서 Beat This의 정확도는 측정된 적이 없고, 다운비트는 JAAH에서도 박보다 약하다(박 F1 95.1, 다운비트 85.0) ([arXiv 2407.21658](https://arxiv.org/abs/2407.21658)) [요약만]. 그래서 마디 위상은 쓰지 않고 박만 쓴다. 로컬 `midi` 트리는 onset이 약 32 ms 격자에 몰려 있고 같은 연주의 `midi_kong` 판이 더 촘촘하므로, 미세 타이밍과 BUR 측정에는 `midi_kong`이 낫다 ([REP_AUDIT.md](REP_AUDIT.md)) [프로젝트 실측]. 원 config의 팝 리드시트(Hooktheory 19,086 / POP909 909 / Nottingham 1,019 / Wikifonia 약 6,000곡, ±6반음 이조) [직접 확인]로 사전학습할 수 있지만, Hooktheory는 CC BY-NC-SA 3.0이고 ([SheetSage](https://github.com/chrisdonahue/sheetsage)) [직접 확인] 스타일이 팝이다.

**학습 비용 [추정치].** 소형 모델도 MPS에서는 고정 오버헤드가 지배한다. 13.4M 비밥 오른손 어댑터는 MPS에서 516 update에 627초였다 ([BEBOP_RH_ADAPTER.md](BEBOP_RH_ADAPTER.md)) [프로젝트 실측]. 이 비율이면 원 설정 30k step은 약 10시간이므로, 첫 실험은 5k step(약 1.5–2시간, 세션 분할)으로 줄인다. RL 단계는 첫 실험에서 뺀다.

**예상 지연 [추정치].** 반 마디가 8프레임(약 16토큰)이고 모델이 약 25M이므로 블록 생성은 수십 ms 수준이다. 반응 지연은 commit 길이가 정한다. ReaLJam 기본(로컬 commit 2박)이면 약 0.94초, commit 1박이면 약 0.47초다.

**첫 실험 E3.** MLE만으로 5k step 학습한 뒤 CMT_CONTINUOUS와 같은 두 진행(P: Gm7–C7–Fmaj7, Q: Gm7b5–C7–Fm7)과 seed 4개로 평가한다. 합격 조건은 F 마디 d > 0이 ≥ 15/16(CMT 수준이지만 미래를 보지 않는 조건), C1 착지율이 null + 0.2 이상, C4 루프 0, 그리고 CMT 연속 생성 대비 블라인드 선호 ≥ 50%다. 통과해도 사용자가 어휘를 거부하면, 3안 모델을 2안의 **학습된 착지 플래너**로 돌려 Aria가 그 사이를 채우게 한다 [추론].

### 4.4 개인화: 연주자 교체 단위는 LoRA보다 '연주자 벡터'가 맞다

Edwards 외는 Aria-medium의 마지막 8개 층 self-attention 뒤에 gated cross-attention(x′ = x + g·CrossAttn(x, E_artist), g 초기값 0.1)을 넣고, 연주자마다 4×1536 컨텍스트 벡터를 학습했다. 이 구조가 "베이스 1개 + 교체형 어댑터"에 가장 직접 맞는 공개 선례다 ([paper.pdf](https://github.com/almostimplemented/jazz-pianist-style/blob/main/docs/paper.pdf), [cross_attention_model.py](https://github.com/almostimplemented/jazz-pianist-style/blob/main/llama_pijama/models/cross_attention_model.py)) [직접 확인]. 교체 단위는 6,144개 숫자이고, 결과는 perplexity 6.82(조건) / 6.96(같은 데이터 무조건 FT) / 11.41(사전학습), 분류기 일치율 70% / 37% / 25%다 [직접 확인]. 이 표가 사용자의 Mehldau LoRA 결과를 설명한다. 도메인 적응만으로 perplexity가 11.41에서 6.96으로 크게 떨어지고 연주자 조건은 6.82로 거의 더하지 않지만, 연주자다움은 일치율에서 두 배가 된다. 사용자의 val −0.297 대 probe −0.283은 차이 0.014 nats로, 연주자 특이 효과가 검출되지 않은 상태다 [추론].

**Tatum.** 공개 생성기 `drewbie/jazz-pianist-style-generator`에 Art Tatum 벡터가 이미 있다(PiJAMA-12에서 118행; 전체 PiJAMA에는 122곡, 8.03시간, 전부 스튜디오) ([pijama12.csv](https://github.com/almostimplemented/jazz-pianist-style/blob/main/data/pijama12.csv), [pijama_30.csv](https://github.com/almostimplemented/jazz-pianist-style/blob/main/data/pijama_30.csv)) [직접 확인]. 학습 없이 기준선과 상한으로 바로 쓸 수 있고, 같은 저장소의 분류기 평가는 Apple Silicon 노트북에서 약 3분이다 [직접 확인]. 다운로드 승인이 필요하다.

**Mehldau.** PiJAMA-12에는 없고 PiJAMA-30에 72곡(라이브 54 / 스튜디오 18, 8.87시간)이 있다 [직접 확인]. 다른 파라미터를 모두 동결하고 Mehldau 벡터 하나만 학습하는 텍스트 반전식 확장을 제안한다. 논문은 "새 연주자 추가"를 실험하지 않았으므로 미검증이다 [추론]. 벡터는 작아도 gradient는 전체 모델을 지나므로 update 비용은 LoRA와 비슷하게 1,024토큰당 약 0.9초, 500 update에 약 8분으로 추정한다 [추정치]. 저샘플 개인화에서 소수 벡터만 학습하는 state tuning이 LoRA r4보다 28명 청취에서 유의하게 선호됐다는 MIDI-RWKV 결과 ([arXiv 2506.13001](https://arxiv.org/abs/2506.13001)) [요약만]가 이 방향의 간접 근거다. 평가는 사전학습 / 도메인만 / 도메인 + 연주자의 3-way 설계와 분류기 일치율(mismatch 모드 포함, [agreement_eval.py](https://github.com/almostimplemented/jazz-pianist-style/blob/main/scripts/agreement_eval.py)) [직접 확인]로 한다. PiJAMA-30 분류기는 공개 스크립트로 재학습해야 한다. Mehldau의 75%가 라이브이고 Tatum은 100% 스튜디오이므로 녹음 조건 층화와 앨범 단위 분할(`recording_condition`, `album_split_*` 열) [직접 확인]이 필요하다. 이 프로젝트의 "Mehldau 제외 base"는 실제로 Mehldau 라이브 54곡을 학습했다 ([ARTIST_LEAK_AUDIT.md](ARTIST_LEAK_AUDIT.md)) [프로젝트 실측].

**티그랑.** PiJAMA 전체에 13곡(앨범 "A Fable", 약 0.89시간)뿐이라 독립 어댑터에는 너무 적다 [직접 확인]. 티그랑은 어댑터가 아니라 프롬프트, 리듬 통계(비대칭 그룹핑 검출), 그리고 사용자가 고를 기준 클립으로 다룬다. T3c에서 이 앨범 구간을 이어 친 결과는 사용자 판정상 원하는 티그랑 솔로잉이 아니었고, 한 출력의 첫 1.5초는 원곡 다음 구간과 거의 같았다 ([T3C_ARIA_TIGRAN.md](T3C_ARIA_TIGRAN.md)) [프로젝트 실측].

**1안과의 결합.** Edwards 생성기는 베이스까지 전체 미세조정한 모델이라([train_cross_attention.py](https://github.com/almostimplemented/jazz-pianist-style/blob/main/scripts/train_cross_attention.py)) [직접 확인] Aria-gen에서 학습한 1안 LoRA를 그대로 얹을 수 없다. 1안이 E1을 통과하면 Edwards 생성기 위에서 1안 LoRA를 다시 학습하되, PiJAMA-12 곡마다 해당 연주자 벡터를 켠 상태로 학습해 두 조건이 함께 있는 분포를 보게 한다 [추론].

**소량 선호 학습.** 사용자 A/B 판정은 먼저 BebopNet식 선호 점수기와 best-of-N 재순위에 쓰고([bebopnet-code](https://github.com/shunithaviv/bebopnet-code)) [직접 확인], 쌍이 수백 개 쌓이면 DPO-Positive로 간다. NotaGen은 CLaMP-DPO에서 `BETA` 0.1, DPOP `LAMBDA` 10, `LR` 1e-6을 쓰고, 정답과 ld_sim > 0.95인 표절 의심 샘플을 chosen에서 뺀다 ([NotaGen RL/config.py](https://github.com/ElectricAlexis/NotaGen)) [직접 확인]. Mac에서는 참조 모델을 "어댑터를 끈 base"로 두고 참조 로그확률을 미리 계산하면 메모리를 아낀다 [추론]. 화성 보상만으로 RL을 하면 반복으로 붕괴한다(GAPT, [arXiv 2511.17879](https://arxiv.org/abs/2511.17879); SMART, [arXiv 2504.16839](https://arxiv.org/abs/2504.16839)) [요약만]. 보상용 분류기와 평가용 분류기는 분리한다.

### 4.5 런타임: 1.8초를 1박 안으로 줄이는 여섯 장치

**현재 지연 수치의 출처.** 사용자가 제시한 "입력→반영 평균 1.8초(128 BPM)"는 13.4M 런타임에 시작 예산을 넣었을 때의 측정(15건, 1.3–2.4초)과 일치한다 ([START_BUDGET.md](START_BUDGET.md)) [프로젝트 실측]. 그런데 저장소의 현황 문서는 그 뒤 반 마디 블록에서 **PC 메시지 기준 평균 0.9초(0.5–1.3초)**, 음 입력은 최신 음에서 블록 시작까지 0.52–0.55초로 기록한다 ([PERSONALIZATION_STATUS.md](../PERSONALIZATION_STATUS.md), [HALF_BAR_BLOCKS.md](HALF_BAR_BLOCKS.md)) [프로젝트 실측]. 둘 다 가상 포트·CPU의 13.4M 경로이고, 실물 키보드·DAW·체감 지연은 재지 않았다. Aria 경로는 블록 계산만 0.674초 중앙값이다. 이 차이는 7절 충돌로 기록한다. **jam_bot의 "반주 100 ms / 멜로디 조건 800 ms" 목표는 이번 세션에서 1차 출처를 찾지 못했다(미확인).** 저장소 문서(GENERATION_LEAD 등)는 출처 없이 인용한다. jam_bot NIME 2026 후속은 Ryzen 9 7950X, RTX 4090, M3 Max에서 ms/token을 쟀다고 하지만 수치는 확인하지 못했다 ([NIME 2026](https://nime.org/proc/nime2026_73/)) [요약만]. 이전 세션에 사용자가 제공한 M3 Max 수치(170M 10.13 ms/token, 416M 26.17 ms/token)도 미확인이다.

여섯 장치는 모두 공개 시스템에서 가져왔다. 첫째, **지속 KV + 증분 prefill**(Aria 데모, mlx-lm `trim_prompt_cache` ([cache.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py))) [직접 확인]. 둘째, **commit 지평**: 확정한 1박은 그대로 재생하고 그 뒤만 다시 생성한다(ReaLJam [직접 확인], StreamMUSE의 GL > I 겹침 생성 ([arXiv 2606.11886](https://arxiv.org/abs/2606.11886)) [요약만]). 셋째, **다음 코드 가설 선생성**: 현재 코드 유지, 진행상 다음 코드, 트라이톤 대리 같은 후보 2–3개를 배치로 미리 만들고 틀리면 롤백한다(SongDriver의 예측 캐시 ([arXiv 2209.06054](https://arxiv.org/abs/2209.06054)) [요약만], mlx-lm `BatchGenerator` [직접 확인]). 넷째, **첫 박 패치**: 코드 직후 1–3음만 제약 샘플링으로 즉시 다시 낸다(Notochord `include_pitch` whitelist ([model.py](https://github.com/Intelligent-Instruments-Lab/notochord/blob/main/src/notochord/model.py)), Aria 첫 두 토큰 3-way beam) [직접 확인]. 다섯째, **MLX와 8-bit**: batch-1 decode는 메모리 대역폭에 묶여 M4 Max의 Qwen3-4B에서 bf16 52.47 → q8 86.91 tok/s(1.66배)였다 ([mlx-lm BENCHMARKS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md)) [직접 확인]. 128 BPM 반 마디에 48토큰을 내려면 51 tok/s, 1박 안에는 102 tok/s가 필요하다 [추론]. 여섯째, **출력 지연 보정과 GPU 깨워 두기**: Aria 데모는 `onset − BASE_OUTPUT_LATENCY_MS(50)`로 미리 보내고([c4dm-disklavier.json](https://github.com/EleutherAI/aria/blob/main/demo/hardware/c4dm-disklavier.json)) [직접 확인], Magenta RealTime 2는 macOS가 작업 공백에 GPU를 다운클록해 frame당 수 ms가 늘어난다며 대기 중 작은 dummy 연산을 돌린다 ([core/README.md](https://github.com/magenta/magenta-realtime/blob/main/core/README.md)) [직접 확인]. 오디오 모델은 핵심 엔진에서 빠지지만 이 운용 아이디어는 그대로 쓴다. FL Studio는 MIDI 클록을 보내기만 하므로 Python 쪽이 클록 slave가 되어 박 그리드에 맞춰 예약 송출한다 [요약만(이전 세션)].

MPS를 떠나야 하는 근거도 쌓였다. MPS에서 GPT-2·BLOOM·OPT의 decode 지연이 특정 구간에서 최대 21배 튀고 그 구간에서 KV 캐시 가속이 무너진다는 보고 ([arXiv 2605.08913](https://arxiv.org/abs/2605.08913)) [요약만], M1 Max에서 MPS 실행 중 GPU가 약 400 MHz에 머물렀다는 단일 사용자 보고 ([PyTorch forum](https://discuss.pytorch.org/t/pytorch-on-mps-doesnt-use-m1-max-gpu-at-full-power-frequency-is-at-minimal-value/194325)) [요약만], M2 Ultra에서 MLX가 지속 생성 처리량이 가장 높았다는 비교 ([arXiv 2511.05502](https://arxiv.org/abs/2511.05502)) [요약만]가 있다. 같은 소형 Transformer의 MPS 대 MLX batch-1 decode를 직접 비교한 측정은 없다.

### 4.6 실행 순서와 승인이 필요한 외부 자원

실행 순서는 순위와 다르다. 2안의 E2를 먼저 만든다. 학습이 없고, 1안이 쓸 런타임과 C1–C6 측정기를 함께 만들며, 순수 Aria의 기준값을 준다. E1은 그 위에서 20분 세션으로 나눠 돌린다. 개인화(Tatum 벡터 사용, Mehldau 벡터 학습)는 E1 결과와 무관하게 2안 런타임에서 바로 시험할 수 있다. E3는 6절의 전환 조건이 걸릴 때만 시작한다.

| 자원 | 쓰는 안 | 라이선스 | 상태 |
|---|---|---|---|
| PiJAMA MIDI (Zenodo 8354955) | 1·2·3안 | CC BY-NC 3.0 [직접 확인] | 로컬 보유. 수익화 영상·재배포는 NC 조건 주의 |
| Aria `model-gen.safetensors` | 1·2안 | Apache-2.0, 학습 데이터 Aria-MIDI CC BY-NC-SA 4.0 [직접 확인] | 로컬 보유 |
| Edwards 생성기·분류기 (HF `drewbie/jazz-pianist-style-*`) | 개인화 | Apache-2.0(저장소 내 카드 사본) [직접 확인], HF 원 카드 미확인 | **승인 필요** |
| `aria-medium-embedding` (HF `loubb`) | 곡 단위 스타일 CFG 대안 | Apache-2.0 [직접 확인 README] | **승인 필요**. 어느 공개 체크포인트에 `embedding_adapter`가 있는지 미확인 [프로젝트 실측] |
| Beat This! 가중치 | 3안 박, C5 측정 | MIT(코드·가중치) [직접 확인] | **승인 필요**. 학습 데이터 일부는 저작권·제한 CC라고 README가 명시 |
| realchords-pytorch | 3안 | MIT [직접 확인] | **승인 필요** |
| Hooktheory(SheetSage 공개본) / POP909 / Nottingham / Wikifonia | 3안 사전학습(선택) | CC BY-NC-SA 3.0 / MIT [직접 확인] / 미확인 / 미확인 | **승인 필요** |
| JTD 주석 | 왼손 PC 라벨 스트레스 시험 | 주석 MIT, 오디오는 연구 목적 요청 [직접 확인] | **승인 필요** |
| chord-tone-generator (Keating & Casey) | 2안 플래너 참고 | MIT [직접 확인] | **승인 필요** |
| frechet-music-distance, deep-pianist-identification | 평가 | MIT [직접 확인] | **승인 필요** |
| WJazzD | 보조 | **충돌**: ODbL [요약만(이전 세션)] 대 "Open Access" 표기만 [요약만] | 로컬 보유 |

## 5. 탈락 후보

| 후보 (연도, venue, ID) | 탈락 이유 (한 줄) | 차용하는 아이디어 |
|---|---|---|
| Suno, Magenta RealTime 1·2 (GitHub magenta-realtime 2026, arXiv 2508.04651), LMDM (arXiv 2605.22717, 2026) | 오디오 출력이라 음 단위 코드 제어와 Serum 전송이 불가능(사용자 제약) | GPU 깨워 두기, windowed attention의 상수 step 비용, block-causal 캐시 |
| HeartMuLa (arXiv 2601.10547, 2026) | 오디오 코덱(12.5 Hz) 모델 | — |
| MIDI-LLM (arXiv 2511.03942, ISMIR 2026) | Llama 3.2 1B 텍스트 조건, 16GB+ CUDA 권장, 인필링에서 CFG 비탄력 | 전역 제어가 음악 문맥에 묻힌다는 교훈 |
| Text2midi (AAAI 2025, arXiv 2412.16526), MuseCoco (2023, muzic), ChatMusician (arXiv 2402.16153), NotaGen (IJCAI 2025, arXiv 2502.18008) | 텍스트·ABC 조건, 실시간 코드 추종 없음 | NotaGen의 표절 제외 규칙과 DPOP 설정 |
| MelodyT5 (ISMIR 2024, arXiv 2407.02277) | ABC 단성 악보 모델, 실시간 아님 | — |
| Polyffusion (ISMIR 2023), Whole-song (ICLR 2024), FGG (ICML 2025), GETMusic (IJCAI 2025, arXiv 2305.10841) | 확산·팝·마디 단위 오프라인 생성 | FGG의 "소프트 조건만으로는 세부 음높이가 보장되지 않는다" → 착지점 제약 |
| ImprovNet (IJCNN 2025 [요약만], arXiv 2502.04522) | LICENSE 파일 없음, 반복 정제가 오프라인 | random_chords 기준선 |
| Jazz Transformer (ISMIR 2020, arXiv 2008.01307) | 코드 입력을 받지 않음 | MusDr 지표 |
| Improv RNN (Magenta, 연도 표기 없음) | 단성, 2026-01 저장소 아카이브 | — |
| BebopNet (ISMIR 2020), MINGUS (ISMIR 2021) | 이미 시험, 단성 관악. MINGUS 전문가 평균 2.45점 대 원곡 3.67점 [직접 확인·계산(이전 세션)] | 반사실·채점기 아이디어 |
| CMT를 스타일 베이스로 (IEEE Access 2021) | 이미 시험, 미래 8마디 lookahead, 티그랑 표상 손실 23–60% [프로젝트 실측] | 프레임별 명시 조건(3안) |
| WJazzD 원연주 검색 (1차 1순위) | 코드 변경 창 coverage 2/4, 관악 [프로젝트 실측] | 착지음 플래너(2안) |
| late-injection `Linear(29→1536)` 어댑터 | 조건 구별 8/8이지만 pitch NLL 순손해, 생성 유효 7/8 < base 8/8 [프로젝트 실측] | 봉인 held-out, 토큰 유효성 검사기 |
| 13.4M Music Transformer 생성기 | 사용자 청취 "무작위적, 틀린 음" | 반 마디 런타임 측정 체계 |
| Moonbeam을 베이스로 (arXiv 2505.15559) | CUDA/DDP 코드, 라이선스·체크포인트 공개 충돌, 코드 사전이 7화음까지 | 음표별 코드 MLP, pitch 임베딩 합 초기화(1안) |
| MIDI-GPT `prism`/`expressive` (AAAI 2025) | "training in progress", 조건이 대상 트랙 자신의 PC 집합, 마디 단위 지연 | 마디 PC 속성 개념 |
| AMT-Small/Medium 직접 미세조정 (TMLR 2024) | Lakh 팝 사전분포, 벨로시티 없음, 학습 코드 없음 | 별도 제어 어휘 interleave(1안) |
| Edwards식 gated cross-attention 코드 어댑터(전체 FT) | 원 레시피 약 18 GPU시간, Function Alignment가 cross-attn이 self-attentive보다 못하다고 기록 | 1안이 실패할 때의 변형 팔 |
| ReaLchords 전체 RL / GAPT를 Aria에 | RL 루프가 길어 열 제약에 맞지 않음 | 반-해킹 판별기·반복 벌점 설계 |
| GAD, SMC, 정확 BP (NeurIPS 2024 arXiv 2405.21047; arXiv 2605.07839, 2026) | 다수 샘플이 필요해 실시간 부담, BP는 Markov 전용 | 형식 마스크까지만 하드로 유지 |
| MuseBarControl 전체 레시피 (arXiv 2407.04331) | 코드 공개 미확인, 2·3차 조사에서 비용 문제로 제외 | 반사실 손실(1안) |
| Function Alignment (ISMIR 2025, arXiv 2506.15548) | 16분 양자화 Roformer, Nottingham chord→melody, 피아노 연주 아님 | "같은 LM concat + LoRA > cross-attn" |
| Notochord (AIMC 2022) 주엔진 | Lakh GRU, 재즈 피아노 품질 근거 없음 | 이벤트별 pitch whitelist(첫 박 패치) |
| MIDI-RWKV (arXiv 2506.13001) | 약 35M 팝 모델 | state tuning → 연주자 벡터 근거 |
| JazzyGPT2 / REMIBlock (Springer DCAI 2025) | 방향 반대(멜로디→반주), 60–120 BPM 제한 | 다음 마디 선행 생성 스케줄 |
| SongDriver (ACM MM 2022, arXiv 2209.06054) | 멜로디→반주, 중국 팝 | 예측 캐시 |
| Pianist Transformer (ICML 2026, arXiv 2512.02652) | 악보→연주 표현 렌더링 | — |
| Frieler & Zaddach (TISMIR 2022) | 코드 공개 미확인, 단성 | 렌디션(아티큘레이션, 미세 타이밍)의 중요성 |
| Impro-Visor grammar (GPL v2) | 기존 손 realizer와 비슷하고 기각 이력 | — |
| SheetSage2 (arXiv 2610.05336, 2026)를 라벨원으로 | 오디오 기반, JAAH tetrads 43.00%, 9th 어휘 없음, 가중치 라이선스 미확인 | — |
| Composer's Assistant 1·2 (ISMIR 2023/2024), Giant·Orpheus Music Transformer | REAPER 다트랙 인필링 / 코드 조건·재즈 근거 없음 | — |
| 대규모 RLHF (MusicRL, ICML 2024, arXiv 2402.04229) | 30만 쌍 규모 | — |

## 6. 놓쳤을 수 있는 것과 추천을 바꾸는 결과

### 놓쳤을 수 있는 것

**측정.** "NLL 약 1%"를 조건 실패로 읽으면 안 된다. 같은 크기의 우도 변화가 MINGUS와 Edwards 외에서 큰 자유생성 차이와 함께 나왔다 [직접 확인]. 반대로 teacher-forced 구별(맞음 < 틀림 8/8)은 생성 능력을 보장하지 않는다는 것을 이 프로젝트가 이미 세 번 확인했다 [프로젝트 실측]. 반사실 손실은 학습 항이면서 그대로 평가 프로브가 되고, ReaLchords 저장소의 곡 중 전조(`multiple_transpose`)·삼전음 대치(`single_transpose_6`) 섭동은 "코드가 바뀐 뒤 몇 박 만에 회복하는가"를 재는 기성 도구다 ([experiment_utils_data_perturbation.py](https://github.com/lukewys/realchords-pytorch/blob/main/realchords/utils/experiment_utils_data_perturbation.py)) [직접 확인]. 2026년 프리프린트는 코드 손상 시험에서 상대 근음 치환이 같은 근음의 품질 변경보다 출력 변화가 2.88배 컸다고 보고했다 ([arXiv 2609.03584](https://arxiv.org/html/2609.03584v1)) [요약만]. 즉 maj7↔m7 짝은 가장 어려운 시험이고, 근음 이동 짝을 함께 써야 감도를 공정하게 잰다.

**런타임.** 0.67초 블록과 "KV 캐시 없는 39분 생성"은 Aria의 한계가 아니라 경로의 문제다. 공식 저장소의 MLX 경로에 KV 캐시와 continuous prefill이 이미 있다 [직접 확인]. T3에서 쓴 로컬 `max_kv_pos` 패치는 logits 최대 차이 0.0으로 검증됐다 [프로젝트 실측]. 1B Llama가 M1 Max에서 MLX로 1k 문맥 217 tok/s를 낸다는 커뮤니티 측정 [요약만]과 비교하면, 프로젝트의 62 tok/s(시퀀스당, 배치 2)에는 최적화 여지가 있다 [추론].

**데이터.** PiJAMA `midi` 트리의 onset은 약 32 ms 격자이고 같은 연주의 `midi_kong`이 더 촘촘하다 [프로젝트 실측]. 비스윙 측정(BUR)과 미세 타이밍 보존에는 이 차이가 크다. Aria 사전학습 데이터(Aria-MIDI, YouTube 전사)에 PiJAMA나 "A Fable"이 들어 있는지는 확인되지 않았으므로 "스타일이 나온다"의 일부는 암기일 수 있다 [프로젝트 실측: T3c A-2]. 솔로 재즈 피아노에서는 리드시트 화성과 편곡이 실제로 함의하는 화성이 자주 다르다 ([Jeremy Siskind](https://jeremysiskind.com/?p=14650)) [요약만]. 실시간 입력은 '연주 화성'이므로, 왼손 PC 자가 라벨의 진짜 장점은 라벨 정확도가 아니라 **런타임 입력 분포와의 일치**다 [추론]. 가장 가치 높은 데이터는 사용자가 FL Studio에서 코드를 직접 치고 그 위에 솔로를 친 기록이다. 코드가 연주와 따로 입력되므로 사후 추정 라벨이 아니다(REP_AUDIT의 제안) [프로젝트 실측]. 규모는 작아도 C1–C3의 금 표준 평가 세트가 된다. QMUL 그룹이 리드시트에 정렬한 솔로 재즈 피아노 대형 데이터셋(부분집합 VAR4000, 4,352쌍)을 개발 중이라고 밝혔으나 공개는 확인되지 않았다 ([arXiv 2412.04610](https://arxiv.org/abs/2412.04610)) [요약만].

**스타일.** 화성 개념이 연주자 식별을 지배하므로(74.4%) [요약만] 코드 추종 지표와 스타일 지표가 서로 얽힌다. 스타일 평가는 코드 진행을 고정한 상태에서 해야 한다 [추론]. "스윙이 싫다"는 BUR 1:1을 뜻하지 않을 수 있다. 재즈 솔로이스트 평균이 이미 약 1.3:1이므로 [요약만], 사용자가 C5 범위를 정해야 한다. 분류기는 "PiJAMA 전사처럼 보이는가"를 "그 연주자처럼 들리는가"로 오인할 수 있고, 실시간 키보드 입력은 전사 MIDI와 통계가 다르다 [추론].

### 추천을 바꾸는 결과

| 관측 | 결정 |
|---|---|
| E2(2안)가 C1·C2·C9를 통과하고 사용자가 선호 | 1안 학습을 보류하고 런타임과 개인화에 투자한다. 2안이 1위가 된다 |
| E1 팔 2가 C2 ≥ 12/16, 피아노 보존 통과 | 1안을 본 학습(약 10k update, 세션 분할)으로 확장하고 Edwards 생성기로 이식한다 |
| E1에서 teacher-forced 구별만 되고 C2 미달 | 1안을 닫는다. 같은 학습 신호로 세 번째 "판별만"이다. 2안을 1위로 올리고 3안을 착지 플래너로 시험한다 |
| E1에서 echo율이 기준 연주를 넘음 | 왼손 누수다. 조건을 동시음 3개 이상 묶음과 한 박 이전 정보로 제한하거나, 사용자 녹음 쌍으로 바꾼다 |
| MLX 지속 KV에서도 블록 p99 > 1박 | 실시간 핵심을 3안으로 바꾸고, Aria는 오프라인 교사·재순위기로 쓴다 |
| Mehldau 벡터가 분류기 일치율에서 도메인만 대비 +15%p 미달 | Mehldau 개인화를 자료 한계로 보고 보류한다(13.4M 기록도 64–128 update에서 포화했다 [프로젝트 실측]). Tatum만 유지한다 |
| 사용자가 블라인드에서 3안 출력을 Aria 계열보다 선호 | 3안을 1위로 올린다 |
| HF에서 Moonbeam ComMU LoRA와 라이선스가 확인되고 MPS에서 돈다 | 코드 조건 참조 구현과 교사로 쓴다 |
| QMUL 리드시트 정렬 데이터 공개 | 1안의 라벨을 연주 화성 + 리드시트 화성 두 층으로 바꾼다 |

## 7. 노트 사이의 충돌 기록

| 사실 | 기록 A | 기록 B | 판단 |
|---|---|---|---|
| Moonbeam GitHub LICENSE | MIT, 마지막 커밋 2025-05-19 [직접 확인] | Apache-2.0, 최종 커밋 2025-06-12 [직접 확인]. 1차 보고서·2·3차 기록도 Apache-2.0 | **충돌.** 커밋 시점이나 브랜치 차이일 수 있다. 쓰기 전에 LICENSE 파일을 커밋 해시와 함께 다시 확인한다 |
| Moonbeam ComMU 조건 LoRA | HF에 309M·839M과 ComMU 조건 LoRA(r8) 실재, 2026-10-03 확인 [내부 기록] | GitHub README의 조건부 체크포인트 링크가 `https://TODO` [직접 확인] | **충돌.** HF 차단으로 이번 세션 재확인 불가. README 미갱신이면 둘 다 사실일 수 있다 |
| CMT venue | 과제 설명·이전 메모의 ISMIR 2021 | README가 IEEE Xplore 9376975로 연결 → IEEE Access 2021 ([CMT-pytorch](https://github.com/ckycky3/CMT-pytorch)) [직접 확인] | **IEEE Access 2021로 정정** |
| REMIBlock 출처 | arXiv 2606.11886 (과제 설명) | Springer DCAI 2025 "Real-Time Symbolic Music Accompaniment Generation for Edge Devices"(JazzyGPT2). 2606.11886은 StreamMUSE [요약만]. 스니펫의 저자명은 ReaLchords 저자와 겹쳐 오류로 보인다 | **정정** |
| Edwards 외 HF 체크포인트 라이선스 | Apache-2.0(저장소 안 HF 카드 사본) [직접 확인] | 미확인(데이터 노트, 1차 보고서) | 카드 사본 기준 Apache-2.0. HF 원 카드는 미확인 |
| Edwards 외 학습 비용 | RTX 6000 1장 약 18시간(논문 PDF) [직접 확인] | 24 GB GPU 1장 약 18시간(README), ≥ 40 GB GPU 약 18시간(스크립트 docstring) [직접 확인] | **충돌.** 어느 쪽이든 이 Mac에서 재현 불가 |
| Edwards 외 논문 본문 열람 | 저장소 `docs/paper.pdf` 직접 열람 | 개인화 노트는 차단으로 못 열었다고 기록 | 저장소 PDF는 열람됨. arXiv·QMUL판은 미열람 |
| Aria-Duet 첫 음 지연 | M 시리즈 200–300 ms, RTX 4090/5090 10–20 ms [요약만(이전 세션)] | README·코드·스니펫에서 찾지 못함(이번 세션) | **미확인으로 강등** |
| jam_bot 쪽수·기록 | 755–762쪽, DOI 10.5281/zenodo.17811478 [요약만] | 769–776쪽, Zenodo 17706584 [요약만] | **충돌** |
| jam_bot 베이스 모델 | Music Transformer(MIT 기사) [요약만] | AMT 미세조정(NIME 2026 본문 스니펫) [요약만] | NIME 본문이 1차 출처에 가깝다 |
| jam_bot 100/800 ms 목표 | 저장소 문서가 출처 없이 인용 | 1차 출처 못 찾음 | **미확인** |
| 현재 입력→반영 지연 | 평균 1.8초(사용자 제시, START_BUDGET) | 평균 0.9초(반 마디 블록, PC 메시지), 음 입력 0.52–0.55초(PERSONALIZATION_STATUS) [프로젝트 실측] | 측정 단계·조건 차이. 둘 다 13.4M·CPU·가상 포트 기준 |
| Moonbeam 사전학습 규모 | 81.6K시간(README) [직접 확인] | 85k시간(데모 페이지) [직접 확인] | **충돌** |
| MIDI-GPT 가중치 라이선스 | GitHub MIT [직접 확인] | HF 카드 CC BY-NC [요약만(이전 세션)], 이번 세션 확인 불가 | **충돌** |
| WJazzD 라이선스 | ODbL [요약만(이전 세션)] | "Open Access" 표기만 [요약만] | **충돌** |
| ImprovNet 청취 | 28명, p = 0.0037(alphaXiv 요약) [요약만] | 참가자 수 미확인(README) | 요약만 |
| Hung 외 2019 전이학습(arXiv 1908.09484) | arXiv 초록: 동시학습이 낫다 | NTU 초록: 사전학습 후 미세조정이 낫다 | **충돌** |
| Beat This JAAH 박 F1 | 95.1(원 논문 요약) | 95.5(AlignBeat 재보고, arXiv 2510.14391) | 작은 충돌 |
| Cheston 해석가능 모델 | top-1 0.767 | 91.3%(다른 출처) | **충돌** |
| JTD 다운비트 F ≈ 0.408 | [직접 확인·계산(이전 세션)] | [사용자 제공, 이번 세션 미확인] | 검증 수준 충돌 |
| PiJAMA 120 BPM 고정 템포 맵 | 사용자 제공, 미확인(데이터 노트) | A Fable 13곡 MIDI에 120 BPM 기본 템포 하나·4/4·박/손 정보 없음 [프로젝트 실측] | 13곡에서 부분 확인 |
| Art Tatum 곡 수 | 122곡(PiJAMA) [직접 확인] | 118행(PiJAMA-12) [직접 확인] | 충돌 아님. Edwards 외가 앙상블 곡을 뺀 부분집합 |
| ReaLchords 학습 규모 | Hooktheory 19,086곡 [직접 확인 config] | Hooktheory 약 3만 스니펫 [요약만] | 단위 차이(곡 대 스니펫)로 보인다 |
| Aria 크기 | "LLaMA 3.2 (1B) 아키텍처 기반"(README) | 658,538,496개(로컬 체크포인트 집계), 659M(Edwards 카드) | 계열 대 실제 수. 실제는 약 6.6억 |
| 손 분리 RNN 정확도 | 93.25% / 94.47% [요약만(이전 세션)] | 수치 미확인 | 요약만 |

## 8. 확인하지 못한 것

이번 세션은 프록시가 arXiv·Hugging Face·ISMIR·Zenodo·NIME·IEEE·ACM 사이트를 막아 GitHub 위주로만 직접 확인했고, 세션 공용 웹 검색 예산 200회도 도중에 소진됐다. 아래는 그 때문에 남은 빈칸과 확인 방법이다.

| 미확인 항목 | 왜 중요한가 | 확인 방법 |
|---|---|---|
| HF 모델 카드: Aria 가중치 라이선스 필드, Edwards 생성기 원 카드, Moonbeam ComMU LoRA 실재·라이선스, MIDI-GPT 가중치 라이선스, SheetSage2 가중치 라이선스, AMT medium·large 파라미터 수 | 승인·공개 범위 | HF 페이지 직접 열람 |
| jam_bot 100/800 ms 목표, NIME 2026의 M3 Max ms/token, 공개 코드 | C6 목표 설정 | nime.org/proceedings/2026/nime2026_73.pdf, Zenodo 17706584 |
| Aria-Duet 첫 음 지연과 측정 칩 | 2안 지연 상한 | arXiv 2511.01663 본문 |
| ReaLJam·ReaLchords ms 지연, MLX tok/s | 3안 지연 | realjam `benchmark_generation_mlx.py`를 로컬 실행 |
| MuseBarControl 표 수치와 venue, 코드 공개 | 1안 반사실 손실의 효과 크기 | arXiv 2407.04331 본문 |
| Moonbeam ComMU 정량 결과, 839M/309M 공식 수치 | 참조 구현 가치 | arXiv 2505.15559 본문 |
| Function Alignment 표 수치와 라이선스 | concat + LoRA 우위의 크기 | arXiv 2506.15548, 저장소 LICENSE |
| CMT 논문 청취 결과·원 학습 데이터, HSE 2026 학위연구 결과표 | 소형 명시 모델 기준선 | IEEE Access 원문, HF `maxkudryashov/*` 카드 |
| Edwards 외 "새 연주자 추가"·청취 연구 유무(논문 전체) | Mehldau 벡터 방식의 선례 | arXiv 2610.02918 |
| 사전학습 피아노 모델에 AMT식 제어를 PEFT로 이식한 결과 | 1안의 핵심 미지수 | 공개 사례 없음. E1이 첫 측정 |
| 동결 backbone + cross-attention 코드 어댑터의 효과 | 1안 변형 팔 | 공개 사례 없음 |
| 자가 라벨(같은 MIDI에서 조건 추출) 모델의 echo·누수 정량 | 1안 위험 크기 | 공개 사례 없음. E1에서 측정 |
| "재타격 대 진짜 반응"을 분리한 공개 지표, 코드 추종 자체를 블라인드 전문가로 판정한 재즈 연구 | C1–C3 정당화 | 찾지 못함. 이 프로젝트가 정의 |
| 솔로 재즈 피아노(드럼 없음, 루바토)의 박·다운비트 정확도, 재즈 피아노 손 분리 정확도(PM2S, Hadjakos) | 3안 데이터, C5 측정 | 사용자 녹음으로 소규모 검증 |
| Aria-MIDI에 PiJAMA·"A Fable" 포함 여부 | 스타일 대 암기 구분 | Aria-MIDI 메타데이터 대조 |
| M1 Max 지속 학습의 열 스로틀링 곡선, MPS 대 MLX 같은 모델 decode 비교 | 학습·지연 추정의 신뢰도 | asitop·powermetrics로 첫 몇 분과 정상 상태 비교 |
| 심볼릭 피아노에서 소규모 사람 쌍(수십–수백)으로 한 DPO의 필요 쌍 수 | 선호 학습 계획 | 찾지 못함 |
| 티그랑 관련 MIR·생성 연구, 공개 전사 데이터, 아르메니아 선법의 계산적 정의 | 티그랑 목표 정의 | 찾지 못함. 사용자 기준 클립으로 대체 |
| WJazzD·FiloBass·DTL1000·Lakh·GiantMIDI·MidiCaps 라이선스 원문, YouTube 오디오 다운로드의 약관상 허용 여부 | 외부 데이터 승인 | 각 배포처 원문 |
| 사용자 측: min_p 0 청취 판정, 기준 클립, 합격 문장 | 모든 합격 기준의 기준점 | 사용자 결정(2절) |

## 결론

이번 조사가 바꾼 핵심 인식은, 지금까지의 실패들이 "Aria는 코드를 못 따른다"가 아니라 "코드를 가장 약한 자리와 가장 복사하기 쉬운 형태로 넣었다"는 쪽으로 설명된다는 점이다. 프롬프트에 피아노 음으로 넣은 코드는 재타격으로 반응했고, 마지막 블록의 선형 주입은 판별만 배웠으며, 연주자 LoRA는 연주자가 아니라 전사 형식(duration)을 배웠다. 세 증상 모두 문헌에 대응하는 처방이 있다. 제어를 생성 불가능한 별도 어휘로 시간에 맞춰 넣고, 틀린 조건을 벌하며, 공통 도메인 성분과 연주자 성분을 분리하는 것이다. 다만 그 처방들을 사전학습 피아노 모델 하나에 함께 얹은 공개 사례는 없으므로, E1의 3팔 대조가 이 분야의 빈칸을 직접 채우는 첫 측정이 된다.

두 번째 함의는 병목의 위치가 옮겨졌다는 것이다. 1차 보고서는 박·코드가 정렬된 피아노 데이터를 병목으로 봤다. 연주 화성을 조건으로 쓰면 런타임 입력과 같은 함수로 라벨을 만들 수 있어 그 병목은 상당 부분 우회된다. 남는 병목은 데이터가 아니라 **판정 기준**이다. "착지", "재타격이 아닌 반응", "같은 릭", "스윙이 아님", "티그랑 같음"을 사용자가 기준 클립으로 고정하기 전에는 어느 안이 이겼는지도 말할 수 없다. 그래서 가장 값싼 다음 행동은 학습이 아니라 2절의 다섯 결정과 E2 측정기이고, 그 위에서 1안의 한 시간짜리 대조가 이 프로젝트의 방향을 정한다.
