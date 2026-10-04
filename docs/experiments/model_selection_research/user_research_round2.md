# 사용자 2차 딥리서치 원문

2026-10-03, 사용자가 다른 딥리서치 도구로 조사해 채팅에 붙여 넣은 원문이다. 마지막 줄처럼 도중에 연결이 끊긴 상태로 받았다. 원 도구의 출처 표시는 출처 이름만 남았다(링크 없음). 주장별 검증 결과와 결정은 `../MODEL_SELECTION_ROUND2_3.md`에 있다.

---

네. 지금 다시 파보니까 이전 리서치에서 빠진 후보가 실제로 있었고, 몇 개는 기존 retrieval → MINGUS식 소형모델 → AMT 순서를 다시 검토할 정도로 중요합니다.
특히 아래 5개는 네 목표와 직접 겹칩니다.
후보    네 목표와 겹치는 부분    문제    지금 판단
Function Alignment, ISMIR 2025    chord → melody를 직접 학습, pretrained LM + lightweight adapter, 코드/가중치 공개    재즈/비밥 전용 아님, 16분음표 quantization    매우 중요. 새 A/B 후보
Moonbeam conditional, 2025    81.6K시간/18B token MIDI foundation model, chord progression 직접 conditioning, LoRA/PEFT, 공개 checkpoint    309M/839M이라 M1 Max 0.4초가 불확실, ComMU 기반    foundation-model 후보 중 가장 직접적
FGG, ICML 2025    symbolic diffusion인데 chord-conditioned melody/accompaniment, interactive generation demo, 코드/checkpoint 공개    POP909, 재즈 아님, diffusion latency 검증 필요    기존 조사에서 놓친 상당히 중요한 후보
Notochord    MIDI 실시간 생성, <10ms inference response, event-level steering, train/fine-tune 가능    native chord-symbol conditioning이 아님, Lakh 기반    실시간 엔진으로 매우 흥미로움
2026 CMT-on-Jazz 연구    CMT를 WJazzD에서 실제 jazz chord-conditioned solo 모델로 재학습, BebopNet/MINGUS와 동일 데이터 비교    학위 연구라 대규모 청취평가는 제한    당장 실험하기 가장 싼 새로운 자료


가장 놀라운 건 Moonbeam입니다. 이전에 “최신 foundation model에는 chord condition이 없다”고 봤다면 그건 수정해야 합니다. Moonbeam에는 실제로 conditional_gen_commu 브랜치가 있고, 코드 MIDI를 <SOC> ... <EOC> 형태의 condition으로 넣어 생성하는 checkpoint가 공개돼 있습니다. 309M과 839M pretrained checkpoint가 있고, ComMU chord-conditioned fine-tune weight도 공개돼 있습니다. 라이선스도 Apache-2.0입니다. GitHub
이건 네 현재 방식과 근본적으로 다릅니다.
현재 네 모델:
[가이드 voicing을 일반 note event로 prompt]
         ↓
Music Transformer

Moonbeam conditional:
[chord sequence = 명시적 condition]
              ↓
conditional architecture
              ↓
music generation

즉 모델이 chord를 condition이라고 학습했습니다.
다만 309M도 현재 13.4M의 23배 규모라서, M1 Max에서 0.4초 p99를 지키는지는 완전히 별개의 문제입니다. 그래서 “최종 엔진”이라고 바로 정할 수는 없지만, 적어도 최신 foundation model을 코드 조건으로 fine-tune해보는 실험은 이제 충분히 정당합니다. Moonbeam 자체가 81.6K시간, 18B MIDI token 사전학습 모델이고 conditional generation까지 공개하고 있습니다. arXiv
오히려 Function Alignment가 네 문제를 더 정확하게 찌른다
ISMIR 2025 Versatile Symbolic Music-for-Music Modeling via Function Alignment는 이전 조사에서 상당히 중요한 누락입니다.
이 연구가 실제로 하는 task 중 하나가:
Chord → Melody

입니다.
기반 모델은 약 35만 개 이상의 정제 MIDI로 사전학습된 RoFormer이고, 같은 pretrained LM을 chord 쪽과 melody 쪽에 놓은 다음 lightweight adapter로 둘의 함수를 연결합니다. cross-attention 방식과 self-attentive adapter 방식 모두 시험했고, 코드와 pretrained weights도 공개했습니다. ResearchGate
이게 네 상황에 흥미로운 이유는:
generic MIDI prior
        +
chord → melody adapter
        +
bebop adapter

라는 구조를 만들 가능성이 있기 때문입니다.
지금처럼:
generic jazz model
→ Tatum LoRA
→ chord를 prompt로 억지 전달

하는 것보다 훨씬 논리적으로 맞습니다.
그리고 네가 원하는 "교체 가능한 adapter" 구조와도 잘 맞습니다.
FGG는 내가 지금 가장 자세히 다시 볼 후보 중 하나다
ICML 2025의 Efficient Fine-Grained Guidance for Diffusion Model Based Symbolic Music Generation도 꽤 중요합니다.
공개 repo 설명을 보면 단순한 “attribute-controlled music” 정도가 아니라 실제로:
chord progression을 조건으로 melody + accompaniment 생성

을 합니다. 별도 모드에서는 melody + chord를 받아 accompaniment만 생성할 수도 있습니다. POP909 기반 pretrained checkpoint도 공개돼 있습니다. Proceedings of Machine Learning Research
즉:
chord progression
       ↓
symbolic diffusion
       ↓
melody + accompaniment

입니다.
그리고 논문은 interactive music creation과 real-time interactive demo까지 명시합니다. Proceedings of Machine Learning Research
이건 네가 전에 말한 Live Music Diffusion Models와 완전히 다릅니다.
Live Music Diffusion Models는 아주 흥미로운 2026 연구이고 block-wise KV cache, streaming diffusion, consumer hardware live performance까지 해결했지만 audio diffusion입니다. 따라서 네 MIDI → Serum, 정확한 음높이 검사, 코드톤/avoid-note 지표라는 장점을 포기하게 됩니다. arXiv
반면 FGG는 symbolic diffusion입니다.
그래서 FGG는 실제 후보에 넣을 가치가 있습니다.
다만 POP909에서 학습했기 때문에 그대로 들으면 아마 비밥은 아닙니다. 핵심 질문은:
FGG의 chord-control mechanism을 네 235곡/자동 chord-labelled bebop 데이터에 fine-tune하면 어떻게 되는가?

입니다.
Notochord도 내가 처음보다 훨씬 높게 평가하게 됐다
Notochord는 2024 모델이라 최신 SOTA foundation model은 아니지만 네 실시간 요구조건과 거의 정확하게 맞습니다.
공개 연구에서 MIDI next-event response가 10ms 이하이고, 실시간 harmonization, machine improvisation, accompaniment를 목적으로 설계됐습니다. pitch/time/velocity/instrument 각각에 직접 intervention도 할 수 있습니다. checkpoint와 training code가 MIT로 공개되어 있고 custom MIDI dataset fine-tuning도 지원합니다. arXiv
즉 지금 네 219ms p99와 비교하면 inference headroom이 엄청나게 큽니다.
문제는:
“C7” 자체를 이해하도록 학습되어 있지 않다.

입니다.
그래서 그대로는 답이 아니지만,
Notochord architecture
+
bebop MIDI training
+
explicit chord features

같은 방향은 꽤 매력적입니다.
특히 96~128 token을 반마디마다 한꺼번에 생성하는 지금 구조와 달리 실제 MIDI event 단위 실시간 interaction을 염두에 둔 모델이라는 것이 큰 차이입니다.
그리고 아주 중요한 자료 하나를 새로 찾았다
2026년에 나온 연구가 하나 있는데, 거의 네가 하려는 비교 실험을 먼저 해놨습니다.
HSE의 Application of Machine Learning Methods to the Analysis and Generation of Symbolic Jazz Improvisations입니다.
이 연구는 동일한 WJazzD split에서:
BebopNet vs MINGUS vs CMT(Chord-Conditioned Melody Transformer)
를 전부 다시 학습해서 동일 조건으로 비교했습니다.
게다가:
- preprocessing
- generation wrapper
- evaluation pipeline
- trained weights
- metric tables
까지 공개했습니다.
연구 결론은 특정 jazz 전용 모델이 무조건 이긴 게 아니고, 일반 chord-conditioned CMT를 jazz에 학습시켜도 specialized model과 경쟁 가능한 수준이었다는 것입니다. HSE University
이건 네게 굉장히 중요한 근거입니다.
왜냐하면 지금 우리가 생각하는:
“13M 정도 작은 모델이라도 명시적 chord conditioning + jazz data가 있으면 충분하지 않을까?”

라는 가설에 2026년에 실제 독립적인 실험 근거가 생긴 것이기 때문입니다.
심지어 MINGUS의 WJazzD 재학습 checkpoint도 Hugging Face에 공개돼 있습니다. MINGUS는 pitch/duration 각각 4-layer, d_model=200 정도이며, 공개된 재학습은 Colab A100에서 두 모델 합쳐 약 2분, CPU에서도 약 18분이었다고 기록돼 있습니다. Hugging Face
즉 이 모델군은 네 M1 Max에서는 계산량 자체가 거의 문제가 아닙니다.
그래서 지금 판단이 바뀌었다
이전 PR #1673의
1. retrieval hybrid
2. MINGUS식 소형 모델
3. AMT
이라는 순위를 그대로 확정하면 안 됩니다.
현재 발견한 자료까지 넣으면 나는 이제 다음 6개를 동일한 progression에서 직접 A/B해봐야 한다고 봅니다.
1. 현재 13.4M MT
2. BebopNet
3. 2026 jazz-CMT checkpoint / 구조
4. Function Alignment chord→melody
5. Moonbeam-309M conditional
6. FGG chord-conditioned symbolic diffusion
Notochord는 별도로 실시간 architecture 실험에 넣는 것이 맞습니다.
그리고 retrieval은 이들과 비교해서 정말 귀로 좋을 때만 채택하는 겁니다.
“혹시 이미 네 목표를 거의 해결한 모델이 나왔나?”에 대한 현재 답
완전히 turnkey인 모델은 아직 못 찾았습니다.
즉:
C7 입력 → Barry Harris 같은 bebop piano lick → MIDI → <400ms M1 Max

를 공개 checkpoint 하나로 바로 해주는 모델은 아직 발견하지 못했습니다.
하지만 이전 연구에서 놓친, 상당히 가까운 부품들은 분명히 있었습니다.
특히:
Moonbeam = 최신 대규모 symbolic prior + explicit chord conditioning
Function Alignment = pretrained symbolic model에 chord→melody adapter
FGG = chord-conditioned symbolic diffusion + interactive generation
Notochord = 진짜 실시간 MIDI neural generator
2026 jazz-CMT = WJazzD에서 chord-conditioned jazz solo를 직접 학습한 공개 실험/weights
입니다.
그래서 지금은 “13M Music Transformer vs retrieval” 둘 중 하나를 택할 단계가 아닙니다.
최신 코드 조건 모델 3~4개를 실제 네 progression과 네 청취 기준으로 한번 돌려본 뒤 결정하는 게 맞습니다.
특히 나는 다음으로 Moonbeam conditional, Function Alignment, FGG, 그리고 2026 jazz-CMT의 실제 코드·checkpoint·입력 형식·라이선스·M1 Max에서 예상 inference cost까지 하나씩 뜯어보는 것이 이 딥리서치에서 가장 중요하다고 봅니다. 이걸 하면 “논문상 좋아 보인다”가 아니라 네 repo에 실제 붙일 수 있는 후보만 남길 수 있습니다.








Connection interrupted. Waiting for the complete answer
