# 사용자 3차 딥리서치 원문

2026-10-03, 사용자가 다른 딥리서치 도구로 조사해 채팅에 붙여 넣은 원문이다. 원 도구의 인용 표시(내부 참조 번호)는 링크로 쓸 수 없어 지웠다. 주장별 검증 결과와 결정은 `../MODEL_SELECTION_ROUND2_3.md`에 있다.

---

# 코드 진행에서 비밥 MIDI를 실시간 생성하는 모델·방법론 재조사

## 핵심 결론

2026년 10월 3일 기준으로 다시 넓게 조사해보면, **이전 답변에서 빠졌거나 과소평가된 후보가 실제로 있습니다.** 그리고 몇 가지는 단순히 후보 하나를 추가하는 수준이 아니라, 지금 프로젝트의 방향 자체를 바꿀 만합니다.

가장 중요한 새 발견은 **MelodyT5**, **Impro-Visor식 grammar/relative-pitch generation**, **Stochastic Control Guidance(SCG)**, 그리고 이들을 이용한 **대형 teacher → 소형 chord-aware student distillation**입니다. 특히 MelodyT5는 이전 답변에서 사실상 빠져 있었는데, 공개 모델 자체가 명시적으로 **chord progression → melody**에 해당하는 *melodization* task를 학습합니다. 113M 파라미터이고, 261,900개의 melody로 구성된 MelodyHub에서 백만 개가 넘는 task pair로 multi-task pretraining되었으며, melodization 데이터만 약 4.2만 건입니다. 코드와 pretrained weights가 공개되어 있고 repo는 MIT 라이선스입니다.

또 하나는 오히려 훨씬 오래된 **Impro-Visor**입니다. 이것은 최신 foundation model이 아니지만, 사용자가 원하는 문제를 놀라울 만큼 정확하게 해결합니다. 코드 진행을 알고 있는 상태에서 jazz solo를 만들고, relative-pitch grammar, chord/approach-tone constraint, guide-tone generation, motif learning, solo에서 grammar 학습, Markov interval learning, 실시간 trading, MIDI 입력·출력을 이미 갖고 있습니다. Woody Shaw, Wes Montgomery처럼 특정 솔로에서 유도한 grammar도 repo에 포함되어 있으며, chord tone과 approach tone으로 음을 제한하는 generation mode도 존재합니다. 즉 **“코드 → 특정 jazz vocabulary → MIDI → 즉시 재생”**이라는 제품 구조는 신경망보다 먼저 거의 완성되어 있었던 셈입니다.

그리고 이전 답변에서 **FGG는 과대평가했습니다.** 공개 데모의 직접 task는 “chord progression만 받아 melody+accompaniment를 생성”하는 것이 아니라, 주로 **melody와 chord를 받아 accompaniment를 생성**하는 쪽입니다. 따라서 현재 목표인 chord→bebop melody의 직접 후보로는 순위를 내려야 합니다. 반대로 2024 ICML의 SCG는 실제로 chord progression 같은 비미분 가능한 규칙을 pretrained symbolic diffusion model에 **재학습 없이 plug-and-play로 강제**할 수 있도록 설계되어 있어, FGG보다 사용자의 harmonic-control 문제에 방법론적으로 더 중요합니다.

결론부터 순위를 매기면, 지금 제가 실제 repo에 붙이는 실험 순서는 다음처럼 바뀝니다.

| 우선도 | 후보/방법 | chord→melody 직접성 | Bebop/Jazz 적합성 | `<400 ms` 가능성 | 제가 보는 역할 |
|---|---|---:|---:|---:|---|
| **최상** | **MINGUS conditioning + CMT rhythm/pitch factorization을 합친 소형 student** | 매우 높음 | 매우 높게 만들 수 있음 | **높음** | 최종 neural engine |
| **최상** | **Chord-relative retrieval/grammar + tiny reranker** | 매우 높음 | **매우 높음** | **매우 높음** | 최종 hybrid engine |
| **최상** | **Impro-Visor식 grammar를 235곡에서 학습** | 매우 높음 | 높음 | **매우 높음** | neural이 못 잡는 bebop language |
| **높음** | **Notochord + future-chord side channel** | 수정 필요 | fine-tune 필요 | **높음** | streaming engine |
| **높음** | **MelodyT5 jazz fine-tune** | **직접 지원** | fine-tune 필요 | 중간 | benchmark / teacher |
| **높음** | **Moonbeam conditional 309M** | 직접 conditioning | fine-tune 필요 | 낮음~불확실 | 대형 teacher |
| 중간 | Function Alignment | **직접 지원** | fine-tune 필요 | 중간 이하 | chord→melody adapter 연구 |
| 중간 | CMT 원형 | **직접 지원** | K-pop 원형 | 높음 | 작은 architecture baseline |
| 중간 | MINGUS 원형 | **직접 지원** | **Jazz 직접 학습** | 높음 | jazz baseline |
| 연구용 | SCG | chord rule 직접 제어 | style 별도 | 낮음 | constraint/teacher |
| 연구용 | MelodyDiffusion | **직접 지원** | 비-jazz | 낮음 | diffusion baseline |
| 낮음 | FGG | 목표 방향과 다름 | POP 계열 | 불확실 | accompaniment 쪽 |
| 낮음 | MuseBarControl | chord control | 비-jazz | **매우 낮음** | latency 때문에 제외 |
| 아이디어용 | MusicRFM | audio steering | 비-symbolic | 목표와 다름 | activation steering 아이디어 |

**“이미 C7을 넣으면 Barry Harris 같은 MIDI lick이 `<400 ms` 안에 바로 나오는 공개 checkpoint가 존재하느냐?”라는 질문에는 여전히 “아직 확인하지 못했다”가 정확합니다.** 하지만 더 중요한 결론은, 그 checkpoint를 찾는 것보다 **이미 공개된 기술을 조합해서 현재 13.4M 모델보다 목표에 훨씬 직접적인 10–20M급 엔진을 만드는 길이 꽤 명확해졌다는 것**입니다.

## 목표를 다시 정의하면 모델 선택이 달라진다

사용자의 문제는 일반적인 “symbolic music generation”이 아닙니다. 실제 요구사항을 분리하면 다음 네 문제가 동시에 있습니다.

첫째, 입력은 단순한 MIDI prompt가 아니라 **미래까지 알려진 chord timeline**입니다. 둘째, 출력은 임의의 음악이 아니라 **monophonic 또는 piano-oriented bebop vocabulary**여야 합니다. 셋째, 결과를 MIDI note 단위로 검사할 수 있어야 해서 audio model은 불리합니다. 넷째, M1 Max에서 전체 request의 p99가 약 400 ms 이하라는 강한 engineering constraint가 있습니다.

이 기준으로 보면 모델 크기나 최신 논문 순위보다 **conditioning topology**가 훨씬 중요합니다. MINGUS는 예를 들어 pitch/duration 생성에서 **현재 chord뿐 아니라 다음 chord, bass line, measure 안의 위치**를 feature로 사용합니다. 즉 사용자의 기존 방식처럼 chord voicing을 일반 MIDI note prompt 안에 넣어서 모델이 “이것이 조건인지 연주할 음인지” 스스로 추론하게 하는 것보다 문제 구조가 훨씬 직접적입니다.

CMT 역시 처음부터 chord-conditioned melody generation을 목표로 만들어져, chord progression을 조건으로 **rhythm decoder를 먼저 학습한 뒤 pitch decoder를 학습하는 두 단계 구조**를 사용합니다. 공개 구현은 melody와 chord를 별도 MIDI track으로 받아들이고, 16분음표 단위 grid와 12-key pitch augmentation을 지원합니다.

이 두 모델을 합쳐 생각하면 상당히 강한 구조가 나옵니다.

```text
chord symbols + chord timing
        │
        ├── current chord
        ├── next chord
        ├── next-next chord
        ├── harmonic function / key
        └── metrical position
                ↓
        small chord encoder
                ↓
       ┌─────────────────┐
       │ rhythm decoder  │
       └─────────────────┘
                ↓
     rhythm + chord context
                ↓
       ┌─────────────────┐
       │ pitch decoder   │
       └─────────────────┘
                ↓
   chord-relative constraint
                ↓
        4~16 candidates
                ↓
      tiny bebop reranker
                ↓
               MIDI
```

이 구조는 Moonbeam을 축소한 것도 아니고 MINGUS를 그대로 복제한 것도 아닙니다. **MINGUS의 미래-harmony-aware conditioning + CMT의 rhythm/pitch factorization + 사용자가 이미 갖고 있는 작은 Transformer latency budget**을 결합하는 것입니다. 이 조합이 현재 조사에서 최종 엔진으로 가장 설득력 있습니다. MINGUS와 CMT는 둘 다 범용 foundation model보다 훨씬 직접적으로 chord-conditioned melodic generation을 모델링합니다.

특히 사용자가 이미 **13.4M 모델로 약 219 ms p99**를 측정했다는 점이 중요합니다. 이는 300M~800M급 모델로 갈 필요가 없다는 강력한 시스템상의 힌트입니다. 모델을 더 크게 만드는 대신 **모델이 해결해야 하는 불확실성을 chord encoder와 constraint layer로 없애는 것**이 400 ms 목표에는 훨씬 유리합니다.

## 이번 조사에서 실제로 중요한 새 후보들

### MelodyT5는 이전 조사에서 가장 큰 누락이다

MelodyT5는 “generic melody model” 정도가 아닙니다. 논문에서 정의한 일곱 task 중 하나인 **melodization**은 chord symbols를 남기고 note들을 rest로 바꾼 score를 입력으로 준 다음, 그 chord progression에 맞는 melody를 복원·생성하는 task입니다. 즉 논문상 task definition 자체가 사용자의 **chord → melody**와 일치합니다.

모델은 patch-level encoder/decoder와 character-level decoder를 가진 encoder-decoder Transformer이고, 9-layer patch encoder/decoder가 weight sharing을 하며 3-layer character decoder, hidden size 768, 총 **113M parameters**입니다. 입력 표현은 ABC notation이고, chord symbols를 표현할 수 있습니다.

더 흥미로운 것은 data regime입니다. MelodyHub에는 261,900개의 unique melodies와 1,067,747개의 task instances가 있고, melodization pair도 약 41,937개 있습니다. 논문은 multi-task pretraining이 특히 데이터가 적은 melodization 같은 task에서 task-specific training보다 큰 이점을 보였다고 보고합니다.

또 objective evaluation에서는 CMT와 chord→melody 조건을 맞춰 비교했고, MelodyT5가 일부 harmonic metric에서는 CMT보다 낫고, CTnCTR에서는 CMT가 약간 앞서는 식의 결과가 나왔습니다. 즉 CMT를 압도한 것은 아니며, 이것이 오히려 사용자의 프로젝트에는 중요한 정보입니다. **113M pretrained generalist조차 잘 설계된 task-specific CMT에 harmonic chord-tone metric에서 반드시 이기지 않습니다.**

코드와 pretrained weights가 공개되어 있고 공식 repo는 MIT 라이선스입니다.

따라서 MelodyT5를 최종 엔진으로 바로 넣기보다는 다음 역할이 훨씬 좋습니다.

> **Chord progression → 다수의 melody candidate를 offline 생성 → bebop 기준으로 필터링 → 사용자의 10–20M student에 distillation**

이렇게 하면 261K melody 규모에서 배운 melodic prior는 가져오면서 production에서는 113M + character-level decoding cost를 지불하지 않아도 됩니다.

### Impro-Visor는 “오래돼서 제외할 것”이 아니라 핵심 아이디어다

이번 재조사에서 가장 관점이 바뀐 대상은 Impro-Visor입니다.

Impro-Visor는 오래전부터 jazz chord progression 위에서 solo를 생성하도록 만들어졌고, 시간이 지나면서 **relative-pitch grammar learning, guide-tone generation, transformational grammar, motifs, Markov-chain interval learning, theme weaving, chord-tone/approach-tone restriction, real-time trading**이 추가됐습니다. repo release history에는 Woody Shaw와 Wes Montgomery의 solo에서 유도된 grammar, chord+approach grammar, Jerry Bergonzi 방식 등을 위한 grammar까지 명시되어 있습니다.

이것이 사용자의 235곡 데이터와 만나는 순간 재미있어집니다.

현재 생각은 아마 다음과 비슷했을 것입니다.

```text
235곡
  ↓
Music Transformer fine-tune
  ↓
확률적 melody
```

하지만 jazz vocabulary에는 다른 decomposition이 가능합니다.

```text
235곡
  ↓
phrase segmentation
  ↓
chord-relative normalization
  ↓
rhythm / interval / enclosure / target-note pattern 추출
  ↓
grammar + retrieval index
  ↓
현재 chord context에 맞게 재결합
```

Impro-Visor의 relative-pitch grammar와 chord-tone/approach-tone 제약은 이런 방향이 실제로 작동 가능한 system design이라는 선례를 줍니다.

사용자의 경우에는 옛날 probabilistic grammar를 그대로 최종 출력기로 쓸 필요도 없습니다. **grammar/retrieval을 candidate generator로 사용하고 현재 13.4M Transformer 또는 더 작은 scorer가 최종 선택만 하게 만들면 됩니다.**

예를 들어 동일 progression에서 16개 candidate를 아주 빠르게 만들고,

```text
score =
    0.30 × bebop_style_score
  + 0.20 × strong_beat_harmony
  + 0.15 × next_chord_resolution
  + 0.15 × contour_score
  + 0.10 × repetition_penalty
  + 0.10 × neural_logprob
```

처럼 rerank할 수 있습니다.

이때 중요한 점은 위 coefficient 자체가 정답이라는 뜻이 아니라, **“모든 것을 한 generative model에게 학습시키는 문제”를 “candidate generation + measurable selection”으로 바꾼다**는 점입니다.

이 방식은 235곡처럼 데이터가 제한된 상황에서는 특히 매력적입니다. 원하는 artist vocabulary가 corpus 안에 있으면 generator가 모델 parameter 안에 그것을 희석시킬 필요가 없기 때문입니다. 반대로 retrieval 그대로 출력하면 copy 문제와 연결부 문제가 생기므로, relative-pitch transform, motif mutation, rhythm alteration, neural reranking을 결합하는 것이 좋습니다.

### SCG는 “avoid-note/chord-tone metric”을 모델 밖으로 꺼내는 방법이다

ICML 2024의 **Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion**은 이번 문제와 상당히 직접적입니다. 저자들은 chord progression, note density처럼 일반적으로 gradient를 직접 흘릴 수 없는 음악 규칙을 diffusion generation에 적용하기 위해 **Stochastic Control Guidance**를 제안했습니다. 규칙을 differentiable neural loss로 바꿀 필요 없이 forward evaluation만 있으면 됩니다.

알고리즘의 핵심은 각 diffusion sampling step에서 여러 다음 상태 후보를 만들고, clean sample을 예측한 뒤, **규칙을 가장 잘 만족하는 후보를 선택**하는 것입니다. 논문은 chord progression rule도 실제 대상으로 다루며, 기존 gradient guidance가 black-box chord rule에는 적용되지 않는 문제를 명시적으로 지적합니다.

다만 **SCG 자체를 production engine으로 쓰라는 뜻은 아닙니다.** 여러 realization을 diffusion step마다 평가해야 하므로 `<400 ms`라는 목표에는 방향이 좋지 않습니다. 제가 가져오고 싶은 것은 그 철학입니다.

사용자의 Transformer에서는 훨씬 싸게 구현할 수 있습니다.

```text
logits from model
      ↓
metrical-position-aware harmony filter
      ↓
soft/hard pitch mask
      ↓
sample
```

강박적으로 “C7이면 C,E,G,B♭만 허용”해서는 비밥이 죽습니다. 대신 제안하는 규칙은 예를 들어 다음과 같습니다.

```text
strong beat / phrase landing:
    chord tone 또는 지정 tension에 강한 prior

weak eighth-note:
    chromatic notes 허용

chord change 직전:
    next chord target으로 semitone / whole-tone approach 허용

next chord downbeat:
    resolution reward

unresolved outside note:
    penalty
```

즉 사용자가 이미 계산하는 chord-tone/avoid-note metric을 **평가용 metric에서 decoding-time controller로 승격**시키는 것입니다.

이것은 새 foundation model을 찾는 것보다 훨씬 큰 성능 개선을 가져올 가능성이 있습니다. 모델이 “어떤 음이 harmonic하게 가능한가”까지 매번 학습할 필요가 없어지고, capacity를 실제 phrase vocabulary와 rhythm/contour에 집중할 수 있기 때문입니다. SCG가 보여주는 핵심 역시 pretrained generator와 별도의 rule evaluator를 분리할 수 있다는 점입니다.

### MusicRFM은 지금 모델에도 적용해볼 만한 2026식 steering 아이디어다

ICLR 2026의 **MusicRFM**은 frozen autoregressive music model의 hidden activation에서 특정 musical concept 방향을 찾아 inference 때 해당 방향을 다시 주입합니다. 저자들은 note, chord, tempo 같은 concept에 대한 lightweight RFM probe를 학습하고, generation 단계에서 per-step optimization 없이 internal activation을 steer합니다. 특정 note generation accuracy를 실험상 0.23에서 0.82로 높이면서 text-prompt adherence 변화는 약 0.02 수준으로 유지했다고 보고합니다.

하지만 바로 적용할 checkpoint는 아닙니다. 실험 대상은 MusicGen 계열의 **audio autoregressive model**이고 symbolic MIDI model이 아닙니다.

그럼에도 방법은 흥미롭습니다. 현재 13.4M Transformer의 layer hidden state에서 다음 concept probe를 학습할 수 있습니다.

```text
"dominant chord tone"
"chromatic approach"
"enclosure"
"ascending"
"descending"
"phrase ending"
"next-chord resolution"
```

그리고 특정 layer activation에 steering direction을 더하는 실험을 할 수 있습니다.

다만 이건 **explicit chord conditioning을 먼저 해결한 뒤** 해야 합니다. 입력 구조가 잘못된 모델의 hidden state를 steering해서 보정하는 것보다 chord encoder를 붙이는 것이 훨씬 단순합니다. MusicRFM은 후순위 R&D입니다.

## 이전 후보들을 다시 뜯어보니 순위가 상당히 바뀐다

### Moonbeam은 중요하지만 최종 엔진보다는 teacher가 더 맞다

Moonbeam은 여전히 매우 중요한 후보입니다. 81.6K hours의 MIDI와 약 18B tokens 규모로 pretraining한 symbolic MIDI foundation model이며, conditional generation downstream setup을 공개했습니다.

공개 repo에는 실제로 `conditional_gen_commu` 경로가 있고, conditional inference에서 chord와 metadata를 Transformer에 추가하는 옵션이 있으며 LoRA/PEFT도 제공됩니다. 즉 “최신 symbolic foundation model에는 chord condition이 없다”는 식의 결론은 틀립니다.

그래서 architecture 관점에서는 매우 매력적입니다.

```text
large generic MIDI prior
          +
explicit musical condition
          +
small PEFT adaptation
```

그러나 **LoRA가 작은 것과 inference model이 작은 것은 전혀 다른 문제**입니다. adapter parameter만 적을 뿐 generation 때는 base model 전체를 실행해야 합니다. 따라서 사용자가 원하는 `<400 ms p99` 관점에서 Moonbeam conditional을 13.4M model의 drop-in replacement로 보는 것은 위험합니다. 공개 결과에서 M1 Max에서 해당 latency를 보장하는 benchmark는 확인되지 않았습니다.

제가 지금 Moonbeam을 쓴다면 production이 아니라 다음 역할입니다.

```text
수십만 개의 synthetic chord progressions
              ↓
Moonbeam conditional
              ↓
수백만 candidate phrases
              ↓
bebop harmonic/style filter
              ↓
high-quality pseudo dataset
              ↓
10~20M student distillation
```

이렇게 하면 foundation model의 broad prior와 사용자의 runtime constraint를 동시에 만족시킬 가능성이 훨씬 큽니다.

### Function Alignment는 맞는 문제를 풀지만 “소형 모델”은 아니다

ISMIR 2025의 Function Alignment는 정말로 **Chord → Melody**를 downstream task로 다룹니다. pretrained symbolic LM을 backbone으로 사용하고 cross-attention 또는 self-attentive adapter를 통해 서로 다른 music functions를 정렬하며, chord→melody adapter와 weights를 공개했습니다.

그러나 이전 답변에서 “lightweight adapter”라는 말이 자칫 전체 inference도 가볍다는 인상을 줄 수 있었습니다. 실제 backbone은 global decoder가 12 layers, hidden size 768, FFN 3072, 12 heads이며 local encoder/decoder도 따로 있습니다. adapter 자체는 작아도 base를 계속 실행해야 합니다.

더 중요한 문제는 chord→melody downstream training이 **Nottingham 1,020곡** 기반이고, jazz/bebop corpus가 아닙니다. representation 역시 16th-note quantized symbolic sequence입니다.

따라서 Function Alignment에서 제가 가져오고 싶은 것은 checkpoint보다 다음 아이디어입니다.

```text
generic musical prior를 그대로 보존
         +
chord→melody mapping만 adapter로 학습
         +
style adapter를 별도로 교체
```

즉 사용자가 원했던 “교체 가능한 Tatum/Bebop adapter” 구조와 논리적으로 잘 맞습니다. 다만 final M1 engine 후보라기보다 **architecture reference와 teacher benchmark**로 보는 것이 더 현실적입니다.

### Notochord는 여전히 실시간 backbone으로 아주 강하다

Notochord는 polyphonic/multitrack MIDI의 다음 event를 직접 모델링하면서 pitch, time, velocity, instrument 같은 sub-event attribute를 intervention할 수 있게 설계되었습니다. Lakh 계열의 약 10만 곡 MIDI로 학습되었고, 논문은 interactive response를 10 ms 이하 수준으로 보고하며 code, checkpoints, training components가 공개되어 있습니다. 공식 repo는 MIT 라이선스입니다.

하지만 여기에도 중요한 함정이 있습니다.

**“<10 ms response” ≠ “96~128개의 MIDI token으로 구성된 반 마디를 <10 ms에 완성한다.”**

Notochord의 강점은 event-by-event interaction입니다. 사용자의 현재 benchmark는 일정 길이 phrase/chunk를 완성하는 전체 latency입니다. 따라서 직접 숫자를 비교하면 안 됩니다.

또한 Notochord는 미래 chord symbol timeline을 native condition으로 받는 모델이 아닙니다. 그러므로 사용자의 문제에서는 다음 변경이 필요합니다.

```text
Notochord event state
       +
current chord embedding
       +
next chord embedding
       +
time-to-next-chord
       ↓
next event prediction
```

이렇게 만들면 상당히 매력적입니다. 특히 performer가 다음 note를 계속 요구하는 streaming interface라면 Moonbeam/Function Alignment보다 architecture 자체가 제품 요구와 훨씬 잘 맞습니다.

### FGG는 이전 답변에서 역할을 잘못 잡았다

FGG는 symbolic diffusion에 fine-grained conditioning과 sampling control을 추가한 흥미로운 연구입니다. 그러나 공개 task 설명을 확인하면 대표적인 조건부 generation은 **melody와 chord를 조건으로 accompaniment를 생성**하는 형태입니다. chord progression 하나만 받아 사용자가 원하는 bebop melody를 생성하는 turnkey model로 분류하는 것은 부정확합니다.

FGG가 쓸모없다는 뜻은 아닙니다. scale/chord constraints와 fine-grained control의 방식은 유용합니다. 다만 **CMT, MelodyT5, MINGUS처럼 입력-출력 방향 자체가 정확한 모델보다 먼저 실험할 이유가 없습니다.**

### MuseBarControl도 chord control은 좋지만 latency에서 탈락한다

MuseBarControl은 bar-level chord/control을 상당히 강하게 집행할 수 있어 논문만 보면 매력적입니다. 그러나 보고된 inference는 RTX 4090에서 MuseCoco 기반 generation이 평균 약 3분, 설정에 따라 약 4~6분 수준까지 올라갑니다. 사용자의 400 ms p99 문제에서는 architecture baseline으로도 거의 탈락입니다.

다만 이 연구의 **counterfactual control learning** 아이디어, 즉 동일 music context에서 조건을 바꾼 반례를 만들어 모델이 control signal을 무시하지 못하게 하는 방식은 소형 model training에 옮길 가치가 있습니다.

예를 들어 같은 melody prefix에 대해:

```text
조건 A: G7
조건 B: Db7
```

를 만들어 pitch target을 다르게 두면, 모델이 chord embedding을 장식품처럼 무시하고 melody prior만 따라가는 문제를 줄일 수 있습니다.

### D3PIA와 MelodyDiffusion도 있지만 역할이 다르다

2026년 D3PIA는 lead sheet, 즉 **melody + chord constraints → piano accompaniment** 문제를 diffusion으로 다룹니다. 따라서 사용자의 chord→solo direction과 반대에 가깝습니다. chord와 melody의 local alignment를 강조한다는 점은 참고할 수 있지만 final candidate 우선순위는 낮습니다.

반대로 MelodyDiffusion은 이름 그대로 **Chord-Conditioned Melody Generation**을 diffusion 방식으로 다루므로 task 방향은 정확합니다. 다만 앞서 나온 CMT와 유사한 비-jazz chord-conditioned melody 문제이고, diffusion sampling이라는 점에서 현재의 strict real-time budget에는 autoregressive small model보다 불리할 가능성이 높습니다.

그래서 이 계열은 quality ceiling을 보는 offline baseline으로는 의미가 있지만 production engine의 첫 후보는 아닙니다.

## 모델보다 더 중요한 방법론이 있다

제가 이번 조사 후 가장 강하게 바뀐 판단은 이것입니다.

**사용자의 목표는 “더 좋은 pretrained model을 찾으면 해결되는 문제”가 아닐 가능성이 큽니다.**

오히려 다음 네 가지를 결합했을 때 목표에 가장 가까워집니다.

### Chord-relative representation으로 235곡의 정보량을 몇 배로 늘린다

CMT 공개 구현 자체가 데이터를 12개 key로 transpose하는 augmentation을 지원합니다. 이는 chord-conditioned generation에서 absolute pitch보다 transposition invariance가 얼마나 중요한지를 잘 보여주는 설계입니다.

사용자는 이보다 한 단계 더 나갈 수 있습니다.

현재:

```text
C4  D4  Eb4  E4  G4
over C7
```

대신 내부적으로:

```text
0   2   b3   3   5
relative to chord root
```

같은 representation을 사용할 수 있습니다.

그리고 chord도:

```text
G7 → Cmaj7
```

를 단순 symbol 두 개가 아니라:

```text
quality = dominant7
function = V
next_function = I
root_motion = P4 up
time_to_change = ...
```

같은 feature로 표현합니다.

그렇게 하면 C7 위에서 학습한 enclosure를 F7, Bb7, Eb7에서 다시 처음부터 학습할 필요가 없습니다. 235곡이라는 absolute MIDI corpus가 **harmonic-function vocabulary corpus**로 바뀝니다.

특히 MINGUS가 current chord뿐 아니라 following chord와 measure position까지 사용한다는 점이 이 설계를 뒷받침합니다.

### 현재 모델의 가장 큰 약점은 “chord를 prompt note로 흉내낸 것”일 가능성이 높다

현재 방식이 다음과 같다면:

```text
[guide voicing notes]
       ↓
same MIDI token stream
       ↓
Music Transformer
```

모델 입장에서는 guide voicing도 일반 MIDI event와 같은 언어에 있습니다.

대신 다음처럼 바꾸는 것이 맞습니다.

```text
melody history ────────────────┐
                               │
chord timeline → chord encoder ├→ decoder
                               │
metrical state ────────────────┘
```

cross-attention까지 부담되면 훨씬 싸게 구현할 수도 있습니다.

```text
h_t = transformer(melody)
c_t = embedding(current_chord, next_chord, beat)
h'_t = h_t + W(c_t)
logits = head(h'_t)
```

혹은 FiLM처럼:

```text
h'_t = γ(c_t) ⊙ h_t + β(c_t)
```

를 사용할 수 있습니다.

이 작은 변경이 300M foundation model로 교체하는 것보다 먼저입니다.

Function Alignment도 본질적으로 musical function 간 연결을 adapter로 따로 학습하는 방향을 택하고 있습니다.

### Rhythm과 pitch를 분리하는 CMT 아이디어가 비밥에 특히 잘 맞는다

CMT는 rhythm decoder를 먼저 훈련한 뒤 pitch decoder가 그 rhythm representation을 활용하게 합니다.

사용자에게 이 방식은 추가 장점이 있습니다.

현재 single-token model은 동시에:

```text
어디에서 note를 칠지
+
몇 박 유지할지
+
어떤 pitch인지
+
chord와 맞는지
+
phrase가 어디로 가는지
```

를 해결해야 합니다.

대신:

```text
Stage A
chord + meter
   ↓
bebop rhythm skeleton

Stage B
rhythm + chord + next chord
   ↓
pitch / chromatic realization
```

로 나누면 Stage B에 harmonic constraint를 매우 쉽게 적용할 수 있습니다.

더 극단적으로는 rhythm을 neural generation조차 하지 않고 corpus에서 retrieve한 뒤 pitch만 neuralize할 수도 있습니다.

### Retrieval은 “최종 답”이 아니라 vocabulary memory로 써야 한다

이전의 “retrieval vs generative model” 구도 자체가 잘못됐을 가능성이 큽니다.

다음처럼 쓰면 둘은 경쟁 관계가 아닙니다.

```text
                 ┌→ neural generator ──┐
chord context ───┤                     ├→ reranker → MIDI
                 └→ lick retrieval ────┘
```

retrieval index의 key를 단순 chord text가 아니라 다음처럼 만듭니다.

```text
[previous harmonic function]
[current quality]
[next quality]
[root movement]
[bar position]
[phrase position]
[target degree]
[rhythm fingerprint]
```

그리고 모든 lick을 chord-relative interval로 저장합니다.

예를 들어 한 solo의 `Dm7 G7 Cmaj7` lick을 그대로 저장하는 것이 아니라,

```text
ii7 → V7 → Imaj7
entry degree
exit degree
relative notes
rhythmic cells
approach structure
```

로 저장합니다.

그러면 같은 vocabulary를 모든 key와 유사 progression에서 reuse할 수 있습니다.

이 방식은 Impro-Visor가 decades 동안 발전시킨 relative-pitch grammar, motif learning, chord/approach constraints와 구조적으로 매우 가깝습니다.

## M1 Max와 400 ms를 기준으로 다시 순위를 매기면

논문 quality와 deployment quality를 분리해야 합니다.

아래의 `<400 ms` 평가는 **공개 M1 Max benchmark가 있다는 뜻이 아니라**, 공개 architecture와 사용자가 이미 측정한 13.4M/219 ms baseline을 기준으로 한 engineering judgement입니다. 실제 p99는 반드시 같은 output length와 runtime으로 측정해야 합니다.

| 후보 | 규모/특성 | 전체 phrase `<400ms` 전망 | 이유 |
|---|---|---|---|
| **Relative retrieval + grammar** | 비신경망/소형 scorer | **매우 좋음** | generation 대부분이 lookup/transform |
| **CMT derivative** | 작은 task-specific Transformer | **좋음** | 거대 prior 없이 chord→melody 직접 학습 |
| **MINGUS derivative** | small jazz Seq2Seq | **좋음** | jazz + explicit chord features |
| **현재 13.4M + chord adapter** | 13.4M + 작은 side module | **가장 안전** | 현재 이미 p99 219ms 측정 |
| **Notochord derivative** | event streaming | **좋음** | 논문 자체가 low-latency interaction 목표 |
| MelodyT5 | 113M + hierarchical decoding | 불확실 | 직접 task는 맞지만 13.4M보다 큼 |
| Function Alignment | 12×768급 backbone + adapter | 불확실/위험 | adapter가 작아도 base inference는 큼 |
| Moonbeam 309M | foundation AR | **위험** | model size와 AR decoding |
| Moonbeam 839M | foundation AR | 매우 위험 | production latency 목적과 반대 |
| MelodyDiffusion | diffusion | 위험 | iterative sampling |
| SCG | diffusion + multi-sample guidance | 매우 위험 | 각 step에서 여러 realization 평가 |
| MuseBarControl | iterative control | 탈락 | 4090에서도 분 단위 보고 |

여기에서 중요한 것은 **메모리와 latency를 분리하는 것**입니다.

M1 Max에 113M이나 309M model이 “들어가는가”는 큰 문제가 아닐 수 있습니다. 그러나 사용자의 핵심 constraint는 모델이 메모리에 올라가느냐가 아니라 **짧은 lick 전체를 p99 400 ms 안에 생성하느냐**입니다. Autoregressive model은 출력 token 수가 직접 latency에 영향을 주므로 96~128 token chunk라면 model size보다 decode loop가 문제일 수 있습니다.

따라서 반드시 다음 조건으로 측정해야 합니다.

```text
warm-up 충분히 수행
same progression
same output duration
same token count cap
same sampling strategy

≥ 1000 requests

report:
p50
p95
p99
max
tokens/sec
time-to-first-note
time-to-complete-phrase
```

Notochord처럼 streaming model에는 `time-to-first-note`가 중요하고, 현재 half-bar pre-generation 방식에는 `time-to-complete-phrase`가 중요합니다. 두 숫자를 섞어 비교하면 안 됩니다. Notochord 논문의 10 ms 이하 응답은 전자에 해당합니다.

## 지금 실제로 만들 모델은 이것이라고 본다

이번 리서치 전체를 종합하면, 저는 더 이상 “Moonbeam vs MINGUS vs retrieval 중 하나를 고르자”고 권하지 않습니다.

**최종 목표 모델은 공개 모델 하나가 아니라 다음 hybrid architecture가 가장 강합니다.**

```text
                FULL CHORD TIMELINE
                        │
          ┌─────────────┴─────────────┐
          │                           │
   chord/function encoder       phrase retrieval
          │                    relative harmony
          │                           │
          ↓                           ↓
     rhythm decoder              lick candidates
          │                           │
          └─────────────┬─────────────┘
                        ↓
             small pitch decoder
            current + future chord
                        ↓
             harmonic controller
       strong/weak beat dependent mask
                        ↓
                N candidates
                        ↓
              tiny style reranker
                        ↓
                       MIDI
```

여기에 각 기존 연구에서 **좋은 부분만** 가져옵니다.

| 출처 | 가져올 것 |
|---|---|
| **MINGUS** | current/next chord, meter-position conditioning |
| **CMT** | rhythm→pitch factorization, 12-key augmentation |
| **Impro-Visor** | relative-pitch grammar, motif, guide-tone, approach-tone vocabulary |
| **Notochord** | event-level low-latency generation/interface philosophy |
| **SCG** | generator와 black-box musical rule scorer의 분리 |
| **Function Alignment** | generic prior와 task adapter를 분리하는 방법 |
| **MelodyT5** | large chord→melody teacher prior |
| **Moonbeam** | very-large MIDI prior와 conditional generation teacher |

그리고 production model 크기는 **현재의 13.4M에서 크게 벗어나지 않는 것**부터 시작하는 게 합리적입니다.

가장 먼저 할 비교는 사실 Moonbeam이 아닙니다.

### 가장 가치 있는 첫 neural 실험

현재 13.4M 모델을 그대로 두고 guide-voicing prompt를 없앤 다음,

```text
current chord
next chord
time to chord change
beat position
key / harmonic function
```

을 별도 embedding으로 각 token hidden state에 넣습니다.

이것만으로:

```text
Current 13.4M prompt-conditioned MT
vs
13.4M explicit chord-conditioned MT
```

를 비교합니다.

여기서 큰 차이가 난다면 대형 모델 조사 대부분이 불필요해집니다.

### 가장 가치 있는 두 번째 neural 실험

CMT처럼 rhythm과 pitch를 분리합니다.

```text
chords → rhythm
chords + rhythm → pitch
```

CMT는 이 task decomposition을 직접 채택한 공개 chord-conditioned melody 모델입니다.

### 가장 가치 있는 비신경망 실험

235곡을 chord-relative phrase DB로 변환하고 현재 progression에 대해 8~32개 후보를 retrieve/transform합니다.

그 결과를 사용자의 현 모델이 rerank합니다.

이 실험이 neural generator보다 청취평가에서 이기면 retrieval을 버릴 이유가 없습니다. 다만 raw-copy retrieval이 아니라 **transposition-invariant generative retrieval**이어야 합니다.

### 가장 가치 있는 foundation-model 실험

Moonbeam과 MelodyT5를 production에 붙이지 말고 **teacher**로 사용합니다.

MelodyT5는 chord→melody task 자체가 이미 존재하고, Moonbeam은 훨씬 큰 MIDI pretraining prior와 conditional path를 갖고 있습니다.

두 모델에 같은 synthetic progression bank를 주고 후보를 많이 만든 뒤, 사용자의 harmonic metrics와 style discriminator로 상위 결과만 남깁니다.

```text
teacher candidates
       ↓
harmonic validity
       ↓
bebop-style validity
       ↓
novelty / anti-copy check
       ↓
pseudo-label dataset
       ↓
small student
```

이것이 **“foundation model quality를 얻으면서 M1 Max latency는 작은 모델 수준으로 유지하는”** 가장 현실적인 경로입니다.

### 최종 A/B 풀

따라서 실제 benchmark에 남길 후보는 저는 이렇게 압축하겠습니다.

| 테스트 | 목적 | 제 판단 |
|---|---|---|
| **A: 현재 13.4M** | baseline | 반드시 유지 |
| **B: 13.4M + explicit chord side-channel** | conditioning 효과 분리 | **최우선** |
| **C: B + harmonic constrained decoding** | chord adherence | **최우선** |
| **D: CMT-style rhythm→pitch small model** | architecture 변경 효과 | **최우선** |
| **E: MINGUS-style future-chord model** | jazz-specific conditioning | **최우선** |
| **F: chord-relative retrieval/grammar + reranker** | style fidelity/latency | **최우선** |
| **G: Notochord + chord context prototype** | true streaming | 높음 |
| **H: MelodyT5 jazz adaptation** | pretrained chord→melody ceiling | 높음 |
| **I: Moonbeam conditional** | foundation teacher ceiling | 높음 |
| **J: Function Alignment** | adapter paradigm | 중간 |
| SCG/MelodyDiffusion | diffusion ceiling | 연구용 |
| FGG/D3PIA | wrong-direction comparison | 후순위 |
| MuseBarControl | latency | 제외 |

그리고 evaluation은 단순 perplexity가 아니라 적어도 다음 다섯 축으로 나누는 것이 좋습니다.

**Harmonic correctness**에서는 strong-beat chord-tone/tension rate, next-chord landing, unresolved chromatic note를 봅니다. **Bebop language**에서는 사용자의 235곡에서 추출한 interval/rhythm/motif 통계 또는 별도 style classifier를 사용합니다. **Musicality**는 blind listening A/B로 평가합니다. **Originality**는 training phrase와의 nearest-neighbor similarity를 측정합니다. 마지막으로 **engineering**에서는 동일 길이 generation에 대한 M1 Max p50/p95/p99를 기록합니다.

이렇게 해야 “Moonbeam이 더 똑똑하다”, “retrieval이 더 jazz 같다”, “MINGUS가 작다” 같은 추상적 비교가 아니라 실제 제품 목표로 순위를 정할 수 있습니다.

가장 중요한 최종 판단은 이것입니다.

**내가 지금 이 시스템을 만든다면 Moonbeam 309M으로 갈아타지 않습니다.**

먼저 현재 13.4M에 **명시적 current/next chord conditioning**을 넣고, **chord-relative representation**, **CMT식 rhythm/pitch 분리**, **metrical harmonic constraint**를 적용합니다. 동시에 235곡으로 **Impro-Visor식 relative grammar/retrieval engine**을 만들어 같은 progression에서 경쟁시킵니다. MINGUS를 세 번째 direct baseline으로 둡니다.

그 다음 MelodyT5와 Moonbeam은 **“최종 모델 후보”가 아니라 “작은 모델이 도달해야 할 teacher ceiling”**으로 사용합니다. MelodyT5는 113M이면서 이미 chord→melody를 학습했고, Moonbeam은 훨씬 큰 MIDI corpus를 본 conditional foundation model이므로 이 역할에 정확히 맞습니다.

그리고 이전 리서치에서 특히 수정해야 하는 부분은 세 가지입니다. **FGG를 chord→melody direct candidate로 높게 둔 것은 수정해야 하고**, **Function Alignment의 lightweight adapter가 전체 inference까지 lightweight하다는 식으로 받아들이면 안 되며**, **Notochord의 `<10 ms`는 완성 phrase latency가 아니라 interactive next-event response라는 점을 분리해야 합니다.**

반대로 이번 조사에서 이전보다 **확실히 승격해야 할 것**은 **MelodyT5**, **Impro-Visor식 grammar/retrieval**, **SCG에서 가져온 rule-guided selection**, 그리고 **CMT+MINGUS를 섞은 소형 explicit-chord student**입니다. 이 네 가지를 포함하면, 현재 문제는 더 이상 “13M Music Transformer냐 retrieval이냐”가 아닙니다.

**가장 유력한 해법은 `작은 explicit-chord neural model + jazz vocabulary memory + deterministic harmonic control + large-model teacher`의 hybrid입니다.** 현재 알려진 공개 연구들을 사용자의 `<400 ms / MIDI / bebop / chord-symbol` 조건까지 동시에 겹쳐 보면, 이 조합이 단일 최신 foundation model보다 목표에 훨씬 가깝습니다.
