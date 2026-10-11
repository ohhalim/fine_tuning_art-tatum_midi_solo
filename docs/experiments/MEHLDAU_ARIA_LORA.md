# Mehldau Aria LoRA 적응 pilot (이슈 #1726, 사전 등록)

작성 2026-10-11. **이 문서의 결과 절 위는 학습 전에 커밋했다.** `style_verified: false` · `musical_quality_verified: false`

## 질문

frozen Aria에 LoRA를 붙여 Mehldau 녹음 일부로 128 update 학습하면,
1. 학습에 쓰지 않은 Mehldau 녹음의 continuation CE가 base보다 내려가는가
2. 다른 피아니스트 곡의 CE는 크게 나빠지지 않는가
3. 같은 prefix의 생성 유효성이 base보다 나빠지지 않는가

이것은 **타당성 pilot**이다. 128 update가 충분하다는 주장(수렴 예산)이 아니다. 성공해도 "Mehldau처럼 들린다"는 뜻이 아니다.

## 자료 (`scripts/mehldau_aria_data.py`, manifest sha256 `0b075bceaa24b9b3`, data sha256 `8501d2aca038d763`)

**녹음과 묶음**
- PiJAMA `midi` 트리의 Mehldau 72곡(studio 18 + live 54). `midi_kong`의 같은 경로 파일은 같은 녹음의 다른 전사라 묶음에 기록만 하고 쓰지 않는다
- 묶음 규칙: 정규화한 제목이 같으면 한 묶음. 그다음 다른 묶음끼리 이조 불변 음정 16-gram 포함률이 0.2 이상이면 합친다. 결과는 **69묶음**이다. 제목으로 묶인 것은 3쌍(From This Moment On studio+live, My Favorite Things live 2개, Things Behind the Sun live 2개)이고, 16-gram으로 합쳐진 것은 없다(최대 포함률 0.0031)
- 제목과 16-gram으로 모든 중복을 찾았다고 보장하지 않는다. 메들리, 다른 이름의 같은 곡, 느슨한 인용은 놓칠 수 있다
- 처음 규칙은 "… - Live At <공연장>, <날짜>" 접미사를 지우지 못해 다른 연주자 곡 중 Mehldau와 같은 제목 5곡을 놓쳤다. 학습 전에 고쳤다(Mehldau 묶음은 변화 없음, probe 한 곡이 바뀜)

**분할** (묶음 키 정렬 → `random.Random(0).shuffle` → test round(0.15G), val round(0.15G), 나머지 train)

| 분할 | 묶음 | 녹음 | studio / live | 창 | 평가 창 | target 토큰 |
|---|---|---|---|---|---|---|
| train | 49 | 50 | 13 / 37 | 597 | 0 | 587,840 |
| val | 10 | 12 | 3 / 9 | 155 | 80 | 152,747 |
| test | 10 | 10 | 2 / 8 | 106 | 66 | 104,306 |

- val: Brahms -  Intermezzo in B-flat major, Op. 76 -  No. 4 - Live(l), Don't Let It Bring You Down(s), From This Moment On(l), From This Moment On(s), Get Happy - Live(l), Holland - Live(l), IV. stopping, listening -  hearing(s), It's All Right With Me - Live(l), Jigsaw Falling into Place - Live(l), My Favorite Things - Live(l), My Favorite Things - Live(l), Teardrop - Live(l)
- test: Junk - Live(l), Knives Out 2 - Live(l), Le Mémoire et la Mer - Live(l), Lost Chords - Live(l), Medley -  Zingaro  Paris - Live(l), Monk's Mood - Live(l), Someone to Watch over Me(l), Think of One - Live(l), VI. uncertainty(s), X. in the kitchen(s)
- studio 비율은 층화하지 않았다. val·test 모두 live가 대부분이다
- 옛 13.4M 모델 개발에 쓴 녹음도 있다. 이번 정의는 "**이번 어댑터 학습에서 제외**"다. Aria 사전학습 미노출이나 완전히 새로운 자료라고 주장하지 않는다

**비대상 probe** (학습에 안 씀)
- Mehldau·Tatum을 뺀 피아니스트 중, Mehldau와 같은 제목을 뺀 녹음이 5개 이상인 사람에서 `random.Random(0).sample` 4명, 사람마다 5곡: **Hiromi, Ted Rosenthal, James Williams, Alan Pasqua** 20곡, 평가 창 144개
- Mehldau와의 음정 16-gram 최대 포함률 0.0007. 재추첨에는 쓰지 않고 기록만 한다
- 작곡·템포·전사 난도가 Mehldau 곡과 달라서, 대상과 비대상의 차이를 "연주자 특화"의 인과로 확정하지 않는다

**창 자르기**
- 녹음마다 원 AbsTokenizer로 한 번 토큰화(페달 연장 켬, `add_dim_tok=False`, `add_eos_tok=False`)하고, 음으로 풀어서 1024토큰 이하 창으로 연속해 자른다
  - 이 왕복이 원 토큰을 그대로 재현하는지 녹음마다 확인했다(assert)
- 창은 음 경계에서 시작하고 끝난다. 같은 onset의 화음은 쪼개지 않는다. 창의 음을 첫 onset 0으로 옮겨 MidiDict로 다시 만들고 토크나이저로 다시 토큰화한다. 시간 정규화는 토크나이저가 한다
  - 모든 창에서 다시 풀었을 때 음이 원래와 정확히 같았다
- 잘린 창에는 `<E>`도 `<D>`도 없다. `<D>`는 토크나이저가 곡 끝 약 100토큰 앞에 넣는 "곧 끝남" 표시라, 잘린 창에 있으면 거짓 종료 신호가 된다. 파일 끝이 실제 연주 끝인지는 확인하지 않았으므로 마지막 창에도 넣지 않는다
- velocity는 Aria가 양자화한 원래 값이다(관악 실험의 v80 변환 아님)
- 32음 미만인 꼬리는 버린다: 12개 녹음에서 169음
- 창 시작 시점에 이미 울리던 음(앞 창에서 시작)은 창 안에 없다: 전체 3,507음. 기록만 한다
- loss target은 `<S>` 다음 토큰부터다
- 그래서 창마다 앞 문맥의 울리던 음이 빠진다. "피아노 문맥 전체를 손실 없이 보존"이 아니라 **"채택한 음을 재토큰화해 보존"**까지만 말한다
- `<D>`·`<E>` 제거는 원 AbsTokenizer의 공개 옵션이며 vocab을 바꾸지 않는다. base와 adapter에 똑같이 적용한 전처리다. `<D>`가 들어 있던 과거 토큰 실험과 CE 수치를 직접 비교하지 않는다

**생성 prompt**: val 묶음 키 정렬 앞 4개의 첫 녹음, 처음 8초의 음. 토큰 수 102, 180, 69, 159, 4개 모두 가용

## 학습 (`scripts/mehldau_aria_lora_run.py --train`, 공용 LoRA `scripts/aria_lora.py`)

- frozen `model-gen.safetensors` F32, MPS, batch 1, gradient checkpoint 켬
- LoRA r16, alpha 16, `mixed_qkv`·`att_proj_linear` 16층(2,359,296개), A seed 0, B 0
- Adam lr 1e-4, **정확히 128 update, 1회**. 매 update: `random.Random(0)`으로 train 녹음을 균등하게 고르고, 그 녹음의 창을 균등하게 고른다
- warmup, 증강, lr sweep, 더 긴 학습은 없다

**학습 중·후 검사** (아스트라 검토로 실행 전에 추가)
- 매 update: backward 뒤 opt.step 전에 LoRA 64개 텐서의 grad가 None이 아니고 유한한지, base 파라미터에 grad가 없는지 본다. step 뒤 LoRA 파라미터가 유한한지 본다. 하나라도 어긋나면 그 자리에서 멈춘다
  - 첫 update의 A grad는 B=0이라 0이 정상이다. 0은 실패로 보지 않고, None·비유한만 실패로 본다
- 128 update 뒤: 64개 텐서 모두 초기값에서 바뀌었는지, base grad 없음, base 해시 불변, checkpoint 파일 SHA 불변을 확인한다
- **파일 재로드:** 고정 train 검산 창(train 첫 녹음의 첫 창)에서 학습 직후 logits를 저장한다. 그다음 LoRA를 파일로 저장하고, 메모리의 LoRA를 0으로 지운 뒤(logits가 달라짐을 확인) 파일에서 다시 불러와 같은 조건으로 비교한다. **허용 오차 0**(실행 전 고정). adapter 평가는 이렇게 다시 불러온 가중치로 한다
- `--test`는 train 보고서의 개발 판정 통과, test 보고서 부재에 더해 **checkpoint·data·manifest·LoRA 파일 SHA가 train 때와 같은지** 확인한 뒤에만 연다
- 회귀: `tests/test_aria_lora_helpers.py` 6개(작은 가짜 모델로 grad 검사, 변화 검사, 파일 재로드, 입력 SHA 대조)

## 평가와 개발 판정

- 같은 창에서 base(LoRA 끔)와 adapter를 비교한다. 녹음당 앞 8개 창까지 쓴다
- **주지표:** 녹음별 continuation CE(녹음 안에서는 토큰 가중) 변화의 녹음 평균. 묶음 평균을 항상 함께 낸다. 대상(Mehldau 묶음)과 비대상(probe 20곡)은 섞지 않고 따로 낸다. 보조로 묶음 평균, 토큰 가중 평균, 토큰 종류별(pitch / onset / dur / `<T>`), pitch를 velocity에 대해 합친 pitch 주변 CE, pitch가 주어졌을 때의 velocity CE를 함께 낸다
- **개발 판정 (모두 충족해야 통과)**
  1. val ΔCE(녹음 평균) < 0
  2. probe ΔCE(녹음 평균) ≤ +0.02 nat/token
  3. 생성 유효 표본 수: adapter ≥ base (각 8개)
- +0.02는 공학적 임시 허용치다. 검증된 음악 품질 한도가 아니다
- **test는 개발 판정을 통과한 뒤 한 번만 연다**(`--test`가 train 보고서의 통과 여부와 기존 test 보고서를 확인한다). 실패하면 열지 않는다

## 생성

- val prompt 4개 × seed 1, 2 × base / adapter = 각 8개. 새 토큰 512개, temperature 1, min_p 없음, `<E>`에서 멈춤. 같은 seed는 base·adapter가 같은 난수열을 쓴다
- 전량 보고하고 고르지 않는다: 유효성(첫 오류, 연쇄 오류, 끝난 방식), 음 수, 음역, 가장 긴 반복, 최대 동시음, 음악 시간, train 16-gram 일치 수
- 16-gram 일치가 0이어도 "복사 위험 0"이라고 하지 않는다. 흔한 반복 패턴, base도 만드는 패턴, 다른 전사의 근사 복사는 이 검사로 못 찾는다

## precheck 결과 (학습 없음)

- 창 구조: 1024 초과 0, `<S>` 위치 불일치 0, `<E>`·`<D>` 포함 0, 첫 토큰이 피아노 prefix가 아닌 창 0
- loss mask: target이 `<S>` 다음부터, 첫 target은 음 토큰
- 0 LoRA logits = base, 차이 0
- 평가 파이프라인: CE 유한, pitch 주변 CE ≤ 토큰 CE
- 생성 파이프라인 16토큰 동작
- 가장 긴 train 창 forward·backward: LoRA grad 32쌍 모두 있음, base grad 없음, 관측 최대 MPS 4.10 GiB

## 비용 상한과 중단 조건

- 예상(precheck 기준 추정): 학습 4분, base·adapter 평가 3분, 생성 9분
- **상한: `--train` wall 60분**, 넘으면 중단하고 그때까지의 기록을 남긴다
- 즉시 중단: OOM, 유한하지 않은 loss, memory pressure level 4, MPS 미지원 연산. OOM이어도 창 길이를 자동으로 줄이거나 sweep하지 않는다
- 새 유료 자원, 외부 다운로드, 공개 업로드 없음. LoRA와 생성 원자료는 저장소 밖 `/Users/ohhalim/git_box/mehldau_aria/run/`에만 둔다

## 결과 — 개발 판정 **미달** (생성 유효성), test 미개봉

원자료: `mehldau_aria/train_report.json`(생성 토큰은 저장소 밖 `generations.json`). 실행 2026-10-11 21:54–22:37(43.6분, 상한 60분 안).

**검사:** 학습 중·후 검사 9개 모두 통과했다(64 텐서 모두 변화, base grad 없음, base 해시·checkpoint SHA 불변, 파일 재로드 logits 차이 0). 학습 관측 최대 MPS 5.17 GiB, 128 update 120초. 학습 loss는 앞 8개 평균 2.453, 뒤 8개 평균 2.372다

| 판정 | 값 | 기준 | 결과 |
|---|---|---|---|
| 1. val ΔCE (녹음 평균) | **-0.2974** (묶음 평균 -0.2978, 12/12 녹음 개선) | < 0 | 통과 |
| 2. probe ΔCE (녹음 평균) | **-0.2833** (20/20 곡 개선) | ≤ +0.02 | 통과 |
| 3. 생성 유효 표본 | adapter **1/8**, base 5/8 | adapter ≥ base | **미달** |

**test는 열지 않는다.** 사전 등록대로 추가 예산, seed 탐색, 판정 재해석도 하지 않는다.

**CE는 Mehldau만이 아니라 다른 피아니스트도 거의 똑같이 내려갔다**

| ΔCE | 전체 | pitch | onset | dur | `<T>` | pitch 주변 | velocity\|pitch |
|---|---|---|---|---|---|---|---|
| val (Mehldau) | -0.297 | -0.130 | -0.194 | -0.556 | -0.083 | -0.078 | -0.052 |
| probe (다른 4명) | -0.283 | -0.090 | -0.180 | -0.572 | -0.106 | -0.042 | -0.048 |

- val과 probe의 차이는 -0.014에 그친다. 이득의 대부분은 Mehldau 고유 성향이 아니라 **두 집단에 공통인 변화**다
- 가장 큰 몫은 duration(−0.56)이다. 후보 원인: PiJAMA 전사·페달 연장의 길이 분포, 또는 `<D>`·`<E>`를 뺀 창 형식에 맞춘 적응. 둘을 분리하지 않았다
- pitch 주변 CE만 보면 Mehldau(−0.078)가 probe(−0.042)보다 조금 더 내려갔다. 이것만으로 연주자 특화를 말하지 않는다(사후 관찰, 작곡·전사 차이 통제 안 함)

**생성 16개 (전량, 고르지 않음)**

| prompt | seed | 모델 | 유효 | 첫 오류(위치: 토큰) | 음 | 최대 동시음 | 최장 반복 | 음악 시간(초) | train 16-gram |
|---|---|---|---|---|---|---|---|---|---|
| Brahms -  Intermezzo in B-fl | 1 | base | 유효 |  | 168 | 11 | 0 | 41.0 | 0 |
| Brahms -  Intermezzo in B-fl | 1 | adapter | 무효 | 39: ('synth_lead', 52, 90) | 166 | 10 | 8 | 42.39 | 0 |
| Brahms -  Intermezzo in B-fl | 2 | base | 무효 | 283: ('ensemble', 70, 30) | 166 | 19 | 6 | 53.24 | 0 |
| Brahms -  Intermezzo in B-fl | 2 | adapter | 무효 | 145: ('organ', 117, 100) | 165 | 10 | 12 | 57.01 | 0 |
| Don't Let It Bring You Down | 1 | base | 유효 |  | 169 | 11 | 4 | 22.41 | 0 |
| Don't Let It Bring You Down | 1 | adapter | 무효 | 152: ('strings', 57, 30) | 166 | 11 | 7 | 51.27 | 0 |
| Don't Let It Bring You Down | 2 | base | 유효 |  | 169 | 10 | 10 | 20.62 | 0 |
| Don't Let It Bring You Down | 2 | adapter | 무효 | 283: ('ensemble', 70, 30) | 160 | 10 | 3 | 148.67 | 0 |
| From This Moment On | 1 | base | 유효 |  | 168 | 41 | 2 | 33.41 | 0 |
| From This Moment On | 1 | adapter | 무효 | 152: ('strings', 57, 30) | 166 | 10 | 10 | 55.98 | 0 |
| From This Moment On | 2 | base | 무효 | 145: ('organ', 117, 100) | 165 | 7 | 6 | 68.44 | 0 |
| From This Moment On | 2 | adapter | 무효 | 41: ('bass', 80, 0) | 164 | 15 | 5 | 37.72 | 0 |
| Get Happy - Live | 1 | base | 유효 |  | 169 | 5 | 6 | 23.44 | 0 |
| Get Happy - Live | 1 | adapter | 유효 |  | 169 | 2 | 8 | 20.89 | 0 |
| Get Happy - Live | 2 | base | 무효 | 145: ('organ', 117, 100) | 168 | 18 | 13 | 20.45 | 0 |
| Get Happy - Live | 2 | adapter | 무효 | 145: ('organ', 117, 100) | 166 | 12 | 5 | 29.16 | 0 |

- 무효 표본의 첫 오류는 모두 **피아노가 아닌 악기 토큰**(organ, strings, ensemble, synth_lead, bass)이다. temperature 1, min_p 없이 전체 vocab(17,727)에서 뽑았기 때문에 꼬리 토큰이 나올 수 있다
- 같은 seed는 prompt가 달라도 같은 위치·같은 토큰에서 오류가 났다(예: seed 2의 145번째 `('organ', 117, 100)`, 283번째 `('ensemble', 70, 30)`). 그 step의 난수가 분포의 꼬리(vocab 끝쪽)를 고른 것으로 보인다. **확인하지 않은 해석이다**
- 이 오류는 base에서도 3/8 났다. adapter가 7/8로 더 많은 것은 꼬리 확률이 늘었기 때문일 수 있으나 재지 않았다
- train 16-gram 일치는 16개 모두 0이다. 이것으로 "복사 위험 0"이라고 하지 않는다

## 해석과 다음

- **이 pilot이 보인 것:** 128 update LoRA는 PiJAMA 피아노 자료 쪽으로 CE를 크게 옮긴다. 다만 그 이동은 Mehldau 고유가 아니라 공통이고, 생성 유효성은 base보다 나빴다
- **보이지 못한 것:** Mehldau 스타일 적응. 개발 판정 미달이라 test는 미개봉이다
- **다음 결정은 아스트라와 한다.** 후보: (a) 공통 이동을 빼고 볼 비교 설계(예: 다른 피아니스트로 같은 예산 학습한 대조 어댑터), (b) 생성 조건 문제(악기 토큰 꼬리)를 판정과 분리할지. 이번 결과를 보고 판정 기준을 바꾸지 않는다
- 비용 추정 실수: 생성은 예상 10분, 실제 39분이었다. KV 캐시 없이 토큰마다 문맥 전체를 다시 계산하는데, 짧은 문맥 측정값에 1.6배만 곱했다. 진행률 로그도 없어서 실행 중에 진행 상태를 볼 수 없었다
