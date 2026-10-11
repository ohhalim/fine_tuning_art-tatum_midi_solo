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

**생성 prompt**: val 묶음 키 정렬 앞 4개의 첫 녹음, 처음 8초의 음. 토큰 수 102, 180, 69, 159, 4개 모두 가용

## 학습 (`scripts/mehldau_aria_lora_run.py --train`, 공용 LoRA `scripts/aria_lora.py`)

- frozen `model-gen.safetensors` F32, MPS, batch 1, gradient checkpoint 켬
- LoRA r16, alpha 16, `mixed_qkv`·`att_proj_linear` 16층(2,359,296개), A seed 0, B 0
- Adam lr 1e-4, **정확히 128 update, 1회**. 매 update: `random.Random(0)`으로 train 녹음을 균등하게 고르고, 그 녹음의 창을 균등하게 고른다
- warmup, 증강, lr sweep, 더 긴 학습은 없다

## 평가와 개발 판정

- 같은 창에서 base(LoRA 끔)와 adapter를 비교한다. 녹음당 앞 8개 창까지 쓴다
- **주지표:** 녹음별 continuation CE(녹음 안에서는 토큰 가중) 변화의 녹음 평균. 보조로 묶음 평균, 토큰 가중 평균, 토큰 종류별(pitch / onset / dur / `<T>`), pitch를 velocity에 대해 합친 pitch 주변 CE, pitch가 주어졌을 때의 velocity CE를 함께 낸다
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

## 결과

(학습 후 작성)
