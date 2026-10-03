# 화성 조건 계약과 기존 모델의 조건 반응 (사전 등록, U1·U2)

작성 2026-10-03. 이슈 #1631. 아스트라와 심층 토의해서 범위를 고정했다. 판정 규칙은 실행 전에 고정했다. `musical_quality_verified: false`.

## 배경
- 사용자 청취: "솔로 같긴 한데 틀린 음을 많이 치는 솔로", "컴핑이 너무 기계적"
- 지시: 사용자가 검증할 필요가 없는 당연한 실패는 사용자에게 가기 전에 거른다
- 자동 검사는 **필요조건**이다. 명백한 결함과 조건 반응 실패를 거를 뿐이다. 통과해도 "듣기 좋다"는 인증이 아니다

## U1 — 계약 (`inference/control/harmony_contract.py`)
- guide: 베이스 pc는 C2–B2에, 나머지 pc는 각 한 번씩 C3–B3에 놓는다. start 0, end 0.85 × 창, velocity 58이다(런타임 guide와 같은 시간 형식)
- 런타임 쪽: 코드 심볼 → (근음, 코드 구성음 pc)
- 학습·평가 쪽: 실제 창의 **accompaniment proxy** → (최저음 pc, 낮은 음 pc)
  - proxy = 솔로 음역(G3) 아래에서 창 안에 시작하거나 창 시작 시점에 울리고 있는 음
  - 손 분리도 아니고 코드 라벨도 아니다
- 예시 형식: `[guide][그 창의 솔로 선율]`. 솔로는 창 시작 기준 시각이고 guide 뒤에 이어진다
- **발견한 결함(확인함):** 지금 런타임 guide(`_voicing_for_chord`)는 C3 위의 코드톤을 아래부터 3개만 고른다
  - G7 → G·D·F·G: 3음 B가 없다
  - Dm7b5 → D·C·D·F: Dm7과 똑같고 b5가 없다
  - Dm7 → 5음 A가 없다
  - 이 단위에서 런타임 guide를 바꾸지는 않는다. 계약 모듈만 정의한다. 런타임 교체는 U2 결과를 본 뒤 별도로 측정한다
- 실제 MIDI에는 박 그리드가 없다(전사본이라 템포가 120으로 고정돼 있다). 그래서 강박 지표는 쓰지 않고 시간 창 기준만 쓴다

## U2 — 기존 모델은 guide를 읽는가 (학습 없음)
- 데이터: `data/bebop_rh` manifest의 **val** 곡(28곡) 원본 양손 MIDI. test는 쓰지 않는다
- 창: 0.9375초(128 BPM 반마디 = 런타임 블록). 2초에서 시작해 4초 간격으로 놓는다. 솔로 음이 3개 이상이고 proxy pc가 2개 이상인 창만 쓴다. 곡마다 최대 20개다
- 솔로: 전체 음의 50 ms 최고음 중 G3 이상(`build_rh_dataset`와 같은 규칙)
- 조건 세 가지(같은 창, paired)
  - **true:** 그 창의 proxy guide
  - **shuffled:** 다른 곡((곡 순번 + 7) mod N)의 같은 순번 창(순번이 모자라면 나머지 연산)의 proxy guide
  - **absent:** guide 없음
- 점수: 솔로 토큰의 평균 NLL(nats/token)이다. 첫 솔로 토큰은 absent에 문맥이 없으므로 모든 조건에서 뺀다
- 모델: base(공통 base), bebop, tatum, mehldau. 추론용으로 병합하고 CPU에서 돌린다
- 통계: 창별 차이 Δshuf = NLL(shuffled) − NLL(true), Δabs = NLL(absent) − NLL(true). 곡 단위 bootstrap 95% CI(2000회, seed 0)
- guide 추출 통계도 보고한다: proxy pc 수 분포, onset 묶음 수(변화 구간 지표), 쓴 창 / 후보 창

## 판정 (실행 전 고정)
- 모델이 "guide를 읽는다" = Δshuf의 CI 하한 > 0
- **A(guide 형식 소규모 재학습) go:** bebop이 다음 중 하나다
  1. guide를 읽지 않는다(Δshuf CI가 0을 포함하거나 0 이하)
  2. Δshuf 평균이 base의 절반보다 작다(적응 과정에서 화성 반응을 잃었다)
- **no-go:** bebop의 Δshuf가 base 이상이고 CI 하한 > 0이면, 모델이 guide를 못 읽어서 틀린 음이 난다는 설명은 기각한다. 그러면 다음 용의자로 넘어간다: 런타임 guide 결함(위)과 디코딩
- 어느 쪽이든 결과는 기록한다. 판정 기준을 다시 고르지 않는다
