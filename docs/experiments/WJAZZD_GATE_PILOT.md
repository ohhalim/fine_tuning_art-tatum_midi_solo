# WJazzD gate pilot: 음높이 예측 위치에만 조건 주입 (사전 등록)

작성 2026-10-09. 아스트라 결정. 관악 단선율로 화성 조건을 검증하는 시험이며, 피아노나 티그랑 스타일 자료가 아니다. **새 학습은 이 설계를 아스트라가 검토한 뒤 시작한다.** `musically_verified: false`.

## 질문 (하나)
이전 pilot과 다른 것은 조건 주입 gate 하나다. gate를 넣으면 다음 셋이 어떻게 되는가.
- 조건 구별(맞음 < 틀림)을 유지하는가
- base 대비 pitch NLL 순손해와 생성 유효성 악화가 줄어드는가

## 바꾸는 것: gate 하나 (`scripts/aria_pitch_gate.py`)
- gate[i] = 1인 경우: tokens[0..i]만 보고, 다음 토큰이 음높이일 수 있는 "음 사이" 상태일 때
  - `<S>` 직후, 음의 `dur` 직후, `<T>` 직후, `<D>` 직후
- gate[i] = 0인 경우
  - 머리(prefix 토큰)
  - 음높이 직후(onset 예측 자리)
  - onset 직후(dur 예측 자리)
  - `<E>` 뒤
  - 예상 밖 토큰 뒤(다음 `dur`·`<T>`·`<D>`에서 재동기화)
- 조건 행에 gate를 곱한다. 정답 다음 토큰을 보고 gate를 켜지 않는다
- 이것은 출력 문법 mask가 아니다. 모든 토큰이 제 확률을 가진다. 음 사이 `<E>` 같은 확률도 바뀔 수 있다. 과거 위치에 넣은 조건은 attention으로 뒤에 남는다
- 현재 pilot adapter에 gate만 붙이는 추론 변경이 아니다. **새 0 초기화 adapter를 gate와 함께 학습한다**

## 그대로인 것
- 자료와 분할: `WJAZZD_PILOT.md`와 같은 24/8/8곡, 덩어리 103개
- 마스크, loss: prefix 16음 target 제외, `<E>` target 제외, continuation CE
- 29차원 v2 조건, 주입점(16번째 block 입력), `Linear(29 → 1536, bias 없음)`
- Adam lr 1e-3, batch 1, seed 0, **정확히 128 update**, 곡 균등 sampling의 순서(`random.Random(0)`)도 같다. 128회 뒤 늘리지 않는다

## 학습 전 검사 (`scripts/wjazzd_gate_run.py --precheck`, `wjazzd_gate/precheck.json`, optimizer 없음)
- gate 시험 5개(`tests/test_aria_pitch_gate.py`): 음 사이에서만 켜짐, 켜진 자리 다음이 음 시작 또는 경계, 앞 토큰 불변성, 한 단계씩 = 한꺼번에, 예상 밖 토큰 뒤 재동기화
- 실제 덩어리 103개 전 위치 38583개 검사, 모두 통과
  - gate 켜짐 13241개
  - 켜진 자리 다음 토큰이 음 시작이 아님: 0
  - 꺼진 자리 다음 토큰이 음 시작임: 0
  - 한 단계씩과 한꺼번에의 불일치: 0
  - 다른 suffix를 붙였을 때 앞부분 gate가 바뀜: 0
- 0 adapter = base(차이 0.0), 가장 긴 덩어리(464토큰) forward·backward 0.205초, base sha256 불변

## 판정 (validation 8곡, 결과 전 고정)
이전 test 8곡은 반복 진단에 쓴 탐색 세트다. 정보로만 내고, 새 확증 성공으로 선언하지 않는다.
- **순효과:** 곡 평균 pitch NLL이 gate 맞음 < base인가. 맞음 < base인 곡 수도 함께 낸다
- **조건 구별:** 곡 평균 맞음 < 틀림이고, 8곡 중 7곡 이상이 같은 방향인가. 틀린 계획은 이전과 같다(현재·다음 chroma +1 반음에 같은 gate)
- **생성 유효성:** validation 앞 4곡 × seed 1·2 × {base, gate 맞음, gate 틀림} = 24개
  - `aria_token_validity` 기준 유효 샘플 수를 센다
  - "base보다 나쁘지 않음" = gate 맞음 유효 수 ≥ base 유효 수
  - 설정은 이전과 같다(prompt, 96토큰, temperature 1, min_p 0, `<E>` 허용). 토큰을 보존한다
- 셋을 따로 보고한다. 하나라도 못 미치면 어느 것인지 적는다
- 참고값: gate 없던 pilot의 validation 곡 평균은 base 1.954, 맞음 1.973, 틀림 2.216이다(맞음 < base 4/8)
- 한계: validation은 덩어리 9개, 음높이 target 645개로 작다. 층(현재·경계 첫 음·anticipation·표현 없음)은 토큰 pooled 평균으로 따로 낸다
