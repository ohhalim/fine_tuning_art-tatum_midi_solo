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

## 결과 (아스트라 검토 뒤 1회 실행, `wjazzd_gate/result.json`, 원 토큰은 로컬 `/Users/ohhalim/git_box/wjazzd/gate/gate_result.json`)
- 안전 확인 전부 통과
  - gate 일관성
  - 0 adapter = base
  - optimizer 안 base 0
  - 저장·재로드 logits 차이 0.0
  - base 파라미터·체크포인트 sha256 불변
- adapter sha256 `53a7183f527ad9c4…`
- 시간: 128 update 14.713초, 평가 20.954초, 생성 90.714초
- 훈련 loss(정보): 처음 32 update 평균 2.544 → 마지막 32 update 평균 2.394
- validation은 이전 pilot 수치를 이미 본 세트다. 독립 확증이 아니라 개발용 탐색 대조다

### validation 8곡
| 조건 | pitch NLL(곡 평균, 덩어리→곡) | continuation NLL |
|---|---|---|
| base | 1.954 | 2.525 |
| correct | 2.071 | 2.518 |
| wrong | 2.367 | 2.617 |
| inactive | 1.954 | 2.525 |

| validation 곡(melid) | base | gate 맞음 | gate 틀림 | 맞음 − base | 맞음 − 틀림 |
|---|---|---|---|---|---|
| 16 | 1.758 | 1.451 | 2.241 | -0.307 | -0.790 |
| 347 | 2.200 | 2.400 | 2.709 | +0.200 | -0.309 |
| 18 | 1.616 | 1.694 | 1.953 | +0.077 | -0.260 |
| 25 | 2.037 | 2.040 | 2.748 | +0.002 | -0.708 |
| 293 | 2.106 | 2.245 | 2.334 | +0.139 | -0.089 |
| 435 | 1.745 | 2.074 | 2.099 | +0.329 | -0.025 |
| 359 | 1.981 | 2.220 | 2.324 | +0.240 | -0.104 |
| 61 | 2.186 | 2.441 | 2.523 | +0.255 | -0.083 |

| 층(토큰 pooled) | n | base | gate 맞음 | gate 틀림 |
|---|---|---|---|---|
| current | 550 | 1.910 | 2.057 | 2.274 |
| boundary_first | 91 | 2.273 | 2.492 | 2.681 |
| anticipation | 19 | 2.853 | 2.749 | 3.031 |
| not_represented | 4 | 2.072 | 2.752 | 2.240 |

### 사전 등록 판정 (셋 따로)
1. **순효과: 미충족.** gate 맞음 2.071 > base 1.954이고, 맞음 < base인 곡은 1/8이다
2. **조건 구별: 관측.** 맞음 < 틀림이 8/8이고, 곡 평균도 맞음 2.071 < 틀림 2.367이다
3. **생성 유효성: 미충족.** 유효 샘플 수가 base 8/8, gate 맞음 4/8, gate 틀림 8/8이다

- 셋째 판정은 이 8개 표본에서 기준을 못 넘었다는 뜻일 뿐이다. 통계적 열등성의 입증이 아니다. base 자신은 8/8 유효였다

### 생성 (validation 앞 4곡 × seed 1·2)
| melid | seed | 생성 | 유효 | 첫 오류(위치: 유형, 토큰) | 첫 오류 전 완성 음 | 연쇄 | 꼬리 |
|---|---|---|---|---|---|---|---|
| 16 | 1 | base | 예 | — | 32 | 0 | — |
| 16 | 1 | correct | 아니오 | 84: token_order, `('dur', 110)` | 28 | 3 | — |
| 16 | 1 | wrong | 예 | — | 32 | 0 | — |
| 16 | 2 | base | 예 | — | 31 | 0 | incomplete_tail |
| 16 | 2 | correct | 아니오 | 70: token_order, `('dur', 30)` | 23 | 0 | incomplete_tail |
| 16 | 2 | wrong | 예 | — | 31 | 0 | incomplete_tail |
| 347 | 1 | base | 예 | — | 31 | 0 | — |
| 347 | 1 | correct | 예 | — | 31 | 0 | — |
| 347 | 1 | wrong | 예 | — | 31 | 0 | incomplete_tail |
| 347 | 2 | base | 예 | — | 31 | 0 | incomplete_tail |
| 347 | 2 | correct | 예 | — | 31 | 0 | — |
| 347 | 2 | wrong | 예 | — | 31 | 0 | — |
| 18 | 1 | base | 예 | — | 31 | 0 | incomplete_tail |
| 18 | 1 | correct | 예 | — | 31 | 0 | incomplete_tail |
| 18 | 1 | wrong | 예 | — | 31 | 0 | — |
| 18 | 2 | base | 예 | — | 31 | 0 | incomplete_tail |
| 18 | 2 | correct | 아니오 | 4: token_order, `('dur', 360)` | 1 | 2 | incomplete_tail |
| 18 | 2 | wrong | 예 | — | 31 | 0 | incomplete_tail |
| 25 | 1 | base | 예 | — | 30 | 0 | incomplete_tail |
| 25 | 1 | correct | 예 | — | 31 | 0 | — |
| 25 | 1 | wrong | 예 | — | 31 | 0 | — |
| 25 | 2 | base | 예 | — | 30 | 0 | incomplete_tail |
| 25 | 2 | correct | 아니오 | 59: onset_reversal, `('onset', 2490)` | 18 | 0 | — |
| 25 | 2 | wrong | 예 | — | 30 | 0 | — |

- gate 맞음의 첫 오류 4개 중 3개는 음높이 바로 뒤 onset 자리에 `dur`가 온 것이고, 1개는 onset 역행이다
  - gate는 그 자리(onset 예측)에서 꺼져 있다. 그런데도 오류가 났다
  - 과거 위치에 넣은 조건이 attention으로 남는 경로나 다른 경로가 후보다(확인 안 함)
- gate 틀림 8/8 유효 대 gate 맞음 4/8은 작은 표본의 관측이다. 원인을 해석하지 않는다

### 그 밖의 관측(정보)
- 참고값(같은 validation, gate 없던 pilot): base 1.954, 맞음 1.973, 틀림 2.216
  - gate 맞음의 pitch NLL(2.071)은 이 참고값보다 높았다
  - gate 없던 pilot의 validation 생성 대조는 이번 등록에 없다. 그래서 "gate가 생성을 개선 또는 악화했다"고 말하지 않는다. 이전 test 생성(5/8)과 직접 비교하지 않는다
- continuation NLL은 gate 맞음 2.518이 base 2.525보다 조금 낮았다. pitch NLL은 높았다. 해석하지 않는다
- test(탐색, 정보): 맞음 < 틀림 8/8, 맞음 < base 0/8
- gate는 음 사이 자리에서 음높이뿐 아니라 `<T>`·`<D>`·`<E>` 확률에도 영향을 준다. 과거 주입의 영향도 남는다. "음높이 확률만 바뀐다"고 말하지 않는다

## 해석 범위
- 조건 구별은 gate를 넣어도 유지됐다(validation 8/8)
- base 대비 pitch NLL 순손해와 생성 유효성 기준 미달은 gate로 해소되지 않았다
- 따라서 "현재 위치 주입만이 원인"이라는 설명은 이 결과와 맞지 않는다. 남은 후보(확인 안 함)
  - 과거 위치 주입의 attention 경로
  - adapter 학습 신호 자체(train 24곡 쪽 공통 이동)
  - 자료 편향
- 예산 추가, seed 변경, 사후 기준 수정은 하지 않았다. 다음 결정은 아스트라와 한다
