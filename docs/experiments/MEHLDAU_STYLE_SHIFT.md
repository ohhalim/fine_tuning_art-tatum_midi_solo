# 멜다우 스타일 이동 — 사전 등록

작성 2026-09-27. 로컬 브랜치 `exp/mehldau-update-budget-diag`. 이전 단계: `MEHLDAU_UPDATE_BUDGET_DIAG.md`.

`mehldau_style_verified: false` · `musical_quality_verified: false`

**이 절은 실행 전에 작성했다. 결과는 아래 결과 절에 따로 붙인다.**

## 질문 순서 (각 실험은 한 질문)

### V1 — 지표가 멜다우와 일반 재즈를 구분하는가
지표: `scripts/style_distance.py`. 여섯 descriptor 히스토그램(음정 간격, IOI, 동시타건 수, 음역, 벨로시티, 음길이)의 평균 JS 거리.

- 멜다우 18곡과, `jazz_full/train`에서 멜다우와 토큰 해시가 겹치는 곡을 뺀 일반 곡을 무작위로 뽑는다(seed 0). 기준용 100곡과 probe용 100곡으로 나눈다
- **1차 판정은 1024토큰 조각 단위다** (생성 길이에 맞춤). 곡마다 결정적 위치의 조각 하나를 쓴다
- 멜다우 probe: 기준 = 나머지 17곡(leave-one-out). 일반 probe: 기준 = 멜다우 18곡 전부
- 정답 조건: 멜다우 probe는 d(멜다우) < d(일반), 일반 probe는 그 반대
- **판정: 균형 정확도 ≥ 0.75면 V2에 사용한다.** 미만이면 V2의 스타일 해석을 하지 않는다
- 특징별 정확도는 탐색용으로만 보고한다. 결과를 보고 특징을 고르지 않는다

### V2 — update를 늘리면 생성이 멜다우 쪽으로 이동하는가 (V1 통과 시)
- 한 연속 run, MPS. `run_mehldau_update_budget_diag.py`와 같은 설정(armB 시작, batch 4, accumulation 4, lr 3e-4, LS 0.1)으로 512 update를 계획한다. snapshot은 0/8/32/128/256/512
- 생성: 중립 primer `outputs/chord_ab/ii_V_I.mid`, seed 8개 × 768토큰 연속 생성, grammar mask, temperature 1.0, top-k 32, top-p 0.95
- 점수: shift = d(일반 기준) − d(멜다우 train 16곡 기준). 높을수록 멜다우 쪽이다
- 복사 위험: 생성 16-gram 중 train 16곡에 정확히 있는 비율
- **판정**
  - 스타일 이동 지지: 어떤 snapshot의 shift가 update 0보다 커야 한다. 차이는 seed 간 부트스트랩 95% 구간이 0을 넘어야 한다
  - 복사 경고: 16-gram 복사율 > 0.10이면 "스타일"보다 "암기"로 기록한다
  - val CE(base가 본 곡)가 update 0보다 나빠지면 과적합 신호로 기록한다
- 들리는 스타일 성공은 이 실험으로 주장하지 않는다

## 결과

(실행 후 추가)
