# 왼손 관측 창 만료 묶음의 suffix 재계산 비용 (사전 등록)

작성 2026-10-19 03시대 KST. 이슈 #1831. 모델·GPU 실행 없음, 비용 분석만. 실시간 계약 변경 없음. #1829(`exp/1820/lh-window-suffix`) 분석 도구 의존.

## 1. 질문

- 왼손 관측 창 시작을 N블록 격자 또는 5 s segment 시작으로 내림해 만료를 묶으면, 블록별 재계산 S의 최대·꼬리와 큰 S 블록 빈도, crop 빈도가 현행 대비 어떻게 달라지는가

## 2. 배경

- #1793: 현행(창 시작 = max(s − 7.5 s, crop base)) b 입력 비crop S 중앙값 116 / p90 248 / 최대 353, 원인 expired 887 / 1,306쌍
- #1829: 15 s 창 S p90 466(악화). crop까지 만료 없음 S p90 50이지만 crop 쌍 8 → 51, 쌍당 재계산 0.785배로 사전 규칙 미달. held_update 갈림 S p90 약 52는 세 변형 공통 바닥 후보
- 현행은 창 시작이 블록마다 0.9375 s 밀려 매 블록 가장 오래된 왼손 음이 빠짐. 마감 위험은 평균이 아니라 블록별 최대 재계산이 결정

## 3. 방법 (실행 전 고정)

- 입력: #1793·#1829와 같음. export `lh_request_export_run3`, velocity `lh_context_run1/human_velocity`, dev 곡 44·68·134·185, `crop_contract.session_requests` 블록 격자
- 요청 생성: `lh_context.build_request(..., window_start=규칙)`. 기본값(규칙 없음)은 기존 max(s − w_ms, base) 그대로. `lh_kv_reuse.build_songs`에 `window_start` 전달 인자만 추가
- 변형 4개(모두 창 길이 7.5 s 이상)
  - `W7.5`: 현행
  - `B2`: 창 시작 = max(t0 + ⌊(s − 7.5 s − t0) / 2B⌋ · 2B, base). t0 = s_k − k·B(블록 원점), B = 937.5 ms. 창 길이 7.5 s ~ 9.375 s 미만
  - `B4`: 같은 식, 격자 4B(3.75 s). 창 길이 7.5 s ~ 11.25 s 미만
  - `SEG5`: 창 시작 = max(⌊(s − 7.5 s) / 5 s⌋ · 5 s, base). 토크나이저 `<T>` 격자(crop base도 이 격자). 창 길이 7.5 s ~ 12.5 s 미만
- 쌍 계산: `lh_kv_reuse.pair_rows`(b 입력, k ≥ 2), 원인 판정 `cause` 그대로, crop 상태가 바뀐 쌍은 P = 0
- 만료 쌍: 연속 두 요청의 창 시작이 다른 비crop 쌍. 나머지 비crop 쌍은 비만료 쌍
- 검증: `W7.5`가 #1793 `count.json` b(쌍 수, 원인 수, P·S·L 요약)와 일치. 불일치면 분석 중단(종료 코드 1)
- 지표(변형별)
  - 쌍 수, 원인 수, crop 쌍 수, 추가 crop 요청 수, build 실패 수
  - 비crop S 중앙값·p90·최대, 만료 쌍·비만료 쌍 각각의 S 요약과 쌍 수
  - 최악 블록: 비crop S 최대 쌍(곡, k, P, S, 원인, 만료 여부, 추정 suffix 시간)
  - 큰 S 쌍: 자기 P에서 추정 suffix 시간 ≥ 100 ms인 비crop 쌍의 수와 비율
  - 쌍당 재계산 토큰 평균(비crop S, crop 쌍 L), 문맥 길이 L, 창 길이, 요청당 왼손 음 수
  - suffix 시간 추정: #1829와 같은 적합(#1793 대표 4지점, T = a + b·S + c·S·P, fp32 teacher forcing). 추정임을 명시

## 4. 판정 규칙 (실행 전 고정, 계약 변경 실험의 가치 판단용)

변형마다 아래 네 조건 모두 충족이면 "계약 변경 실험 가치 있음".

- 비만료 쌍 S p90 ≤ 0.5 × `W7.5` 비crop S p90(대부분 블록이 싸짐)
- 최악 블록 S ≤ 1.25 × `W7.5` 비crop S 최대(만료가 몰린 블록이 현행 최악보다 크게 나빠지지 않음)
- crop 쌍 ≤ 1.5 × `W7.5` crop 쌍
- 큰 S 쌍 비율 ≤ 0.5 × `W7.5` 큰 S 쌍 비율

- 쌍당 재계산 토큰 평균은 기록만(마감은 최대가 결정)
- 어느 경우든 오래 남는 왼손 음의 조건·음악 효과는 이 단위로 판단하지 않음

## 5. 범위 밖과 후속 후보

- 모델 forward, 실시간 실행, 생성 품질, 왼손 문맥 길이 변화의 조건 효과
- 실시간 producer의 crop 방식(`SessionCrop`, #1806)과 오프라인 `extra_cut`의 차이(이 분석은 #1773 오프라인 계약 기준)
- 후속 후보(구현하지 않음): 만료 블록을 미리 알 수 있으므로(격자 이동 시점이 정해짐), 직전 블록의 추측 생성 구간에 만료 뒤 suffix를 미리 계산해 두는 방식. 만료 블록의 큰 S를 마감 밖으로 옮길 수 있는지 별도 단위

## 6. 재현

- `scripts/lh_batched_expiry.py <export dir> <velocity dir> <#1793 count json> <#1793 cost json> <out json>`
- 시험 `tests/test_lh_batched_expiry.py`
- 보고 `docs/experiments/lh_batched_expiry/report.json`
