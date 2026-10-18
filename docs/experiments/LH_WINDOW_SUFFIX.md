# 왼손 관측 창 변형별 suffix 재계산 비용 (사전 등록)

작성 2026-10-18 23시대 KST. 이슈 #1820. 모델·GPU 실행 없음, 비용 분석만. 실시간 계약 변경 없음.

## 1. 질문

- 왼손 관측 창을 바꾸면(현행 7.5 s, 15 s, crop까지 만료 없음) 연속 요청 쌍의 재계산 토큰 수 S, 공통 앞부분 P, 문맥 길이, crop 빈도, 갈림 원인이 어떻게 달라지는가

## 2. 배경

- #1793 공통 앞부분 감사(dev 4곡, b 입력, 1,306쌍): 비crop S 중앙값 116 / p90 248 / 최대 353. 원인 expired 887, held_update 284, insert 73, none 54, crop 8
- #1813 v2 실시간 q8: suffix 중앙값 42~48 ms, 최대 131 ms(S 218). 연산 쪽 비용이라 정밀도로 줄지 않음(#1811 suffix 48토큰 34~37 ms)
- expired는 창 시작이 블록마다 0.9375 s씩 밀리며 가장 오래된 왼손 음이 빠지는 갈림. held_update는 앞 요청에서 s로 잘랐던 held 음이 더 울려 길이 토큰이 바뀌는 갈림이라 창 변형과 무관하게 남을 후보

## 3. 방법 (실행 전 고정)

- 입력: #1793과 같음. export `lh_request_export_run3`, velocity `lh_context_run1/human_velocity`, dev 곡 44·68·134·185, `crop_contract.session_requests` 블록 격자
- 요청 생성: `lh_context.build_request(..., w_ms=W)` 그대로(창 시작 = max(s − W, crop base), crop 판단은 a·b·c 세 입력 중 가장 긴 것). `lh_kv_reuse.build_songs`에 `w_ms` 인자만 추가(기본값 7.5 s 그대로)
- 변형 3개: `W7.5`(현행), `W15`, `no_expiry`(W = ∞, 창 시작 = crop base)
- 쌍 계산: `lh_kv_reuse.pair_rows`(b 입력, k ≥ 2, 연속 요청), 원인 판정 `cause` 그대로. crop 상태가 바뀐 쌍은 P = 0
- 검증: `W7.5` 결과가 #1793 `count.json` b(쌍 수, 원인 수, P·S·L 요약)와 일치해야 함. 불일치면 분석 중단
- 지표(변형별)
  - 쌍 수, 원인 수, crop 쌍 수, build 실패 수, 추가 crop(`extra_crop`) 요청 수
  - 비crop S 중앙값·p90·최대, P 중앙값, 문맥 길이 L(= len(N)) 중앙값·p90·최대, 요청당 왼손 음 수 중앙값·최대
  - 쌍당 재계산 토큰 평균 = 모든 쌍 평균(비crop은 S, crop 쌍은 L)
  - suffix 시간 추정: #1793 대표 4지점 T(P, S)(fp32, teacher forcing)에 T = a + b·S + c·S·P 최소제곱 적합. 변형별 비crop S p90·최대를 각 변형 P 중앙값에서 평가. crop 쌍은 #1793 전체 prefill 측정값(562~615 ms) 범위로 표시. 추정임을 명시
- 판정 규칙(실행 전 고정, 계약 변경 실험의 가치 판단용)
  - "계약 변경 실험 가치 있음": `no_expiry` 또는 `W15`가 `W7.5` 대비 비crop S p90을 50% 이상 줄이고, 쌍당 재계산 토큰 평균을 30% 이상 줄임
  - crop 쌍이 늘어 쌍당 재계산 토큰 평균이 줄지 않으면 "가치 없음(crop 비용으로 상쇄)"
  - 어느 경우든 음악·조건 효과는 이 단위로 판단하지 않음

## 4. 범위 밖

- 모델 forward, 실시간 실행, 생성 품질, 오래된 왼손 음을 문맥에 남기는 것의 조건 효과
- 실시간 producer의 crop 방식(`SessionCrop`, #1806)과 오프라인 `extra_cut`의 차이. 이 분석은 오프라인 #1773 계약 기준

## 5. 재현

- `scripts/lh_window_suffix.py <export dir> <velocity dir> <#1793 count json> <#1793 cost json> <out json>`
- 시험 `tests/test_lh_window_suffix.py`
- 보고 `docs/experiments/lh_window_suffix/report.json`
