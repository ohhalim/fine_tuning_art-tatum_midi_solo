# 8비트 Aria 실시간 왼손 smoke (사전 등록)

작성 2026-10-18 18시대 KST. 이슈 #1813. 학습, 다운로드, 청취 없음. 실시간 기본값 변경 없음. `quality_claimed: false`.

## 1. 질문

- 실시간 왼손 producer를 8비트 Aria로 돌릴 때 계약 검사 유지 여부, KV 재사용 동등성, 루프 안 샘플당 시간 변화

## 2. 배경

- #1811: 문맥 512 디코드 중앙값 fp32 11.32 ms, 8비트(q8) 6.07 ms(0.54배). fp32 대비 KL 평균 0.0009, TV 평균 0.0139
- #1807: fp32 실시간 재실행 사전 등록 기준 전부 통과. 정상 블록 최소 여유 35 ms, 중앙값 151 ms
- 블록 행 `tokens`는 지평 도달 뒤 추측 생성분 포함 샘플 수. 루프 안 토큰당 시간 산출 불가
- producer 동등성 검사(캐시 suffix 대 전체 프리필, #1793)는 top-1 전부 일치와 마지막 위치 최대 절대 차이 1e-3. bf16 KV 캐시에는 맞지 않는 기준 후보

## 3. 변경 (실행 전 고정)

- `lh_kv_reuse.load_mlx(..., precision)`: fp32(기본, 기존 경로 그대로), bf16, q8, q4. #1811과 같은 방법(가중치와 KV 캐시 bf16, `model.model`만 그룹 32 양자화)
- `rt_lh_producer.py --precision`: 보고서에 `precision`, 블록 행에 `samples_to_ready`(suffix 준비부터 지평 도달까지 샘플 수)
- 동등성 검사에 위치 평균 KL(전체 프리필‖캐시 suffix)과 TV 추가. fp32는 기존 규칙, 그 외 정밀도는 KL 평균 ≤ 0.0009, TV 평균 ≤ 0.0139(#1811 q8 대 fp32 평균. KV 재사용 차이가 양자화 차이 자체보다 크지 않을 것)
- `rt_lh_smoke_run.py --precision`: 모든 시나리오 producer에 전달, 시나리오마다 `loop_speed`(블록별 `ready_s / samples_to_ready`, Python 샘플링·문법 마스크 포함)

## 4. 실행 (실행 전 고정)

- #1807과 같은 입력(data `8501d2ac…`, 왼손 음 목록 `lh_input_smoke_run2/lh_notes.json`, checkpoint `5e6b07ee…`)과 시나리오 세트(`--rerun`: A 정상, B 용량 260·keep 120 이른 crop, C 블록 2에서 600 ms 지연). 동등성 표본 A 블록 2, 3
- 팔 2개: fp32, q8. fp32 먼저, 이어서 q8. 같은 seed지만 logits가 달라 샘플 토큰은 팔마다 다름(블록 대응 비교 아님)
- 팔 시작 전·끝 뒤 열 압력 기록
- 오케스트레이터 `fine_tuning_art-tatum_midi_solo/.venv` python, producer는 `t3_aria/.venv` python(#1807 편차 3 재발 방지)

## 5. 판정 기준 (실행 전 고정)

- K1 계약: q8의 A, B, C 모두 #1807 `rerun_checks` 전부 참(확정 지평 앞 음 0, 접수 지각 거부 0, 용량 안, 기록된 중단 또는 정상 종료, 안전 종료, 기존 검사). B crop 1회 이상, C 블록 2 `deadline_miss` 중단. fp32도 같은 기준(새 계측 코드 회귀 확인)
- K2 KV 재사용: q8 동등성 표본 2개 모두 KL·TV 상한 안. fp32 표본은 기존 규칙
- K3 루프 속도: A와 B의 warm 블록을 합친 `ms_per_sample` 중앙값. q8 ≤ 0.75 × fp32면 루프 안 속도 이득
- 마감 여유 최소·중앙값은 기록만(샘플 토큰이 달라 판정 없음)
- 결론 규칙
  - K1, K2, K3 모두 충족: q8 실시간 기본값 전환을 별도 단위로 제안(전환 전 짧은 청취 비교 포함 여부는 사용자 결정)
  - K2 미충족: q8 실시간 보류, 동등성 차이 원인 분리 단위
  - K1 미충족: 실패 기록, 해당 검사 원인 분리
  - K3 미충족: #1811 forward 이득이 루프에서 사라지는 지점(샘플링·마스크 비용 등) 분리

## 6. 범위 밖

- 촘촘한 블록 재현, 밀도 상한, q4, `mx.compile`
- 실시간 기본값 변경, 청취

## 7. 재현

- `rt_lh_smoke_run.py <t3 python> <aria repo> <ckpt> <data json> <lh notes json> <out dir> <report json> --rerun --precision fp32|q8`
- 시험 `tests/test_lh_kv_reuse.py`(정밀도 동등성 규칙), `tests/test_rt_lh_input.py`(`loop_speed`)
- 원자료 저장소 밖 `/Users/ohhalim/git_box/q8_realtime_smoke_run1/`, 보고 `docs/experiments/q8_realtime_smoke/{fp32,q8}.json`

## 8. 첫 실행 무효 (2026-10-18 21:56–21:58, #1806 미포함)

- 실행 브랜치(15ddd115)가 #1808 첫 head(3e9c3a54) 위라 #1806(crop 위치에 왼손 길이 반영) 미포함. #1807 첫 재실행과 같은 원인(아스트라 H1)
- 원자료 저장소 밖 `q8_realtime_smoke_run1/`, 보고 `q8_realtime_smoke/run1_without_1806_{fp32,q8}.json`으로 보존. 결론·기본값 전환 근거로 쓰지 않음
- K2(사전 지정 KL·TV 구간)는 구간 충족 여부로만 기록: q8 표본 2개 KL 평균 0, TV 평균 0, 마지막 위치 최대 절대 차이 0으로 구간 안. 이 기준은 모델 정밀도 차이를 계산 경로 차이의 상한으로 쓴 것이라 동등성 판단으로 읽지 않음(아스트라 M2). 동등성은 #1815 계약으로 분리
- 당시 판정기(fail-open) 기준 관측, 판정 아님
  - fp32: A·B·C 기준 참, 최소 여유 A 47.0 / B 109.9 ms, 루프 ms/sample 중앙값 A 11.12 / B 10.99
  - q8: A 최소 여유 174.5 ms, B 블록 15 `deadline_miss`(여유 −8.8 ms)로 중단, C 블록 2 중단. 루프 ms/sample 중앙값 A 6.59 / B 6.13
  - q8 B의 블록 15 마감 초과는 q8에서도 마감 초과가 남을 수 있다는 관측. 원인 후보(블록 15 생성 샘플 수, crop 뒤 문맥) 미분리
- 다음: develop 병합(33a29227, #1806·#1808 판정기 포함) 완료. #1815 계약 결과 뒤 producer 동등성 표본을 #1815 규칙(같은 정밀도 바닥 대비)으로 바꾼 v2 사전 등록 후 새 디렉터리로 재실행
