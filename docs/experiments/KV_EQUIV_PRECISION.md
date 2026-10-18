# 저정밀도 Aria 캐시 suffix 동등성 검증 계약 (사전 등록)

작성 2026-10-18 22시대 KST. 이슈 #1815. 오프라인 forward만. 실시간 실행, 기본값 전환, 청취 없음. `quality_claimed: false`.

## 1. 질문

- 같은 정밀도(fp32, bf16, q8) 안에서 producer식 캐시 suffix 경로가 정상 계산 경로 변동 범위 안인지
- 같은 기준이 실제 참조 가능한 과거 KV를 잘못 유지하는 결함을 검출하는지

## 2. 배경

- #1793 동등성 기준(top-1 전부 일치, 마지막 logits 최대 절대 차이 1e-3)은 fp32 전제
- #1811: q8 디코드 0.54배, fp32 대비 KL 평균 0.0009
- #1813 사전 등록 K2: q8 대 fp32 평균(모델 정밀도 차이)을 q8 캐시 대 q8 전체 prefill(계산 경로 차이)의 상한으로 사용. 검증 대상이 다른 오차의 혼용(아스트라 M2). #1813 첫 실행은 #1806 미포함 기반이라 무효, 원자료만 보존
- 아스트라 설계 검토 반영: 미래 위치 stale KV는 causal mask로 가려져 결함 대조가 아님(비가시성 양성 대조로 분리), 결함 대조는 생략 위치를 채우지 않고 자기 위치에서 비교, 바닥은 정밀도×사례×지표별, top-1은 배수 판정 제외

## 3. 사례 (실행 전 고정)

- #1793 `--equiv` 사례: 합성 6(insert, expired, held_update, none, empty, same_onset) + dev 원인별 첫 비crop 쌍 4
- #1795 사례: 긴 P 4(1692, 2142, 2877, 3343 근처), S = 0(`s0`), 첫 crop 쌍
- 입력: export `lh_request_export_run3`, velocity `lh_context_run1/human_velocity`, #1793 `count.json`, checkpoint `5e6b07ee…`
- 재사용 시작 p: #1795 `reuse_start`(공통 앞부분과 KV 유효 끝 중 작은 값, crop이면 0, S = 0이면 마지막 토큰 재계산)

## 4. 계산 (정밀도·사례마다, 모든 변형은 poison 4,096토큰으로 캐시를 덮은 뒤 시작)

| 이름 | 계산 | 역할 |
|---|---|---|
| R0 | N 전체 prefill | 기준 |
| R1 | R0 반복 | 완전 재현 확인 |
| R2 | N[:p] prefill 후 N[p:] 한 조각 | 바닥(분할점 p) |
| R3 | N[:p] prefill 후 N[p:] 토큰별 | 바닥(incremental) |
| R4 | N[:p//2] prefill 후 N[p//2:] | 바닥(분할점 민감도) |
| C | Pseq prefill 후 N[p:]를 p부터, max_kv_pos = len(N) − 1 | 시험 대상(producer 경로) |
| PC_future | N prefill(뒤는 poison) 후 N[p:]를 p부터, max_kv_pos = 4,095 | 비가시성 양성 대조 |
| NC_unfed | Pseq[:lcp − 8]만 쓴 캐시에서 N[lcp:]를 lcp부터 | 결함: KV 유효 끝 무시 |
| NC_skip | Pseq 전체 캐시에서 N[s:]를 s부터, s = min(len(Pseq), len(N)) − 1 | 결함: 첫 변경 뒤 재계산 누락 |
| NC_crop | crop 쌍에서 NC_skip과 같은 계산 | 결함: crop 뒤 이전 위치 KV 유지 |

- 결함 대조 적용 조건: 재시작 위치 앞에 N과 다른 토큰이 실제로 있고, suffix가 비어 있지 않음. NC_unfed는 p = lcp이고 lcp − 8 ≥ 1
- 결함 대조는 계산한 위치(재시작 위치부터 끝)에서만 비교. 생략 위치를 채우지 않음
- 마지막 위치 문법 마스크: N 전체 뒤의 next-token 문법 상태, `<D>`·`<E>` 제외, 모든 변형에 같은 마스크

## 5. 지표

- R0 대비, 위치 p..len(N) − 1: 마지막 위치 KL·TV, 위치별 KL 최대·p95, TV 최대, 마지막 위치 마스크 재정규화 KL·TV
- float64 안정 log-softmax. logits 비유한 값, 허용 토큰 0개는 실패(작은 차이로 처리하지 않음)
- top-1 마지막 위치 일치 여부와 R0의 top-2 확률 차이는 기록만

## 6. 판정 기준 (실행 전 고정, 경험적 운영 기준이며 수학적 동등성 보장 아님)

- 바닥: 정밀도×사례×지표별로 R1~R4 중 적용 가능한 것의 최대값. 결함 대조는 자기 위치로 자른 R1~R4로 따로 계산
- 무효: R1이 R0와 완전 일치하지 않는 사례는 측정 무효(캐시 결함 확정 아님)
- C 통과: 모든 지표 ≤ 2 × 바닥 + ε(KL 1e-7, TV 1e-6)
- PC_future 통과: 같은 규칙
- 결함 검출: 결함 대조의 KL 최대 > 10 × 자기 위치 바닥 KL 최대 + 1e-4
- 결함 미검출은 그 사례에서 검출력 불충분. C 통과로 바꾸지 않음. 결과를 본 뒤 배수 완화 없음
- 결론 규칙(정밀도별)
  - 무효 0, C 전부 통과, PC_future 전부 통과, 적용된 결함 대조 전부 검출: 그 정밀도에서 이 계약으로 캐시 suffix 경로 이상 미관측
  - 결함 미검출 있음: 해당 사례·정밀도에서 계약 검출력 불충분으로 기록, C 결론 보류
  - C 미통과: 해당 정밀도 캐시 경로 점검 단위로 분리
- fp32도 같은 규칙으로 판정(기존 #1793 기준과 함께 기록)

## 7. 범위 밖

- #1813 실시간 재실행과 q8 기본값 전환. 이 계약 결과와 무관하게 별도 단위
- 실시간 producer의 여러 조각에 걸친 Pseq 기록 순서 재현(이 계약은 Pseq를 한 조각으로 씀)

## 8. 재현

- `scripts/kv_equiv_precision.py <aria repo> <ckpt> <export dir> <velocity dir> <count json> <out json> [--precisions fp32,bf16,q8]`
- 시험 `tests/test_kv_equiv_precision.py`
- 보고 `docs/experiments/kv_equiv_precision/report.json`
