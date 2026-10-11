# Aria MLX KV 캐시 기반 점검 (#1737) — 사전 등록

종합 조사(#1735, PR #1736)는 실시간 런타임을 Aria MLX의 지속 KV 캐시로 옮기자고 제안했다. 아스트라 리뷰는 착지 바이어스·CFG·재순위·새 학습을 한꺼번에 하지 말고, **기반이 실제로 동작하는지부터** 보자고 했다. 이 문서는 그 기반 점검이다. 소리나 코드 제어가 나아지는지는 보지 않는다.

## 질문

1. 같은 토큰열에서 문맥을 한 번에 넣은 결과(full prefill)와 KV 캐시에 한 토큰씩 이어 넣은 결과(incremental)가 같은가?
2. 캐시를 앞 위치로 되돌려(rollback) 다른 토큰을 이어 넣거나, 쓰던 캐시를 다시 쓸 때(reset)도 같은가?
3. 같은 결정적(argmax) 생성이 두 경로에서 같은 토큰·같은 음 시각을 내는가?
4. batch 1의 실제 지연은 얼마인가? cold와 warm을 나눈다

## 고정 조건

- 로컬 `t3_aria` Aria 저장소(커밋 c2f67bc, mlx 0.25.2, `max_kv_pos` 로컬 패치는 #T3에 기록), `model-gen.safetensors`
- 가중치와 KV 캐시 모두 float32. `aria/inference/sample_mlx.py`와 같다
- 토큰열은 이미 저장된 #1731 마스크 생성에서 고정한다. 그 base 스트림은 #1726 마스크 없는 생성과 같다
  - A: prompt 0 + base seed 1 스트림
  - B: prompt 0 + base seed 2 스트림
- 캐시는 고정 크기다. rollback은 앞 위치부터 다시 써 넣고, `max_kv_pos`로 보이는 범위를 자르는 방식이다

## 검사와 기준 (실행 전 고정)

| ID | 무엇 | 기준 |
|---|---|---|
| E1 동등성 | A의 full prefill 대 prompt prefill 뒤 한 토큰씩 decode | 비교한 모든 위치에서 top-1이 같고, 최대 \|logit 차\| ≤ 1e-3 |
| E2 rollback | E1 뒤 같은 캐시에서 A와 B가 처음 갈리는 위치부터 B를 덮어쓴 결과 대 B의 새 full prefill | 같음 |
| E3 reset | E2에서 쓰던 캐시로 A를 위치 0부터 다시 prefill한 결과 대 E1의 full prefill | 같음 |
| E4 결정적 생성 | prompt 뒤 argmax 128토큰: incremental 대 매 step full prefill | 토큰이 모두 같고, 풀어낸 음(pitch, onset, end, velocity)도 같음 |
| L 지연 | cold 첫 prefill / warm prefill 128·256·512·1024토큰(5회 중앙값) / 문맥 256·512·1024에서 한 토큰 decode(64회 중앙값) / 512토큰 뒤 24토큰 decode 블록 | 보고만 |
| R 참고 | MLX full prefill 대 PyTorch MPS 같은 토큰열 logits | 보고만 |

- 기준에 못 미치면 숫자를 그대로 적고 "동등성 미확인"으로 둔다. 허용 오차를 결과를 보고 바꾸지 않는다
- 지연은 128 BPM 기준 1박 0.469초, 반 마디 0.9375초, 지금 런타임의 PyTorch MPS 블록 0.674초와 나란히 적는다. 합격선은 두지 않는다

## v1 무효와 수정 (아스트라 #1738 리뷰 H1)

- v1 실행(085b9f95)은 캐시 용량을 750으로 잡고 1024토큰 prefill과 문맥 1024 decode를 쟀다. 캐시 크기를 넘은 위치에 써 넣은 것이라, 예외가 나지 않았어도 유효한 1024 문맥이 아니다. **v1의 1024 지연 행은 무효다.** 원자료는 `report_v1_capacity750_invalid1024.json`으로 남긴다
- 수정: 캐시 용량을 모든 검사가 쓰는 최대 위치 + 1(1024)로 미리 계산한다. 모든 호출에서 forward 전에 `0 ≤ start`, `start + 길이 ≤ 용량`을 검사하고, 범위를 넘는 호출이 거절되는지 시험한다. 실제 캐시 shape와 재현 정보(checkpoint·data·generation SHA, MLX 버전, Aria 커밋, 로컬 패치 hash)를 기록한다
- 용량이 바뀌면 다른 길이의 비용도 달라질 수 있으므로 **전부 다시 돌린다.** E1~E4 기준은 그대로다

## 하지 않는 것

착지 바이어스, CFG, 양자화(bf16 포함), 후보 재순위, 새 학습, 다운로드, 소리·제어 개선 주장. 실시간 런타임 계약(확정 송출 경계, 입력 삽입 뒤 캐시 무효화, 어댑터 교체 재prefill, note-off 소유권)은 다음 단위에서 다룬다.

## 실행

```
scripts/aria_mlx_kv_smoke.py <aria repo> <checkpoint> <masked generations json> <data json> <report json>
```
t3_aria `.venv`. 결과는 `aria_mlx_kv_smoke/report.json`.

## 결과 (v2) — 다섯 검사 모두 통과. 시험한 토큰열에서 이어 생성한 결과가 처음부터 계산한 것과 허용 오차 안에서 같다

원자료: `aria_mlx_kv_smoke/report.json`(v2). 실행 2026-10-12, 44초. 열 압력은 시작과 끝 모두 0(Nominal)이었다.

**재현 정보**
- 캐시 용량 1,024, 실제 shape: k_cache [1, 24, 1024, 64], causal_mask [1024, 1024]
- checkpoint `5e6b07ee2680f004`, data `8501d2aca038d763`, generation `99ebff90047254bf`
- MLX 0.25.2, Aria 커밋 `c2f67bc90b49`, 로컬 패치 sha256 `f648d574362f3f55`(20줄)

| 검사 | 비교 위치 | 최대 \|logit 차\| | top-1 일치 | 판정 |
|---|---|---|---|---|
| 범위 밖 호출 거절 | 위치 1023~1024, 용량 1024 | — | forward 전에 거절 | 통과 |
| E1 incremental = full | 101..612 | 1.3e-04 | 512/512 | 통과 |
| E2 rollback | 101..612 | 6.6e-05 | 512/512 | 통과 |
| E3 쓰던 캐시 재사용 | 0..613 | 0.0e+00 | 614/614 | 통과 |
| E4 argmax 128토큰 | 생성 128토큰 | — | 토큰 128/128, 음 75개 동일 | 통과 |

- E1~E4는 위치 613 이하만 쓰므로 v1(용량 750)과 수치가 같았다
- **말할 수 있는 범위:** 시험한 prefix에서 logit이 허용 오차 안이고 top-1이 일치하며, argmax 128토큰이 같다. 확률 샘플링에서 토큰이 일치한다는 보장은 아니다(아스트라 리뷰 LOW)
- **한계:** A와 B는 prompt 바로 다음 토큰(위치 102)부터 갈라졌다. 그래서 E2는 prompt 끝으로 되돌린 경우만 시험했다. 생성 중간 rollback과, 취소한 뒤쪽 토큰이 KV에서 보이지 않는지는 다음 단위에서 시험한다
- **참고 R:** MLX full prefill과 PyTorch MPS의 같은 토큰열 logits는 최대 차이 1.2e-04, top-1 614/614였다(float32, base 가중치). LoRA나 새 임베딩을 얹은 경우는 확인하지 않았다

**지연 (batch 1, float32, warm은 중앙값)**

| 항목 | v2 | v1 (참고) |
|---|---|---|
| cold 첫 prefill (614토큰) | 0.221 s | 0.244 s |
| warm prefill 128 / 256 / 512토큰 | 0.043 / 0.073 / 0.135 s | 0.042 / 0.073 / 0.133 s |
| warm prefill 1024토큰 | 0.280 s | 무효 |
| 한 토큰 decode, 문맥 256 / 512 | 10.9 / 10.5 ms | 10.3 / 10.6 ms |
| 한 토큰 decode, 문맥 1024 | 11.0 ms | 무효 |
| 512토큰 뒤 저장 토큰 24개 decode | 0.252 s | 0.252 s |
| warm prefill 128토큰 최대 | 0.184 s | 0.852 s |

- 토큰 하나는 문맥 256~1024에서 약 11 ms다
- **24토큰 0.25초의 의미:** 이미 prefill된 512토큰 뒤에 저장된 토큰 24개를 teacher-force로 decode한 시간이다. 동기화와 numpy 복사를 포함하지만, prefill이나 샘플링은 빠져 있다. 지금 런타임의 0.674초(T4a: prefix 237~408토큰, 48토큰 생성, prefill 포함, 블록 준비까지)와는 작업량과 포함 구간이 달라서, 비율을 성능 개선율로 쓰지 않는다. 128 BPM 1박(0.469초)보다 짧다는 것도 warm decode 부분만의 이야기다. 키 입력에서 소리까지의 마감을 통과했다는 뜻이 아니다(아스트라 리뷰 M1)
- warm prefill 128토큰의 최대는 v1 0.85초, v2 0.18초로 튀었다. 원인은 확인하지 않았다. 다음 단위에서 prefix 길이별 첫 호출 비용을 따로 잰다

## 해석

- **확인한 것:** 로컬 Aria MLX(float32)에서 KV 캐시로 이어 넣는 경로는, 시험한 토큰열에서 처음부터 계산하는 경로와 logit이 허용 오차 안이고 top-1과 argmax 출력이 같다. 캐시를 덮어쓰거나 다시 써도 같았다. 범위 밖 위치는 forward 전에 거절된다
- **확인하지 않은 것:** 생성 중간 rollback, 취소된 suffix의 비가시성, prefix 길이별 첫 호출 비용, bf16, LoRA·새 제어 임베딩을 얹은 MLX 모델, 실시간 계약(확정 송출 경계, 입력 삽입 뒤 캐시 무효화, 어댑터 교체 시 full reset, note-off 소유권), 키보드 입력부터 소리까지의 지연
- 소리나 코드 제어가 나아졌다는 주장은 하지 않는다. 이것은 기반 점검이다
