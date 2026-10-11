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

## 하지 않는 것

착지 바이어스, CFG, 양자화(bf16 포함), 후보 재순위, 새 학습, 다운로드, 소리·제어 개선 주장. 실시간 런타임 계약(확정 송출 경계, 입력 삽입 뒤 캐시 무효화, 어댑터 교체 재prefill, note-off 소유권)은 다음 단위에서 다룬다.

## 실행

```
scripts/aria_mlx_kv_smoke.py <aria repo> <checkpoint> <masked generations json> <data json> <report json>
```
t3_aria `.venv`. 결과는 `aria_mlx_kv_smoke/report.json`.

## 결과

(실행 뒤 추가)
