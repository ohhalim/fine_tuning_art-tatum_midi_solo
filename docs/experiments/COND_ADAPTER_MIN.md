# 왼손 조건 adapter 최소 학습 (#1760) — 사전 등록

작성 2026-10-16. 아스트라가 #1759 리뷰에서 제안한 A–G를 따른다. 이 문서는 단계별로 고정한다. 지금 고정하는 것은 **F(학습 전 KV 검증)**다. E(메모리·시간 확인)와 학습·비교 단계는 F를 통과한 뒤 수치를 채우고, 실행 전에 다시 고정한다.

## 목표

블록 요청 단위 입력(#1756)과 crop 계약(#1758)으로 작은 adapter를 32 step 학습한다. 그 뒤 같은 문맥에서 현재 요청의 조건을 실제 조건과 다른 곡 조건으로 바꿨을 때 예측 손실이 달라지는지 본다. 조건을 켠 상태가 끈 상태보다 나빠지지 않는지도 본다. 음악성이나 개인화가 좋아졌다는 판단은 하지 않는다.

## 경로 (A)

- Aria medium은 고정한다(eval, 가중치 해시 불변)
- 조건 입력은 13차원이다(pc12 multihot + available). bias 없는 `Linear(13 → 1536)`이 마지막 encoder 층의 입력에 더해진다. 기존 torch adapter(`aria_cond_smoke.py`의 pre-hook)와 같은 위치다
- 실시간 경로(MLX)에서는 `scripts/aria_mlx_cond.py`의 `cond_forward`가 upstream forward를 그대로 따라가고 이 주입만 더한다. upstream 코드는 바꾸지 않는다
- 이 경로를 다시 쓴다고 안전한 것은 아니다. 기존 late-injection은 손해와 문법 악화를 낸 기록이 있다

## F. 학습 전 KV 검증 — `scripts/aria_mlx_cond_kv_check.py` (지금 고정)

0 가중치로 비교하면 잘못된 주입도 통과한다. 그래서 **고정된 비0 가중치**(seed 1760, 정규분포 × 0.05)를 쓴다.

| 항목 | 값 |
|---|---|
| 곡 | #1756 3회차 내보내기의 train 곡 0번, 7번 (실행 전 고정) |
| 구간 | 곡마다 앞 500음, 그 뒤 `<E>` |
| crop 설정 | CAP 16, 용량 384, keep 192. 짧은 구간에서도 crop이 여러 번 일어나게 하려고 작게 잡았다 |
| 실시간 경로 | 세션 하나에 KV 캐시 하나를 쓴다. crop이면 캐시를 비우고 `[prefix, <S>]` + 남긴 문맥을 원래 조건으로 prefill한다. 그다음 p0를 현재 조건으로 다시 쓰고 continuation을 한 토큰씩 넣는다 |
| 기준 | 새 캐시에 같은 세션 토큰과 같은 위치별 조건을 한 번에 prefill한다 |
| 통과 | 비교한 모든 위치에서 top-1이 같고 최대 \|logit 차\| ≤ 1e-3이다(#1741 KV smoke와 같은 기준) |
| 함께 확인 | 주입이 실제로 작동한다(실제 조건과 0 조건의 logit 차 > 1e-2). 비교 구간에 crop ≥ 2, 조건 변경 > 0, cold > 0, warm-empty > 0이 들어 있다 |

- 결과를 보고 허용 오차나 곡을 바꾸지 않는다. 기준에 못 미치면 숫자를 그대로 적고 "동등성 미확인"으로 둔다
- 위치별 조건 배정(어느 위치가 어느 요청의 조건을 받는지)이 맞는지는 #1758에서 순차 재연으로 이미 확인했다. F는 그 배정이 주어졌을 때 KV 경로가 전체 재계산과 같은지만 본다
- 실시간 producer(`rt_producer.py`)에는 아직 조건 주입이 없다. F는 MLX 모델 위의 검증이고, 실시간 준비가 끝났다는 뜻이 아니다
- 학습 뒤에는 저장했다가 다시 불러온 가중치로 같은 검증을 한 번 더 한다

```
/Users/ohhalim/git_box/t3_aria/.venv/bin/python scripts/aria_mlx_cond_kv_check.py /Users/ohhalim/git_box/t3_aria/aria \
  /Users/ohhalim/git_box/t3_aria/ckpt/model-gen.safetensors /Users/ohhalim/git_box/lh_request_export_run3 \
  docs/experiments/cond_adapter_min/kv_check.json
```

### 실행 전에 마친 시험

- `tests/test_aria_mlx_cond.py`(Aria venv, 작은 무작위 모델): 4개 통과
  - 0 가중치면 upstream forward와 같다
  - 비0 가중치면 logit이 바뀌고, 한 토큰씩 넣어도 전체 계산과 같다
  - 위치를 새 조건으로 다시 쓰면 처음부터 계산한 것과 같다

## 이후 단계 (F 통과 뒤 수치를 채워 다시 고정)

- **E. 메모리·시간:** 가장 긴 허용 입력(세션 문맥 최대 3,965 + window)으로 batch 1, fp32 forward/backward/1 step을 확인한다
  - 조건이 마지막 층에만 들어가므로, 아래 15층은 기울기 없이 계산할 수 있다(Claude 판단, 검토 대상)
  - OOM이나 시간 상한에 걸리면 기록하고 설계를 바꾼다. 긴 예제를 몰래 지우지 않는다
- **데이터(B):** train 189곡 안에서 학습 8곡과 확인용 4곡을 고른다. 고르는 규칙은 출력을 보기 전에 정한다(같은 곡 제목은 한쪽에만 둔다). val/test는 열지 않는다. 빈 window와 CAP 초과 요청은 빼고 수를 적는다. boundary에는 loss를 걸지 않는다
- **학습:** window CE로 32 update를 1회 한다. LR은 실행 전에 적는다
- **비교(C):** (a) adapter off (b) 실제 조건 (c) 다른 곡 조건. 과거 문맥 조건은 같게 두고 현재 요청 조건만 바꾼다
  - wrong 조건은 available이 같고 PC 수가 같은 다른 집합으로 정한다. 정답이나 결과를 보고 고르지 않는다
  - pitch/onset/dur/`<T>`별 NLL과 곡 평균을 따로 낸다. cold와 warm-empty는 따로 본다
- **해석(D):** b < c인데 둘 다 a보다 나쁘면 "조건은 쓰지만 손해"로 기록한다. available = 1이 만드는 공통 효과를 음악 조건 효과로 보지 않는다
- **생성(G):** 확인용 곡에서 2 primer × 2 seed로 a/b/c 짧은 생성을 하고, 문법 오류와 cap 종료를 확인한다
