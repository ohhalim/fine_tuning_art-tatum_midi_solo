# 추론 전 LoRA 병합 — 사전 등록

작성 2026-09-29. 이슈 #1512, 브랜치 `perf/issue-1512-lora-merge`.

**이 절은 구현·측정 전에 작성했다.**

## 문제
Tatum 완성 모델(#1509, out_proj+QKV)의 16마디 생성 p50은 약 690 ms로, out_proj 모델(약 358 ms)의 약 2배다. LoRA는 forward마다 delta를 다시 계산한다. QKV는 `in_proj_weight` property가 `B@A` 세 개를 계산하고 이어 붙이며, out_proj는 `weight` property를 쓴다.

## 설계
`merge_lora_for_inference(model)`
- out_proj·FFN의 `LoRALayer`는 병합한 가중치를 담은 `nn.Linear`로 바꾼다
- QKV는 `in_proj_weight` Parameter에 delta를 더하고, LoRA 파라미터와 서브클래스를 제거한다

추론(eval) 전용이고 되돌리지 않는다. 학습·평가 스크립트는 바꾸지 않는다. 런타임 옵션은 `--merge-lora/--no-merge-lora`다.

## 채택 기준 (모두 충족해야 런타임 기본값을 켠다)
1. **logits:** 3개 체크포인트(Tatum 완성, Tatum16, 멜다우16)에서 병합 전후 최대 절대 차이 **< 1e-4**(CPU)
2. **토큰:** 코드 primer 블록 조건, seed 1–8 × 코드 4개에서 생성 토큰이 **전부 동일**(KV 캐시 켠 상태)
3. **속도:** Tatum 완성 모델 16마디(seed 42/43/44)에서 fallback이 늘지 않고, 생성 p50이 병합 전보다 **20% 이상 감소**

1·2에 실패하면 합치지 않는다. 3에 미달하면 opt-in으로만 합친다.
