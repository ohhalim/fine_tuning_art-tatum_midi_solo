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

---

## 결과 (측정 후 추가)

증거는 `docs/experiments/lora_merge/`에 있다. 원본 런타임 리포트는 `outputs/lora_merge/runtime_{nomerge,merge}/`(gitignore)에 있다.

### 기준 1·2 — 동등성 (`verify.json`, CPU)

| 모델 | logits 최대 차이 | 동일 블록 | wall 병합 전→후 |
|---|---|---|---|
| Tatum 완성 (out_proj+QKV) | 0.0 | 32/32 | 11.7 s → 7.8 s (1.50배) |
| Tatum16 (out_proj) | 0.0 | 32/32 | 6.48 s → 6.39 s (1.01배) |
| 멜다우16 (out_proj) | 0.0 | 32/32 | 4.66 s → 4.64 s (1.01배) |

~~차이가 정확히 0인 이유: … 속도 이득은 QKV 모델에서만 크다. out_proj만 쓰는 모델은 delta 계산이 층당 한 번뿐이어서 차이가 거의 없다.~~ **이 해석은 틀렸다. 아래 「정정 (#1520)」을 볼 것.** 실제로는 out_proj LoRA가 병합되지 않았다. 표의 out_proj 모델 행은 병합 전후가 같은 모델을 비교한 것이다.

### 기준 3 — 런타임 (Tatum 완성, 128 BPM, 코드 primer, 16마디, CPU, KV 캐시 켬)

| arm | p50 ms (s42/43/44) | p95 ms (s42/43/44) | fallback | deadline miss |
|---|---|---|---|---|
| 병합 안 함 | 640 / 589 / 646 | 953 / 969 / 921 | 0 | 2 (s42) |
| 병합 | 436 / 371 / 415 | 613 / 582 / 586 | 0 | 0 |

- p50 평균 625 → 408 ms (**−35%**, 기준 −20%)
- p95 평균 948 → 594 ms (−37%)
- fallback은 둘 다 0이다
- **연주된 16마디 음높이 시퀀스가 seed 3개 모두 병합 전과 동일**하다(16/16마디). `played_bars.json`의 chord-tone 비율 0.462, 음 수 390도 같다

### 판정
기준 1–3을 모두 충족했다. `run_continuous_jazz.py`의 `--merge-lora` 기본값을 **True**로 바꿨다(`--no-merge-lora`로 이전 경로를 쓸 수 있다). LoRA가 없는 체크포인트에서는 병합해도 아무것도 바뀌지 않는다(단위 테스트로 확인).

한계
- 측정은 M1 Max CPU 한 대에서 seed 3개로 했다
- 같은 머신에서 다른 무거운 작업과 동시에 측정하지 않았다
- 스타일·음악 품질은 검증하지 않았다(`style_verified: false`, `musical_quality_verified: false`)

---

## 정정 (#1520) — out_proj·FFN LoRA가 병합되지 않았다
**관측:** 어댑터 스왑(#1519)을 구현하면서 병합한 Tatum 완성 모델에 `out_proj.lora_A/B`가 남아 있는 것을 발견했다. 병합 결과는 `{"linear": 0, "qkv": 6}`이었다.

**원인:**
- `scripts/generate.py`는 이 파일을 `from train_qlora import ...`(최상위 모듈)로 가져온다
- 런타임과 검증 스크립트는 `scripts.train_qlora`로 가져온다
- 같은 파일이 두 모듈이 되어 `LoRALayer` 클래스가 두 개 생긴다. `isinstance(child, LoRALayer)`가 항상 거짓이었다
- 단위 테스트는 두 곳 모두 `scripts.train_qlora`로 만든 모델을 써서 이를 잡지 못했다
- QKV는 파라미터 이름으로 감지해서 병합됐다

**영향**
- 출력은 처음부터 맞았다. 병합되지 않은 out_proj는 기존 경로 그대로 계산됐다. 사전 기준 1–3의 판정도 유지된다
- 틀린 것은 내 보고다
  - "out_proj·FFN도 병합한다"는 설명
  - out_proj 모델에서 속도 차이가 없는 이유를 "층당 한 번이라서"라고 한 해석
  - 기준 1의 out_proj 모델 행(병합 전후가 같은 모델)

**수정**
- 래퍼를 `isinstance`가 아니라 형태로 감지한다(`original_layer`가 `nn.Linear`이고 `lora_A`, `lora_B`, `scaling`이 있음)
- 병합 후 `lora_` 텐서가 하나라도 남으면 `RuntimeError`를 낸다
- 회귀 테스트는 최상위 `train_qlora`로 만든 모델을 병합한다. 수정 전 코드에서 실패하는 것을 확인했다(`{'linear': 0, 'qkv': 2}`)

**재검증 (`lora_merge/verify_fix1520.json`, CPU)**

| 모델 | 병합 수 | 남은 LoRA 텐서 | logits 최대 차이 | 동일 블록 | 속도 (수정 전 → 후) |
|---|---|---|---|---|---|
| Tatum 완성 | linear 6 + QKV 6 | 0 | 0.0 | 32/32 | 1.50배 → **2.11배** |
| Tatum16 | linear 6 | 0 | 0.0 | 32/32 | 1.01배 → **1.31배** |
| 멜다우16 | linear 6 | 0 | 0.0 | 32/32 | 1.01배 → **1.31배** |

**런타임 재측정 (Tatum 완성, 128 BPM, 16마디, seed 42/43/44, `lora_merge/played_bars_fix1520.json`)**

| arm | p50 ms | p95 ms | 최대 ms | fallback | miss |
|---|---|---|---|---|---|
| 병합 안 함 | 625 | 948 | 1,037 | 0 | 2 |
| QKV만 병합(수정 전) | 408 | 594 | 648 | 0 | 0 |
| **전부 병합(수정 후)** | **307** | **463** | **496** | 0 | 0 |

- 연주된 16마디 음높이는 seed 3개 모두 병합 안 함과 16/16 동일하다
- 기준 3의 p50 감소는 −35%가 아니라 **−51%**다
