# Tatum 어댑터 용량 — LoRA 타깃 확장 (사전 등록)

작성 2026-09-27. 이슈 #1492, 브랜치 `exp/issue-1492-lora-targets`. 선행: `TATUM_PERSONALIZATION.md`(T1).

`tatum_style_verified: false` · `musical_quality_verified: false` · 청취 없음

**이 절은 실행 전에 작성했다.**

## 질문
같은 update 예산에서 LoRA를 QKV, 나아가 FFN까지 넓히면 **어댑터가 보지 않은 Tatum val 12곡**의 특화가 커지는가?

## 팔
| 팔 | LoRA 타깃 | 비고 |
|---|---|---|
| A | out_proj | T1 run 재사용(`outputs/tatum_diag/update_budget_t1`). 같은 스크립트·설정·seed |
| B | out_proj + QKV(in_proj의 q/k/v 각각 r16) | 신규 |
| C | out_proj + QKV + FFN(linear1, linear2) | 신규 |

공통: armB ep8 시작, r16/α32, lr 3e-4, cosine(518 update), batch 4, accumulation 4, LS 0.1, seed 42, MPS, `data/tatum_full`. 학습 스크립트는 `run_mehldau_update_budget_diag.py`다.

## 평가
- `eval_mehldau_snapshots.py`, CE만 잰다(`--no-generate`). T1과 같은 crop과 같은 일반 probe(`tatum_generic_probe_list.json`)를 쓴다
- snapshot은 T1과 같은 update(0/14/35/70/133/259/385/518)다

## 판정
- 각 팔의 **최종 update 518** 특화도(val) = ΔCE_Tatum_val − ΔCE_일반
- **B 또는 C가 A보다 특화도를 0.01 이상 더 낮추고(더 특화), 그 팔의 ΔCE_일반 ≤ +0.02이면 "타깃 확장이 특화를 키운다"로 기록한다**
- 채택: 조건을 만족하는 팔 중 **파라미터가 적은 쪽**을 새 배포 후보로 한다. 없으면 A(u518)를 유지한다
- 채택 팔은 런타임 스모크(240 BPM, 8마디)로 생성 p95가 A 대비 1.2배 이하인지 확인한다. 초과하면 채택을 보류하고 기록한다
- 생성 기반 탐색 지표(속주 밀도 등)는 채택 팔에만, 판정과 별개로 보고한다

## 결과 — B(out_proj + QKV) 채택

원시 결과: `docs/experiments/tatum_targets/*.json`. 팔 A는 T1(`mehldau_diag/tatum_snapshot_eval_t1_report.json`)이다.

| 팔 | 타깃 | 학습 파라미터 | 학습 wall (MPS) | ΔCE Tatum val | ΔCE 일반 | **특화도 (u518)** | 판정 |
|---|---|---|---|---|---|---|---|
| A | out_proj | 98,304 (0.72%) | 850 s | −0.068 | −0.004 | −0.063 | 기준 |
| **B** | + QKV | 393,216 | 924 s | **−0.096** | **+0.015** | **−0.111** | **충족** (A보다 −0.048, 일반 ≤ +0.02) |
| C | + QKV + FFN | 688,128 | 1,182 s* | −0.111 | **+0.025** | −0.136 | **미충족** (일반 +0.025 > +0.02) |

\* C의 학습 시간은 B 평가와 겹쳐 실행돼 부풀려졌다. CE 값에는 영향이 없다.

update별 특화도(val):

| update | 14 | 35 | 70 | 133 | 259 | 385 | 518 |
|---|---|---|---|---|---|---|---|
| A | −0.007 | −0.016 | −0.027 | −0.042 | −0.058 | −0.062 | −0.063 |
| B | −0.018 | −0.037 | −0.056 | −0.078 | −0.100 | −0.110 | −0.111 |
| C | −0.023 | −0.045 | −0.068 | −0.095 | −0.121 | −0.135 | −0.136 |

| update | 14 | 35 | 70 | 133 | 259 | 385 | 518 |
|---|---|---|---|---|---|---|---|
| ΔCE 일반 B | −0.016 | −0.012 | −0.007 | +0.001 | +0.012 | +0.016 | +0.015 |
| ΔCE 일반 C | −0.015 | −0.011 | −0.005 | +0.008 | +0.018 | +0.026 | +0.025 |

- **타깃 확장은 보지 않은 Tatum 곡의 특화를 키운다.** 모든 update에서 특화 크기가 A < B < C 순서다
- 대가로 일반 재즈 CE가 오른다. 용량이 커질수록 일반 능력과 교환하는 폭이 크다. C는 update 259 이후 허용치를 넘는다(탐색 관찰: C u259는 특화도 −0.121, 일반 +0.018로 허용치 안이지만, 사전 규칙은 최종 update 기준이라 채택하지 않는다)
- **사전 규칙대로 B를 새 배포 후보로 채택한다**(조건을 만족하는 유일한 팔)

### 런타임 비용 (240 BPM, 8마디, CPU, 각 3회)
| | 생성 p50 ms | 생성 p95 ms | 중앙값 p95 | fallback | 미스 |
|---|---|---|---|---|---|
| A u518 | 360 / 448 / 510 | 643 / 664 / 628 | 643 | 0 | 0 |
| **B u518** | 562 / 512 / 579 | 667 / 801 / 712 | **712 (1.11배)** | 0 | 0 |

- 기준(1.2배 이하) 충족. QKV LoRA는 매 forward마다 `B@A`를 더하므로 생성이 느려진다. 추론 전에 가중치를 합치면 줄일 수 있다(미적용, 후속 후보)

### 탐색 지표 — 속주 밀도 (판정과 별개)
| | 런 비율 | onset/초 |
|---|---|---|
| 실제 Tatum val | 0.442 | 7.16 |
| update 0 | 0.330 | 6.34 |
| A u518 | 0.428 | 7.34 |
| **B u518** | **0.519** [Δ CI +0.117, +0.252] | 7.95 |

B는 실제 Tatum 값을 **넘어섰다**. "Tatum보다 더 속주"일 수 있으며, 좋다는 뜻은 아니다. 청취로 확인해야 한다.

### 배포 후보
- 체크포인트: `outputs/tatum_lora_v2_qkv/u518/checkpoint_update518.pt`(gitignore). `model_config.lora_targets = [out_proj, qkv]`, 재로드 logits 동일
- 청취 세트(준비만 함, 미청취): `outputs/tatum_listening_v2_qkv/`. X는 Tatum val, A/B는 update 0 vs B u518, shuffle seed 2029

## 한계
- 단일 seed(42) 학습이다. 팔 간 차이(0.048)가 seed 편차보다 큰지는 검증하지 않았다
- Tatum 곡은 base 사전학습셋에 있다. 들리는 스타일은 미검증이다
- 일반 CE 허용치(+0.02)는 임의 기준이다

## 후속 수정 — LoRA 로딩 fail-closed (Codex 리뷰 H1/M1)
브랜치 `fix/lora-target-loading`. 현재 코드에서 먼저 재현했다(tiny 모델).
- **H1:** `eval_mehldau_snapshots.py`가 snapshot을 `strict=False`로 로드했다. `--lora-targets`를 생략하면(기본 out_proj) QKV snapshot의 키 **6개가 조용히 무시**됐다
- **M1:** 학습·평가·내보내기 스크립트가 out_proj 기본 구조에 base를 strict 로드한 뒤 확장했다. QKV를 포함한 B 체크포인트는 base로 쓸 수 없었다(`strict load` 실패)

수정 (`scripts/train_qlora.py`):
- `build_lora_model_from_state`: base의 저장 구조를 먼저 복원해 strict 로드하고, 요청한 타깃 중 없는 것만 붙인다
- `load_lora_snapshot`: 모델과 snapshot의 LoRA 키가 정확히 일치하지 않으면(누락 또는 초과) `ValueError`로 멈춘다
- `lora_targets_in_state_dict`: LoRA만 담긴 snapshot에서도 FFN을 인식하도록 고쳤다

세 스크립트가 이 두 함수를 쓴다. 평가·내보내기는 snapshot에서 타깃을 추론하고, `--lora-targets`는 검증용 선택 옵션이 됐다(불일치하면 오류). 평가 리포트에 `lora_targets`를 기록한다.
LoRA가 적용된 base의 RNG 소비 순서(생성 → out_proj LoRA → 로드 → 추가 타깃)는 기존과 같다.

**과거 수치 영향: 없음.** 당시 B/C 평가는 `--lora-targets`를 올바르게 명시했다. 새 로더(자동 추론)로 A/B/C의 update 0과 518을 재평가하니 CE가 이전 리포트와 **차이 0.00e+00**으로 같았다(`docs/experiments/lora_fix_recheck/`).
실제 CLI에서 B snapshot에 `--lora-targets out_proj`를 주면 불일치 오류로 멈춘다.

검증: 회귀 테스트 6개 추가(QKV→out_proj 거부, 부분 snapshot 거부, 추론 복원 logits 동일, QKV base 재사용과 FFN 추가, plain base 기본값, export 추론과 잘못된 선언 거부). `agent_harness.sh quick` 213 tests OK, `demo` 통과.
