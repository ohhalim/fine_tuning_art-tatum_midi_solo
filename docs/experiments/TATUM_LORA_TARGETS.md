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
