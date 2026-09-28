# Tatum 완성 모델 — 사전 등록

작성 2026-09-29. 이슈 #1509, 브랜치 `exp/issue-1509-final-tatum`. `style_verified: false` · `musical_quality_verified: false`

**이 절은 실행 전에 작성했다.**

## 목적
실제로 쓸 Tatum 개인화 모델을 만든다. 16곡 비교용 모델(#1499)이 아니라 가능한 자료를 다 쓰고, 검증된 구성을 적용한다.

## 구성
- base: 공통 base(`outputs/tvm/common_base/checkpoint_epoch8.pt`, Tatum·멜다우 미학습, 누출 교집합 0)
- 어댑터: out_proj + **QKV** LoRA r16. 110곡에서 out_proj보다 우세했고(#1493), seed 3개에서 강건했다(#1496)
- 데이터: **학습 98곡** = `tatum_full/train` 110곡 − fresh12. **선택용 val12** = `tatum_full/val`. **최종 평가 fresh12**는 학습·선택 모두에 쓰지 않는다
- 학습: lr 3e-4, batch 4, accumulation 4(epoch당 7 update), cosine 518 update(74 epoch), LS 0.1, seed 42, MPS. snapshot 0 / 70 / 133 / 259 / 385 / 518

## 선택과 판정
- **선택(val12만 사용):** ΔCE_일반 ≤ +0.02인 snapshot 중 ΔCE_val12가 가장 낮은 것. 일반은 `tatum_generic_probe_list.json` 100곡이다
- **최종 평가(선택 뒤 1회):** fresh12, 일반 100곡, 멜다우 16+2곡에서 base 대비 ΔCE(token·macro)를 잰다. 비교 기준은 #1499의 Tatum16 out_proj 모델(같은 base, 같은 fresh12)이다
- **완성 모델 채택:** fresh12 ΔCE < 0이고, 특화도(fresh12 − 일반)가 Tatum16 모델보다 낮으며(더 특화), ΔCE_일반 ≤ +0.02이면 새 Tatum 기본 모델로 채택한다. 아니면 Tatum16 모델을 유지하고 기록한다
- 생성·런타임: 생성 문법 유효성과 16-gram 복사(학습 98곡 대비 ≤ 0.10), 128 BPM 16마디 fallback 0, KV 캐시 기본
