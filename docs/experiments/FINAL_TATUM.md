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

## 결과 — 채택 기준 충족 → **Tatum 기본 모델로 채택**
원시값: `docs/experiments/final_tatum/`(선택 리포트, 3×3, 생성 통계, 런타임, 데이터 manifest)

### 학습과 선택 (val12만 사용)
학습 wall 829 s(MPS). update 518에서 train CE −0.142, val12 CE −0.125이고, 끝까지 단조 감소했다.

| update | ΔCE train | **ΔCE val12** | ΔCE 일반 | 선택 조건(일반 ≤ +0.02) |
|---|---|---|---|---|
| 70 | −0.099 | −0.086 | −0.014 | ✓ |
| 133 | −0.116 | −0.105 | −0.008 | ✓ |
| 259 | −0.134 | −0.119 | +0.005 | ✓ |
| 385 | −0.140 | −0.124 | +0.009 | ✓ |
| **518** | −0.142 | **−0.125** | +0.009 | ✓ ← **선택** |

### 최종 평가 (fresh12는 선택 뒤 1회만 사용, base 대비 ΔCE, token (macro) [곡 CI])
| 모델 | **Tatum fresh12** | 멜다우 16곡 | 멜다우 val2 | 일반 재즈 100 |
|---|---|---|---|---|
| Tatum16 out_proj(#1499) | −0.055 (−0.056) [−0.062, −0.050] | −0.005 | −0.028 | −0.021 |
| **Tatum 완성(98곡+QKV)** | **−0.126 (−0.126) [−0.135, −0.117]** | **+0.052** | +0.003 | +0.009 |

- **채택 기준**
  - fresh12 ΔCE −0.126 < 0 ✓
  - 특화도(fresh12 − 일반) **−0.134** vs Tatum16 −0.035 → 더 특화 ✓
  - 일반 +0.009 ≤ +0.02 ✓
- 미학습 Tatum 곡 개선이 16곡 모델의 약 **2.3배**다
- **연주자 특이성이 강하다.** 멜다우 곡은 오히려 나빠졌다(+0.052). 일반 재즈는 거의 그대로다(+0.009). Tatum 전문 모델이다

### 생성과 실시간
- 생성(seed 1–4, 768토큰): 문법 4/4, 빈 출력 0, 학습 98곡 대비 16-gram 복사 **0**
- 기술통계(판정 아님): 속주 런 비율 0.30 → **0.41**(실제 Tatum fresh12 0.36), 동시타건 onset 비율 0.30 → **0.43**(실제 0.39), IOI 중앙값 135 → 108 ms
- 128 BPM 16마디, 코드 primer, KV 캐시, 3회: **fallback 0, 미스 0**, 생성 p50 약 690 ms, p95 약 990 ms(마디의 53%)
  - QKV LoRA 때문에 out_proj 모델(p50 358 ms)보다 느리다. 추론 전 LoRA 병합은 후속 후보다

### 쓰는 법
```sh
FORCE_CPU=1 .venv/bin/python scripts/run_continuous_jazz.py \
    --checkpoint outputs/final_tatum/export/checkpoint_update518.pt \
    --conditioning-midi <primer.mid> --chords Dm7,G7,Cmaj7,A7 --bars 16 --bpm 128 \
    --chord-primer --chord-blocks-per-bar 2 --capture --output-dir outputs/continuous/tatum
```
- 체크포인트(gitignore): `outputs/final_tatum/export/checkpoint_update518.pt`(재로드 logits 동일, `lora_targets = [out_proj, qkv]`)
- MIDI: `outputs/final_tatum/midi/`(생성 4, 실시간 16마디 3)
- 한계: 청취 없음. Tatum 곡은 base가 보지 않았지만 val12는 선택에 쓰였고, fresh12는 과거 다른 Tatum 어댑터의 학습 데이터였다(이번 모델과 base는 보지 않음)
