# Art Tatum 개인화 — 사전 등록

작성 2026-09-27. 로컬 브랜치 `exp/mehldau-update-budget-diag`. 선행: `MEHLDAU_STYLE_SHIFT.md`.

`tatum_style_verified: false` · `musical_quality_verified: false`

**이 절은 실행 전에 작성했다.**

## 왜 Tatum인가
- 멜다우 청취에서 사용자는 스타일 판별이 "사투리 구분처럼 모호하다"고 했다. Tatum은 스트라이드 왼손, 빠른 런, 리하모니가 뚜렷해 판별이 쉬울 것으로 기대한다(가설)
- 데이터가 122곡, 약 113만 토큰으로 멜다우(18곡, 약 7만 토큰)의 약 17배다
- 이전 "Tatum-adapted" 체크포인트는 오기였다. 실제로는 멜다우 lead 적응이며, Tatum 전용 어댑터는 이번이 처음이다

## 데이터
`data/tatum_full` (gitignore), `scripts/build_artist_dataset.py`. 인코더는 `jazz_full`과 같다(멜다우 18곡으로 해시 일치를 검증했다: train 17 / val 1).
- 122곡 → **train 110 / val 12**, 곡 단위, seed 42. 목록은 `manifest.json`
- **122곡 전부 `jazz_full`에 있다**(train 109 / val 13). base가 본 곡이다
- 멜다우와 다른 점: **val 12곡은 어댑터 학습에서 제외된다.** "어댑터가 보지 않은 Tatum 곡"에 대한 효과는 잴 수 있다. 단 base는 그 곡들을 봤다

## T1 — Tatum 어댑터가 보지 않은 Tatum 곡으로 특화되는가
- 학습: armB ep8 시작, out_proj LoRA r16, batch 4, accumulation 4, lr 3e-4, LS 0.1, seed 42, MPS. 한 연속 run으로 약 512 update(epoch당 7 update). snapshot은 약 0/8/32/64/128/256/384/512 update다(epoch 경계에서 가장 가까운 값)
- 평가 CE(LS·dropout 없음): Tatum val 12곡 고정 512 crop 전부, Tatum train 곡당 2 crop, 일반 재즈 probe 100곡. 일반 probe는 V1 probe 표본과 같은 방식(seed 0)으로 뽑되, **Tatum과 멜다우를 모두 제외**한 `jazz_full/train`에서 뽑는다
- **1차 판정**: 어떤 snapshot에서 (ΔCE_Tatum_val − ΔCE_일반) ≤ −0.02이고 ΔCE_일반 ≤ +0.05면 **"보지 않은 Tatum 곡으로 특화"**로 기록한다
- 과적합 기록: ΔCE_Tatum_val이 최저점 이후 +0.02 넘게 되돌아오면 기록한다
- **배포 snapshot 선정 규칙**: ΔCE_일반 ≤ +0.02인 snapshot 중 ΔCE_Tatum_val이 가장 낮은 것
- 탐색용(해석 제한): 생성 JS shift(Tatum train 기준 vs 일반 기준), 16-gram 복사율(경고 기준 0.10)

## T2 — 들리는가 (T1 통과 시)
- 멜다우 청취 v2와 같은 형식이다. X = **Tatum val(어댑터가 보지 않은) 곡** 가운데 15초, A/B = update 0 vs 배포 snapshot, seed 1–6, 15초, A/B 무작위(shuffle seed 2028)
- **판정: 6쌍 중 5쌍 이상 배포 snapshot → 청취 지지.** 4쌍은 불확실, 3쌍 이하는 지지 없음
- "모름"은 배포 snapshot을 고르지 않은 것으로 센다

## 결과

### T1 학습 — 518 update, MPS
`outputs/tatum_diag/update_budget_t1/`. epoch당 7 update(28배치, accumulation 4), 74 epoch, 학습 wall 850 s.
평가 CE는 LS·dropout 없이 쟀다. train은 110곡 × 2 고정 512 crop(112,640토큰), val은 **어댑터가 보지 않은 12곡** 전체 고정 crop(95,744토큰)이다.

| update | lr | Tatum train CE | Δ | **Tatum val CE** | **Δ** |
|---|---|---|---|---|---|
| 0 | 3.0e-4 | 2.5692 | — | 2.5122 | — |
| 14 | 3.0e-4 | 2.5485 | −0.021 | 2.4913 | −0.021 |
| 35 | 3.0e-4 | 2.5370 | −0.032 | 2.4814 | −0.031 |
| 70 | 2.9e-4 | 2.5242 | −0.045 | 2.4693 | −0.043 |
| 133 | 2.5e-4 | 2.5114 | −0.058 | 2.4577 | −0.055 |
| 259 | 1.5e-4 | 2.5010 | −0.068 | 2.4485 | −0.064 |
| 385 | 4.7e-5 | 2.4967 | −0.073 | 2.4450 | −0.067 |
| 518 | 1e-6 | 2.4961 | −0.073 | 2.4445 | **−0.068** |

- **어댑터가 보지 않은 Tatum 곡의 CE가 train과 거의 같은 폭으로 단조 감소했다.** 되돌아옴(과적합 기록 조건)은 없다
- 멜다우와 대조된다. 멜다우는 train −0.20 vs val(2곡) 64 이후 정체였다. Tatum은 train −0.073 vs val −0.068이다. 곡 수가 많아(110) 특정 곡 암기보다 공통 패턴을 학습하는 쪽으로 보인다(해석)
- 일반 probe 대비 판정은 snapshot 평가 후 붙인다
