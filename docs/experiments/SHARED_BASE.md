# 공통 base 하나로 두 연주자 — 멜다우 base 비교 (사전 등록)

작성 2026-09-29. 이슈 #1517. `style_verified: false` · `musical_quality_verified: false`

**이 절은 실행 전에 작성했다.**

## 문제
두 완성 모델이 서로 다른 base 위에 있다.
- Tatum 완성(#1509): **공통 base**(Tatum·멜다우 미학습) + out_proj+QKV
- 멜다우 완성(#1497): **멜다우 제외 clean base**(Tatum 포함) + out_proj

재설계 문서의 층 2 방향("base 1개 + 런타임 어댑터 스왑")을 따르려면 두 어댑터가 같은 base를 써야 한다. 공통 base 위의 멜다우 어댑터는 이미 있다(#1499/#1501 `outputs/tvm/cv_mehldau_fold{k}`, 최종 `outputs/tvm/export_mehldau/checkpoint_update128.pt`).

두 CV의 ΔCE(#1497 −0.046, #1501 −0.047)는 **각자 다른 base 대비**라 직접 비교할 수 없다. 이번에는 같은 곡·같은 crop에서 **절대 CE**를 비교한다.

## 비교 조건 (새 학습 없음)
- 곡: 멜다우 train16. 4-fold 분할은 두 CV가 같다(`data/mehldau_cv/folds.json` = `data/tvm/cv_mehldau/folds.json`, 파일 sha1 동일)
- 학습 레시피도 같다: out_proj r16, seed 42, cosine 128, 128 update. **다른 것은 base뿐이다**
- arm R(기준): clean base + `outputs/clean_base/c1_fold{k}/lora_update128.pt`
- arm S(후보): 공통 base + `outputs/tvm/cv_mehldau_fold{k}/lora_update128.pt`
- 각 곡은 그 곡을 held-out으로 둔 fold 어댑터로 잰다. 곡마다 512 비중첩 crop을 쓰고 곡 macro로 평균한다
- 일반 재즈는 `outputs/tatum_diag/generic_probe_list.json`(100곡, Tatum·멜다우 제외)의 가운데 1024 토큰이다. 4 fold 어댑터의 평균을 쓴다
- 스크립트: `scripts/compare_base_cv.py`. 차이 CI는 곡 짝 재표집 2,000회이며, 고정된 fold 모델에 조건부다

## 판정 (모두 충족하면 공통 base로 통합)
1. **멜다우 held-out 절대 CE:** S − R ≤ **+0.01**(비열등 한계)
2. **일반 재즈 절대 CE:** S − R ≤ **+0.02**
3. S의 held-out ΔCE(자기 base 대비) < 0

충족하면 멜다우 기본 모델을 `outputs/tvm/export_mehldau/checkpoint_update128.pt`(공통 base)로 바꾸고, 두 연주자를 공통 base 하나로 운영한다. 미달하면 #1497을 유지하고 "base 두 개"를 한계로 기록한다.

## 해석 주의
- 128은 각 CV에서 고른 예산이다. 선택 후 비교이므로 탐색 분석이다
- 두 base는 학습곡 수가 다르다(공통 base는 Tatum 곡이 빠짐). 차이가 base 자료 차이인지 무작위성인지는 구분하지 않는다
