# Tatum vs 멜다우 개인화 정도 비교 — 사전 등록

작성 2026-09-28. 이슈 #1499, 브랜치 `exp/issue-1499-tatum-vs-mehldau`.
`musical_quality_verified: false` · `style_verified: false` · 청취 없음 · 목표는 "특정 연주자처럼 들린다"는 선언이 아니다. 두 개인화 체크포인트와 재현 가능한 비교표·MIDI를 만드는 것이다.

**이 절은 실행 전에 작성했다.**

## 0. 왜 새로 하는가 — manifest 감사
기존 두 모델은 직접 비교할 수 없다.
- Tatum QKV 모델: base는 armB(Tatum을 봄), 타깃은 out_proj+QKV, 학습 110곡
- 멜다우 모델: base는 clean(멜다우만 제외), 타깃은 out_proj, 학습 16곡

토큰 해시 감사 결과(2026-09-28)
| | Tatum 122곡 | 멜다우 18곡 |
|---|---|---|
| armB 학습셋(2,777) | 122 겹침 | 18 겹침 |
| clean base 학습셋(2,759) | **122 겹침** | 0 |

Tatum과 멜다우의 겹침은 0이다. **두 연주자를 모두 제외한 base가 없으므로 1회 재학습한다.**

## 1. 공통 base (P0)
- 데이터: `jazz_full`에서 Tatum 122곡과 멜다우 18곡을 해시로 제외 → `data/jazz_full_notatum_nomehldau`(manifest 기록)
- 학습: armB 레시피(무작위 초기화, full model, 8 epoch, batch 4, accumulation 4, lr 3e-4, LS 0.1, seed 42, max_seq 1024, `legacy_batches`, `val_crop_seed -1`). MPS, 새 디렉터리, 기존 체크포인트 불변
- **C0 게이트(고정):** 일반 probe 100곡(`tatum_generic_probe_list.json`, 두 연주자 제외, 가운데 1024토큰)의 CE가 armB 대비 **+0.05 이내**. 실패하면 별도 실패 보고로 남기고 이후 비교는 "base 품질 차이 가능"으로 해석을 제한한다
- **중복 전사 점검:** 해시 일치가 아닌 다른 전사·다른 녹음 가능성을 본다. 각 연주자 곡의 16-gram 중 공통 base 학습셋에 있는 비율을 곡마다 잰다. 0.2를 넘는 곡은 목록으로 보고한다(제외하지 않고 기록만)

## 2. 데이터 — 데이터 수 맞추기
- **멜다우 train 16곡:** `data/mehldau_full/train` 전부
- **Tatum train 16곡:** `data/tatum_full/train` 110곡 중 곡 이름 정렬 후 `random.Random(0).sample(·, 16)`
- **Tatum holdout(주):** Tatum train 나머지 94곡 중 `random.Random(1).sample(·, 12)` = **fresh12**. 과거에 평가·선정에 쓰인 적이 없다. 다만 과거 다른 Tatum 어댑터(T1 등)의 학습 데이터였고, 이번 모델들과는 무관하다
- **Tatum holdout(보조):** `data/tatum_full/val` 12곡 = **val12**. T1/T4/M-S1의 평가와 선정에 쓰였다(재사용)
- **멜다우 holdout:** `data/mehldau_full/val` 2곡. V2/M-A1/#1497의 평가와 선정에 쓰였다(재사용)
- **일반 재즈:** `tatum_generic_probe_list.json` probe 100곡(두 연주자 제외)
- 학습 토큰 수: 학습 루프는 곡마다 epoch당 1024토큰 crop 하나를 쓰므로 **update당 학습 토큰은 두 연주자가 같다**(16 crop). 다만 곡 길이가 다르다(Tatum 곡이 더 길다). 그래서 학습셋의 전체·고유 토큰 수를 기록하고, 이 차이는 해소하지 않은 비대칭으로 명시한다

## 3. 적응과 선택 (P1 CV → P2 최종)
- 구조: out_proj LoRA r16/α32, lr 3e-4, batch 4, accumulation 4, LS 0.1, seed 42. 두 연주자가 같다
- **CV:** 연주자마다 train16을 곡 단위 4-fold로 나눈다(`make_song_folds.py`, seed 0, held-out 4곡). fold마다 **공통 base에서 새로 시작**한다(optimizer 독립). cosine 128, 후보 update는 **16 / 32 / 64 / 128**
- **예산 선택 규칙(연주자별):**
  1. fold 평균 ΔCE_일반 ≤ +0.02
  2. fold 평균 자기 held-out ΔCE < 0

  위 둘을 만족하는 후보 중 fold 평균 특화도(= ΔCE_held-out − ΔCE_일반)가 최저인 update를 고른다. **동률**(차이 < 0.001)이면 작은 update를 고른다. 만족하는 후보가 없으면 그 연주자는 "적응 실패"로 기록한다
- **최종:** 연주자마다 train16 전부로 cosine 128을 학습하고 선택 update의 snapshot을 쓴다

## 4. 핵심 결과 — 3×3 교차 평가 (P3)
- 행: base / Tatum 어댑터 / 멜다우 어댑터
- 열: Tatum holdout fresh12 / 멜다우 holdout val2 / 일반 재즈 100곡. 보조 열로 Tatum val12를 둔다
- 각 칸
  - CE는 **token 가중**과 **곡 macro** 두 가지로 쟀다. 같은 곡, 같은 crop이다(holdout은 겹치지 않는 512 crop 전부, 일반 재즈는 가운데 1024토큰)
  - base 대비 ΔCE와, 곡 단위 bootstrap 95% CI(2,000회)를 붙인다
- 어댑터별 해석 지표
  - **자기 개선:** 자기 연주자 holdout ΔCE. **< 0이 필수**다(일반 성능이 나빠진 덕분에 특화도만 좋아진 경우는 통과가 아니다)
  - **특화도:** 자기 ΔCE − 일반 ΔCE
  - **상대 대비 특이성:** 자기 ΔCE − 상대 연주자 ΔCE. 두 어댑터가 음악 예측을 비슷하게 좋게 만든 것뿐인지 가린다
- 연주자 간에 raw CE는 비교하지 않는다(곡 난이도가 다르다). "개인화 정도"는 **base 대비 Δ와 특이성**으로만 비교한다
- CV fold 평균·곡별 값·불확실성도 보고한다. 검증셋이 작고 일부는 재사용이라는 한계는 결론에 그대로 남긴다

## 5. 생성 비교 세트 (P4)
- 3개 모델, 같은 중립 primer(`outputs/chord_ab/ii_V_I.mid`), seed 1–4, 768토큰, temperature 1.0, top-k 32, top-p 0.95, grammar mask
- 기록 항목
  - 문법 유효성, 빈 출력, 끝에 열린 음(stuck), 중복 note-on(overlap)
  - 학습곡 exact 16-gram(자기 연주자 train16과 상대 train16 각각)
  - 노트 수, 음역(최저·최고·범위), IOI 중앙값, 동시타건 비율, 속주 런 비율
- 런타임: 128 BPM, 8마디, `--chord-primer --chord-blocks-per-bar 2`, 코드 `Dm7,G7,Cmaj7,A7`, seed 42. 기록 항목은 fallback, 오류, 미스, 생성 p50
- 반복 최적화나 지표 맞추기는 하지 않는다. 한 번 만들고 기록한다

## 6. 예산 상한
- 실험 수 고정: 공통 base 1 + CV 8 run(2 연주자 × 4 fold) + 최종 2 run + 평가·생성
- 시간 상한은 총 **8시간**(base 약 2시간 + 나머지). 넘으면 그 시점까지 확보한 결과로 보고한다
- 새 유료 GPU·API, 외부 데이터, 원본 덮어쓰기는 하지 않는다

## 결과

### P0 준비 — 제외 manifest, 분할, 중복 전사 점검
- 공통 base 학습셋: `jazz_full`(2,777곡)에서 Tatum과 멜다우 해시 일치 곡을 뺐다(train 126곡, val 14곡 제외) → **2,637곡**. 제외 목록: `docs/experiments/tvm/common_base_exclusion_manifest.json`
- 분할(`docs/experiments/tvm/splits_manifest.json`, 곡 제목·해시·토큰 수 포함)

| 세트 | 곡 | 토큰 | 과거 관측 |
|---|---|---|---|
| Tatum train16 | 16 | **150,865** | 과거 Tatum 어댑터의 학습 데이터(이번 모델과 무관) |
| 멜다우 train16 | 16 | **59,654** | 과거 멜다우 어댑터의 학습 데이터 |
| Tatum holdout fresh12 | 12 | 122,734 | 평가·선정에 쓰인 적 없음 |
| Tatum holdout val12 | 12 | 99,701 | **재사용**(T1/T4/M-S1 평가·선정) |
| 멜다우 holdout val2 | 2 | 6,398 | **재사용**(V2/M-A1/#1497 평가·선정) |

- **해소하지 않은 비대칭:** 학습 곡 수는 같지만 Tatum 곡이 길어 **학습셋 토큰이 2.5배**다. update당 학습 토큰은 같다(곡당 1024 crop 1개). 즉 같은 update 수에서 Tatum은 더 다양한 재료를 본다
- **중복 전사 점검**(`docs/experiments/tvm/near_duplicates_summary.json`): 공통 base 학습셋의 고유 16-gram 2,304만 개와 비교했다. 곡별 16-gram 공유 비율은 Tatum 122곡 최대 **0.0002**, 멜다우 18곡 최대 **0**이다. 기준 0.2를 넘는 곡이 없다. 다른 파일명의 동일 전사는 보이지 않는다(짧은 구절 수준의 유사성까지 배제하지는 않는다)

### P0 공통 base — C0 게이트 통과
`outputs/tvm/common_base/checkpoint_epoch8.pt`(gitignore, 기존 체크포인트 불변). 무작위 초기화, 8 epoch, MPS. 최종 train 3.317 / val 3.212(armB 3.285 / 3.193).

| CE (LS 없음, 같은 crop) | armB | **공통 base** | 차이 |
|---|---|---|---|
| 일반 재즈 probe 100곡(두 연주자 제외) | 2.5711 | 2.6076 | **+0.037** (게이트 +0.05 이내 → 통과) |
| Tatum train16 | 2.5567 | 2.6267 | +0.070 |
| Tatum fresh12 | 2.5486 | 2.6131 | +0.065 |
| 멜다우 train16 | 2.6386 | 2.6994 | +0.061 |
| 멜다우 val2 | 2.4408 | 2.5105 | +0.070 |

- 두 연주자 CE가 일반 재즈보다 크게 올랐다(+0.06~0.07 vs +0.037). 공통 base가 두 연주자를 보지 않았다는 방증이다
- 통과했지만 멜다우 전용 clean base(+0.009)보다 여유가 작다. Tatum 122곡(피아노 솔로 데이터의 상당 부분)을 추가로 뺀 영향으로 보인다(추정)
