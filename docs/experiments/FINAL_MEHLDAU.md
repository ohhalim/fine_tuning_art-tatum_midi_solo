# 멜다우 완성 모델 — 학습량 확장 (사전 등록)

작성 2026-09-29. 이슈 #1511. `style_verified: false` · `musical_quality_verified: false`

**이 절은 실행 전에 작성했다.**

## 목적
현재 멜다우 모델(#1497: 멜다우 제외 clean base + out_proj, 128 update)은 CV에서 128이 **시험 범위의 끝값**이었다(#1497, #1499 모두). 곡선이 아직 내려가는 중이었다. 범위를 넓혀 다시 고른다. 멜다우 곡은 16곡뿐이라 자료는 늘릴 수 없다.

## 구성
- base: 멜다우 제외 clean base(`outputs/clean_base/nomehldau_full/checkpoint_epoch8.pt`, 멜다우 미학습, Tatum 포함 2,759곡)
- 어댑터: out_proj r16. 16곡에서는 QKV가 과적합했다(M-A1)
- CV: `data/mehldau_cv` 4-fold(#1497과 같은 분할, held-out 4곡). fold마다 base에서 새로 시작한다. **cosine 384**, 후보 update **64 / 128 / 256 / 384**
- 선택 규칙: fold 평균 ΔCE_일반 ≤ +0.02이고 fold 평균 held-out ΔCE < 0인 후보 중 fold 평균 특화도 최저. 동률(차이 < 0.001)이면 작은 update. 일반은 V1 probe 100곡(멜다우 제외)이다
- 최종: 16곡 전부, cosine 384, 선택한 update의 snapshot

## 채택 판정
- **새 멜다우 기본 모델로 채택하는 조건:** 선택 예산의 CV 평균 특화도가 #1497 CV의 u128 값(−0.0223)보다 **0.005 이상 낮고**, CV 평균 held-out ΔCE가 #1497 값(−0.0460)보다 낮을 것. 아니면 #1497 u128을 유지한다
- 비교는 같은 fold·같은 base에서 cosine 길이만 다르다. #1497 수치는 cosine 128에서 나온 값이라는 차이를 명시한다
- 최종 모델 확인(보조): val2(재사용 holdout) ΔCE < 0, 생성 유효성 100%, 16-gram 복사 ≤ 0.10, 128 BPM 16마디 fallback 0
