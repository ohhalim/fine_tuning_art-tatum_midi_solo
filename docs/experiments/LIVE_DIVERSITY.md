# 루프 안 다양성 지표 — 사전 등록

작성 2026-09-29. 이슈 #1546. `style_verified: false` · `musical_quality_verified: false`

**이 절은 측정 전에 작성했다.** 구현과 단위 테스트는 측정 전에 끝냈다.

## 목적
`block_metrics`(#1544)에는 화성 지표(코드톤 비율)만 있다. 재설계 문서가 말한 **다양성** 지표를 루프 안에 더한다. D1(다양성 붕괴 판정)에서 쓴 보이싱 정의를 그대로 쓴다.

## 구현
- `block_voicings(block)`: 블록의 note_on 시각으로 `diversity_metrics.group_voicings`(50 ms 창, pitch-class 집합)를 부른다
- `VoicingPool`: 어댑터별로 세션 누적 보이싱을 쌓는다
- `block_metrics`에 `voicings`, `unique_voicings_so_far`, `distinct_voicing_ratio_so_far`를 더한다. `--live-metrics` 출력에도 나온다

## 측정
- `play_personalized.py --preset swap`(Tatum 완성 ↔ 멜다우 #1497, 4마디씩, 반 마디 블록), 128 BPM, 16마디, seed 42/43/44
- 오프라인 기준: 같은 실행의 `played_bars`를 반 마디로 나눠 `group_voicings`로 다시 계산한다

## 판정 (모두 충족하면 합친다)
1. 모든 모델 블록에서 `voicings`가 오프라인 값과 같다. 각 실행 끝의 어댑터별 `unique_voicings_so_far`도 오프라인 누적값과 같다
2. 3회 모두 fallback 0이고, `producer_busy: true`인 미스가 0이다. 생성과 무관한 드문 멈춤(#1530, #1544)은 판정에서 빼고 수만 보고한다
3. 연주 마디가 같은 seed 단독 실행(#1523 `final_tempo`, 마디별 어댑터 기준)과 전부 같다

---

## 결과 — 기준 1–3 모두 충족, **합침**
- 기준 1
  - 모델 블록 **96/96**에서 `voicings`가 오프라인 `group_voicings`와 같다
  - 실행 끝 어댑터별 누적 고유 보이싱도 같다(아래 표)
- 기준 2: 3회 모두 fallback 0, 미스 0
- 기준 3: 연주 마디 **48/48**이 단독 실행(마디별 어댑터)과 같다

| seed | Tatum 누적 고유 보이싱 (8마디) | 멜다우 누적 고유 보이싱 (8마디) |
|---|---|---|
| 42 | 50 | 42 |
| 43 | 50 | 39 |
| 44 | 60 | 39 |

- 기술 관측(판정 아님): 같은 세션 같은 길이에서 Tatum 쪽 고유 보이싱이 더 많다. 음 수가 더 많은 것(#1528 쇼케이스)과 맞는다. 다양성이 "좋다"는 뜻은 아니다
- 이제 루프 안에서 화성(코드톤)과 다양성(보이싱) 지표가 함께 블록마다 찍힌다
  - D1이 오프라인에서 쓰던 지표를 연주 중에 볼 수 있다(`--live-metrics`)
