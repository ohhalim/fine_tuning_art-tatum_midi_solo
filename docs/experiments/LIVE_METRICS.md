# 루프 안 블록 지표 — 사전 등록

작성 2026-09-29. 이슈 #1544. `style_verified: false` · `musical_quality_verified: false`

**이 절은 측정 전에 작성했다.** 구현과 8마디 스모크 1회 뒤에 작성했다.

## 목적
재설계 층 2의 세 번째 차별점은 "지표가 루프 안에서 계속 찍힌다"이다(jam_bot은 재지 않는다). 지금까지 지표는 세션이 끝난 뒤 `played_bars`로 계산했다.

## 구현
- `block_metrics(block, chord, adapter, input_events)`: producer 스레드에서 모델 블록을 만든 직후 계산한다
  - 어댑터, 코드, 음 수, 평균·최저·최고 음높이, 코드톤 비율
  - 스냅샷의 입력 음 수와 평균 음높이
- 리포트 `block_metrics`: 블록마다 한 항목이다. fallback 블록은 None이다
- `--live-metrics`: 블록마다 한 줄을 출력한다. 프리셋에서는 `play_personalized.py ... -- --live-metrics`로 켠다

## 측정
- Tatum 완성, `--half-bar-blocks`, 16마디, 128 / 240 BPM × seed 42/43/44(지표 기록, 출력 끔)
- 추가로 128 BPM × 3 seed를 `--live-metrics`(출력 켬)로 잰다

## 판정 (모두 충족하면 합친다)
1. 정확성: 모든 모델 블록에서 `block_metrics`의 음 수와 코드톤 비율이 `played_bars`로 따로 계산한 값과 같다
2. 실시간성: 9회 모두 fallback 0 · 오류 0 · 미스 0
3. 무간섭: 연주 마디가 #1532 sweep(같은 모델·템포·seed)과 전부 같다(출력 켠 실행 포함)
