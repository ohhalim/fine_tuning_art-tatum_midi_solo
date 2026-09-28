# 최종 두 모델의 실시간 성립 템포 — 사전 등록

작성 2026-09-29. 이슈 #1523. `style_verified: false` · `musical_quality_verified: false` · 가상 CoreMIDI 포트, 실물 키보드·DAW·청취 없음

**이 절은 실행 전에 작성했다.**

## 목적
`TATUM_REALTIME_TEMPO.md` R1(#1488)은 옛 Tatum 어댑터(u518 armB, KV 캐시·병합 없음)로 쟀다. 지금 기본 런타임에는 KV 캐시(#1507)와 LoRA 병합(#1512, #1520)이 켜져 있다. 개인화를 마무리한 두 모델이 이 설정에서 몇 BPM까지 버티는지 기록한다.

## 조건
- 모델
  - Tatum 완성: `outputs/final_tatum/export/checkpoint_update518.pt`(공통 base, out_proj+QKV)
  - 멜다우 완성 #1497: `outputs/clean_base/c2_export/checkpoint_update128.pt`
- 런타임: `run_continuous_jazz.py` 기본값(KV 캐시, 병합, spin 5 ms, QoS user-interactive) + `--chord-primer --chord-blocks-per-bar 2 --chords Dm7,G7,Cmaj7,A7`, 16마디, `--capture`, `FORCE_CPU=1`, primer `outputs/chord_ab/ii_V_I.mid`
- 템포 128 / 160 / 200 / 240 BPM × seed 42 / 43 / 44 = 모델당 12회. `scripts/run_tempo_sweep.py`로 순차 실행하고, 다른 무거운 작업과 겹치지 않게 한다

## 판정 (R1과 같음)
- 한 템포에서 3회 모두 fallback 0 · 오류 0 · 데드라인 미스 0이면 **성립**이다
- 모델별 최대 성립 템포와, 템포별 생성 p95/마디 비율을 보고한다
- 미스가 나면 `deadline_miss_detail`로 생성 중에 났는지 기록한다(원인 판정은 하지 않는다)

---

## 결과 — 두 모델 모두 **240 BPM까지 성립**(24회 중 24회)
원시값: `docs/experiments/final_tempo/sweep_report.json`. 원본: `outputs/final_tempo/`. 24회, 순차 실행, CPU, 약 14분.

| 모델 | BPM | 판정 | fallback | 미스 | 생성 p50 최대 ms | p95 최대 ms | 최대 ms | 마디 ms | p95/마디 | 지각 최대 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| Tatum 완성 | 128 | 성립 | 0 | 0 | 319 | 483 | 532 | 1,875 | 0.26 | 16.6 |
| Tatum 완성 | 160 | 성립 | 0 | 0 | 249 | 395 | 398 | 1,500 | 0.26 | 10.1 |
| Tatum 완성 | 200 | 성립 | 0 | 0 | 236 | 347 | 357 | 1,200 | 0.29 | 13.8 |
| Tatum 완성 | 240 | 성립 | 0 | 0 | 192 | 278 | 317 | 1,000 | 0.28 | 11.1 |
| 멜다우 #1497 | 128 | 성립 | 0 | 0 | 224 | 379 | 490 | 1,875 | 0.20 | 7.4 |
| 멜다우 #1497 | 160 | 성립 | 0 | 0 | 201 | 309 | 309 | 1,500 | 0.21 | 11.0 |
| 멜다우 #1497 | 200 | 성립 | 0 | 0 | 167 | 277 | 356 | 1,200 | 0.23 | 8.6 |
| 멜다우 #1497 | 240 | 성립 | 0 | 0 | 151 | 262 | 273 | 1,000 | 0.26 | 6.0 |

- 생성 p95는 마디 길이의 최대 29%다
  - 비교: R1(#1488, 옛 Tatum u518, KV 캐시·병합 없음)은 최대 74%였다
  - Tatum 완성 모델은 옛 u518보다 LoRA가 크지만(out_proj+QKV) 생성 여유가 더 크다
- 데드라인 미스는 24회 모두 0이다(R1은 base 200, Tatum 128에서 미성립). 스케줄러 지각 최대는 16.6 ms로 허용치 20 ms 안이다
- 해석 한계: 가상 포트에서 쟀다. DAW·실물 키보드 부하는 없다. 생성 여유가 다른 부하를 흡수할 수 있는지는 재지 않았다
