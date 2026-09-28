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
