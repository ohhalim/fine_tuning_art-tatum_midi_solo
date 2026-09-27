# Tatum 어댑터 실시간 성립 템포 — 사전 등록

작성 2026-09-27. 이슈 #1488, 브랜치 `exp/issue-1488-tatum-realtime-tempo`.
선행: `TATUM_PERSONALIZATION.md` (u518 어댑터, 128 BPM 스모크 생성 p50 778 ms).

`quality_claimed: false` · 가상 CoreMIDI 포트 측정 · 실물 키보드·DAW·청취 없음

**이 절은 실행 전에 작성했다.**

## R1 — 몇 BPM까지 fallback 없이 버티는가

- 런타임: `scripts/run_continuous_jazz.py`. 권장 설정(`--chord-primer --chord-blocks-per-bar 2`), `--bars 16`, `--capture`, `FORCE_CPU=1`, primer `outputs/chord_ab/ii_V_I.mid`
- 모델: base(armB ep8) vs Tatum u518(`outputs/tatum_lora_v1/u518/checkpoint_update518.pt`)
- 템포: 128 / 160 / 200 / 240 BPM(런타임 상한 240). seed 42 / 43 / 44, 조건당 3회
- 실행 스크립트: `scripts/run_tempo_sweep.py`(런마다 별도 프로세스, 순차 실행, 다른 무거운 작업 없이)
- 기록: fallback 마디 수, 생성 오류, 스케줄러 데드라인 미스, 마디 생성 시간 p50/p95/최대, 마디 길이 대비 p95 비율
- **판정: 한 템포에서 3회 모두 fallback 0 · 오류 0 · 데드라인 미스 0이면 그 템포에서 "성립"으로 기록한다.** 모델별 최대 성립 템포를 보고한다
- R2(지연 단축)는 **Tatum u518이 240 BPM에서 성립하지 않을 때만** 진행한다. 목표는 해당 템포를 성립으로 바꾸는 것이다

## 결과

(실행 후 추가)
