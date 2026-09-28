# 반 마디 스케줄러 블록 — 입력 반영 하한 절반 (사전 등록)

작성 2026-09-29. 이슈 #1532. `style_verified: false` · `musical_quality_verified: false` · 가상 CoreMIDI 포트

**이 절은 측정 전에 작성했다.** 구현과 스모크(1회) 뒤, 본 측정 전에 작성했다.

## 문제
늦은 fetch(#1526)와 시작 예산(#1530) 뒤, 입력 도착에서 반영 마디 시작까지 평균 1.8초다(128 BPM). 하한은 스케줄러 블록 길이(1마디)가 정한다. 그런데 코드 primer 모드는 이미 한 마디를 반 마디 sub-block 2개로 따로 생성한다.

## 변경 (opt-in)
- `run_continuous_jazz.py --half-bar-blocks`(`--chord-primer --chord-blocks-per-bar 2` 필요)
  - 스케줄러와 producer가 반 마디 블록(2박)을 한 단위로 다룬다
  - 늦은 fetch, 시작 예산(β는 **블록**의 비율), 입력 스냅샷, 어댑터 선택이 반 마디마다 동작한다
- 생성 seed는 그대로 (마디, sub)로 정해진다. 따라서 **입력 음이 없으면 토큰이 기존 경로와 같다.** 입력이 있으면 둘째 sub-block도 자기 스냅샷의 입력을 primer에 넣는다(기존에는 첫 sub-block만)
- 리포트
  - `played_bars`는 마디 단위로 합쳐 기존 분석 도구와 호환한다
  - `bars_detail`, `production`, 어댑터 `per_bar`는 **블록 단위**다(`block_beats: 2`, `blocks`)
- 스모크(Tatum 완성, 128 BPM, seed 42, 1회): 16/16마디가 기존과 같았다. fallback 0, 미스 0, 블록 생성 p50 179 ms, 최대 327 ms(반 마디 937 ms)

## 측정
- (a) 지연: `run_live_select_probe.py`
  - 쌍 S, 128 BPM, 16마디, seed 42/43/44
  - PC 1/0/1/0/1을 실행 후 11.0/15.4/19.9/24.1/28.2초에 보낸다
  - 기준은 같은 계획으로 잰 현재 기본값(#1530의 β = 0.5 arm, `docs/experiments/start_budget/probe_beta05_seed*.json`)이다
- (b) 실시간성: Tatum 완성 · 멜다우 #1497 × 128 / 240 BPM × seed 3 = 12회, `--half-bar-blocks`
- (c) 마디 동일성: (b)와 (a)의 연주 마디를 기존 실행(#1523 sweep, 단독 실행)과 비교한다

## 판정 (모두 충족하면 프리셋 실행기(`play_personalized.py`)가 이 모드를 쓴다. 런타임 CLI 기본값은 그대로 둔다)
1. PC 15/15 적용
2. 모든 `latency_ms` ≤ 1.5 × 반 마디 + 150 ms(= 1,556 ms)이고, 평균이 기준(1,832 ms)보다 0.25마디(469 ms) 이상 짧다(≤ 1,363 ms)
3. (b) 12회 모두 fallback 0, 오류 0, 미스 0
4. (c) 연주 마디 전부 동일
