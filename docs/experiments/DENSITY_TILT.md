# 생성 단계 밀도 기울기 (사전 등록)

작성 2026-10-19 KST. 이슈 #1844. 학습, 다운로드 없음. 실시간 기본값 변경 없음. 음악 변화는 청취 전 품질 주장 없음(`quality_claimed: false`).

## 1. 질문

- 블록 창의 임시 음 수가 늘수록 다음 onset을 늦은 쪽으로 기울이는 logit 가산(잘라내기 없음)이 q8 실시간 경로의 창당 음 수, 지평 전 샘플 수, 마감 초과를 줄이는가

## 2. 배경

- #1819·#1835: 창당 12음에서 다음 창으로 미는 상한은 지평 전 샘플을 40으로 묶고, 결정적 왼손 짝 비교에서 상한 없음 팔의 마감 초과 3건이 상한 팔에서는 없음(짝 관측 3 대 0, 인과 효과 미확정)
- #1843 청취: 상한 12 발췌 6개 전부 "끊김·몰림 어색함", 상한 없는 촘촘한 블록 3개 중 2개 "빽빽해서 별로"
- 밀도는 낮추되 잘라내지 않는 방식 필요

## 3. 방식 (실행 전 고정)

- `rt_lh_input.tilt_bias(onset, last_onset, n_window, lam, n0)` = lam × max(0, n_window − n0) × min(onset − last_onset, 468.75) / 100. n_window는 현재 시각이 속한 블록 확정 창의 임시 음 수(`window_count`, `cap_floor`와 같은 격자). 같은 onset(동시음)은 0
- producer `--density-tilt LAM --density-n0 N0`(기본 0 = 끔): 문법이 onset 토큰을 기다릴 때만, 확률에 exp(bias)를 곱함(= logit 가산). 확정 하한, 문법 마스크, `masked_pick` 그대로. 블록 행 `tilted_to_ready`, `tilted_samples`
- 오케스트레이터·실행기 `--density-tilt`, `--density-n0` 전달(기본 끔)
- n0 = 8 고정

## 4. 보정 (판정 아님, 실행 전 고정)

- q8, `--lh-replay`, A 정상, seed 101~108, 기울기 끔과 lam 0.1, 0.25, 0.5, 1.0
- 지표: warm 블록 확정 음 수(블록 행 `notes`) 중앙값과 최대, 지평 전 샘플 최대
- 선택 규칙: 끔 대비 중앙값 감소율 15~25%이고 최대 ≤ 12인 lam 중 가장 작은 값. 없으면 최대 ≤ 12이면서 감소율이 20%에 가장 가까운 값, 그것도 없으면 감소율 20%에 가장 가까운 값을 고르고 기준 미달을 기록
- 선택한 lam을 이 문서 6절에 적고 커밋한 뒤 5절 시험 실행

## 5. 짝 시험 (실행 전 고정)

- q8, `--lh-replay`, A 정상과 B 이른 crop, 끔 대 선택 lam, seed 1~16, seed마다 팔 순서 교대(홀수 끔 먼저), 64 실행. 원자료 `/Users/ohhalim/git_box/density_tilt_run2/`, 보고 `docs/experiments/density_tilt/test_{off,tilt}_s{seed}.json`
- P0 짝 검증: 블록 행(tokens·notes·S)과 송출 note-on(음높이·세기·예정 시각) 기준 첫 분기 블록 ≥ 기울기 팔의 첫 기울기 활성 블록(`tilted_samples` 첫 증가). 위반 짝은 제외하고 기록
- K1: 64 실행 fail-closed `rerun_checks` 전부 참
- T1 밀도: 기울기 팔 warm 블록 확정 음 수 중앙값이 끔 팔보다 작고, 창당 최대 ≤ 끔 팔
- T2 샘플: 지평 전 샘플 최대·p95 기록, 기울기 팔 최대 < 끔 팔 최대면 충족
- T3 마감: 짝별 첫 실패 블록·완주 여부. 기울기 활성 짝 중 끔 팔만 마감 초과인 짝 수와 반대 짝 수(부호 검정 양측, 표본 작으면 기록만). capacity_failure(B 강제 소용량)는 불완전으로 분리
- 기록만: 쉼 비율(창 안 가장 긴 쉼), 동시음 비율(30 ms 안 onset 묶음), 지평 전 기울기 활성 블록 비율, 루프 ms/sample
- 결론 규칙: K1 미충족이면 실패 기록. K1·T1 충족이면 청취 자료로 사용자 판단(T2·T3는 같이 보고). T1 미충족이면 강도 부족 또는 방식 무효로 기록
- 소표본 한계: 짝 수가 작아 마감 효과는 확정 불가, 시간 잡음(같은 토큰 짝에서도 최소 여유 30 ms 안팎) 있음

## 6. 보정 결과와 고정 lam

(보정 뒤 기록)

## 7. 청취

- `/Users/ohhalim/git_box/density_tilt_listen_run1/index.html`: 기울기 활성 블록 발췌 짝(끔 대 기울기, 순서 무작위) + 긴 발췌 1~2개. 항목마다 선택 버튼·메모·답 만들기 버튼(읽기 전용 상자 아님). 정답 `key.json`과 보고서

## 8. 재현

- `rt_lh_smoke_run.py ... --rerun --precision q8 --lh-replay --seed N --scenarios A_normal,B_crop_early --density-tilt LAM --density-n0 8`
- 시험 `tests/test_rt_lh_input.py`(`DensityTiltTest`), `tests/test_rt_play.py`(`DensityTiltArgsTest`)
