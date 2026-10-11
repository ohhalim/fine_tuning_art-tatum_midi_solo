# 멜다우 제외 base에 멜다우 라이브 녹음이 들어 있었나 (이슈 #1722)

작성 2026-10-11. 학습 없음. 해시 대조만.

## 요약

**들어 있었다.** "멜다우 제외" base 두 개 모두 Mehldau **라이브 녹음 54곡 전부**를 학습했다(train 45 / val 9). 뺀 것은 스튜디오 18곡뿐이었다.

- 그래서 #1497의 "멜다우를 한 번도 안 본 base"는 틀린 설명이다. 정확히는 **"멜다우 스튜디오 18곡을 안 본 base"**다. 같은 연주자의 다른 녹음 54곡은 base가 이미 봤다
- 곡 단위 held-out은 대부분 유지된다. 스튜디오 18곡은 어느 base에도 없었다. 단, CV fold 3 held-out인 "From This Moment On"은 라이브 버전(Live in Tokyo)이 base에 있었다
- Tatum 쪽 "Tatum 미학습"은 맞다. PiJAMA에 Tatum 라이브 녹음은 없고, 공통 base에 Tatum 122곡은 0곡이다
- 이전 결과를 통째로 무효로 보지 않는다. 바뀌는 것은 **"연주자 미노출"에서 "스튜디오 곡 미노출"로의 해석 범위**다

## 어떻게 확인했나

- `jazz_full`을 만든 인코더(`archive/scripts/preprocess_jazz.py:process_midi_file`)로 PiJAMA의 두 연주자 녹음을 모두 다시 인코딩했다
- 각 토큰 배열의 sha1을 실제 학습 입력 폴더의 `.npy`와 대조했다. manifest 추정이 아니라, 같은 토큰 배열이 폴더에 있을 때만 "들어 있음"으로 센다
- **재현 확인:** Mehldau 스튜디오 18/18곡, Tatum 122/122곡이 기존 어댑터 자료와 해시까지 일치했다. 인코딩 실패 0곡
- `midi_kong`(같은 연주의 다른 전사)은 어느 학습셋과도 일치하지 않았다. 모든 학습셋은 `midi` 트리에서 만들었다

## 결과

| 학습셋 | 크기 | Mehldau 스튜디오 18 | Mehldau 라이브 54 | Tatum 스튜디오 122 |
|---|---|---|---|---|
| 원래 base `jazz_full` (armB) | 2777 | 18 (train 17 / val 1) | 54 (train 45 / val 9) | 122 (train 109 / val 13) |
| 멜다우 제외 clean base `jazz_full_nomehldau` (#1497) | 2759 | 0 (train 0 / val 0) | 54 (train 45 / val 9) | 122 (train 109 / val 13) |
| 공통 base `jazz_full_notatum_nomehldau` (#1499·#1509) | 2637 | 0 (train 0 / val 0) | 54 (train 45 / val 9) | 0 (train 0 / val 0) |
| 멜다우 어댑터 `mehldau_full` | 18 | 18 (train 16 / val 2) | 0 (train 0 / val 0) | 0 (train 0 / val 0) |
| Tatum 어댑터 `tatum_full` | 122 | 0 (train 0 / val 0) | 0 (train 0 / val 0) | 122 (train 110 / val 12) |

**같은 제목의 다른 녹음** (해시는 다르지만 같은 곡을 다시 연주한 것)
- Mehldau: "From This Moment On"은 스튜디오(CV fold 3 held-out)와 라이브(base 학습)에 모두 있다. 라이브끼리 겹치는 곡이 2개 있다(My Favorite Things, Things Behind the Sun). 원래 val 2곡(조곡 I, X)은 라이브에 같은 제목이 없다
- Tatum: 같은 제목 두 녹음이 9곡이다. 그중 5곡은 분할을 넘는다. 4곡은 train98과 val12(선택용)에, "Tea For Two" 1곡은 train98과 fresh12(평가용)에 나뉘었다

**일반 재즈 probe에도 섞였다** ("Tatum·멜다우 제외"로 만든 100곡 목록)
- `outputs/tatum_diag/generic_probe_list.json`: probe 100곡 중 1곡, reference 100곡 중 2곡이 Mehldau 라이브
- `outputs/mehldau_diag/style_validity_v1.json`(V1): probe 100곡 중 2곡, reference 100곡 중 1곡이 Mehldau 라이브
- 1–2% 수준이라 특화도(ΔCE_대상 − ΔCE_일반)를 조금 흐리는 정도로 본다. 다시 재지 않았다

## 무엇이 바뀌나

| 기존 표현 | 고친 표현 |
|---|---|
| 멜다우 미학습 base / 멜다우를 한 번도 안 본 base | 멜다우 **스튜디오** 미학습 base. 라이브 54곡은 base 학습에 포함 |
| 처음 보는 멜다우 곡 적응 | base가 같은 연주자의 라이브 녹음을 본 상태에서, **처음 보는 스튜디오 곡** 적응 |
| 공통 base(Tatum·멜다우 미학습) | 공통 base(Tatum 미학습, 멜다우 스튜디오 미학습·라이브 54곡 포함) |
| C0에서 멜다우 CE가 0.04 높다 = 멜다우를 안 봤다는 방증 | 스튜디오 18곡을 뺀 효과다. 연주자 미노출의 근거가 아니다 |

- 그대로인 것: Mehldau held-out 스튜디오 곡의 CE 감소(−0.046, 4/4 fold), 재사용 holdout 수치, 복사 0, 실시간 동작. 이 측정값 자체는 해시상 base에 없는 곡에서 나왔다
- 약해지는 것: "연주자를 처음 만난 모델이 적응했다"는 해석. 라이브 54곡을 본 base였으므로, 어댑터 이득의 일부는 이미 base에 있던 Mehldau 성향을 끌어낸 것일 수 있다. 분리하지 않았다
- 다음 Aria 설계에는 녹음 단위가 아니라 **연주자 단위로 녹음 종류(studio/live)를 묶어** 학습셋 교집합을 기록한다. 같은 제목의 다른 녹음은 한쪽 분할에만 둔다

## 재현

- `scripts/artist_leak_audit.py docs/experiments/artist_leak_audit/report.json` (본체 `.venv`, 약 15초)
- 보고 파일에는 녹음별 해시, 들어간 학습셋, CV fold가 있다
