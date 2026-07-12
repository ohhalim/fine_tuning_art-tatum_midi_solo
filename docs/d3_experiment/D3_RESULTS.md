# D3 결과 (1차): 두손 데이터 FT 처방 실패 — 병목은 base의 화음 생성 능력

> Issue #1458 / 로드맵 L4. 실행: 2026-07-12. 판정 기준은 실행 전 사전 등록(#1458).

## 확정된 것 (H-D3a 측정)

단선율의 1차 원인은 lead 분리 데이터가 맞다 — 증거 사슬:

| 단계 | mean_voicing_size | 노트 밀도 /s |
|---|---|---|
| 원본 두손 Mehldau (18곡) | **1.657** | 5.52 |
| lead 분리 학습 데이터 (split_pitch=60) | 1.323 | 2.55 |
| 생성 (Arm D + mask) | 1.305 | 2.58 |

생성이 lead 데이터 상한에 도달해 있으므로, 데이터 상한을 올리면 생성도 오를 것이라는
처방(Arm F)을 시험했다.

## Arm F 처방 실험 — 실패

구성: Arm B ckpt → LoRA-only FT on 두손 Mehldau 토큰(16/2) → 24샘플 `--grammar_mask` 생성.

| 지표 | Arm D masked | Arm F | 기준 | 판정 |
|---|---|---|---|---|
| mean_voicing_size | 1.305 | **1.290** | ≥1.49 | ❌ |
| pooled 고유 보이싱 | 149 | 151 | ≥135 | ✅ |
| val loss (mehldau_full val) | — | 3.084→3.032 | 하강 | △ 미미 |

**두손 데이터로 FT해도 생성 화음이 전혀 늘지 않았다.**

## 실패 분석 — 두 가지 발견

1. **FT가 사실상 no-op였다.** val loss가 3.08→3.03으로 거의 안 움직임 (Arm D의 FT는
   3.37→3.14로 움직였음). 원인: **두손 Mehldau 18곡은 사전학습 코퍼스(midi_dataset 전체
   → jazz_full 2,777)에 이미 포함돼 있다.** base가 이미 아는 데이터로 FT한 셈이라
   모델이 거의 변하지 않았고, 생성도 그대로였다. lead-split FT(Arm D)가 효과 있었던 건
   그 분리 형태가 사전학습에 없는 새 분포였기 때문.

2. **진짜 병목은 base의 화음 생성 갭이다.** base(Arm B)는 자기 학습 데이터의 voicing
   상한이 1.66인데 생성은 1.32에 머문다 (D0에서도 동일: armB gen 1.317). 데이터에는
   화음이 있는데 **모델이 화음을 치지 않는다**. 따라서 FT 데이터를 바꾸는 것으로는
   해결되지 않고, base 수준에서 해결해야 한다.

## 갱신된 가설 (다음 실험 후보)

- **H-D3c (최우선 부상)**: 사전학습 부족 — 8ep에서 val이 계속 하강 중이었음(D0).
  화음(무간격 연속 note_on)은 고차 구조라 늦게 학습될 수 있다. **사전학습 16ep+ 연장 후
  base 생성의 voicing_size 재측정.**
- **H-D3d (신규)**: 샘플링 원인 — 화음은 time_shift 없는 연속 note_on인데 temp 1.0
  샘플링이 화음 연쇄를 끊을 수 있다. temp/top_p 스윕으로 저비용 선행 검증 가능.
- H-D3b(디코딩)는 D2에서 기각, H-D3a(FT 데이터)는 필요조건이지만 충분조건 아님(본 실험).

권장 순서: H-D3d 먼저 (학습 불필요, 분 단위) → 기각되면 H-D3c (사전학습 연장, ~30분).

## 재현

```bash
# 데이터: preprocess_jazz.py --input_dir "midi_dataset/midi/studio/Brad Mehldau" --output_dir data/mehldau_full
# 학습·생성: D1 Arm D와 동일 커맨드에서 --data_dir만 data/mehldau_full로 교체
```

원자료: `outputs/d3_experiment/`, 비교 JSON 사본은 이 디렉터리.
