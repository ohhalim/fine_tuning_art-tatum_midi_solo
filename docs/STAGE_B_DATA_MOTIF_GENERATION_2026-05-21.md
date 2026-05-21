# Stage B Data Motif Generation

작성일: 2026-05-21

## 목적

Issue #63은 실제 Stage B phrase window에서 rhythm/contour/full motif templates를 추출했다.

이번 작업은 그 catalog를 실제 8-bar generation baseline으로 연결한다.

핵심은 hand-written `swing_motif_approach`를 더 늘리는 것이 아니라, 실제 MIDI에서 뽑은 motif material을 사용해 다음을 비교하는 것이다.

- hand-written swing rhythm baseline
- data-derived motif baseline

이 작업은 아직 broad training이 아니다.
또한 "재즈 솔로 모델 완성"도 아니다.

## 구현

추가 파일:

- `scripts/run_stage_b_data_motif_generation_compare.py`
- `tests/test_stage_b_data_motif_generation_compare.py`

하네스:

```bash
bash scripts/agent_harness.sh stage-b-data-motif-compare
```

비교 방식:

- 먼저 `scripts/run_stage_b_motif_template_extraction.py`로 motif catalog를 만든다.
- `hand_written_swing`은 기존 static swing/motif position-duration grammar를 사용한다.
- `data_motif`은 extracted rhythm template을 position/duration 후보로 사용한다.
- `data_motif`은 extracted contour template을 pitch interval 후보로 사용한다.
- pitch는 current chord/tension/approach 후보군 위에 가장 가까운 pitch로 투영한다.

## Duration Fit

초기 실행에서 data-derived motif는 grammar/collapse gate는 통과했지만, duration을 그대로 쓰면서 note overlap이 생겼고 postprocess가 많은 note를 제거했다.

그 결과 dead-air ratio가 `0.821`로 gate를 실패했다.

따라서 data-derived baseline은 motif duration을 다음 onset 전까지로 제한한다.

이것은 음악적 성공을 꾸미는 postprocess가 아니라, solo-line baseline에서 동시에 겹치는 sustain을 만들지 않기 위한 constraint다.

## Local Result

실행:

```bash
bash scripts/agent_harness.sh stage-b-data-motif-compare
```

결과:

- compare gate: `true`
- `hand_written_swing`: strict `3/3`
- `data_motif`: strict `3/3`
- data minus hand duration diversity delta: `+0.016`
- data minus hand IOI diversity delta: `+0.016`
- data minus hand bar-pattern delta: `+0.500`
- data minus hand syncopation delta: `-0.125`

요약 표:

| mode | strict | sync | bar-var | dur-var | dur-rep | ioi-var | ioi-rep | tension | root |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| data_motif | 3/3 | 0.625 | 1.000 | 0.062 | 0.375 | 0.079 | 0.429 | 0.177 | 0.000 |
| hand_written_swing | 3/3 | 0.750 | 0.500 | 0.047 | 0.750 | 0.063 | 0.476 | 0.198 | 0.000 |

## 해석

좋아진 점:

- `data_motif`도 strict review gate를 통과한다.
- bar-position pattern variation이 `0.500`에서 `1.000`으로 오른다.
- most-common duration ratio가 `0.750`에서 `0.375`로 낮아진다.
- IOI repetition도 약간 낮아진다.

나빠졌거나 아직 부족한 점:

- syncopation은 `0.750`에서 `0.625`로 낮아졌다.
- duration diversity와 IOI diversity 상승폭은 아직 작다.
- 이것만으로 jazz vocabulary가 생겼다고 말할 수 없다.
- 실제 piano-roll/listening review가 필요하다.

판단:

> data-derived motif baseline은 hand-written baseline보다 bar-to-bar variation과 duration repetition 측면에서 낫지만, 아직 "재즈 솔로" 품질을 증명하지는 않는다.

## 다음 작업

다음 단계는 생성된 `data_motif` MIDI를 review export 대상으로 분리하고, hand-written swing 후보와 실제 piano roll에서 비교하는 것이다.

확인할 것:

- data_motif가 "패턴 변화"만 늘린 것인지, 실제 phrase처럼 들리는지
- syncopation 하락이 체감상 groove 약화로 들리는지
- contour가 chord-tone 나열을 넘어선 phrase motion으로 들리는지
- extracted contour를 더 길게 연결해야 하는지

통과 기준:

- data_motif MIDI가 초급 scale exercise처럼만 들리지 않는다.
- 8-bar phrase 안에 call/continuation/landing 느낌이 일부라도 있다.
- 다음 broad generic training 전에 사용할 constraint candidate로 남길 가치가 있다.

## Validation

```bash
./.venv/bin/python -m unittest tests.test_stage_b_data_motif_generation_compare
bash scripts/agent_harness.sh stage-b-data-motif-compare
```
