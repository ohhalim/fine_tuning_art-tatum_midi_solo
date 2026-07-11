# Stage B Reference Pitch-Role Landing Stats

작성일: 2026-05-22

## 배경

Issue #73의 `data_motif_guide_tones`는 strict `3/3`을 통과했고 root-tone ratio는 `0.000`, chord-tone ratio는 `0.797`이었다.

하지만 이 수치만으로는 좋은 jazz solo 후보인지 판단할 수 없다.

- chord-tone ratio가 높으면 안정적일 수 있다.
- 동시에 너무 안전해서 초급 멜로디처럼 들릴 수 있다.
- tension/approach가 낮으면 jazz vocabulary가 약할 수 있다.

따라서 실제 Stage B jazz phrase window에서 bar/position별 pitch role landing 분포를 reference로 만들려고 했다.

## 구현

`scripts/run_stage_b_reference_stats.py`에 다음을 추가했다.

- embedded Stage B chord token에서 bar chord 추출
- note group별 pitch role 분류:
  - `root`
  - `guide`
  - `chord`
  - `tension`
  - `approach`
  - `outside`
  - `unknown_chord`
- strong/eighth/offgrid bucket별 role 분포
- generated report와 reference rhythm/pitch-role delta 비교
- reference chord coverage가 부족하면 pitch-role delta를 의도적으로 생략하는 guard

새 harness:

```bash
bash scripts/agent_harness.sh stage-b-reference-pitch-roles
```

## 결과

Reference rhythm/contour 통계는 정상적으로 생성됐다.

- record count: `57`
- note group mean: `32.649`
- unique pitch mean: `11.000`
- syncopated onset ratio mean: `0.736`
- unique bar-position pattern ratio mean: `0.996`
- duration diversity ratio mean: `0.379`
- IOI diversity ratio mean: `0.341`

하지만 pitch-role reference는 아직 사용할 수 없다.

- known chord note ratio: `0.000`
- unknown chord ratio: `1.000`
- known chord note count: `0`
- unknown chord note count: `1861`

즉 현재 Stage B reference tokenized records에는 chord progression annotation이 들어있지 않다.

## Generated Comparison

Pitch-role delta는 의도적으로 생략했다.

이유:

```text
Reference tokenized records do not have enough chord annotations for pitch-role comparison
```

현재 비교 가능한 것은 rhythm 계열뿐이다.

`data_motif_guide_tones` vs reference mean:

- syncopated onset ratio delta: `-0.111`
- unique bar-position pattern ratio delta: `+0.004`
- duration diversity ratio delta: `-0.317`
- most common duration ratio delta: `+0.115`
- IOI diversity ratio delta: `-0.262`
- most common IOI ratio delta: `+0.090`

해석:

- bar-position pattern 다양성은 reference에 가깝다.
- syncopation은 reference보다 낮다.
- duration/IOI diversity는 reference보다 많이 낮다.
- pitch-role은 reference chord annotation이 없어서 아직 비교 불가다.

## 판단

다음에 generator를 더 고치는 것은 순서가 아니다.

먼저 reference dataset에 chord annotation을 넣거나, 최소한 audit해서 chord progression이 있는 subset을 찾아야 한다.

다음 작업:

1. Stage B source metadata에서 chord progression coverage를 audit한다.
2. chord progression이 있는 파일만 reference pitch-role stats에 사용한다.
3. 없으면 chord inference/lead-sheet alignment는 별도 이슈로 분리한다.
4. 그 뒤에야 `data_motif_guide_tones`의 tension/approach 비율이 reference보다 낮은지 판단한다.

## Output

```text
outputs/stage_b_reference_stats/harness_stage_b_reference_pitch_roles/reference_stats_report.json
outputs/stage_b_reference_stats/harness_stage_b_reference_pitch_roles/reference_stats_report.md
```

## Validation

```bash
./.venv/bin/python -m unittest tests.test_stage_b_reference_stats
bash scripts/agent_harness.sh stage-b-reference-pitch-roles
bash scripts/agent_harness.sh quick
```
