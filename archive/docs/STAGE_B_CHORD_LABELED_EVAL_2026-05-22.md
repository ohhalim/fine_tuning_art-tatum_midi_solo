# Stage B Chord-Labeled Evaluation Subset Contract

작성일: 2026-05-22

## 배경

Issue #77에서 현재 local dataset에는 바로 사용할 수 있는 chord progression annotation이 없다는 점을 확인했다.

따라서 이번 단계의 목적은 실제 Brad/reference 곡에 코드를 추측해서 붙이는 것이 아니다.

이번 단계는 다음을 먼저 만든다.

- chord-labeled evaluation manifest schema
- manifest validator
- inline-note tiny fixture
- bar-level chord label 기반 pitch-role summary
- 다음 수동 라벨링이 들어갈 파일 포맷

## 구현

새 스크립트:

```bash
python scripts/evaluate_chord_labeled_subset.py
```

기본 manifest:

```text
data/eval/stage_b_chord_labeled_tiny/manifest.json
```

새 harness:

```bash
bash scripts/agent_harness.sh stage-b-chord-labeled-eval
```

Manifest schema:

- `schema_version`: `stage_b_chord_labeled_eval_v1`
- `samples[].sample_id`
- `samples[].bar_count`
- `samples[].chords`
- exactly one of:
  - `samples[].midi_path`
  - `samples[].notes`

중요한 제약:

- `chords.length == bar_count`
- unsupported chord symbol은 실패 처리한다.
- 실제 MIDI는 `midi_path`로 넣을 수 있지만, repo에는 raw MIDI를 커밋하지 않는다.
- 현재 committed fixture는 `inline_notes`만 사용한다.

## 결과

실행:

```bash
bash scripts/agent_harness.sh stage-b-chord-labeled-eval
```

출력:

```text
outputs/stage_b_chord_labeled_eval/harness_stage_b_chord_labeled_eval/chord_labeled_eval_report.json
outputs/stage_b_chord_labeled_eval/harness_stage_b_chord_labeled_eval/chord_labeled_eval_report.md
```

요약:

| metric | value |
|---|---:|
| sample count | 2 |
| note count | 32 |
| chord-tone ratio | 0.844 |
| tension ratio | 0.156 |
| approach ratio | 0.000 |
| outside ratio | 0.000 |

Sample summary:

| sample | bars | notes | chord-tone | tension | outside |
|---|---:|---:|---:|---:|---:|
| `tiny_ii_v_i_inline_01` | 4 | 16 | 0.812 | 0.188 | 0.000 |
| `tiny_minor_turnaround_inline_01` | 4 | 16 | 0.875 | 0.125 | 0.000 |

## 판단

이 결과는 real jazz reference가 라벨링됐다는 뜻이 아니다.

의미는 다음이다.

- pitch-role evaluator가 known chord labels를 받으면 정상 동작한다.
- 앞으로 사람이 3-5개 phrase에 chord labels를 붙이면 같은 contract로 sanity check를 돌릴 수 있다.
- generated MIDI의 chord-tone/tension/outside 비율을 평가할 최소한의 기준 파일 포맷이 생겼다.

## 다음 작업

다음 단계는 두 갈래 중 하나다.

우선순위 1:

```text
3-5개 실제 review phrase에 수동 chord labels를 붙인다.
```

조건:

- 코드가 확실한 phrase만 넣는다.
- 불확실하면 manifest에 넣지 않는다.
- raw MIDI 파일은 커밋하지 않는다.

우선순위 2:

```text
generated candidate report를 chord-labeled eval manifest와 비교하는 bridge를 만든다.
```

조건:

- generated candidate가 어떤 chord progression 위에서 만들어졌는지 metadata가 있어야 한다.
- solo-only MIDI만으로 in/out 판단하지 않는다.

## Validation

```bash
./.venv/bin/python -m unittest tests.test_chord_labeled_subset_eval
bash scripts/agent_harness.sh stage-b-chord-labeled-eval
bash scripts/agent_harness.sh quick
```
