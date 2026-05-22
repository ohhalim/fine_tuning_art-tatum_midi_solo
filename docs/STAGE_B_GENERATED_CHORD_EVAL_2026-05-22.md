# Stage B Generated Candidate Chord Evaluation Bridge

작성일: 2026-05-22

## 배경

Issue #79에서 chord-labeled evaluation contract를 만들었다.

이번 단계는 generated candidate report가 이미 알고 있는 chord progression metadata를 그 evaluator에 연결한다.

이 작업은 real reference chord labels 문제를 해결하지 않는다.

목적은 다음이다.

- generated MIDI candidate가 어떤 chord progression 위에서 만들어졌는지 metadata가 있을 때만 평가한다.
- solo-only MIDI를 억지로 in/out 판단하지 않는다.
- generated candidate의 chord-tone/tension/outside ratio를 같은 evaluator contract로 산출한다.

## 구현

새 스크립트:

```bash
python scripts/evaluate_generated_candidate_chords.py
```

새 harness:

```bash
bash scripts/agent_harness.sh stage-b-generated-chord-eval
```

지원 입력:

- review manifest with:
  - `chord_progression`
  - `candidates[].review_midi_path`
  - `candidates[].midi_path`
- generation report with:
  - `request.chord_progression`
  - `request.bars`
  - `samples[].midi_path`
- review manifest that points to `source_report`

Harness는 raw MIDI를 커밋하지 않고, `outputs/` 아래 tiny generated-candidate fixture를 만들어 bridge를 검증한다.

## 결과

실행:

```bash
bash scripts/agent_harness.sh stage-b-generated-chord-eval
```

출력:

```text
outputs/stage_b_generated_chord_eval/harness_stage_b_generated_chord_eval/generated_chord_eval_report.json
outputs/stage_b_generated_chord_eval/harness_stage_b_generated_chord_eval/generated_chord_eval_report.md
```

요약:

| metric | value |
|---|---:|
| sample count | 1 |
| note count | 16 |
| chord-tone ratio | 1.000 |
| tension ratio | 0.000 |
| outside ratio | 0.000 |

Fixture chord progression:

```text
Cm7, F7, Bbmaj7, G7
```

## 판단

이 결과는 다음을 의미한다.

- generated candidate report에 chord metadata가 있으면 pitch-role evaluator로 연결할 수 있다.
- 기존 candidate ranking의 chord-tone proxy와 별도로, manifest/evaluator 기반 report를 만들 수 있다.
- 앞으로 generated review package마다 같은 bridge를 붙이면 청취 전 수치 sanity check가 쉬워진다.

이 결과가 의미하지 않는 것:

- real Brad/reference phrase chord labels가 생긴 것은 아니다.
- generated sample이 jazz solo quality를 만족한다는 뜻도 아니다.
- fixture chord-tone ratio `1.000`은 bridge smoke result일 뿐 model score가 아니다.

## 다음 작업

다음 단계는 실제 generated review package에 bridge를 적용하는 것이다.

우선순위:

```text
stage-b-data-guide-hybrid review_manifest를 generated chord eval bridge에 연결한다.
```

조건:

- `review_manifest.json`에 chord progression metadata가 있어야 한다.
- raw generated MIDI는 계속 `outputs/` artifact로만 둔다.
- PR에는 script/docs/test만 커밋한다.

그 다음:

```text
generated chord eval report를 review markdown에 같이 붙인다.
```

## Validation

```bash
./.venv/bin/python -m unittest tests.test_generated_candidate_chord_eval
bash scripts/agent_harness.sh stage-b-generated-chord-eval
bash scripts/agent_harness.sh quick
```
