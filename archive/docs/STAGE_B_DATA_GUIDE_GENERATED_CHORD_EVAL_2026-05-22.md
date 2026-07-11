# Stage B Data-Guide Hybrid Generated Chord Evaluation

작성일: 2026-05-22

## 배경

Issue #81에서 generated candidate report를 chord-labeled evaluator에 연결하는 bridge를 만들었다.

이번 단계는 그 bridge를 실제 Stage B data-guide hybrid review package에 적용한다.

목적은 다음이다.

- `data_motif`와 `data_motif_guide_tones` 후보의 chord-role profile을 같은 evaluator로 비교한다.
- review MIDI를 듣기 전에 chord-tone/tension/approach/outside 비율을 확인한다.
- raw generated MIDI는 계속 `outputs/` artifact로만 둔다.

## 구현

새 harness:

```bash
bash scripts/agent_harness.sh stage-b-data-guide-generated-chord-eval
```

실행 순서:

1. `stage-b-data-guide-hybrid` review package를 생성한다.
2. 생성된 `review_manifest.json`을 `evaluate_generated_candidate_chords.py`에 넣는다.
3. top generated review MIDI 후보를 chord-labeled evaluator로 분석한다.

입력:

```text
outputs/stage_b_data_motif_review/harness_stage_b_data_guide_generated_chord_eval/review_manifest.json
```

출력:

```text
outputs/stage_b_generated_chord_eval/harness_stage_b_data_guide_generated_chord_eval/generated_chord_eval_report.json
outputs/stage_b_generated_chord_eval/harness_stage_b_data_guide_generated_chord_eval/generated_chord_eval_report.md
```

## 결과

요약:

| metric | value |
|---|---:|
| sample count | 6 |
| note count | 192 |
| chord-tone ratio | 0.656 |
| tension ratio | 0.120 |
| outside ratio | 0.000 |

Candidate summary:

| sample | notes | chord-tone | tension | approach | outside |
|---|---:|---:|---:|---:|---:|
| `data_motif_rank_1_sample_1` | 32 | 0.500 | 0.219 | 0.281 | 0.000 |
| `data_motif_rank_2_sample_2` | 32 | 0.500 | 0.188 | 0.312 | 0.000 |
| `data_motif_rank_3_sample_3` | 32 | 0.500 | 0.219 | 0.281 | 0.000 |
| `data_motif_guide_tones_rank_1_sample_1` | 32 | 0.812 | 0.031 | 0.156 | 0.000 |
| `data_motif_guide_tones_rank_2_sample_2` | 32 | 0.812 | 0.031 | 0.156 | 0.000 |
| `data_motif_guide_tones_rank_3_sample_3` | 32 | 0.812 | 0.031 | 0.156 | 0.000 |

## 판단

이번 결과는 `data_motif_guide_tones`의 pitch grammar가 실제로 chord-tone 쪽으로 더 강하게 기울어져 있음을 보여준다.

구체적으로:

- `data_motif`는 chord-tone `0.500`이고 approach `0.281-0.312`가 많다.
- `data_motif_guide_tones`는 chord-tone `0.812`이고 approach `0.156`으로 낮다.
- outside ratio는 둘 다 `0.000`이다.

따라서 현재 "초급 멜로디처럼 들린다"는 문제는 outside note 폭주가 아니다.

더 정확한 문제 후보는 다음이다.

- chord-tone safety가 너무 강함
- tension 비율이 낮음
- phrase vocabulary와 motif development가 아직 부족함

## 다음 작업

다음은 generated review markdown에 이 chord eval summary를 자동 첨부하는 것이다.

그 다음 판단:

- chord-tone이 너무 낮은 후보는 review 우선순위에서 내린다.
- chord-tone이 높아도 초급스럽게 들리면 tension/approach/cadence vocabulary를 조정한다.
- real reference chord labels는 계속 임의로 만들지 않는다.

## Validation

```bash
bash scripts/agent_harness.sh stage-b-data-guide-generated-chord-eval
bash scripts/agent_harness.sh quick
```
