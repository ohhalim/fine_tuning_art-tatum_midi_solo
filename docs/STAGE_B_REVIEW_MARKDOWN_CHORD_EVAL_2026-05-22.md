# Stage B Review Markdown With Chord Eval Summary

작성일: 2026-05-22

## 배경

Issue #83에서 data-guide hybrid 후보의 generated chord eval report를 만들었다.

이번 단계는 그 chord-role summary를 기존 review markdown과 결합한다.

이유:

- MIDI 파일명과 context MIDI path는 review markdown에 있다.
- chord-tone/tension/approach/outside summary는 generated chord eval report에 있다.
- 두 파일을 따로 보면 청취 리뷰 중 판단이 끊긴다.

## 구현

변경:

- `scripts/evaluate_generated_candidate_chords.py`
  - `--review_markdown`
  - `--combined_review_markdown_name`
  - `chord_eval_review_append_markdown()`
  - `write_combined_review_markdown()`
- `scripts/agent_harness.sh stage-b-review-markdown-chord-eval`

새 harness:

```bash
bash scripts/agent_harness.sh stage-b-review-markdown-chord-eval
```

출력:

```text
outputs/stage_b_generated_chord_eval/harness_stage_b_review_markdown_chord_eval/review_candidates_with_chord_eval.md
```

원본 review markdown은 수정하지 않는다.

## 결과

Combined markdown에는 다음이 함께 들어간다.

- 기존 review candidates table
- solo MIDI path
- context MIDI path
- rhythm metrics
- strict gate result
- generated chord eval summary
- candidate별 chord-tone/tension/approach/outside ratio

요약:

| metric | value |
|---|---:|
| evaluated candidates | 6 |
| note count | 192 |
| chord-tone ratio | 0.656 |
| tension ratio | 0.120 |
| approach ratio | 0.224 |
| outside ratio | 0.000 |

Candidate chord-role summary:

| sample | chord-tone | tension | approach | outside |
|---|---:|---:|---:|---:|
| `data_motif_rank_1_sample_1` | 0.500 | 0.219 | 0.281 | 0.000 |
| `data_motif_rank_2_sample_2` | 0.500 | 0.188 | 0.312 | 0.000 |
| `data_motif_rank_3_sample_3` | 0.500 | 0.219 | 0.281 | 0.000 |
| `data_motif_guide_tones_rank_1_sample_1` | 0.812 | 0.031 | 0.156 | 0.000 |
| `data_motif_guide_tones_rank_2_sample_2` | 0.812 | 0.031 | 0.156 | 0.000 |
| `data_motif_guide_tones_rank_3_sample_3` | 0.812 | 0.031 | 0.156 | 0.000 |

## 판단

이제 청취 리뷰용 markdown에서 다음을 한 번에 볼 수 있다.

- 어떤 MIDI를 들어야 하는지
- context MIDI가 어디 있는지
- rhythm profile이 어떤지
- chord-role profile이 어떤지

이는 모델 품질 성공 선언이 아니다.

의미는 다음이다.

- generated 후보를 듣기 전에 수치상 이상한 후보를 빠르게 걸러낼 수 있다.
- `data_motif_guide_tones`가 더 안전한 chord-tone line인지, 너무 안전해서 초급스럽게 들리는지 청취 리뷰로 확인할 수 있다.

## 다음 작업

다음은 listening review 결과를 구조화하는 것이다.

후보:

```text
Stage B listening review notes schema 추가
```

필요한 필드:

- candidate id
- heard as phrase or exercise
- timing acceptable
- chord context fit
- too safe / too scalar / too mechanical
- keep / reject / needs follow-up

## Validation

```bash
./.venv/bin/python -m unittest tests.test_generated_candidate_chord_eval
bash scripts/agent_harness.sh stage-b-review-markdown-chord-eval
bash scripts/agent_harness.sh quick
```
