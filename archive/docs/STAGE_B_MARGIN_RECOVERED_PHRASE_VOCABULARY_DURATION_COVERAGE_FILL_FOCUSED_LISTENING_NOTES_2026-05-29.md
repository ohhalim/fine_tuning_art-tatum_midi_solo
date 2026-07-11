# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Focused Listening Notes

Issue #318은 Issue #316 focused context keep candidate의 focused listening notes template을 생성한 작업이다.

## Context

- candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- prior decision: `keep_for_focused_listening`
- context flags: `[]`
- note count: `18`
- unique pitch count: `15`
- range: `D#4-G#5`
- phrase span: `7.000` beats
- dead-air ratio: `0.2941`
- adjacent pitch repeats: `0`
- max interval: `7`
- final note: `F4` over `Fm7`, chord tone

## Change

- duration/coverage fill focused listening notes harness 추가
- focused package/context decision dependency check 추가
- focused listening notes template 생성
- listening fields pending 유지

## Result

| item | value |
|---|---:|
| candidate count | `1` |
| reviewed count | `0` |
| pending count | `1` |
| prior decision | `keep_for_focused_listening` |
| listening decision | `pending` |
| review risks | `sustained_coverage_review` |

Focused context metrics:

| item | value |
|---|---:|
| note count | `18` |
| unique pitch count | `15` |
| phrase span | `7.000` beats |
| dead-air ratio | `0.2941` |
| adjacent pitch repeats | `0` |
| duplicated 3-note pitch-class chunks | `0` |
| max active notes | `1` |
| max interval | `7` |
| final note role | `chord_tone` |

## Judgment

- focused listening review template 생성 완료.
- pending 상태 유지.
- notes template은 human/audio listening proof가 아님.
- 남은 review risk: `sustained_coverage_review`.

## Validation

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-listening-notes
```

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill focused listening fill`
- evidence fill에서 timing, phrase continuation, landing, vocabulary decision 기록
