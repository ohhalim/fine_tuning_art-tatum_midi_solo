# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Focused Listening Fill

Issue #320은 duration/coverage fill candidate의 focused listening notes를 MIDI/context evidence로 채운 작업이다.

## Context

- Issue #318 notes result: pending `1`
- candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- prior decision: `keep_for_focused_listening`
- notes risk before fix: `sustained_coverage_review`
- source issue: token-derived onset/sustained coverage metric absent

## Change

- source coverage metric 부재 시 solo MIDI grid 기반 coverage 산출
- focused context decision 재생성
- focused listening notes 재생성
- focused listening evidence fill 실행

## Result

| item | value |
|---|---:|
| candidate count | `1` |
| reviewed count | `1` |
| pending count | `0` |
| decision | `keep` |
| review risks | `[]` |
| timing | `acceptable` |
| chord fit | `strong` |
| phrase continuation | `acceptable` |
| landing | `strong` |
| jazz vocabulary | `acceptable` |

Focused context metrics:

| item | value |
|---|---:|
| note count | `18` |
| unique pitch count | `15` |
| range | `D#4-G#5` |
| phrase span | `7.000` beats |
| dead-air ratio | `0.2941` |
| onset coverage | `0.5625` |
| sustained coverage | `0.6250` |
| adjacent pitch repeats | `0` |
| duplicated 3-note pitch-class chunks | `0` |
| max active notes | `1` |
| max interval | `7` |
| final note | `F4` over `Fm7`, chord tone |

## Judgment

- duration/coverage fill candidate remains keep under MIDI/context evidence fill.
- adjacent repeat and wide interval blockers remain repaired.
- review risk list cleared after MIDI-derived coverage metric.
- claim boundary: MIDI/context evidence fill, not human audio proof.
- broad trained-model quality and Brad style adaptation remain unverified.

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_margin_recovered_focused_context_decision.py tests/test_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill.py
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-listening-fill
```

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill keep consolidation`
- postprocess candidate boundary, human review boundary, broad training boundary 분리
