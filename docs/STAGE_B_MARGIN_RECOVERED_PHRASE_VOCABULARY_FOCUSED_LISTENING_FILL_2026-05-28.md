# Stage B Margin-Recovered Phrase/Vocabulary Focused Listening Fill

## 요약

Issue #278은 Issue #276 focused listening notes를 MIDI/context evidence 기준으로 채운 작업이다.

이 단계는 pending 상태였던 timing, chord fit, phrase continuation, landing, jazz vocabulary, decision field를 metric/context evidence로 채우되, human audio listening proof나 broad model quality로 과장하지 않는 경계다.

## 변경

- phrase/vocabulary focused listening notes fill script 추가
- focused context metrics를 listening field로 변환
- sustained coverage risk를 evidence로 보존하고, adjacent repeat / wide interval blocker repair 상태를 함께 기록
- result docs와 current plan 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` |
| reviewed count | `1` |
| pending count | `0` |
| timing | `acceptable` |
| chord fit | `strong` |
| phrase continuation | `acceptable` |
| landing | `strong` |
| jazz vocabulary | `acceptable` |
| final decision | `keep` |
| dead-air ratio | `0.333` |
| sustained coverage | `0.594` |
| adjacent pitch repeats | `0` |
| max interval | `7` |
| final landing | `C5` over `Fm7`, chord tone |
| review risks | `sustained_coverage_review` |

## 해석

- Issue #270의 blocker였던 adjacent repeats와 wide interval은 repair 상태로 유지됐다.
- final landing은 chord tone이고, focused context blocker도 없다.
- sustained coverage가 `0.594`로 threshold 근처라 evidence에는 risk로 남긴다.
- decision `keep`은 MIDI/context evidence fill 기준의 keep이며, human/audio preference나 broad model quality proof는 아니다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-focused-listening-fill
```

## 산출물

- script: `scripts/fill_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill.py`
- filled notes: `outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill/harness_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill/focused_listening_review_notes_filled.json`
