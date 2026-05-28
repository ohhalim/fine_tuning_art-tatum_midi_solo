# Stage B Margin-Recovered Phrase/Vocabulary Focused Listening Notes

## 요약

Issue #276은 Issue #274 focused context keep 후보를 focused listening review notes template으로 넘긴 작업이다.

이 단계는 context decision을 최종 청감 판단으로 과장하지 않고, timing, chord fit, phrase continuation, landing, jazz vocabulary, decision field를 pending 상태로 분리해 기록하기 위한 경계다.

## 변경

- phrase/vocabulary focused package와 focused context decision을 함께 읽는 notes wrapper 추가
- focused context metrics, prior decision, review risks를 notes candidate에 보존
- adjacent repeat / wide interval repair 상태가 notes risk에 다시 올라오지 않는지 단위 테스트로 확인
- result docs와 current plan 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` |
| candidate count | `1` |
| reviewed count | `0` |
| pending count | `1` |
| prior decision | `keep_for_focused_listening` |
| listening decision | `pending` |
| note count | `13` |
| unique pitch count | `8` |
| range | `G4-E5` |
| phrase span | `7.000` beats |
| dead-air ratio | `0.333` |
| onset coverage | `0.500` |
| sustained coverage | `0.594` |
| adjacent pitch repeats | `0` |
| max interval | `7` |
| final landing | `C5` over `Fm7`, chord tone |
| review risks | `sustained_coverage_review` |

## 해석

- focused context keep 후보를 listening notes template으로 넘겼다.
- 실제 청감 판단 필드는 모두 `pending`이다.
- Issue #270의 blocker였던 adjacent repeat와 wide interval은 notes risk로 재등장하지 않았다.
- sustained coverage가 `0.594`라서 청감 fill에서 phrase continuity를 다시 확인해야 한다.
- 아직 human/audio preference나 broad model quality proof는 아니다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-focused-listening-notes
```

## 산출물

- script: `scripts/build_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py`
- notes: `outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes/harness_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes/focused_listening_review_notes_template.json`
