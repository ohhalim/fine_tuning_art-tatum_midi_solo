# Stage B Margin-Recovered Phrase/Vocabulary Keep Consolidation

## 요약

Issue #280은 Issue #278에서 나온 filled listening `keep` 결과를 current margin-recovered evidence keep candidate로 정리한 작업이다.

이 문서는 후보를 최종 음악 품질이나 human audio preference로 과장하지 않고, 지금까지 검증된 범위와 아직 검증되지 않은 범위를 분리한다.

## Current Margin-Recovered Evidence Keep Candidate

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` |
| source run | `harness_stage_b_margin_recovered_phrase_vocab_seed43_topk7_temp082_n48` |
| sample seed | `85` |
| decision path | objective repair -> focused context -> focused listening notes -> evidence fill |
| focused context decision | `keep_for_focused_listening` |
| filled listening decision | `keep` |
| note count | `13` |
| unique pitch count | `8` |
| range | `G4-E5` |
| phrase span | `7.000` beats |
| max active notes | `1` |
| dead-air ratio | `0.333` |
| sustained coverage | `0.594` |
| adjacent pitch repeats | `0` |
| max interval | `7` |
| duplicated 3-note chunks | `0` |
| final landing | `C5` over `Fm7`, chord tone |
| remaining evidence risk | `sustained_coverage_review` |

## Proven

- `.mid` existence가 아니라 note-level metric과 focused context decision을 거쳐 후보를 검증했다.
- Issue #270에서 남은 adjacent repeat `2`와 max interval `16` blocker를 각각 `0`, `7`로 줄였다.
- focused context package에서 chord guide, bass guide, solo track 존재와 max active notes `1`을 확인했다.
- final note는 outside가 아니라 `Fm7` 위 chord tone이다.
- MIDI/context evidence fill 기준으로 timing, phrase continuation, landing, jazz vocabulary가 모두 keep 조건을 통과했다.

## Not Proven

- human/audio listening preference
- broad trained-model quality
- Brad style adaptation
- robust repeatability across broader seeds/files
- production-ready improviser

## 다음 경계

- current keep candidate를 반복 가능한 품질로 주장하려면 stability comparison 또는 listening comparison이 필요하다.
- broad training으로 바로 넘어가기보다, 같은 candidate path가 다른 seeds/files에서도 유지되는지 확인하는 issue를 별도로 둔다.

## 검증

```bash
bash scripts/agent_harness.sh quick
```

## 관련 문서

- `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_REPAIR_2026-05-28.md`
- `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_FOCUSED_CONTEXT_2026-05-28.md`
- `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_FOCUSED_LISTENING_NOTES_2026-05-28.md`
- `docs/STAGE_B_MARGIN_RECOVERED_PHRASE_VOCABULARY_FOCUSED_LISTENING_FILL_2026-05-28.md`
