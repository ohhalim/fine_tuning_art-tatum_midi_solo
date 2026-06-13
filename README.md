# Jazz Piano MIDI Solo Generator MVP

입력 MIDI/chord context를 받아 짧은 jazz piano solo-line MIDI 후보를 만들고, 후보별 WAV와 objective review package까지 생성하는 symbolic MIDI model-core MVP.

## 현재 상태

- 생성 방식: Music Transformer 계열 symbolic checkpoint + constrained decoding
- 출력 단위: 2-8마디 solo-line MIDI 후보
- 최신 대표 review package: strict-listen top4 stepwise-large-leap-guard, MIDI `4`, WAV `4`
- 최신 frontier review package: strict-listen top8 case-balanced motion interval-guard, MIDI `8`, solo/context WAV `8 / 8`, all-selected note review `8`
- 최신 interval guard: cap `0.0125` selected `8` 불가, cap `0.0164` selected `8`, case counts `2/2/2/2`, interval repeat avg `0.0123`
- 최신 review handoff: case-balanced motion interval-guard, review ready `true`, solo/context WAV `8 / 8`, listen-first audio pair `4`, note review validated `true`
- 최신 guard feasibility: motion-balance guard/motion feasible config `20`, stricter offbeat feasible `false`, stricter bar similarity feasible `true`
- 최신 pool expansion review package: strict-listen top8 safety-pool-expansion, representative replacement `false`
- 최신 feasibility sweep: safety-gate motion/leap strict filter feasible config `0`
- 최신 objective gate: max gate penalty `0.0000`
- 최신 chord/rhythm 지표: strong-beat chord-tone `1.0000`, offbeat non-chord `0.3984`, offbeat resolution `1.0000`, unresolved offbeat `0.0000`, large leap `0.0437`, adjacent repeat `0.0000`, enclosure proxy `0.3203`, bar pitch-class similarity `0.6131`
- 최신 contour 지표: step motion `0.3849 -> 0.4087`, chromatic step `0.1905 -> 0.2143`, third/fourth motion `0.5635 -> 0.5476`
- 최신 articulation 지표: duration template repeat `0.8750 -> 0.3750`, most common duration `0.5000 -> 0.2031`, duration bucket `3 -> 16`, velocity count `4 -> 17`
- 기준 best-of review package: MIDI `16`, WAV `16`
- residual-aware objective rubric: pass/fail `6 / 2`
- validated listening input: `false`
- human/audio preference claim: `false`
- musical quality claim: `false`

## 구현 범위

- MIDI dataset tokenization
- Music Transformer 계열 symbolic checkpoint 학습 경로
- checkpoint logits 기반 candidate generation
- chord/rhythm 범위 기반 constrained decoding
- MIDI 문법 gate
- objective metric 계산
- 후보 ranking 및 MIDI export
- FluidSynth 기반 WAV render
- review package 생성
- listening input guard
- residual risk 문서화

## 최근 개선 흐름

- broader repaired sampling repeatability: strict `40 / 40`, grammar-valid `40 / 40`
- final handoff audit: MIDI/WAV `8 / 8`, missing `0`, checksum mismatch `0`
- objective quality rubric baseline: pass/fail `3 / 5`
- dead-air density repair: dead-air avg `0.6583 -> 0.5806`, high count `2 -> 0`
- phrase direction repair: direction avg `0.4912 -> 0.6016`, direction low count `4 -> 0`
- tension color repair: tension avg `0.1979 -> 0.2431`, low count `4 -> 2`
- residual tension feasibility decision: tension repeat feasible `false`
- rhythm syncopation repair: syncopation avg `0.7951 -> 0.8160`, low count `1 -> 0`
- residual-aware final review package: candidate `8`, MIDI/WAV `8 / 8`, proxy pass/fail `6 / 2`
- residual-aware listening input guard: preference fill `false`, listening review completed `false`
- residual-aware MVP handoff freeze: local artifacts verified `true`, checksum mismatch `0`
- residual-aware listening review pending: pending candidate `8`, quality claim `false`
- residual-aware completion audit: technical MVP complete `true`, local review ready `true`
- residual-aware final status sync: synced `true`, next `listening_review_input_wait`
- residual-aware listening review input wait: quality claim blocked `true`
- bebop language best-of strict consonance listen-first package: source package `107`, pool `1519`, selection pool `64`, selected `16`, score `0.1905`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.4219`, offbeat resolution `0.9256`, unresolved offbeat non-chord `0.0313`, altered offbeat `0.1797`, two-note cycle `0.0092`, interval trigram repeat `0.0369`, max gate penalty `0.0000`, listen-first mode `consonance`
- bebop language strict-listen top4 bar-repair package: source package `119`, pool `1711`, selection pool `16`, selected `4`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.3984`, offbeat resolution `0.9615`, unresolved offbeat non-chord `0.0156`, altered offbeat `0.1875`, two-note cycle `0.0000`, interval trigram repeat `0.0000`, bar pitch-class similarity `0.6012`, max gate penalty `0.0000`, major ii-V turnaround bar similarity `0.8000 -> 0.5000`
- bebop language strict-listen top4 enclosure-repair package: source package `119`, pool `1711`, selection pool `16`, selected `4`, strong-beat chord-tone `1.0000`, enclosure proxy `0.3281`, offbeat non-chord `0.3750`, offbeat resolution `0.9583`, unresolved offbeat non-chord `0.0156`, altered offbeat `0.1563`, two-note cycle `0.0000`, interval trigram repeat `0.0082`, bar pitch-class similarity `0.6429`, max gate penalty `0.0000`
- bebop language strict-listen top4 enclosure context-strong package: source package `119`, pool `1711`, selection pool `16`, selected `4`, context bass velocity boost `6`, context comp velocity boost `10`, WAV RMS range `736.76-786.69`, max gate penalty `0.0000`, quality claim `false`
- bebop language strict-listen top4 post-repair-select package: source package `119`, pool `1711`, selection pool `16`, selected `4`, select-after-repair `true`, strong-beat chord-tone `1.0000`, enclosure proxy `0.3125`, offbeat non-chord `0.3750`, offbeat resolution `0.9792`, unresolved offbeat non-chord `0.0078`, altered offbeat `0.1406`, two-note cycle `0.0000`, interval trigram repeat `0.0082`, bar pitch-class similarity `0.6429`, max gate penalty `0.0000`
- bebop language strict-listen top4 unresolved-repair package: source package `119`, pool `1711`, selection pool `16`, selected `4`, select-after-repair `true`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.3750`, offbeat resolution `1.0000`, unresolved offbeat non-chord `0.0000`, enclosure proxy `0.3281`, altered offbeat `0.1563`, two-note cycle `0.0000`, interval trigram repeat `0.0082`, max gate penalty `0.0000`
- bebop language strict-listen top4 large-leap-repair package: source package `119`, pool `1711`, selection pool `16`, selected `4`, select-after-repair `true`, strong-beat chord-tone `1.0000`, offbeat resolution `1.0000`, unresolved offbeat non-chord `0.0000`, large leap `0.1151 -> 0.0595`, enclosure proxy `0.3203`, two-note cycle `0.0000`, max gate penalty `0.0000`
- bebop language strict-listen top4 bebop-selection-profile package: source package `125`, pool `1807`, selection pool `18`, selected `4`, select-after-repair `true`, selection profile `bebop_language`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.3906`, offbeat resolution `1.0000`, unresolved offbeat non-chord `0.0000`, large leap `0.0476`, enclosure proxy `0.3516`, two-note cycle `0.0000`, interval trigram repeat `0.0123`, bar pitch-class similarity `0.5774`, max gate penalty `0.0000`
- bebop language strict-listen top4 adjacent-repeat-final-repair package: source package `125`, pool `1807`, selection pool `18`, selected `4`, select-after-repair `true`, selection profile `bebop_language`, adjacent repeat `0.0119 -> 0.0040`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.3906`, offbeat resolution `1.0000`, unresolved offbeat non-chord `0.0000`, large leap `0.0476`, enclosure proxy `0.3438`, two-note cycle `0.0000`, interval trigram repeat `0.0123`, bar pitch-class similarity `0.5774`, max gate penalty `0.0000`
- bebop language strict-listen top4 residual-adjacent-search-repair package: source package `125`, pool `1807`, selection pool `18`, selected `4`, select-after-repair `true`, selection profile `bebop_language`, adjacent repeat `0.0040 -> 0.0000`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.3906`, offbeat resolution `1.0000`, unresolved offbeat non-chord `0.0000`, large leap `0.0516`, enclosure proxy `0.3516`, two-note cycle `0.0000`, interval trigram repeat `0.0123`, bar pitch-class similarity `0.5774`, max gate penalty `0.0000`
- bebop language strict-listen top4 rhythm-articulation-repair package: source package `125`, pool `1807`, selection pool `18`, selected `4`, articulation accepted `4 / 4`, duration template repeat `0.8750 -> 0.3750`, most common duration `0.5000 -> 0.2031`, duration bucket `3 -> 16`, velocity count `4 -> 17`, strong-beat chord-tone `1.0000`, offbeat resolution `1.0000`, unresolved offbeat non-chord `0.0000`, adjacent repeat `0.0000`, max gate penalty `0.0000`
- bebop language strict-listen top4 stepwise-chromatic-selection package: source package `125`, pool `1807`, selection pool `18`, selected `4`, selection profile `bebop_stepwise_chromatic`, step motion `0.3849 -> 0.4048`, chromatic step `0.1905 -> 0.2183`, third/fourth motion `0.5635 -> 0.5397`, large leap `0.0516 -> 0.0556`, bar pitch-class similarity `0.5774 -> 0.6369`, max gate penalty `0.0000`, offbeat resolution `1.0000`, unresolved offbeat non-chord `0.0000`, adjacent repeat `0.0000`
- bebop language strict-listen top4 stepwise-large-leap-guard package: source package `125`, pool `1807`, selection pool `18`, selected `4`, selection profile `bebop_stepwise_chromatic`, large-leap repair iterations `8`, step motion `0.3849 -> 0.4087`, chromatic step `0.1905 -> 0.2143`, third/fourth motion `0.5635 -> 0.5476`, large leap `0.0556 -> 0.0437`, bar pitch-class similarity `0.6369 -> 0.6131`, max gate penalty `0.0000`, offbeat resolution `1.0000`, unresolved offbeat non-chord `0.0000`, adjacent repeat `0.0000`
- bebop language strict-listen top8 stepwise-frontier review package: source package `125`, pool `1807`, selection pool `18`, selected `8`, max-per-case `2`, selection profile `bebop_stepwise_chromatic`, large-leap repair iterations `8`, max gate penalty `0.0000`, offbeat resolution `1.0000`, unresolved offbeat non-chord `0.0000`, step motion `0.3968`, chromatic step `0.2202`, third/fourth motion `0.5575`, large leap `0.0456`, adjacent repeat `0.0020`, bar pitch-class similarity `0.6458`, duration template repeat `0.3750`, most common duration `0.2031`, quality claim `false`
- bebop language strict-listen top8 safety-gate review package: source package `125`, pool `1807`, selection pool `11`, selected `8`, max-per-case `3`, max adjacent repeat `0.0000`, max bar pitch-class similarity `0.7000`, adjacent repeat `0.0020 -> 0.0000`, bar pitch-class similarity `0.6458 -> 0.6131`, max gate penalty `0.0000`, offbeat resolution `1.0000`, unresolved offbeat non-chord `0.0000`, step motion `0.3968 -> 0.3790`, chromatic step `0.2202 -> 0.2044`, large leap `0.0456 -> 0.0595`, quality claim `false`
- bebop language safety-gate motion feasibility sweep: repaired pool `18`, safety baseline `11`, selectable max-per-case `2 / 3` = `7 / 9`, selected target `8`, strict motion config feasible `0`, best strict config selectable max-per-case `3` = `6`, next boundary `selection_score_reweight_or_pool_expansion`
- bebop language safety motion pool expansion package: source generated `4800`, source selected `64`, best-of pool `1871`, selection pool `12`, selected `8`, max gate penalty `0.0000`, adjacent repeat `0.0000`, offbeat resolution `1.0000`, unresolved `0.0000`, large leap `0.0595 -> 0.0516`, enclosure proxy `0.2969 -> 0.3086`, step motion `0.3790 -> 0.3770`, chromatic step `0.2044 -> 0.2004`, bar pitch-class similarity `0.6131 -> 0.6256`, representative replacement `false`
- bebop language strict-listen top8 motion-balance repair package: pool `1871`, selection pool `19`, selected `8`, changed candidates `7 / 8`, pitch repair steps `26`, step motion `0.3770 -> 0.4226`, chromatic step `0.2004 -> 0.2440`, large leap `0.0516 -> 0.0437`, enclosure proxy `0.3086 -> 0.3125`, max gate penalty `0.0000`, adjacent repeat `0.0000`, offbeat resolution `1.0000`, unresolved `0.0000`, bar pitch-class similarity `0.6256 -> 0.6339`, quality claim `false`
- bebop language motion-balance review handoff: review ready `true`, selected `8`, solo/context WAV `8 / 8`, listen-first audio pair `4`, note review validated `true`, baseline improved `4`, tradeoff watch `2`, unchanged guard `4`, quality claim `false`
- bebop language motion-balance guard feasibility sweep: repaired pool `21`, safety baseline `19`, selectable max-per-case `2 / 3` = `8 / 11`, feasible guard/motion config `20`, min feasible offbeat max `0.40625`, min feasible bar similarity max `0.675`, stricter offbeat feasible `false`, stricter bar similarity feasible `true`, next `motion_balance_guard_tightening_candidate_package`
- bebop language targeted-low-offbeat package: generated candidate pool `1999`, selected `8`, max gate penalty `0.0000`, offbeat non-chord `0.4063 -> 0.3711`, offbeat resolution `1.0000`, unresolved `0.0000`, step motion `0.4226 -> 0.4345`, chromatic step `0.2440 -> 0.2401`, large leap `0.0437 -> 0.0437`, enclosure proxy `0.3125 -> 0.3242`, bar pitch-class similarity `0.6339 -> 0.6429`, motion-balance changed candidates `8 / 8`, pitch repair steps `36`, quality claim `false`
- bebop language targeted-low-offbeat case-balanced package: candidate pool `1999`, selected `8`, max-per-case `2`, selected case counts `3/3/1/1 -> 2/2/2/2`, offbeat non-chord `0.3711 -> 0.3711`, offbeat resolution `1.0000`, unresolved `0.0000`, max gate penalty `0.0000`, adjacent repeat `0.0000`, step motion `0.4345 -> 0.4325`, chromatic step `0.2401 -> 0.2321`, large leap `0.0437 -> 0.0456`, enclosure proxy `0.3242 -> 0.3164`, bar pitch-class similarity `0.6429 -> 0.6429`, quality claim `false`
- bebop language case-balanced motion-tight package: candidate pool `1999`, selected `8`, selected case counts `2/2/2/2`, offbeat non-chord `0.3711 -> 0.3672`, offbeat resolution `1.0000`, unresolved `0.0000`, max gate penalty `0.0000`, adjacent repeat `0.0000`, step motion `0.4325 -> 0.4583`, chromatic step `0.2321 -> 0.2599`, large leap `0.0456 -> 0.0337`, enclosure proxy `0.3164 -> 0.3242`, bar pitch-class similarity `0.6429 -> 0.6345`, dominant altered offbeat `0.1719 -> 0.0703`, motion-balance pitch repair steps `60`, quality claim `false`
- bebop language interval-repeat guard audit: strict cap `0.0125` failed with selected count shortage under max-per-case `2`, feasible cap `0.0164`, selected `8`, selected case counts `2/2/2/2`, interval trigram repeat avg `0.0123`, max selected interval trigram repeat `0.0164`, gate/resolution/unresolved/motion metrics preserved, quality claim `false`
- bebop language interval-repeat source expansion: source generated/selected `4800 / 128`, strict interval `<=0.0125` rows `21`, strict interval + offbeat `<=0.390625` rows `2`, strict interval + resolution `1.0000` + unresolved `<=0.1000` rows `0`, strict interval best-of selected `8` failed under max-per-case `2` and `3`, accepted frontier unchanged
- bebop language best-of with-sweeps package: source package `91`, pool `1263`, selected `16`, score `0.2143`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.4277`, offbeat resolution `0.9177`, unresolved offbeat non-chord `0.0352`, altered offbeat `0.1836`, two-note cycle `0.0082`, max gate penalty `0.0639`
- bebop language best-of balanced package: source package `33`, pool `335`, selected `16`, score `0.2728`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.4414`, offbeat resolution `0.8983`, unresolved offbeat non-chord `0.0449`, altered offbeat `0.1602`, two-note cycle `0.0092`, max-per-case `4`
- bebop language altered-color balanced package: generated `4000`, selected `16`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.4512`, offbeat resolution `0.8914`, unresolved offbeat non-chord `0.0488`, unique pitch avg `14.8125`, 3rd/4th motion `0.4831`, large leap `0.0863`, altered offbeat `0.1445`, bar pitch-class similarity `0.7027`, half-repeat `0.0000`
- bebop language bar-similarity package: generated `1440`, selected `16`, strong-beat chord-tone `1.0000`, offbeat non-chord `0.4746`, offbeat resolution `0.8891`, unresolved offbeat non-chord `0.0527`, bar pitch-class similarity `0.7280`, half-repeat `0.0000`
- bebop language data-contour sweep: config `18`, best `config_01`, strong-beat chord-tone `1.0000`, offbeat resolution `0.8598`, unresolved offbeat non-chord `0.0625`, unique pitch avg `13.9375`, 3rd/4th motion `0.4931`, bar pitch-class similarity `0.7449`, half-repeat `0.0000`

## 최신 산출물

- best-of strict consonance 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_v8_interval_repeat_tight/listen_first_by_progression/`
- best-of strict consonance 전체 WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_v8_interval_repeat_tight/audio_with_context/`
- best-of strict consonance package report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_v8_interval_repeat_tight/bebop_language_best_of_package.md`
- best-of strict consonance note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_v8_listen_first_note_review/bebop_language_note_review.md`
- strict-listen top4 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_strict_listen/listen_first_by_progression/`
- strict-listen top4 note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_strict_listen_note_review/bebop_language_note_review.md`
- strict-listen top4 walking-context 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_walking_context/listen_first_by_progression/`
- strict-listen top4 walking-context note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_walking_context_note_review/bebop_language_note_review.md`
- strict-listen top4 bar-repair 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_bar_repair_probe/listen_first_by_progression/`
- strict-listen top4 bar-repair note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_bar_repair_note_review/bebop_language_note_review.md`
- strict-listen top4 enclosure-repair 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_enclosure_repair_probe/listen_first_by_progression/`
- strict-listen top4 enclosure-repair note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_enclosure_repair_note_review/bebop_language_note_review.md`
- strict-listen top4 enclosure context-strong 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_enclosure_context_strong_probe/listen_first_by_progression/`
- strict-listen top4 enclosure context-strong note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_enclosure_context_strong_note_review/bebop_language_note_review.md`
- strict-listen top4 post-repair-select 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_post_repair_select_probe/listen_first_by_progression/`
- strict-listen top4 post-repair-select note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_post_repair_select_note_review/bebop_language_note_review.md`
- strict-listen top4 unresolved-repair 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_unresolved_repair_probe/listen_first_by_progression/`
- strict-listen top4 unresolved-repair note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_unresolved_repair_note_review/bebop_language_note_review.md`
- strict-listen top4 large-leap-repair 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_large_leap_repair_probe/listen_first_by_progression/`
- strict-listen top4 large-leap-repair note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_large_leap_repair_note_review/bebop_language_note_review.md`
- strict-listen top4 bebop-selection-profile 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_bebop_selection_profile_probe/listen_first_by_progression/`
- strict-listen top4 bebop-selection-profile note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_bebop_selection_profile_note_review/bebop_language_note_review.md`
- strict-listen top4 adjacent-repeat-final-repair 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_adjacent_repeat_final_repair_probe/listen_first_by_progression/`
- strict-listen top4 adjacent-repeat-final-repair note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_adjacent_repeat_final_repair_note_review/bebop_language_note_review.md`
- strict-listen top4 residual-adjacent-search-repair 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_residual_adjacent_search_repair_probe/listen_first_by_progression/`
- strict-listen top4 residual-adjacent-search-repair note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_residual_adjacent_search_repair_note_review/bebop_language_note_review.md`
- strict-listen top4 rhythm-articulation-repair 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_rhythm_articulation_repair_probe/listen_first_by_progression/`
- strict-listen top4 rhythm-articulation-repair note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_rhythm_articulation_repair_note_review/bebop_language_note_review.md`
- strict-listen top4 stepwise-chromatic-selection 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_stepwise_chromatic_selection_probe/listen_first_by_progression/`
- strict-listen top4 stepwise-chromatic-selection note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_stepwise_chromatic_selection_note_review/bebop_language_note_review.md`
- strict-listen top4 stepwise-large-leap-guard 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_stepwise_large_leap_guard_probe/listen_first_by_progression/`
- strict-listen top4 stepwise-large-leap-guard note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top4_stepwise_large_leap_guard_note_review/bebop_language_note_review.md`
- strict-listen top8 stepwise-frontier 전체 context WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_stepwise_frontier_probe/audio_with_context/`
- strict-listen top8 stepwise-frontier 전체 solo WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_stepwise_frontier_probe/audio/`
- strict-listen top8 stepwise-frontier package report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_stepwise_frontier_probe/bebop_language_best_of_package.md`
- strict-listen top8 stepwise-frontier all-selected note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_stepwise_frontier_all_selected_note_review/bebop_language_note_review.md`
- strict-listen top8 safety-gate 전체 context WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_safety_gate_probe/audio_with_context/`
- strict-listen top8 safety-gate 전체 solo WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_safety_gate_probe/audio/`
- strict-listen top8 safety-gate package report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_safety_gate_probe/bebop_language_best_of_package.md`
- strict-listen top8 safety-gate all-selected note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_safety_gate_all_selected_note_review/bebop_language_note_review.md`
- safety-gate motion feasibility sweep report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of_feasibility/manual_2026_06_13_bebop_language_top8_safety_motion_feasibility_sweep/bebop_language_safety_gate_feasibility_sweep.md`
- safety motion pool expansion source package: `outputs/stage_b_midi_to_solo_bebop_language_package/manual_2026_06_13_bebop_language_v23_safety_motion_pool_expansion/bebop_language_package.md`
- strict-listen top8 safety-pool-expansion 전체 context WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_safety_pool_expansion_probe/audio_with_context/`
- strict-listen top8 safety-pool-expansion package report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_safety_pool_expansion_probe/bebop_language_best_of_package.md`
- strict-listen top8 safety-pool-expansion all-selected note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_safety_pool_expansion_all_selected_note_review/bebop_language_note_review.md`
- strict-listen top8 motion-balance 전체 context WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_motion_balance_probe/audio_with_context/`
- strict-listen top8 motion-balance 전체 solo WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_motion_balance_probe/audio/`
- strict-listen top8 motion-balance package report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_motion_balance_probe/bebop_language_best_of_package.md`
- strict-listen top8 motion-balance all-selected note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_motion_balance_all_selected_note_review/bebop_language_note_review.md`
- strict-listen top8 motion-balance review handoff: `outputs/stage_b_midi_to_solo_bebop_language_review_handoff/manual_2026_06_13_bebop_language_motion_balance_review_handoff/bebop_language_review_handoff.md`
- motion-balance guard feasibility sweep report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of_feasibility/manual_2026_06_13_bebop_language_motion_balance_guard_feasibility_sweep/bebop_language_safety_gate_feasibility_sweep.md`
- targeted-low-offbeat source package: `outputs/stage_b_midi_to_solo_bebop_language_package/manual_2026_06_13_bebop_language_v24_targeted_low_offbeat_pool/bebop_language_package.md`
- targeted-low-offbeat top8 전체 context WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_dedup_probe/audio_with_context/`
- targeted-low-offbeat top8 전체 solo WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_dedup_probe/audio/`
- targeted-low-offbeat top8 package report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_dedup_probe/bebop_language_best_of_package.md`
- targeted-low-offbeat top8 all-selected note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_targeted_low_offbeat_dedup_all_selected_note_review/bebop_language_note_review.md`
- targeted-low-offbeat review handoff: `outputs/stage_b_midi_to_solo_bebop_language_review_handoff/manual_2026_06_13_bebop_language_targeted_low_offbeat_review_handoff/bebop_language_review_handoff.md`
- targeted-low-offbeat case-balanced top8 전체 context WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_case_balanced_probe/audio_with_context/`
- targeted-low-offbeat case-balanced top8 package report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_case_balanced_probe/bebop_language_best_of_package.md`
- targeted-low-offbeat case-balanced top8 all-selected note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_targeted_low_offbeat_case_balanced_all_selected_note_review/bebop_language_note_review.md`
- targeted-low-offbeat case-balanced review handoff: `outputs/stage_b_midi_to_solo_bebop_language_review_handoff/manual_2026_06_13_bebop_language_targeted_low_offbeat_case_balanced_review_handoff/bebop_language_review_handoff.md`
- case-balanced motion-tight top8 전체 context WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_tight_probe/audio_with_context/`
- case-balanced motion-tight top8 package report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_tight_probe/bebop_language_best_of_package.md`
- case-balanced motion-tight top8 all-selected note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_case_balanced_motion_tight_all_selected_note_review/bebop_language_note_review.md`
- case-balanced motion-tight review handoff: `outputs/stage_b_midi_to_solo_bebop_language_review_handoff/manual_2026_06_13_bebop_language_case_balanced_motion_tight_review_handoff/bebop_language_review_handoff.md`
- case-balanced motion interval-guard top8 전체 context WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe/audio_with_context/`
- case-balanced motion interval-guard top8 package report: `outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe/bebop_language_best_of_package.md`
- case-balanced motion interval-guard top8 all-selected note review: `outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_case_balanced_motion_interval_guard_all_selected_note_review/bebop_language_note_review.md`
- case-balanced motion interval-guard review handoff: `outputs/stage_b_midi_to_solo_bebop_language_review_handoff/manual_2026_06_13_bebop_language_case_balanced_motion_interval_guard_review_handoff/bebop_language_review_handoff.md`
- interval-repeat source expansion package report: `outputs/stage_b_midi_to_solo_bebop_language_package/manual_2026_06_13_bebop_language_v25_interval_repeat_source_expansion/bebop_language_package.md`
- altered-color balanced 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/manual_2026_06_13_bebop_language_v22_altered_color_balanced/listen_first_by_progression/`
- altered-color balanced 전체 WAV: `outputs/stage_b_midi_to_solo_bebop_language_package/manual_2026_06_13_bebop_language_v22_altered_color_balanced/audio_with_context/`
- altered-color balanced package report: `outputs/stage_b_midi_to_solo_bebop_language_package/manual_2026_06_13_bebop_language_v22_altered_color_balanced/bebop_language_package.md`
- data-contour focused sweep 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/best_listen_first_by_progression/`
- data-contour focused sweep report: `outputs/stage_b_midi_to_solo_bebop_language_package/parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/bebop_language_parameter_sweep.md`
- previous bar-similarity 대표 청취: `outputs/stage_b_midi_to_solo_bebop_language_package/manual_2026_06_13_bebop_language_v14_bar_similarity_rank/listen_first_by_progression/`
- final review package: `outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_review/issue_1388_residual_aware_final_review_package/residual_aware_final_review_package.md`
- review input template: `outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_review/issue_1388_residual_aware_final_review_package/residual_aware_review_input_template.json`
- MIDI 후보: `outputs/music_transformer_finetune_mvp/solo_yield_rhythm_syncopation_balance_repair/issue_1384_rhythm_syncopation_balance_repair_package/midi/`
- WAV 후보: `outputs/music_transformer_finetune_mvp/solo_yield_rhythm_syncopation_balance_repair/issue_1384_rhythm_syncopation_balance_repair_package/audio/`
- final review doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_FINAL_REVIEW_PACKAGE_2026-06-11.md`
- listening input guard doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_INPUT_GUARD_2026-06-11.md`
- MVP handoff freeze doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_MVP_HANDOFF_FREEZE_2026-06-11.md`
- listening review pending doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_REVIEW_PENDING_2026-06-11.md`
- completion audit doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_COMPLETION_AUDIT_2026-06-11.md`
- final status sync doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_FINAL_STATUS_SYNC_2026-06-11.md`
- listening review input wait doc: `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_REVIEW_INPUT_WAIT_2026-06-11.md`

## 재현 명령

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py \
  --run_id manual_2026_06_13_bebop_language_best_of_top4_stepwise_large_leap_guard_probe \
  --package_globs 'manual_2026_06_13_bebop_language_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v10_interval_repeat_rank/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v11_large_leap_pool/config_*/bebop_language_package.json' \
  --selected_count 4 \
  --max_per_case 1 \
  --bars 8 \
  --bpm 124 \
  --target_chord_tone_ratio 0.78 \
  --target_offbeat_non_chord_ratio 0.38 \
  --max_gate_penalty 0 \
  --max_offbeat_non_chord_ratio 0.40625 \
  --max_unresolved_offbeat_non_chord_ratio 0.03125 \
  --max_dominant_altered_offbeat_ratio 0.25 \
  --listen_first_mode consonance \
  --repair_bar_similarity \
  --repair_bar_similarity_iterations 4 \
  --repair_enclosure_density \
  --repair_enclosure_density_iterations 8 \
  --repair_unresolved_offbeat \
  --repair_unresolved_offbeat_iterations 4 \
  --repair_adjacent_repeats \
  --repair_adjacent_repeats_iterations 4 \
  --repair_large_leaps \
  --repair_large_leaps_iterations 8 \
  --repair_rhythm_articulation \
  --min_large_leap_repair_enclosure_proxy_ratio 0.28125 \
  --max_enclosure_repair_offbeat_non_chord_ratio 0.421875 \
  --context_bass_velocity_boost 6 \
  --context_comp_velocity_boost 10 \
  --select_after_repair \
  --selection_profile bebop_stepwise_chromatic
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py \
  --run_id manual_2026_06_13_bebop_language_top4_stepwise_large_leap_guard_note_review \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top4_stepwise_large_leap_guard_probe/bebop_language_best_of_package.json \
  --max_notes 32
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py \
  --run_id manual_2026_06_13_bebop_language_best_of_top8_safety_gate_probe \
  --package_globs 'manual_2026_06_13_bebop_language_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v10_interval_repeat_rank/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v11_large_leap_pool/config_*/bebop_language_package.json' \
  --selected_count 8 \
  --max_per_case 3 \
  --bars 8 \
  --bpm 124 \
  --target_chord_tone_ratio 0.78 \
  --target_offbeat_non_chord_ratio 0.38 \
  --max_gate_penalty 0 \
  --max_offbeat_non_chord_ratio 0.40625 \
  --max_unresolved_offbeat_non_chord_ratio 0.03125 \
  --max_dominant_altered_offbeat_ratio 0.25 \
  --max_adjacent_repeat_ratio 0 \
  --max_bar_pitch_class_jaccard 0.70 \
  --listen_first_mode consonance \
  --repair_bar_similarity \
  --repair_bar_similarity_iterations 4 \
  --repair_enclosure_density \
  --repair_enclosure_density_iterations 8 \
  --repair_unresolved_offbeat \
  --repair_unresolved_offbeat_iterations 4 \
  --repair_adjacent_repeats \
  --repair_adjacent_repeats_iterations 4 \
  --repair_large_leaps \
  --repair_large_leaps_iterations 8 \
  --repair_rhythm_articulation \
  --min_large_leap_repair_enclosure_proxy_ratio 0.28125 \
  --max_enclosure_repair_offbeat_non_chord_ratio 0.421875 \
  --context_bass_velocity_boost 6 \
  --context_comp_velocity_boost 10 \
  --select_after_repair \
  --selection_profile bebop_stepwise_chromatic
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py \
  --run_id manual_2026_06_13_bebop_language_top8_safety_gate_all_selected_note_review \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_safety_gate_probe/bebop_language_best_of_package.json \
  --all_candidates \
  --max_notes 32
```

```bash
.venv/bin/python scripts/run_stage_b_midi_to_solo_bebop_language_safety_gate_feasibility_sweep.py \
  --run_id manual_2026_06_13_bebop_language_top8_safety_motion_feasibility_sweep
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_package.py \
  --run_id manual_2026_06_13_bebop_language_v23_safety_motion_pool_expansion \
  --variants_per_progression 1200 \
  --selected_count 64 \
  --bars 8 \
  --bpm 124 \
  --seed_base 5010000 \
  --non_chord_probability 0.28 \
  --target_chord_tone_ratio 0.78 \
  --target_offbeat_non_chord_ratio 0.38
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py \
  --run_id manual_2026_06_13_bebop_language_best_of_top8_motion_balance_probe \
  --package_globs 'manual_2026_06_13_bebop_language_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v10_interval_repeat_rank/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v11_large_leap_pool/config_*/bebop_language_package.json' \
  --selected_count 8 \
  --max_per_case 3 \
  --bars 8 \
  --bpm 124 \
  --target_chord_tone_ratio 0.78 \
  --target_offbeat_non_chord_ratio 0.38 \
  --max_gate_penalty 0 \
  --max_offbeat_non_chord_ratio 0.40625 \
  --max_unresolved_offbeat_non_chord_ratio 0.03125 \
  --max_dominant_altered_offbeat_ratio 0.25 \
  --max_adjacent_repeat_ratio 0 \
  --max_bar_pitch_class_jaccard 0.70 \
  --listen_first_mode consonance \
  --repair_bar_similarity \
  --repair_bar_similarity_iterations 4 \
  --repair_enclosure_density \
  --repair_enclosure_density_iterations 8 \
  --repair_unresolved_offbeat \
  --repair_unresolved_offbeat_iterations 4 \
  --repair_adjacent_repeats \
  --repair_adjacent_repeats_iterations 4 \
  --repair_large_leaps \
  --repair_large_leaps_iterations 8 \
  --repair_motion_balance \
  --repair_motion_balance_iterations 12 \
  --target_min_step_motion_ratio 0.40 \
  --target_min_chromatic_step_ratio 0.22 \
  --target_max_large_leap_ratio 0.055 \
  --max_motion_balance_bar_pitch_class_jaccard 0.70 \
  --repair_rhythm_articulation \
  --min_large_leap_repair_enclosure_proxy_ratio 0.28125 \
  --max_enclosure_repair_offbeat_non_chord_ratio 0.421875 \
  --context_bass_velocity_boost 6 \
  --context_comp_velocity_boost 10 \
  --select_after_repair \
  --selection_profile bebop_stepwise_chromatic
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py \
  --run_id manual_2026_06_13_bebop_language_top8_motion_balance_all_selected_note_review \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_motion_balance_probe/bebop_language_best_of_package.json \
  --all_candidates \
  --max_notes 32
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_review_handoff.py \
  --run_id manual_2026_06_13_bebop_language_motion_balance_review_handoff \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_motion_balance_probe/bebop_language_best_of_package.json \
  --baseline_package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_safety_pool_expansion_probe/bebop_language_best_of_package.json \
  --note_review outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_motion_balance_all_selected_note_review/bebop_language_note_review.json \
  --expected_candidate_count 8
```

```bash
.venv/bin/python scripts/run_stage_b_midi_to_solo_bebop_language_safety_gate_feasibility_sweep.py \
  --run_id manual_2026_06_13_bebop_language_motion_balance_guard_feasibility_sweep \
  --selected_count 8 \
  --max_per_case_values 2,3 \
  --max_offbeat_non_chord_ratio 0.40625 \
  --max_offbeat_non_chord_ratios 0.3828125,0.390625,0.3984375,0.40625 \
  --max_bar_pitch_class_jaccard 0.70 \
  --max_bar_pitch_class_jaccards 0.625,0.65,0.675,0.70 \
  --min_step_motion_ratios 0.38,0.40,0.42 \
  --min_chromatic_step_ratios 0.20,0.22,0.24 \
  --max_large_leap_ratios 0.045,0.055,0.065 \
  --repair_motion_balance \
  --repair_motion_balance_iterations 12 \
  --target_min_step_motion_ratio 0.40 \
  --target_min_chromatic_step_ratio 0.22 \
  --target_max_large_leap_ratio 0.055 \
  --max_motion_balance_bar_pitch_class_jaccard 0.70
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_package.py \
  --run_id manual_2026_06_13_bebop_language_v24_targeted_low_offbeat_pool \
  --variants_per_progression 1000 \
  --selected_count 128 \
  --seed_base 6100000 \
  --non_chord_probability 0.08 \
  --target_chord_tone_ratio 0.82 \
  --target_offbeat_non_chord_ratio 0.30 \
  --bars 8 \
  --bpm 124
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py \
  --run_id manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_dedup_probe \
  --package_globs 'manual_2026_06_13_bebop_language_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v10_interval_repeat_rank/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v11_large_leap_pool/config_*/bebop_language_package.json' \
  --selected_count 8 \
  --max_per_case 3 \
  --bars 8 \
  --bpm 124 \
  --target_chord_tone_ratio 0.82 \
  --target_offbeat_non_chord_ratio 0.34 \
  --max_gate_penalty 1.0 \
  --max_offbeat_non_chord_ratio 0.390625 \
  --max_unresolved_offbeat_non_chord_ratio 0.10 \
  --max_dominant_altered_offbeat_ratio 0.25 \
  --max_adjacent_repeat_ratio 0 \
  --max_bar_pitch_class_jaccard 0.675 \
  --listen_first_mode consonance \
  --repair_bar_similarity \
  --repair_bar_similarity_iterations 4 \
  --repair_enclosure_density \
  --repair_enclosure_density_iterations 8 \
  --repair_unresolved_offbeat \
  --repair_unresolved_offbeat_iterations 8 \
  --repair_adjacent_repeats \
  --repair_adjacent_repeats_iterations 4 \
  --repair_large_leaps \
  --repair_large_leaps_iterations 8 \
  --repair_motion_balance \
  --repair_motion_balance_iterations 12 \
  --target_min_step_motion_ratio 0.40 \
  --target_min_chromatic_step_ratio 0.22 \
  --target_max_large_leap_ratio 0.055 \
  --max_motion_balance_bar_pitch_class_jaccard 0.675 \
  --repair_rhythm_articulation \
  --min_large_leap_repair_enclosure_proxy_ratio 0.28125 \
  --max_enclosure_repair_offbeat_non_chord_ratio 0.421875 \
  --context_bass_velocity_boost 6 \
  --context_comp_velocity_boost 10 \
  --select_after_repair \
  --selection_profile bebop_stepwise_chromatic
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py \
  --run_id manual_2026_06_13_bebop_language_top8_targeted_low_offbeat_dedup_all_selected_note_review \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_dedup_probe/bebop_language_best_of_package.json \
  --all_candidates \
  --max_notes 32
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_review_handoff.py \
  --run_id manual_2026_06_13_bebop_language_targeted_low_offbeat_review_handoff \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_dedup_probe/bebop_language_best_of_package.json \
  --baseline_package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_motion_balance_probe/bebop_language_best_of_package.json \
  --note_review outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_targeted_low_offbeat_dedup_all_selected_note_review/bebop_language_note_review.json \
  --expected_candidate_count 8
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py \
  --run_id manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_case_balanced_probe \
  --package_globs 'manual_2026_06_13_bebop_language_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v10_interval_repeat_rank/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v11_large_leap_pool/config_*/bebop_language_package.json' \
  --selected_count 8 \
  --max_per_case 2 \
  --bars 8 \
  --bpm 124 \
  --target_chord_tone_ratio 0.82 \
  --target_offbeat_non_chord_ratio 0.34 \
  --max_gate_penalty 1.0 \
  --max_offbeat_non_chord_ratio 0.390625 \
  --max_unresolved_offbeat_non_chord_ratio 0.10 \
  --max_dominant_altered_offbeat_ratio 0.25 \
  --max_adjacent_repeat_ratio 0 \
  --max_bar_pitch_class_jaccard 0.675 \
  --listen_first_mode consonance \
  --repair_bar_similarity \
  --repair_bar_similarity_iterations 4 \
  --repair_enclosure_density \
  --repair_enclosure_density_iterations 8 \
  --repair_unresolved_offbeat \
  --repair_unresolved_offbeat_iterations 8 \
  --repair_adjacent_repeats \
  --repair_adjacent_repeats_iterations 4 \
  --repair_large_leaps \
  --repair_large_leaps_iterations 8 \
  --repair_motion_balance \
  --repair_motion_balance_iterations 12 \
  --target_min_step_motion_ratio 0.40 \
  --target_min_chromatic_step_ratio 0.22 \
  --target_max_large_leap_ratio 0.055 \
  --max_motion_balance_bar_pitch_class_jaccard 0.675 \
  --repair_rhythm_articulation \
  --min_large_leap_repair_enclosure_proxy_ratio 0.28125 \
  --max_enclosure_repair_offbeat_non_chord_ratio 0.421875 \
  --context_bass_velocity_boost 6 \
  --context_comp_velocity_boost 10 \
  --select_after_repair \
  --selection_profile bebop_stepwise_chromatic
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py \
  --run_id manual_2026_06_13_bebop_language_top8_targeted_low_offbeat_case_balanced_all_selected_note_review \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_case_balanced_probe/bebop_language_best_of_package.json \
  --all_candidates \
  --max_notes 32
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_review_handoff.py \
  --run_id manual_2026_06_13_bebop_language_targeted_low_offbeat_case_balanced_review_handoff \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_case_balanced_probe/bebop_language_best_of_package.json \
  --baseline_package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_dedup_probe/bebop_language_best_of_package.json \
  --note_review outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_targeted_low_offbeat_case_balanced_all_selected_note_review/bebop_language_note_review.json \
  --expected_candidate_count 8
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py \
  --run_id manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_tight_probe \
  --package_globs 'manual_2026_06_13_bebop_language_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v10_interval_repeat_rank/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v11_large_leap_pool/config_*/bebop_language_package.json' \
  --selected_count 8 \
  --max_per_case 2 \
  --bars 8 \
  --bpm 124 \
  --target_chord_tone_ratio 0.82 \
  --target_offbeat_non_chord_ratio 0.34 \
  --max_gate_penalty 1.0 \
  --max_offbeat_non_chord_ratio 0.390625 \
  --max_unresolved_offbeat_non_chord_ratio 0.10 \
  --max_dominant_altered_offbeat_ratio 0.25 \
  --max_adjacent_repeat_ratio 0 \
  --max_bar_pitch_class_jaccard 0.675 \
  --listen_first_mode consonance \
  --repair_bar_similarity \
  --repair_bar_similarity_iterations 4 \
  --repair_enclosure_density \
  --repair_enclosure_density_iterations 8 \
  --repair_unresolved_offbeat \
  --repair_unresolved_offbeat_iterations 8 \
  --repair_adjacent_repeats \
  --repair_adjacent_repeats_iterations 4 \
  --repair_large_leaps \
  --repair_large_leaps_iterations 12 \
  --repair_motion_balance \
  --repair_motion_balance_iterations 24 \
  --target_min_step_motion_ratio 0.43 \
  --target_min_chromatic_step_ratio 0.24 \
  --target_max_large_leap_ratio 0.045 \
  --max_motion_balance_bar_pitch_class_jaccard 0.675 \
  --repair_rhythm_articulation \
  --min_large_leap_repair_enclosure_proxy_ratio 0.28125 \
  --max_enclosure_repair_offbeat_non_chord_ratio 0.421875 \
  --context_bass_velocity_boost 6 \
  --context_comp_velocity_boost 10 \
  --select_after_repair \
  --selection_profile bebop_stepwise_chromatic
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py \
  --run_id manual_2026_06_13_bebop_language_top8_case_balanced_motion_tight_all_selected_note_review \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_tight_probe/bebop_language_best_of_package.json \
  --all_candidates \
  --max_notes 32
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_review_handoff.py \
  --run_id manual_2026_06_13_bebop_language_case_balanced_motion_tight_review_handoff \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_tight_probe/bebop_language_best_of_package.json \
  --baseline_package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_targeted_low_offbeat_case_balanced_probe/bebop_language_best_of_package.json \
  --note_review outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_case_balanced_motion_tight_all_selected_note_review/bebop_language_note_review.json \
  --expected_candidate_count 8
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py \
  --run_id manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe \
  --package_globs 'manual_2026_06_13_bebop_language_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v10_interval_repeat_rank/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v11_large_leap_pool/config_*/bebop_language_package.json' \
  --selected_count 8 \
  --max_per_case 2 \
  --bars 8 \
  --bpm 124 \
  --target_chord_tone_ratio 0.82 \
  --target_offbeat_non_chord_ratio 0.34 \
  --max_gate_penalty 1.0 \
  --max_offbeat_non_chord_ratio 0.390625 \
  --max_unresolved_offbeat_non_chord_ratio 0.10 \
  --max_dominant_altered_offbeat_ratio 0.25 \
  --max_adjacent_repeat_ratio 0 \
  --max_interval_trigram_repeat_ratio 0.0164 \
  --max_bar_pitch_class_jaccard 0.675 \
  --listen_first_mode consonance \
  --repair_bar_similarity \
  --repair_bar_similarity_iterations 4 \
  --repair_enclosure_density \
  --repair_enclosure_density_iterations 8 \
  --repair_unresolved_offbeat \
  --repair_unresolved_offbeat_iterations 8 \
  --repair_adjacent_repeats \
  --repair_adjacent_repeats_iterations 4 \
  --repair_large_leaps \
  --repair_large_leaps_iterations 12 \
  --repair_motion_balance \
  --repair_motion_balance_iterations 24 \
  --target_min_step_motion_ratio 0.43 \
  --target_min_chromatic_step_ratio 0.24 \
  --target_max_large_leap_ratio 0.045 \
  --max_motion_balance_bar_pitch_class_jaccard 0.675 \
  --repair_rhythm_articulation \
  --min_large_leap_repair_enclosure_proxy_ratio 0.28125 \
  --max_enclosure_repair_offbeat_non_chord_ratio 0.421875 \
  --context_bass_velocity_boost 6 \
  --context_comp_velocity_boost 10 \
  --select_after_repair \
  --selection_profile bebop_stepwise_chromatic
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py \
  --run_id manual_2026_06_13_bebop_language_top8_case_balanced_motion_interval_guard_all_selected_note_review \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe/bebop_language_best_of_package.json \
  --all_candidates \
  --max_notes 32
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_review_handoff.py \
  --run_id manual_2026_06_13_bebop_language_case_balanced_motion_interval_guard_review_handoff \
  --package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_interval_guard_feasible_probe/bebop_language_best_of_package.json \
  --baseline_package outputs/stage_b_midi_to_solo_bebop_language_package/best_of/manual_2026_06_13_bebop_language_best_of_top8_case_balanced_motion_tight_probe/bebop_language_best_of_package.json \
  --note_review outputs/stage_b_midi_to_solo_bebop_language_note_review/manual_2026_06_13_bebop_language_top8_case_balanced_motion_interval_guard_all_selected_note_review/bebop_language_note_review.json \
  --expected_candidate_count 8
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_package.py \
  --run_id manual_2026_06_13_bebop_language_v25_interval_repeat_source_expansion \
  --variants_per_progression 1200 \
  --selected_count 128 \
  --seed_base 7200000 \
  --non_chord_probability 0.22 \
  --target_chord_tone_ratio 0.82 \
  --target_offbeat_non_chord_ratio 0.34 \
  --bars 8 \
  --bpm 124
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py \
  --run_id manual_2026_06_13_bebop_language_best_of_v8_interval_repeat_tight \
  --package_globs 'manual_2026_06_13_bebop_language_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v7_altered_balanced/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v8_v22_micro/config_*/bebop_language_package.json,parameter_sweep/manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance/config_*/bebop_language_package.json' \
  --selected_count 16 \
  --max_per_case 4 \
  --bars 8 \
  --bpm 124 \
  --target_chord_tone_ratio 0.78 \
  --target_offbeat_non_chord_ratio 0.38 \
  --max_gate_penalty 0 \
  --max_offbeat_non_chord_ratio 0.46875 \
  --max_unresolved_offbeat_non_chord_ratio 0.03125 \
  --max_dominant_altered_offbeat_ratio 0.25 \
  --listen_first_mode consonance
```

```bash
.venv/bin/python scripts/run_stage_b_midi_to_solo_bebop_language_parameter_sweep.py \
  --run_id manual_2026_06_13_bebop_language_param_sweep_v9_strict_consonance \
  --variants_per_progression 240 \
  --selected_count 16 \
  --bars 8 \
  --bpm 124 \
  --seed_base 4010000 \
  --non_chord_probabilities 0.22,0.24,0.26,0.28 \
  --target_chord_tone_ratios 0.78,0.80 \
  --target_offbeat_non_chord_ratios 0.36,0.38,0.40 \
  --max_configs 16
```

```bash
.venv/bin/python scripts/run_stage_b_midi_to_solo_bebop_language_parameter_sweep.py \
  --run_id manual_2026_06_13_bebop_language_param_sweep_v6_data_contour_resolution \
  --variants_per_progression 160 \
  --selected_count 16 \
  --bars 8 \
  --bpm 124 \
  --seed_base 1010000 \
  --non_chord_probabilities 0.28,0.32,0.34 \
  --target_chord_tone_ratios 0.76,0.78 \
  --target_offbeat_non_chord_ratios 0.38,0.40,0.44 \
  --max_configs 18
```

```bash
.venv/bin/python scripts/build_stage_b_midi_to_solo_bebop_language_package.py \
  --run_id manual_2026_06_13_bebop_language_v22_altered_color_balanced \
  --variants_per_progression 1000 \
  --selected_count 16 \
  --bars 8 \
  --bpm 124 \
  --seed_base 910000 \
  --non_chord_probability 0.28 \
  --target_chord_tone_ratio 0.78 \
  --target_offbeat_non_chord_ratio 0.38
```

```bash
.venv/bin/python scripts/build_music_transformer_solo_yield_residual_aware_final_review_package.py \
  --run_id issue_1388_residual_aware_final_review_package \
  --source_package outputs/music_transformer_finetune_mvp/solo_yield_rhythm_syncopation_balance_repair/issue_1384_rhythm_syncopation_balance_repair_package/rhythm_syncopation_balance_repair_package.json \
  --rubric_report outputs/music_transformer_finetune_mvp/solo_yield_objective_quality_rubric/issue_1386_rhythm_syncopation_repair_rubric_review/objective_quality_rubric_baseline.json \
  --feasibility_decision outputs/music_transformer_finetune_mvp/solo_yield_residual_tension_decision/issue_1382_residual_tension_feasibility_decision/residual_tension_target_decision.json \
  --min_candidate_count 8 \
  --require_residual_context \
  --require_no_quality_claim
```

```bash
.venv/bin/python scripts/guard_music_transformer_solo_yield_residual_aware_listening_input.py \
  --run_id issue_1390_residual_aware_listening_input_guard \
  --source_package outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_review/issue_1388_residual_aware_final_review_package/residual_aware_final_review_package.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_status_sync \
  --require_pending_input \
  --require_no_quality_claim
```

```bash
.venv/bin/python scripts/freeze_music_transformer_solo_yield_residual_aware_mvp_handoff.py \
  --run_id issue_1396_residual_aware_mvp_handoff_freeze \
  --final_review_package outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_review/issue_1388_residual_aware_final_review_package/residual_aware_final_review_package.json \
  --input_guard_report outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_listening_input_guard/issue_1390_residual_aware_listening_input_guard/residual_aware_listening_input_guard.json \
  --status_audit_report outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_status_audit/issue_1394_residual_aware_status_audit/residual_aware_status_audit.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_listening_review_pending \
  --require_local_artifacts_verified \
  --require_pending_input \
  --require_no_quality_claim
```

```bash
.venv/bin/python scripts/mark_music_transformer_solo_yield_residual_aware_listening_review_pending.py \
  --run_id issue_1398_residual_aware_listening_review_pending \
  --handoff_freeze_report outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_mvp_handoff_freeze/issue_1396_residual_aware_mvp_handoff_freeze/residual_aware_mvp_handoff_freeze.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_completion_audit \
  --require_pending_review \
  --require_no_quality_claim
```

```bash
.venv/bin/python scripts/audit_music_transformer_solo_yield_residual_aware_completion.py \
  --run_id issue_1400_residual_aware_completion_audit \
  --pending_report outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_listening_review_pending/issue_1398_residual_aware_listening_review_pending/residual_aware_listening_review_pending.json \
  --handoff_freeze_report outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_mvp_handoff_freeze/issue_1396_residual_aware_mvp_handoff_freeze/residual_aware_mvp_handoff_freeze.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_final_status_sync \
  --require_technical_complete \
  --require_no_quality_claim
```

```bash
.venv/bin/python scripts/sync_music_transformer_solo_yield_residual_aware_final_status.py \
  --run_id issue_1402_residual_aware_final_status_sync \
  --completion_audit outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_completion_audit/issue_1400_residual_aware_completion_audit/residual_aware_completion_audit.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_listening_review_input_wait \
  --require_final_status_synced \
  --require_no_quality_claim
```

```bash
.venv/bin/python scripts/mark_music_transformer_solo_yield_residual_aware_listening_review_input_wait.py \
  --run_id issue_1404_residual_aware_listening_review_input_wait \
  --final_status_sync outputs/music_transformer_finetune_mvp/solo_yield_residual_aware_final_status_sync/issue_1402_residual_aware_final_status_sync/residual_aware_final_status_sync.json \
  --expected_next_boundary music_transformer_solo_yield_residual_aware_user_listening_review_fill \
  --require_wait_recorded \
  --require_no_quality_claim
```

## 아닌 것

- 완성형 jazz solo quality claim 아님
- 특정 pianist style adaptation 완료 아님
- production-ready improviser 아님
- human listening preference 검증 완료 아님
- raw audio generation 프로젝트 아님
- bebop language package는 직접 모델 품질 claim이 아니라 청감 수율 개선용 chord-guided comparison package
- parameter sweep는 생성 후보의 음악적 완성도 판정이 아니라 offbeat 해소율과 미해소 비율 기준의 로컬 비교
- chromatic/enclosure/altered 지표는 비밥 장치 proxy이며 human listening preference 대체 아님
- half-repeat 지표는 단순 패턴 반복 proxy이며 실제 청감 반복감 판정 아님
- bar pitch-class similarity 지표는 마디 간 pitch-class 중복 proxy이며 실제 motif 반복감 판정 아님
- best-of package는 기존 후보 재랭킹/재렌더 산출물이며 새 모델 학습 결과 claim 아님

## 주요 파일

- `scripts/build_stage_b_midi_to_solo_bebop_language_best_of_package.py`
- `scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py`
- `scripts/run_stage_b_midi_to_solo_bebop_language_parameter_sweep.py`
- `scripts/build_stage_b_midi_to_solo_bebop_language_package.py`
- `scripts/build_music_transformer_solo_yield_rhythm_syncopation_balance_repair_package.py`
- `scripts/build_music_transformer_solo_yield_residual_aware_final_review_package.py`
- `scripts/guard_music_transformer_solo_yield_residual_aware_listening_input.py`
- `scripts/freeze_music_transformer_solo_yield_residual_aware_mvp_handoff.py`
- `scripts/mark_music_transformer_solo_yield_residual_aware_listening_review_pending.py`
- `scripts/audit_music_transformer_solo_yield_residual_aware_completion.py`
- `scripts/sync_music_transformer_solo_yield_residual_aware_final_status.py`
- `scripts/mark_music_transformer_solo_yield_residual_aware_listening_review_input_wait.py`
- `scripts/build_music_transformer_solo_yield_objective_quality_rubric_baseline.py`
- `scripts/decide_music_transformer_solo_yield_residual_tension_target.py`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_FINAL_REVIEW_PACKAGE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_INPUT_GUARD_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_MVP_HANDOFF_FREEZE_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_REVIEW_PENDING_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_COMPLETION_AUDIT_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_FINAL_STATUS_SYNC_2026-06-11.md`
- `docs/STAGE_B_MIDI_TO_SOLO_RESIDUAL_AWARE_LISTENING_REVIEW_INPUT_WAIT_2026-06-11.md`
