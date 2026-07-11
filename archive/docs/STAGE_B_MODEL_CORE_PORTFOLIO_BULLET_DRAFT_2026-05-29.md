# Stage B Model-Core Portfolio Bullet Draft

## Project Entry

**Jazz Piano MIDI 생성 검증 파이프라인 | Model-Core / ML Pipeline | 2026.05**

Python, PyTorch, pretty_midi, music21, mido, NumPy, Pandas

## Resume Bullets

- `.mid` 파일 생성만으로 성공을 판단하던 문제를 note count, unique pitch, dead-air, max active notes, repeated cell, chord/tension/outside ratio, final landing 기반 objective MIDI review gate로 전환해 one-note collapse, long sustain block, chord block 실패를 분리
- `NOTE_ON/OFF` 중심 Stage A representation에서 duration/phrase 제어가 어려운 문제를 `BAR`, `POSITION`, `CHORD_ROOT`, `CHORD_QUALITY`, `NOTE_PITCH`, `NOTE_DURATION`, `VELOCITY` 기반 Stage B tokenization으로 전환해 dataset -> training -> generation -> decode -> review pipeline 검증
- 6-file / 5-sample repeatability sweep에서 strict `12/15`, warning seed `0`을 기록하고, seed-level margin과 dead-air outlier를 sample 단위로 분리해 aggregate pass-rate로 가려지는 후보 안정성 리스크를 추적
- constrained adjacent repair 후 dead-air가 `0.5714`로 악화되던 문제를 duration/coverage fill로 보정해 qualified `2/4`, fill additions `6`, dead-air `0.5714 -> 0.2941`, focused unique pitch `15`, adjacent repeat `0` 후보 확보
- 사용자 청취에서 repeatability 후보가 outside-soloing-like로 판정된 문제를 pitch-role repair sweep과 policy-level repeatability consolidation으로 보정해 repaired source `2/2`, qualified variants `6/6`, policy support `3/3` 확인
- MIDI metric 우세와 human/audio preference가 섞이는 문제를 review input guard, MIDI evidence consolidation, final decision boundary로 분리해 human/audio preference claim `false`, broad model quality claim `false` 상태 유지

## Interview Summary

| 질문 | 답변 근거 |
|---|---|
| 무엇을 만들었는가 | symbolic MIDI 기반 jazz piano solo-line 생성 모델 개발을 위한 dataset, tokenization, generation, decode, objective review, focused review pipeline |
| 가장 큰 문제 | `.mid` 파일은 생성되지만 one-note collapse, long sustain, chord block, dead-air outlier로 solo-line 품질 판단 불가 |
| 핵심 해결 | Stage B duration-explicit tokenization, objective MIDI review gate, repeatability sweep, repair candidate selection, claim boundary 분리 |
| 수치 근거 | dataset readable `2777`, vocab `547`, 6-file 5-sample strict `12/15`, duration fill dead-air `0.5714 -> 0.2941`, outside-soloing repair policy support `3/3` |
| 현재 한계 | human/audio preference 미검증, broad trained-model quality 미검증, Brad style adaptation 미진행 |

## Claim Boundary

사용 가능:

- symbolic MIDI 생성 검증 파이프라인
- model-core validation pipeline
- objective MIDI review gate
- repeatability sweep 및 repair boundary
- outside-soloing repair objective support

사용 금지:

- 완성된 재즈 피아노 생성 모델
- Brad Mehldau style adaptation 완료
- production-ready improviser
- broad trained-model quality 검증 완료
- human/audio preference 검증 완료
