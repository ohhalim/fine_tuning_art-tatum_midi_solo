# Stage B Model-Core Portfolio Bullet Refresh

## Project Entry

**Jazz Piano MIDI 모델 검증 파이프라인 | Model-Core / ML Pipeline | 2026.05 - 2026.06**

Python, PyTorch, pretty_midi, music21, mido, NumPy, Pandas

## Resume Bullets

- `.mid` 파일 생성만으로 성공을 판단하던 문제를 note count, unique pitch, phrase coverage, dead-air, max active notes, long-note ratio 기반 objective MIDI review gate로 전환해 one-note collapse, long sustain block, chord block 실패를 분리
- Brad style adaptation 이전 generic base 검증 기준 부재를 generic/Brad manifest split과 leakage guard로 분리하고, generic train/val `2433/270`, Brad split `47/11/14`, full generic tokenized records `154136/21845` 생성
- Stage B duration-explicit representation으로 `BAR`, `POSITION`, `CHORD_ROOT`, `CHORD_QUALITY`, `NOTE_PITCH`, `NOTE_DURATION`, `VELOCITY` token 구조를 구성하고 max token id/vocab `544/547` guard로 window dataset 검증
- generic base scale checkpoint training smoke에서 selected train/val `128/32`, best validation loss `5.9031`, checkpoint `1`을 확보한 뒤 raw generation valid/strict/grammar `0/0/0` 실패를 note count `2-4` 문제로 분리
- raw checkpoint generation 실패 이후 density/coverage, duration/long-note, sustained coverage/dead-air repair boundary를 순차 검증해 constrained objective repair valid/strict/grammar `3/3/3`, dead-air/long-note `0/0` 달성
- 단일 seed objective gate 과장 위험을 seed `44/52/60` repeatability sweep으로 확장해 sample `9`, valid/strict/grammar `9/9/9`, failure reasons none을 확인하고 musical quality, human/audio preference, Brad style adaptation claim은 제외

## Short Resume Version

- Symbolic MIDI 기반 jazz piano solo-line 생성 model-core pipeline 구축: dataset audit, Stage B tokenization, training smoke, generation decode, objective MIDI review gate 연결
- generic base scale checkpoint raw generation valid/strict/grammar `0/0/0` 실패를 density, duration, dead-air repair boundary로 분리하고 constrained objective repair `3/3/3`까지 개선
- seed `44/52/60` repeatability sweep에서 objective MIDI gate `9/9/9` 통과를 확인하되 broad model quality, human/audio preference, Brad style adaptation claim은 제외

## Interview Summary

| 질문 | 답변 근거 |
|---|---|
| 무엇을 만들었는가 | symbolic MIDI 기반 jazz piano solo-line 생성 모델을 검증하기 위한 dataset -> window -> training smoke -> checkpoint -> generation -> decode -> objective review pipeline |
| 가장 큰 문제 | `.mid` 파일은 생성되지만 raw checkpoint generation valid/strict/grammar `0/0/0`, note count `2-4`, long sustain, dead-air 등으로 solo-line 품질 판단 불가 |
| 핵심 해결 | Stage B duration-explicit representation, objective MIDI review gate, repair boundary 분리, seed repeatability sweep |
| 수치 근거 | generic train/val `2433/270`, tokenized records `154136/21845`, best validation loss `5.9031`, constrained repair `3/3/3`, repeatability `9/9/9` |
| 현재 한계 | human/audio preference 미검증, broad trained-model quality 미검증, Brad style adaptation 미진행 |

## Claim Boundary

사용 가능:

- symbolic MIDI model-core validation pipeline
- Stage B duration-explicit tokenization
- generic base scale checkpoint training smoke
- objective MIDI review gate
- configured seed sweep objective repeatability

사용 금지:

- 완성된 재즈 피아노 생성 모델
- Brad style adaptation 완료
- production-ready improviser
- broad trained-model quality 검증 완료
- human/audio preference 검증 완료
