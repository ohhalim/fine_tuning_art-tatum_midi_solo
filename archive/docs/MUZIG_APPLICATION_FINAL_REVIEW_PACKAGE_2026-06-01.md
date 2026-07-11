# Muzig Application Final Review Package

## 제출용 프로젝트 제목

**Jazz Piano MIDI 모델 검증 파이프라인 | Model-Core / ML Pipeline | 2026.05 - 2026.06**

Python, PyTorch, pretty_midi, music21, mido, NumPy, Pandas

## 이력서 Long Bullet 후보

- `.mid` 파일 생성만으로 성공을 판단하던 문제를 note count, unique pitch, phrase coverage, dead-air, max active notes, long-note ratio 기반 objective MIDI review gate로 전환해 one-note collapse, long sustain block, chord block 실패를 분리
- AI 음악 모델의 Brad style adaptation 이전 generic base 검증 기준을 만들기 위해 generic/Brad manifest split과 leakage guard를 구성하고 generic train/val `2433/270`, full generic tokenized records `154136/21845` 생성
- Stage B duration-explicit tokenization과 generic base scale checkpoint training smoke를 연결해 selected train/val `128/32`, best validation loss `5.9031`, checkpoint `1` 검증
- raw checkpoint generation valid/strict/grammar `0/0/0` 실패를 note count `2-4`, density, duration, dead-air 문제로 분리하고 constrained objective repair valid/strict/grammar `3/3/3`, dead-air/long-note `0/0`까지 개선
- seed `44/52/60` repeatability sweep에서 objective MIDI gate valid/strict/grammar `9/9/9`, failure reasons none을 확인하되 musical quality, human/audio preference, Brad style adaptation claim은 제외

## 이력서 Short Bullet 후보

- Symbolic MIDI jazz solo-line 생성 실험에서 `.mid` 파일 존재 기반 성공 판단을 objective MIDI review gate로 전환하고 one-note collapse, long sustain block, chord block 실패를 note-level metric으로 분리
- generic base 검증을 위해 manifest split, duration-explicit tokenization, training smoke, generation/decode/review pipeline을 구축하고 tokenized records `154136/21845`, best validation loss `5.9031` 기록
- raw checkpoint generation `0/0/0` 실패를 density/duration/dead-air repair boundary로 분리해 constrained objective repair `3/3/3`, seed sweep repeatability `9/9/9` 확인

## 1분 자기소개 후보

AI 음악 서비스에서 중요한 것은 결과물이 생성됐다는 사실보다, 그 결과가 실제로 사용할 수 있는지 판단할 수 있는 기준이라고 생각합니다. 저는 jazz piano MIDI 생성 프로젝트에서 `.mid` 파일 생성 여부를 성공 기준으로 두지 않고, note count, phrase coverage, dead-air, long-note ratio 같은 objective metric으로 실패 원인을 분리했습니다.

이후 generic/Brad manifest split, Stage B tokenization, training smoke, checkpoint generation, MIDI decode, objective review gate를 연결해 model-core validation pipeline을 만들었습니다. raw checkpoint generation이 valid/strict/grammar `0/0/0`으로 실패한 뒤에는 density, duration, dead-air repair boundary를 분리했고, constrained objective repair `3/3/3`, seed sweep `9/9/9`까지 검증했습니다.

Muzig에서도 AI tool과 음악 모델을 활용한 아이디어를 빠르게 실험하되, 결과를 측정 가능한 기준으로 남기고 다음 개선 대상을 정의하는 개발자로 기여하고 싶습니다.

## 지원 동기 후보

Muzig의 AI 음악 서비스는 모델 출력, 콘텐츠 기획, 기능 실험이 함께 맞물리는 영역이라고 보았습니다. 제 프로젝트는 완성된 음악 모델을 주장하는 작업이 아니라, AI 음악 생성 결과를 검증 가능한 단위로 쪼개고 실패 원인을 수치로 분리한 작업입니다.

이 경험은 AI tool을 활용한 신규 기능 실험, 음악 콘텐츠 아이디어 구현, 서비스 확장 과정에서 빠르게 만들고 근거를 남기는 방식으로 연결될 수 있습니다.

## 최종 Claim Check

사용 가능:

- MIDI 모델 검증 파이프라인
- model-core validation pipeline
- objective MIDI review gate
- configured seed sweep objective repeatability
- AI 음악 도메인 실험

사용 금지:

- 완성된 재즈 피아노 생성 모델
- Brad style adaptation 완료
- broad trained-model quality 검증 완료
- human/audio preference 검증 완료
- production-ready improviser
