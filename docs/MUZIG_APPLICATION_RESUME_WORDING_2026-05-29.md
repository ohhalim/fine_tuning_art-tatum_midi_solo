# Muzig Application Resume Wording

## Target Context

- 회사/서비스: AI 음악 스타트업 Muzig
- 포지션: 풀스택 개발자 크루
- 공고 키워드: AI 음악 서비스, AI tool 실험, 아이디어 구현, 신규 기능/콘텐츠 기획, 서비스 확장
- 적용 프로젝트: Jazz Piano MIDI 생성 검증 파이프라인
- claim boundary: model-core validation pipeline

## Resume Project Entry

**Jazz Piano MIDI 생성 검증 파이프라인 | Model-Core / ML Pipeline | 2026.05**

Python, PyTorch, pretty_midi, music21, mido, NumPy, Pandas

- Symbolic MIDI 기반 jazz piano solo-line 생성 실험에서 `.mid` 파일 존재만으로 성공을 판단하던 문제를 note count, unique pitch, dead-air, max active notes, repeated cell, chord/tension/outside ratio, final landing 기반 objective MIDI review gate로 전환
- `NOTE_ON/OFF` 중심 Stage A representation의 duration/phrase 제어 한계를 `BAR`, `POSITION`, `CHORD_ROOT`, `CHORD_QUALITY`, `NOTE_PITCH`, `NOTE_DURATION`, `VELOCITY` 기반 Stage B tokenization으로 개선
- Dataset audit, tokenization, tiny training, generation probe, MIDI decode, objective review, focused review package를 연결해 model-core validation pipeline 구축
- 6-file / 5-sample repeatability sweep에서 strict `12/15`, warning seed `0`을 기록하고 seed-level margin과 dead-air outlier를 sample 단위로 분리
- Constrained adjacent repair 후 dead-air `0.5714`로 악화된 후보를 duration/coverage fill로 보정해 qualified `2/4`, fill additions `6`, dead-air `0.5714 -> 0.2941`, focused unique pitch `15`, adjacent repeat `0` 후보 확보
- 사용자 청취에서 repeatability 후보가 outside-soloing-like로 판정된 문제를 pitch-role repair sweep과 policy-level repeatability consolidation으로 보정해 repaired source `2/2`, qualified variants `6/6`, policy support `3/3` 확인
- MIDI metric 우세와 human/audio preference claim을 review input guard, MIDI evidence consolidation, final decision boundary로 분리해 broad model quality claim 없이 검증 범위 관리

## Short Resume Version

- Symbolic MIDI jazz solo-line 생성 실험에서 `.mid` 파일 존재 기반 성공 판단을 objective MIDI review gate로 전환하고, one-note collapse, long sustain block, chord block, dead-air outlier를 note-level metric으로 분리
- Stage B duration-explicit tokenization과 generation/decode/review pipeline을 구축해 dataset readable `2777`, vocab `547`, 6-file 5-sample strict `12/15`, duration fill dead-air `0.5714 -> 0.2941` 검증
- 사용자 청취에서 난해한 outside-soloing 후보를 pitch-role repair와 policy repeatability sweep으로 보정해 repaired source `2/2`, qualified variants `6/6`, policy support `3/3` 확인

## Self-Introduction Draft

### 1분 자기소개

AI 음악 서비스를 개발할 때 모델 출력이 실제 사용 가능한 결과인지 판단하는 검증 구조가 중요하다고 생각합니다. 저는 jazz piano MIDI 생성 프로젝트에서 `.mid` 파일이 생성됐다는 사실만으로 성공을 판단하지 않고, note count, dead-air, 반복 pitch-cell, chord fit, final landing 같은 objective metric으로 실패 원인을 분리했습니다. 이후 Stage B tokenization, generation probe, MIDI decode, focused review package, repeatability sweep을 연결해 model-core validation pipeline을 만들었습니다. 이 경험을 바탕으로 Muzig의 AI 음악 서비스에서 아이디어를 빠르게 실험하되, 결과를 측정 가능한 기준으로 검증하는 개발자로 기여하고 싶습니다.

### 지원 동기

Muzig의 AI 음악 서비스 방향은 단순 기능 구현보다 실험, 검증, 콘텐츠 확장 흐름이 중요하다고 판단했습니다. 제 MIDI 생성 프로젝트는 완성된 음악 모델을 주장하기보다, 생성 결과의 실패 원인을 metric으로 분리하고 다음 repair target을 정의한 작업입니다. AI tool을 활용한 아이디어 구현 과정에서도 같은 방식으로, 빠른 prototype과 검증 가능한 결과 기록을 함께 가져갈 수 있습니다.

### 강점 요약

- AI 음악 도메인 실험 경험: symbolic MIDI generation, tokenization, MIDI decode, review metric
- 문제 원인 분리: one-note collapse, dead-air outlier, outside-soloing-like output, duplicate output claim 위험
- 검증 기반 개선: repeatability sweep, repair boundary, review input guard, claim boundary 관리
- 스타트업 업무 적합성: 기능 구현, 실험 결과 정리, 다음 개선 target 정의, 문서화

## Caution

지원 문서에서 사용 가능:

- MIDI 생성 검증 파이프라인
- model-core validation pipeline
- objective MIDI review gate
- AI 음악 도메인 실험
- repeatability sweep 및 repair boundary

지원 문서에서 사용 금지:

- 완성된 재즈 피아노 생성 모델
- Brad Mehldau style adaptation 완료
- broad trained-model quality 검증 완료
- human/audio preference 검증 완료
- production-ready improviser
