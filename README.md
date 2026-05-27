# Jazz Piano MIDI 생성 검증 파이프라인

> 생성된 MIDI 파일을 성공으로 보지 않고, note-level metric과 review gate로 실패 원인을 분리하는 symbolic MIDI 실험 프로젝트입니다.

## 프로젝트 요약

이 프로젝트는 재즈 피아노 솔로 MIDI를 바로 "잘 생성하는 모델"로 포장하기보다, 모델 출력이 왜 실패하는지 재현하고 측정하는 파이프라인을 만드는 데 집중했습니다.

핵심 목표는 완성된 음악 생성 제품이 아니라, **reviewable jazz solo-line MIDI를 만들 수 있는 tokenization, generation, decoding, validation loop를 증명하는 것**입니다.

현재 가장 의미 있는 결과는 Stage B constrained generation과 focused review loop를 거쳐, 단일 후보를 current best focused review candidate로 유지한 것입니다. 단, 이 결과는 broad model quality, human listening preference, Brad Mehldau style adaptation을 증명하지 않습니다.

## 문제 정의

초기 목표는 Brad Mehldau 스타일의 jazz piano MIDI generator였습니다. 하지만 작은 dataset으로 바로 학습을 키우면, `.mid` 파일은 생성되지만 실제 piano roll에서는 다음 문제가 반복됐습니다.

- note count가 너무 적음
- 긴 sustain block
- chord block처럼 보이는 출력
- solo-line이 아닌 동시 발음 구조
- 반복 pitch-class cell
- grid-derived timing stiffness
- final landing이 음악적으로 어색함

그래서 방향을 바꿨습니다.

> 모델을 키우기 전에, 실패를 측정할 수 있는 generation/evaluation pipeline을 먼저 만든다.

## 핵심 성과

Issue #210 기준 current best focused review candidate:

| 항목 | 결과 |
|---|---|
| candidate | `data_motif_rhythm_phrase_variation_rank_2_sample_2` |
| decision | current best focused review candidate |
| note count | `64` |
| unique pitch count | `19` |
| range | `G3-G5` |
| phrase span | `32.0` beats |
| max active notes | `1` |
| max interval | `4` |
| objective flags | `[]` |
| adjacent pitch repeats | `0` |
| duplicated 3/4/8-note pitch-class chunks | `0 / 0 / 0` |
| final landing | `D5` over `Ebmaj7` |
| focused timing | `acceptable` |
| focused chord fit | `strong` |
| focused landing | `strong` |
| focused jazz vocabulary | `acceptable` |

이 결과의 의미:

- MIDI 파일 존재 여부가 아니라 note-level metric으로 검증했습니다.
- one-note, long sustain, chord block, repeated cell 같은 실패를 gate에서 분리했습니다.
- Stage A 실패를 representation 문제로 재정의하고 Stage B duration-explicit token으로 전환했습니다.
- proxy review, focused context decision, focused listening fill까지 이어지는 review workflow를 만들었습니다.

이 결과가 아직 의미하지 않는 것:

- broad multi-seed model quality
- human/audio listening preference
- Brad Mehldau style adaptation
- realtime DAW/plugin readiness
- backend/API/product MVP readiness

## 접근 방식

```mermaid
flowchart LR
    A["Dataset audit"] --> B["Stage B tokenization"]
    B --> C["Generation probe"]
    C --> D["MIDI decode"]
    D --> E["Overlap-free postprocess"]
    E --> F["Objective MIDI review"]
    F --> G["Proxy / focused review"]
    G --> H["Repair or consolidate"]
```

### 1. Stage A 실패 확인

초기 Stage A는 `NOTE_ON/OFF` 중심의 control token 방식이었습니다. 학습/생성 경로는 runnable했지만 결과 MIDI가 솔로 라인으로 보기 어려웠습니다.

대표 실패:

- one-note collapse
- long sustain block
- chord block output
- phrase coverage 부족

결론:

- Stage A를 더 강하게 postprocess하지 않는다.
- duration과 position을 명시하는 Stage B tokenization으로 전환한다.

### 2. Stage B duration-explicit tokenization

Stage B에서는 REMI/Jazz Transformer 계열 판단에 맞춰 token family를 명시했습니다.

- `BAR`
- `POSITION`
- `CHORD_ROOT`
- `CHORD_QUALITY`
- `NOTE_PITCH`
- `NOTE_DURATION`
- `VELOCITY`

이후 2-bar, 4-bar, 8-bar phrase window 단위로 generation probe를 만들고, 생성된 MIDI를 다시 note-level로 읽어 평가했습니다.

### 3. Objective MIDI review

생성된 MIDI를 다음 기준으로 다시 검사합니다.

- non-zero note count
- unique pitch count
- max simultaneous notes
- polyphonic tick ratio
- phrase coverage
- dead-air ratio
- max note duration ratio
- repeated pitch/cell ratio
- max interval
- unresolved large leap ratio
- chord-tone/tension/outside/root ratio
- final guide/chord landing
- IOI/duration diversity

`valid .mid exists`는 성공 조건이 아닙니다.

### 4. Focused review loop

최근 loop는 다음 순서로 진행했습니다.

1. proxy review에서 후보를 좁힘
2. focused package로 solo/context MIDI를 격리
3. context MIDI note 기준으로 register, cadence, repeated cell, final landing 확인
4. focused listening review notes template 생성
5. pending fields를 채워 `keep` 또는 `needs_followup` 결정
6. keep 후보의 의미를 consolidation 문서로 정리

## 구현 범위

### Dataset audit

- active dataset tree: `midi_dataset/midi`
- readable files: `2777`
- candidate files: `2775`
- candidate non-Brad files: `2703`
- candidate Brad files: `72`
- exact duplicate hash groups: `0`

Brad dataset은 바로 scratch training에 쓰지 않고, generic jazz base 이후 adaptation/holdout 후보로 분리했습니다.

### Generation probes

구현한 probe와 review 흐름:

- grammar-constrained generation
- overlap/dedup gate
- temporal coverage diagnostics
- coverage-aware constrained generation
- chord-aware pitch constrained generation
- data-derived motif rhythm generation
- phrase/cadence review baseline
- register-safe final landing repair
- focused context package
- focused listening review notes/fill

각 probe는 "좋은 MIDI 하나"가 아니라 여러 sample의 pass-rate와 failure reason을 남기도록 만들었습니다.

## 실행 방법

환경 설치:

```bash
pip install -r requirements.txt
```

빠른 검증:

```bash
bash scripts/agent_harness.sh quick
```

Stage B rhythm/phrase variation probe:

```bash
bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation
```

Focused listening review notes:

```bash
bash scripts/agent_harness.sh stage-b-focused-listening-review-notes
```

## 주요 파일

```text
scripts/
  run_stage_b_generation_probe.py
  run_stage_b_data_motif_generation_compare.py
  review_midi_note_objectives.py
  build_listening_review_notes.py
  build_focused_review_package.py
  build_focused_listening_review_notes.py
  agent_harness.sh

inference/app/
  generator.py
  metrics.py
  postprocess.py

docs/
  CORE_PLAN.md
  CURRENT_STATUS_AND_PLAN.md
  STAGE_B_FOCUSED_TIMING_VOCABULARY_KEEP_CANDIDATE_CONSOLIDATION_2026-05-27.md
```

## 포트폴리오에서 보여줄 수 있는 점

이 프로젝트의 핵심은 "음악 생성 모델을 만들었다"가 아니라, 모델 출력 실패를 엔지니어링 문제로 분해한 과정입니다.

- 표면적으로 `.mid`가 생성됐다는 사실을 성공으로 처리하지 않음
- piano-roll에서 보이는 실패를 metric과 gate로 분리
- 실패한 Stage A를 버리고 representation 문제로 재정의
- Stage B tokenization, constrained generation, review gate를 단계적으로 구축
- 실험 결과가 나쁘면 다음 issue의 repair target으로 연결
- 품질을 과장하지 않고 current best candidate와 남은 한계를 함께 기록

## 현재 한계

- focused review `keep` 후보는 1개뿐이며 multi-seed 품질 증명은 아닙니다.
- timing과 jazz vocabulary는 `acceptable` 수준이며 strong claim은 아닙니다.
- source IOI diversity는 아직 낮고 `too_mechanical` risk가 남아 있습니다.
- Brad style adaptation을 주장할 단계가 아닙니다.
- realtime DAW/plugin, backend/API, product MVP는 아직 범위 밖입니다.

## 현재 상태와 다음 작업

현재 main 기준 최신 판단:

- latest completed: Issue #212
- current best candidate evidence: Issue #210 consolidation
- broad training: 아직 진행하지 않음
- Brad style adaptation: 아직 진행하지 않음

다음 작업 후보:

- 이 README를 기반으로 이력서 프로젝트 bullet 정리
- Stage B focused timing vocabulary keep repeatability sweep

## 문서

- [Current Status and Plan](docs/CURRENT_STATUS_AND_PLAN.md)
- [Core Plan](docs/CORE_PLAN.md)
- [References](docs/REFERENCES.md)
