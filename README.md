# Jazz Piano MIDI 생성 검증 파이프라인

> Symbolic MIDI 생성 결과의 note-level 검증, 실패 원인 분리, review gate 구축 프로젝트

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 주제 | Jazz piano solo-line MIDI 생성 검증 |
| 목표 | reviewable jazz solo-line MIDI 생성을 위한 tokenization / generation / decoding / validation loop 검증 |
| 핵심 관점 | `.mid` 파일 생성 여부가 아닌 note-level 품질 검증 |
| 현재 범위 | model-core 실험, objective review, focused review |
| 제외 범위 | broad model-quality claim, Brad Mehldau style adaptation claim, realtime DAW/plugin, backend/API, product MVP |

## 2. 문제 정의

초기 Stage A 생성 결과의 주요 실패 유형:

| 실패 유형 | 관찰 내용 |
|---|---|
| note sparsity | note count 부족 |
| sustain collapse | 긴 sustain block 출력 |
| chord block | solo-line이 아닌 동시 발음 구조 |
| pitch repetition | 반복 pitch / pitch-class cell |
| timing stiffness | grid-derived timing |
| weak landing | 어색한 final landing |

핵심 판단:

- 모델 확장 전 실패 재현 및 측정 체계 필요
- `.mid` 생성 성공과 음악적 성공의 분리 필요
- representation 문제로 인한 Stage A 한계 확인
- duration / position 명시형 Stage B tokenization 전환

## 3. 접근 방식

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

작업 원칙:

- 작은 probe 단위 실험
- 실패 유형별 metric 분리
- objective review와 focused review 분리
- generated artifact와 문서화 결과 분리
- 품질 주장보다 검증 근거 우선

## 4. Stage 전환

| 단계 | 방식 | 판단 |
|---|---|---|
| Stage A | `NOTE_ON/OFF` 중심 control token | runnable pipeline 확인, musical output 실패 |
| Stage B | duration-explicit symbolic token | phrase/window 단위 생성 및 review gate 구축 |

Stage B token family:

- `BAR`
- `POSITION`
- `CHORD_ROOT`
- `CHORD_QUALITY`
- `NOTE_PITCH`
- `NOTE_DURATION`
- `VELOCITY`

## 5. 검증 기준

Objective MIDI review 기준:

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

성공 조건 제외:

- `.mid` 파일 존재만으로 성공 처리
- one-note / two-note output
- long sustain block
- chord block output
- repeated-cell collapse
- final landing 미검증 결과

## 6. 핵심 결과

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

결과 의미:

- reviewable MIDI outcome 확보
- objective-clean focused candidate 확보
- repeated-cell blocker 제거
- proxy review -> focused context decision -> focused listening fill 경로 검증
- 단일 후보 기준 current best candidate 정리

결과 한계:

- broad multi-seed model quality 미증명
- human/audio listening preference 미증명
- Brad Mehldau style adaptation 미증명
- realtime DAW/plugin readiness 미증명
- backend/API/product MVP readiness 미증명

## 7. 구현 범위

### Dataset audit

| 항목 | 값 |
|---|---:|
| active dataset tree | `midi_dataset/midi` |
| readable files | `2777` |
| candidate files | `2775` |
| candidate non-Brad files | `2703` |
| candidate Brad files | `72` |
| exact duplicate hash groups | `0` |

Dataset 판단:

- Brad dataset 직접 scratch training 제외
- generic jazz base 이후 adaptation / holdout 후보 분리
- dataset audit 선행 후 generation probe 진행

### Generation / review probes

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

## 8. 실행 방법

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

## 9. 주요 파일

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

## 10. 포트폴리오 포인트

- 모델 출력 실패의 원인 단위 분해
- MIDI 파일 생성 여부와 품질 검증의 분리
- Stage A 실패 후 representation 재설계
- Stage B tokenization / constrained generation / review gate 구축
- issue 단위 실험, 검증, 문서화 흐름
- 품질 과장 없이 current best candidate와 한계 동시 기록

## 11. 현재 한계

| 구분 | 상태 |
|---|---|
| focused keep candidate | 단일 후보 |
| multi-seed reliability | 미검증 |
| timing / vocabulary | `acceptable`, strong claim 제외 |
| source IOI diversity | 낮음 |
| proxy risk | `too_mechanical` 잔존 |
| style adaptation | 미주장 |
| realtime/product scope | 범위 밖 |

## 12. 현재 상태

| 항목 | 상태 |
|---|---|
| latest completed | Issue #214 |
| current best candidate evidence | Issue #210 consolidation |
| broad training | 미진행 |
| Brad style adaptation | 미진행 |
| next candidate task | 이력서 프로젝트 bullet 정리 |
| alternative research task | Stage B focused timing vocabulary keep repeatability sweep |

## 13. 문서

- [Current Status and Plan](docs/CURRENT_STATUS_AND_PLAN.md)
- [Core Plan](docs/CORE_PLAN.md)
- [References](docs/REFERENCES.md)
