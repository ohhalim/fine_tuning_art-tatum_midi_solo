# References for Brad Mehldau MIDI Fine-Tuning

작성일: 2026-05-18

이 문서는 논문 목록이 아니라, 현재 Brad Mehldau MIDI fine-tuning probe에 직접 영향을 주는 reference decision map이다.

현재 문제는 "Transformer를 쓸 수 있는가"가 아니다.

현재 문제는 다음에 가깝다.

- `control_v1` full-song token sequence가 너무 길다.
- NOTE_ON/OFF 중심 representation이 duration을 안정적으로 학습하지 못할 수 있다.
- 생성 MIDI가 solo-line이 아니라 sustain block/chord block처럼 무너진다.
- chord/position/phrase 정보를 모델이 명시적으로 보기 어렵다.

따라서 다음 reference는 모델 이름보다 tokenization, conditioning, sequence length, jazz-specific representation 관점에서 읽는다.

## Current Takeaway

지금 당장 적용할 판단:

1. 먼저 `max_files=2` Brad Mehldau `control_v1` training probe를 끝낸다.
2. 그 probe가 sustain/chord block을 계속 만들면 postprocess를 더 세게 걸지 않는다.
3. 다음 설계는 REMI/Jazz Transformer 계열의 duration-explicit, bar-position-aware tokenization으로 간다.
4. `control_v1` full-song training은 control-aware crop 없이는 의미가 없다.
5. 긴 곡 전체를 한 sequence로 넣는 대신 phrase/window dataset을 준비해야 할 가능성이 높다.

## Priority References

### 1. Music Transformer

Link:

- https://research.google/pubs/music-transformer-generating-music-with-long-term-structure/

Why it matters:

- 현재 repo의 Music Transformer 계열 코드가 기대고 있는 기본 연구 라인이다.
- symbolic music sequence에 self-attention을 쓰는 근거가 된다.
- relative timing과 long-range structure가 핵심 문제라는 점을 확인한다.

What it means for this repo:

- Transformer 자체는 타당한 방향이다.
- 하지만 지금 failure는 architecture보다 representation/data/crop 문제에 더 가깝다.

Do not copy blindly:

- 논문이 성공했다고 해서 현재 NOTE_ON/OFF tokenization과 작은 Brad Mehldau dataset이 바로 좋은 solo-line을 만든다고 보면 안 된다.

### 2. Pop Music Transformer / REMI

Link:

- https://arxiv.org/abs/2002.00212
- https://github.com/YatingMusic/remi

Why it matters:

- REMI는 BAR, POSITION, TEMPO, CHORD 같은 metric-aware events를 사용한다.
- music score를 Transformer input으로 바꾸는 방식 자체가 성능에 큰 영향을 준다는 점을 전면에 둔다.

What it means for this repo:

- 현재 `control_v1`은 role/tempo/bar/conditioning separator만 추가한 Stage A 포맷이다.
- sustain block 문제가 계속되면 Stage B는 `BAR + POSITION + NOTE_PITCH + NOTE_DURATION + VELOCITY + CHORD` 쪽으로 가야 한다.
- chord progression을 low-register primer hack으로만 넣는 대신 symbolic chord event로 넣을 필요가 있다.

Implementation implication:

- Stage B tokenizer 후보:
  - `BAR`
  - `POSITION_0..63` or 16th/32nd grid
  - `CHORD_ROOT_*`
  - `CHORD_QUALITY_*`
  - `NOTE_PITCH_*`
  - `NOTE_DURATION_*`
  - `VELOCITY_*`
  - `TEMPO_*`

### 3. The Jazz Transformer on the Front Line

Link:

- https://archives.ismir.net/ismir2020/paper/000339.pdf

Why it matters:

- jazz-specific symbolic Transformer reference다.
- NOTE_DURATION, BAR, POSITION, TEMPO, CHORD, PHRASE 같은 event가 실제 jazz improvisation modeling에서 쓰였다.
- WJazzD solo sequence가 길어서 전체 piece를 Transformer에 넣기 어렵다는 문제를 명시한다.

What it means for this repo:

- 지금 Brad Mehldau audit에서 18/18 files가 `max_sequence=512`를 초과한 것은 예상 가능한 문제다.
- duration-explicit tokenization은 선택 사항이 아니라 다음 유력 경로다.
- phrase boundary 또는 phrase-window dataset이 필요할 수 있다.

Implementation implication:

- `control_v1` probe가 실패하면 바로 이 방향을 Stage B spec으로 문서화한다.
- 특히 NOTE_OFF를 모델이 암묵적으로 맞히게 두는 대신 duration token을 직접 예측하게 만드는 것이 우선이다.

### 4. Compound Word Transformer

Link:

- https://arxiv.org/abs/2101.02402

Why it matters:

- pitch, duration, velocity, onset 같은 서로 다른 token type을 모두 같은 flat token처럼 취급하는 문제를 지적한다.
- 여러 token을 compound word로 묶어 sequence length를 줄이는 방향을 제시한다.

What it means for this repo:

- 현재 Brad Mehldau `control_v1_token_count`는 평균 약 `3931`, 최대 `10653`이다.
- flat event token만 계속 쓰면 sequence length 문제가 반복될 가능성이 높다.

Implementation implication:

- 당장 Compound Word architecture를 구현하지 않는다.
- 하지만 Stage B tokenizer 설계 시 token type을 명시하고, 나중에 grouped representation으로 갈 수 있게 naming을 정리한다.

### 5. Anticipatory Music Transformer

Link:

- https://openreview.net/forum?id=EBNJ33Fcrl

Why it matters:

- symbolic music generation에서 control process와 event process를 interleave하는 관점을 제공한다.
- chord/control 정보를 generation stream과 어떻게 섞을지 참고할 수 있다.

What it means for this repo:

- `control_v1`의 `ROLE/TEMPO/BAR + conditioning + COND_SEP`는 아주 단순한 control format이다.
- 나중에는 chord, section, density, phrase target 같은 control token을 event stream 중간중간 넣는 방식이 필요할 수 있다.

Implementation implication:

- 지금은 `control_v1`을 유지한다.
- Stage B에서 chord event를 bar/position 위치에 직접 배치하는 설계를 검토한다.

### 6. ImprovNet

Link:

- https://arxiv.org/abs/2502.04522

Why it matters:

- controllable symbolic improvisation과 jazz style transfer를 직접 다룬다.
- limited jazz dataset, short continuation, infilling 같은 문제가 현재 목표와 가깝다.

What it means for this repo:

- 최종 방향은 단순 next-token generation보다 controllable improvisation task에 가깝다.
- 하지만 지금 단계에서는 architecture를 따라가기보다 evaluation criteria와 task framing을 참고한다.

Implementation implication:

- Stage A/Stage B가 valid MIDI를 만들기 전에는 corruption-refinement 같은 큰 구조 변경을 하지 않는다.

### 7. Jazz Trio Database

Link:

- https://transactions.ismir.net/articles/10.5334/tismir.186

Why it matters:

- jazz piano trio recordings에서 piano soloist MIDI annotation을 제공하는 dataset reference다.
- Brad Mehldau 18-file dataset이 너무 작을 때 확장 후보가 된다.

What it means for this repo:

- Brad Mehldau-only fine-tuning은 데이터가 매우 작다.
- style-specific fine-tuning 전에 broader jazz piano solo prior가 필요할 수 있다.

Implementation implication:

- 지금은 dataset을 추가하지 않는다.
- 2-file/5-file/full-18 probe가 실패하면 broader jazz piano MIDI pretraining/fine-tuning data를 검토한다.

### 8. Aria / Aria-MIDI

Link:

- https://github.com/EleutherAI/aria

Why it matters:

- pretrained autoregressive symbolic piano model과 large piano MIDI dataset 방향을 보여준다.
- scratch training이 너무 약할 때 pretrained piano continuation model을 쓰는 대안이다.

What it means for this repo:

- Brad Mehldau 18 MIDI만으로 from-scratch model을 만드는 것은 매우 약한 기준선이다.
- 장기적으로는 pretrained symbolic piano model + small style adaptation이 더 현실적일 수 있다.

Implementation implication:

- 현재 branch에서는 Aria를 도입하지 않는다.
- `control_v1` probe 결과가 나쁘면 "pretrained symbolic piano model migration"을 별도 issue로 분리한다.

## Local References

### Existing Research Memo

Path:

- `docs/archive/VELOG_PRIOR_RESEARCH_SYNTHESIS.md`

Use:

- realtime/DAW/JAM_BOT/Continuator 같은 장기 방향을 보관한다.
- 현재 active plan은 아니지만, 나중에 live co-performer 방향으로 돌아갈 때 참고한다.

### Third-Party Music Transformer Reference

Path:

- `music_transformer/third_party/references.txt`

Current content:

- `jason9693/midi-neural-processor`
- https://github.com/jason9693/midi-neural-processor

Use:

- 현재 MIDI processor 계열 코드의 ancestry를 확인할 때 참고한다.

## Stage B Direction If Control V1 Fails

If the Brad Mehldau `control_v1` probe still generates invalid MIDI, create a new Stage B tokenization issue instead of adding more postprocess.

Candidate Stage B representation:

```text
STYLE_PERSONAL_JAZZ
ROLE_LEAD
TEMPO_120
BAR
CHORD_ROOT_C
CHORD_QUALITY_MIN7
POSITION_0
NOTE_PITCH_60
NOTE_DURATION_4
VELOCITY_80
POSITION_8
NOTE_PITCH_63
NOTE_DURATION_2
VELOCITY_72
...
END
```

Required properties:

- duration is explicit
- onset position is explicit
- chord is explicit
- bar boundary is explicit
- no implicit long note from missing NOTE_OFF
- phrase/window segmentation is possible

Acceptance criteria:

- generated MIDI has enough notes
- generated MIDI has enough unique pitches
- generated MIDI covers the requested phrase span
- max note duration ratio stays under solo-line gate
- max simultaneous notes stays under solo-line gate
- no one-note/two-note files
- no long sustain block
- no chord block pretending to be a solo line

## Reading Order

Use this order before changing tokenization:

1. `docs/STAGE_A_CODE_REVIEW_2026-05-18.md`
2. `docs/BRAD_MEHLDAU_FINETUNING_PLAN.md`
3. Jazz Transformer paper
4. REMI / Pop Music Transformer
5. Compound Word Transformer
6. Anticipatory Music Transformer
7. Aria only if scratch training remains too weak

## Non-Goals

Do not use these references to justify scope creep in this branch.

- Do not restart backend/API work.
- Do not build realtime DAW integration.
- Do not add a new large model dependency before the 2-file probe.
- Do not claim artist cloning.
- Do not treat valid file creation as musical success.
