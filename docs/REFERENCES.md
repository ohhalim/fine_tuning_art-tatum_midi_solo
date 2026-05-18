# References for Brad Mehldau MIDI Fine-Tuning

작성일: 2026-05-18

이 문서는 논문 목록이 아니라, 현재 Brad Mehldau MIDI fine-tuning probe에 직접 영향을 주는 reference decision map이다.

최신성 기준:

- 2026-05-18 기준으로 2024-2026 symbolic MIDI generation, tokenization, dataset research를 반영한다.
- 최신 논문이라도 현재 2-file probe를 건너뛰게 만들지는 않는다.

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

## Latest Research Synthesis

2024-2026 흐름을 종합하면, 지금 repo의 다음 구현 방향은 크게 세 갈래다.

### A. Keep Current Control V1 Only For The Probe

`control_v1`은 현재 코드를 빨리 검증하기 위한 최소 format이다.

Use it for:

- Brad Mehldau `max_files=2` prepare
- Brad Mehldau `max_files=2` training
- loader/checkpoint/generation/gate smoke

Do not extend it too far:

- `control_v1` has no explicit note duration token.
- `control_v1` has no explicit bar-position note onset.
- `control_v1` uses conditioning MIDI as a primer rather than learned chord events.
- `control_v1` full-song sequences are too long for naive `max_sequence=512` training.

### B. If Control V1 Fails, Build Stage B Tokenization

The strongest implementation signal comes from REMI, Jazz Transformer, BEAT, and MidiTok.

Stage B should make these musical facts explicit:

- bar boundary
- onset position
- chord at the current position
- note pitch
- note duration
- velocity
- tempo
- optional phrase/window boundary

This directly targets the current failure:

- long sustain block from unstable NOTE_OFF prediction
- chord block instead of solo-line phrase
- weak chord conditioning
- long sequence crop losing conditioning information

### C. If From-Scratch Training Is Too Weak, Stop Training Tiny Models From Scratch

Aria, Moonbeam, MIDI-LLM, Text2midi, and recent piano Transformer studies all point in the same direction:

- large symbolic MIDI pretraining matters
- small task-specific fine-tuning works better after a broad prior exists
- 18 Brad Mehldau MIDI files are not enough to learn a style from scratch

If the full 18-file probe still produces invalid MIDI, split a new issue for:

- pretrained symbolic piano continuation model evaluation
- broader jazz/piano MIDI pretraining data
- small style adapter on top of a real symbolic base

## Implementation Decision Matrix

| Problem Observed | Reference Direction | Implementation Move |
|---|---|---|
| generated MIDI has very long notes | Jazz Transformer, REMI, MidiTok TSD/REMI | use explicit `NOTE_DURATION_*`; stop relying on NOTE_OFF |
| generated MIDI ignores bar grid | REMI, BEAT | add `BAR` and `POSITION_*` or uniform time-step groups |
| generated MIDI ignores chords | Jazz Transformer, AMT, Text2midi/MIDI-LLM control results | add symbolic chord events, not only low-register primer MIDI |
| sequence length is too long | Compound Word, BEAT, PianoCoRe/Aria large-scale practice | phrase/window dataset; consider grouped/time-step representation |
| tiny dataset cannot learn style | Aria, Moonbeam, PianoCoRe, piano Transformer scale study | use pretrained symbolic piano model or broader pretraining corpus |
| need text/style controls later | Text2midi, MIDI-LLM, FIGARO-style control | defer; use structured tokens before natural language |
| need live accompaniment later | AMT, BEAT, JAM_BOT line | defer until offline generation is valid |

## Implementation Order Recommended By The Literature

Do not jump directly to a new foundation model.

Use this order:

1. Finish current `control_v1` 2-file probe.
2. If it fails musically, add Stage B tokenization docs/tests before training again.
3. Implement a REMI-like tokenizer locally or through MidiTok.
4. Rebuild Brad Mehldau phrase/window dataset with explicit duration and position.
5. Run tiny-overfit on Stage B representation.
6. Run 2-file Brad Mehldau Stage B probe.
7. Only then evaluate pretrained symbolic models such as Aria/Moonbeam.

Reason:

- A pretrained model will not fix a broken evaluation gate.
- A better tokenizer will not help if the pipeline cannot overfit a tiny sample.
- More data will not help if decoding still produces invalid MIDI.

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
- https://arxiv.org/abs/2506.23869
- https://arxiv.org/abs/2504.15071

Why it matters:

- pretrained autoregressive symbolic piano model과 large piano MIDI dataset 방향을 보여준다.
- scratch training이 너무 약할 때 pretrained piano continuation model을 쓰는 대안이다.
- Aria model line reports pretraining on roughly 60,000 hours of symbolic solo-piano transcriptions.
- Aria-MIDI reports a much larger piano MIDI corpus built from transcribed piano audio.

What it means for this repo:

- Brad Mehldau 18 MIDI만으로 from-scratch model을 만드는 것은 매우 약한 기준선이다.
- 장기적으로는 pretrained symbolic piano model + small style adaptation이 더 현실적일 수 있다.

Implementation implication:

- 현재 branch에서는 Aria를 도입하지 않는다.
- `control_v1` probe 결과가 나쁘면 "pretrained symbolic piano model migration"을 별도 issue로 분리한다.

### 9. Moonbeam

Link:

- https://arxiv.org/abs/2505.15559
- https://aim-qmul.github.io/moonbeam-midi-foundation-model/

Why it matters:

- large MIDI foundation model 방향이다.
- absolute musical attributes and relative musical attributes를 모두 쓰는 tokenization/modeling을 강조한다.
- conditional generation and infilling fine-tuning을 downstream task로 둔다.

What it means for this repo:

- jazz solo는 absolute pitch만이 아니라 interval/motif contour가 중요하다.
- Brad Mehldau-like phrasing을 원하면 relative interval/melodic motion feature가 나중에 필요할 수 있다.

Implementation implication:

- 지금은 Moonbeam을 도입하지 않는다.
- Stage B tokenizer naming은 나중에 relative interval features를 추가할 수 있게 열어둔다.
- example future tokens:
  - `INTERVAL_UP_2`
  - `INTERVAL_DOWN_1`
  - `CONTOUR_ASC`
  - `MOTIF_REPEAT`

### 10. BEAT

Link:

- https://arxiv.org/abs/2604.19532

Why it matters:

- 2026 tokenization reference다.
- event-based sequence가 musical time regularity를 implicit하게 다루는 문제를 지적한다.
- uniform time-step group을 basic unit으로 삼아 symbolic events를 묶는 방향이다.

What it means for this repo:

- 현재 출력이 sustain/chord block으로 무너지는 것은 time representation 문제가 섞여 있을 수 있다.
- REMI-like event sequence가 실패하면 uniform time-step representation도 후보가 된다.

Implementation implication:

- Stage B first choice는 REMI-like duration/position tokens.
- If REMI-like Stage B still has sequence/generation instability, Stage C can evaluate BEAT-like time-step grouping.

Potential Stage C shape:

```text
BAR
STEP_000 CHORD_C_MIN7 NOTE_60_DUR_2_VEL_72 NOTE_63_DUR_1_VEL_68
STEP_001 REST
STEP_002 NOTE_67_DUR_1_VEL_80
...
```

### 11. PianoCoRe

Link:

- https://arxiv.org/abs/2605.06627
- https://huggingface.co/datasets/SyMuPe/PianoCoRe
- https://github.com/ilya16/PianoCoRe

Why it matters:

- 2026 large-scale refined piano MIDI dataset reference다.
- Reports 250,046 performances, 5,625 pieces, and 21,763 hours of performed music.
- Includes quality classification and alignment refinement concepts.

What it means for this repo:

- The dataset problem is not just "more MIDI".
- Quality filtering matters before training.
- Brad Mehldau 18-file style data should probably become a small adaptation layer after broader piano/jazz prior.

Implementation implication:

- Add dataset audit fields before expanding data:
  - corrupted/unreadable
  - score-like vs performance-like
  - note density distribution
  - tempo outliers
  - duration outliers
  - repeated identical files / duplicates
- Do not add PianoCoRe dependency in this branch.

### 12. MIDI-LLM

Link:

- https://arxiv.org/abs/2511.03942

Why it matters:

- LLM vocabulary expansion with MIDI tokens and two-stage training is a current text-to-MIDI direction.
- It preserves the text LLM parameter structure and targets faster inference through existing LLM serving stacks.

What it means for this repo:

- Natural-language prompting is not needed yet.
- But the "extend vocabulary with MIDI/control tokens" direction supports our current control-token approach.

Implementation implication:

- Defer text prompting.
- Keep structured controls first:
  - chord
  - section
  - density
  - energy
  - role
- If adding text later, map text to structured controls first instead of training direct free-form text-to-MIDI.

### 13. Text2midi

Link:

- https://arxiv.org/abs/2412.16526
- https://github.com/AMAAI-Lab/Text2midi

Why it matters:

- Uses a pretrained LLM encoder plus autoregressive Transformer decoder to generate MIDI from captions.
- Captions can include music-theory terms such as chords, keys, and tempo.

What it means for this repo:

- User-facing prompt control can come later, but it should not replace explicit symbolic conditioning now.

Implementation implication:

- Keep API/request fields structured.
- Later, optional text prompt can be parsed into:
  - `key`
  - `chords`
  - `tempo`
  - `style`
  - `density`

### 14. PerTok / Expressive MIDI Performance Tokenization

Link:

- https://ismir2025program.ismir.net/lbd_392.html
- https://researchtrend.ai/papers/2410.02060

Why it matters:

- PerTok focuses on expressive MIDI performance details such as micro-timing and duration nuance.
- ISMIR 2025 ornament generation work uses PerTok with a domain-specific Transformer.

What it means for this repo:

- Brad Mehldau-like feel eventually needs microtiming and velocity nuance.
- But current problem is more basic: valid solo-line notes first.

Implementation implication:

- Do not implement microtiming in Stage B.
- Reserve Stage C/D for expressive performance rendering after note/duration/chord correctness is stable.

### 15. Generating Piano Music With Transformers: Scale, Data, Metrics

Link:

- https://arxiv.org/abs/2511.07268

Why it matters:

- It explicitly compares datasets, model sizes, architectures, training strategies, and metrics for symbolic piano generation.
- It emphasizes that quantitative metrics should be checked against human listening judgment.

What it means for this repo:

- Our current gate is necessary but not sufficient.
- Metrics can prevent nonsense MIDI, but piano-roll/listening review remains required.

Implementation implication:

- Keep automatic gates:
  - note count
  - unique pitch
  - phrase coverage
  - max note duration ratio
  - max simultaneous notes
  - dead air
- Add manual review checkpoints after each probe.
- Do not claim improvement only from lower validation loss.

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

Candidate Stage B representation, REMI/Jazz-Transformer-like:

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

## Concrete Stage B Implementation Sketch

If Stage B is opened, implement it in small files rather than rewriting the repo.

Suggested files:

```text
scripts/stage_b_tokenizer.py
scripts/prepare_stage_b_dataset.py
scripts/run_stage_b_tiny_overfit.py
tests/test_stage_b_tokenizer.py
docs/STAGE_B_TOKENIZATION_PLAN.md
```

Minimum tokenizer API:

```python
def encode_midi_stage_b(
    midi_path: str,
    chord_progression: list[str] | None,
    bpm: float,
    bars_per_window: int = 2,
) -> list[int]:
    ...

def decode_stage_b_tokens(tokens: list[int], output_path: str, bpm: float) -> None:
    ...
```

Minimum generated token contract:

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
POSITION_4
NOTE_PITCH_62
NOTE_DURATION_2
VELOCITY_76
END
```

Minimum tests:

- round-trip one 2-bar phrase through encode/decode
- duration tokens decode to bounded notes
- no note can last beyond the requested phrase unless tied explicitly
- chord tokens align to bar/position windows
- token count is lower or at least more musically structured than current `control_v1`
- tiny-overfit can produce non-fallback MIDI

Consider MidiTok before custom implementation:

- MidiTok supports REMI, TSD, CPWord, Octuple, MuMIDI, and other tokenizations.
- REMI/TSD are better first candidates than CPWord/Octuple for a small model because CPWord/Octuple need multi-head output losses and more delicate decoding.

## Reading Order

Use this order before changing tokenization:

1. `docs/STAGE_A_CODE_REVIEW_2026-05-18.md`
2. `docs/BRAD_MEHLDAU_FINETUNING_PLAN.md`
3. Jazz Transformer paper
4. REMI / Pop Music Transformer
5. MidiTok docs for REMI/TSD implementation
6. BEAT if time-step grouping is needed
7. Compound Word Transformer if sequence length remains a blocker
8. Anticipatory Music Transformer if control/infilling is needed
9. Aria/Moonbeam only if scratch training remains too weak
10. Text2midi/MIDI-LLM only when natural-language control becomes relevant

## Non-Goals

Do not use these references to justify scope creep in this branch.

- Do not restart backend/API work.
- Do not build realtime DAW integration.
- Do not add a new large model dependency before the 2-file probe.
- Do not claim artist cloning.
- Do not treat valid file creation as musical success.
