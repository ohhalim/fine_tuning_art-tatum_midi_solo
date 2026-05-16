# Inference and Model Specification

작성일: 2026-05-16

## 1. 목표

Python inference service는 structured musical request를 받아 valid MIDI와 metrics를 반환한다.

핵심 목표:

- output MIDI가 항상 valid하거나 명확히 실패한다.
- model output이 실패하면 fallback phrase를 생성한다.
- metrics로 결과를 설명한다.

## 2. Generator Priority

MVP generator 우선순위:

1. 기존 `scripts/generate.py` 기반 LoRA Music Transformer.
2. 실패 시 rule/template fallback generator.
3. 나중에 control-token model로 개선.

`music_transformer/generate.py`는 legacy generator로 보고 MVP inference의 직접 entrypoint로 사용하지 않는다.

## 3. Request to Conditioning 변환

초기 MVP는 chord progression을 완전한 모델 condition token으로 학습하지 않았을 수 있다. 따라서 두 경로를 둔다.

### Path A: Existing Stage A Model

- `conditioning_midi`가 명시되면 기존 Stage A 방식으로 해당 MIDI를 primer로 사용.
- `conditioning_midi`가 없으면 request의 `chord_progression`, `bpm`, `bars`, `time_signature`를 low-register chord guide MIDI로 변환해 primer로 사용.
- chord progression은 전체 phrase duration에 균등 배치한다. 예를 들어 `bars=2`와 코드 4개가 들어오면 각 코드는 half-bar 구간을 담당한다.
- `primer_max_tokens=64` 기본.
- `temperature`, `top_k`, `top_p`를 model sampling에 전달한다.
- `model_candidates` 개수만큼 후보 MIDI를 생성하고, repair/metrics gate를 통과한 후보 중 dead-air, repetition, target density 이탈, 낮은 chord-tone ratio penalty를 기준으로 가장 낮은 score의 후보를 최종 선택한다.
- MVP inference 기본 `max_sequence`는 256이다. 기존 512-token 생성은 더 느린 Stage A 비교/실험값으로 유지한다.
- LoRA checkpoint:
  - `checkpoints/jazz_lora_stage_a`

### Path B: Rule-Based Fallback

모델 사용 불가 또는 output 실패 시 request field로 phrase를 만든다.

규칙:

- BPM과 bars로 phrase duration 계산.
- chord root/quality를 파싱한다.
- density에 따라 notes per bar를 정한다.
- energy에 따라 register, velocity, rhythmic subdivision을 조정한다.
- section에 따라 phrase start/end를 조정한다.

## 4. Density Mapping

초기값:

| Density | Notes per bar |
|---|---:|
| sparse | 3~5 |
| medium | 6~10 |
| dense | 10~16 |

## 5. Energy Mapping

초기값:

| Energy | Register | Velocity | Rhythm |
|---|---|---|---|
| low | mid | 55~75 | longer notes |
| mid | mid-high | 65~90 | mixed eighth/sixteenth |
| high | high | 80~110 | more sixteenth motion |

## 6. Piano Range

MVP range:

- hard clamp: MIDI 21~108
- preferred solo range: MIDI 48~88
- avoid too much low register in dance context: keep most notes above MIDI 48

## 7. Post-Processing

Before saving:

- remove notes with non-positive duration.
- clamp pitch range.
- clamp velocity 1~127.
- quantize starts to 16th grid unless model output already has expressive timing.
- limit max simultaneous notes.
- remove extreme overlaps of same pitch.
- ensure MIDI has tempo.
- ensure duration roughly matches requested bars.

## 8. Metrics

Required:

- `generationTimeMs`
- `noteCount`
- `durationSec`
- `noteDensity`
- `deadAirRatio`
- `repetitionScore`
- `pitchMin`
- `pitchMax`
- `fallbackUsed`
- `chordToneRatio`
- `chordToneCount`
- `nonChordToneCount`

Optional:

- `scaleToneRatio`
- `barBoundaryError`
- `avgVelocity`

`chordToneRatio` is computed as a pitch-class hit ratio against the request chord that is active at each note start time. It contributes a weak candidate-selection penalty below `0.55`, but it is not an acceptance gate in the current MVP.

## 9. Failure Detection

Fail if:

- MIDI file was not written.
- MIDI cannot be read by `pretty_midi`.
- note count is 0.
- duration is 0.
- medium/dense request has near-zero note density.
- pitch range is invalid after post-processing.

Fallback if:

- model checkpoint missing.
- model generation raises exception.
- model output has no decodable tokens.
- generated MIDI fails accept gate.

## 10. Result Contract

```json
{
  "jobId": "uuid",
  "status": "COMPLETED",
  "midiPath": "outputs/generated/uuid.mid",
  "metricsPath": "outputs/metrics/uuid.json",
  "fallbackUsed": false,
  "conditioning_midi_path": "outputs/generated/_conditioning/uuid_conditioning.mid",
  "metrics": {
    "generationTimeMs": 842,
    "noteCount": 43,
    "durationSec": 3.87,
    "noteDensity": 11.11,
    "deadAirRatio": 0.12,
    "repetitionScore": 0.08,
    "pitchMin": 48,
    "pitchMax": 84
  }
}
```

## 11. Later Model Improvements

After MVP works:

- explicit control tokens:
  - `STYLE`
  - `SECTION`
  - `ENERGY`
  - `DENSITY`
  - `BPM`
  - `CHORD`
  - `BAR`
  - `POSITION`
- role-conditioned training format.
- KV cache for lower latency.
- phrase memory / retrieval.
- realtime 1-bar lookahead queue.
