# API Specification

작성일: 2026-05-16

상태: 모델 MVP 기준 문서. Spring Boot API는 이번 MVP 범위가 아니며, 필요할 때 포트폴리오 확장으로 붙인다.

## 1. Current MVP Interfaces

현재 MVP의 primary interface는 Python CLI다.

```bash
python -m inference.app.generator \
  --bpm 124 \
  --chords Cm7,Fm7,Bb7,Ebmaj7 \
  --bars 2 \
  --section drop \
  --energy high \
  --density medium \
  --output_dir outputs/generated
```

출력:

```json
{
  "job_id": "uuid",
  "status": "COMPLETED",
  "midi_path": "outputs/generated/uuid.mid",
  "metrics_path": "outputs/metrics/uuid.json",
  "fallback_used": false,
  "model_repaired": true,
  "metrics": {
    "generation_time_ms": 15027,
    "note_count": 5,
    "duration_sec": 3.87,
    "note_density": 1.29,
    "dead_air_ratio": 0.75,
    "repetition_score": 0.0,
    "pitch_min": 63,
    "pitch_max": 70,
    "fallback_used": false
  }
}
```

## 2. Optional FastAPI Wrapper

FastAPI는 CLI generation contract를 감싸는 얇은 wrapper다. 모델 MVP 검증에는 필수는 아니지만, 외부 호출 실험용으로 유지한다.

### Health

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Generate MIDI

```http
POST /infer/midi
Content-Type: application/json
```

Request:

```json
{
  "jobId": "6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c",
  "bpm": 124,
  "timeSignature": "4/4",
  "key": "C minor",
  "chordProgression": ["Cm7", "Fm7", "Bb7", "Ebmaj7"],
  "bars": 2,
  "section": "drop",
  "energy": "high",
  "density": "medium",
  "style": "personal_jazz",
  "temperature": 0.9,
  "topP": 0.95,
  "modelCandidates": 2,
  "useModel": true,
  "seed": 42
}
```

Response:

```json
{
  "jobId": "6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c",
  "status": "COMPLETED",
  "midiPath": "outputs/generated/6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c.mid",
  "metricsPath": "outputs/metrics/6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c.json",
  "conditioningMidiPath": "outputs/generated/_conditioning/6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c_conditioning.mid",
  "fallbackUsed": false,
  "modelRepaired": true,
  "metrics": {
    "generationTimeMs": 15027,
    "noteCount": 5,
    "durationSec": 3.87,
    "noteDensity": 1.29,
    "deadAirRatio": 0.75,
    "repetitionScore": 0.0,
    "pitchMin": 63,
    "pitchMax": 70,
    "fallbackUsed": false
  }
}
```

## 3. Deferred Spring Boot Public API

아래 내용은 이번 모델 MVP가 끝난 뒤, backend portfolio 확장이 필요할 때만 사용한다.

Base path:

```text
/api
```

## 2. Create Generation Job

```http
POST /api/generation-jobs
Content-Type: application/json
```

Request:

```json
{
  "bpm": 124,
  "timeSignature": "4/4",
  "key": "C minor",
  "chordProgression": ["Cm7", "Fm7", "Bb7", "Ebmaj7"],
  "bars": 2,
  "section": "drop",
  "energy": "high",
  "density": "medium",
  "style": "personal_jazz",
  "temperature": 0.9,
  "topK": 40,
  "topP": 0.95
}
```

Response:

```json
{
  "jobId": "6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c",
  "status": "PENDING",
  "createdAt": "2026-05-16T14:00:00+09:00"
}
```

Validation:

- `bpm`: 40~240
- `bars`: 1~4 for MVP
- `chordProgression`: non-empty
- `section`: `intro`, `build`, `breakdown`, `drop`
- `energy`: `low`, `mid`, `high`
- `density`: `sparse`, `medium`, `dense`

## 3. Get Generation Job

```http
GET /api/generation-jobs/{jobId}
```

Response:

```json
{
  "jobId": "6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c",
  "status": "COMPLETED",
  "createdAt": "2026-05-16T14:00:00+09:00",
  "startedAt": "2026-05-16T14:00:01+09:00",
  "completedAt": "2026-05-16T14:00:02+09:00",
  "resultMidiPath": "outputs/generated/6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c.mid",
  "fallbackUsed": false,
  "metrics": {
    "generationTimeMs": 842,
    "noteCount": 43,
    "noteDensity": 21.5,
    "deadAirRatio": 0.12,
    "repetitionScore": 0.08,
    "pitchMin": 48,
    "pitchMax": 84,
    "durationSec": 3.87,
    "chordToneCount": 31,
    "nonChordToneCount": 12,
    "chordToneRatio": 0.72
  }
}
```

## 4. List Generation Jobs

```http
GET /api/generation-jobs?page=0&size=20
```

Response:

```json
{
  "items": [
    {
      "jobId": "6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c",
      "status": "COMPLETED",
      "bpm": 124,
      "bars": 2,
      "section": "drop",
      "energy": "high",
      "density": "medium",
      "createdAt": "2026-05-16T14:00:00+09:00"
    }
  ],
  "page": 0,
  "size": 20,
  "total": 1
}
```

## 5. Download MIDI

```http
GET /api/generation-jobs/{jobId}/download
```

Response:

- `200 OK`
- `Content-Type: audio/midi` or `application/octet-stream`
- `Content-Disposition: attachment; filename="<jobId>.mid"`

Errors:

- `404` if job not found.
- `409` if job is not completed.
- `404` if MIDI file is missing.

## 6. Job Status Enum

```text
PENDING
RUNNING
COMPLETED
FAILED
```

## 9. Legacy Python FastAPI Inference API Draft

Base path:

```text
/
```

### Health

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Generate MIDI

```http
POST /infer/midi
Content-Type: application/json
```

Request:

```json
{
  "jobId": "6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c",
  "bpm": 124,
  "timeSignature": "4/4",
  "key": "C minor",
  "chordProgression": ["Cm7", "Fm7", "Bb7", "Ebmaj7"],
  "bars": 2,
  "section": "drop",
  "energy": "high",
  "density": "medium",
  "style": "personal_jazz",
  "temperature": 0.9,
  "topK": 40,
  "topP": 0.95
}
```

Response:

```json
{
  "jobId": "6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c",
  "status": "COMPLETED",
  "midiPath": "outputs/generated/6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c.mid",
  "metricsPath": "outputs/metrics/6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c.json",
  "fallbackUsed": false,
  "metrics": {
    "generationTimeMs": 842,
    "noteCount": 43,
    "noteDensity": 21.5,
    "deadAirRatio": 0.12,
    "repetitionScore": 0.08,
    "pitchMin": 48,
    "pitchMax": 84,
    "durationSec": 3.87
  }
}
```

Failure response:

```json
{
  "jobId": "6a9f6da7-0be5-4c99-a0bf-ff21696f6d6c",
  "status": "FAILED",
  "errorCode": "GENERATION_FAILED",
  "message": "Model output was empty and fallback generation failed."
}
```
