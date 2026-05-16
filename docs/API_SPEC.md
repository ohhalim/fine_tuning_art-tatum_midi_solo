# API Specification

작성일: 2026-05-16

## 1. Spring Boot Public API

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
    "durationSec": 3.87
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

## 7. Python FastAPI Inference API

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
