# Phase 1 — 연속 재생 경로

작성 2026-09-20. `scripts/run_continuous_jazz.py`.

`quality_claimed: false` — 능력 측정이고 음악적 품질 주장이 아니다.

## 무엇이 다른가

`run_jazz_mvp.py` 는 **전부 생성한 뒤 재생한다.** 그 경로는 그대로 두었다.
이 경로는 **재생하면서 생성한다.**

```
producer 스레드        마디 N+1 생성          (모델 실행은 여기서만)
      ↓ ready dict
scheduler 스레드       마디 N 디스패치        (dict 읽기만, 블로킹 없음)
      ↓ 가상 포트
독립 CoreMIDI 입력     --capture 시 관측
```

`OneBarMidiScheduler.run` 은 `blocks.get(bar_index)` 만 호출하므로,
**스케줄러를 고치지 않고** live view 를 끼워 넣었다.

## 설계 규칙

| 규칙 | 이유 |
|---|---|
| 스케줄러 스레드에서 모델 실행 금지 | `get` 이 블로킹하면 마디 경계를 놓친다 |
| 늦은 결과 폐기 | 이미 지나간 입력을 조건으로 만든 블록이다 |
| fallback 은 **미리** 만든다 | `get` 안에서 만들면 그것도 지연이 된다 |
| 생성 예외는 기록만 | producer 스레드가 죽으면 이후 전 마디가 무음이 된다 |
| 선행 생성 상한(`max_lead_bars`) | 무제한이면 입력 반영이 늦어진다 |
| 입력 콜백은 타임스탬프·적재만 | 콜백에서 일하면 CoreMIDI 가 막힌다 |

## 측정 (16마디 @128BPM, `FORCE_CPU=1`, seed 7)

| | |
|---|---|
| model 마디 | **16/16** (fallback 0, 늦은 폐기 0, 오류 0) |
| 생성 | p50 **103.3ms**, 최대 **188.8ms** (마디 1875ms) |
| 캡처 노트 이벤트 | **82/82** — 손실 0, 중복 0, 순서 오류 0 |
| scheduled → capture | p50 **4.20ms**, 최대 **7.20ms** |
| queue underrun / 데드라인 미스 / 전송 실패 | 0 / 0 / 0 |

### 지연 항목을 섞지 말 것

리포트는 네 가지를 **따로** 적는다.

| 필드 | 무엇 |
|---|---|
| `generation_ms` | 모델 시간만 |
| `input_to_ready_ms` | 입력 도달 → 생성 완료 |
| `input_to_bar_start_ms` | 위 + **마디 그리드 대기(lookahead)**. 마디 길이에 묶이므로 모델 속도로 읽으면 안 된다 |
| `scheduled_to_capture_ms` | 예정 재생 시각 → 독립 입력이 관측한 시각. 출력 경로만 |

합치면 왕복이 되지 않는다. 입력 구간이 비어 있다.

### 캡처 비교에서 걸렸던 것

첫 실행은 `duplicate_output_count: 48` 로 나왔다. 48 은 종료 시
`reset()`+`panic()` 이 내보내는 control_change 수다. 스케줄러 `records` 에는
없는 트래픽이라 전부 중복으로 잡혔다. **노트 이벤트만 비교**하도록 고쳤다.

## 아직 아닌 것

- **왕복이 아니다.** 입력 이벤트는 수신·타임스탬프까지만 하고
  **primer 에 반영하지 않는다.** `--input-port` 로 받은 이벤트는 개수만
  기록된다. 따라서 `input_to_ready_ms` 는 아직 항상 null 이다.
- **실물 키보드 미검증.** 외부 장비로 시험하지 않았다.
- **FL Studio / DAW 오디오 미검증.** 가상 포트로만 냈다.
- **음악적 품질 미검증.** 청취 리뷰 없음. chord conditioning 없음
  (`model_chord_conditioning: false`).
- **장시간 soak 미실시.** 8~16 마디 단발만 돌렸다.
- **부하 상태 미측정.** DAW·신스 동시 구동 조건에서 재측정 필요.

## 실행

```
# 가상 포트로 내면서 독립 입력으로 캡처
FORCE_CPU=1 uv run python scripts/run_continuous_jazz.py \
    --checkpoint <armD_lora/ckpt/checkpoint_epoch8.pt> \
    --conditioning-midi <data/roles/lead/000002/conditioning.mid> \
    --bars 16 --bpm 128 --seed 7 --capture \
    --output-dir outputs/continuous/final

# 기존 MIDI 출력 포트로 (DAW 등)
FORCE_CPU=1 uv run python scripts/run_continuous_jazz.py ... --port "<포트 이름>"

# 모델 없이 fallback 경로만
uv run python scripts/run_continuous_jazz.py --fallback-only --bars 8 \
    --output-dir outputs/continuous/fallback
```

산출물: `continuous_report.json`, `played.mid`
(`played.mid` 는 생성된 블록이 아니라 **스케줄러가 실제로 디스패치한 기록**으로
만든다. fallback 으로 나간 마디도 나간 그대로 남는다.)
