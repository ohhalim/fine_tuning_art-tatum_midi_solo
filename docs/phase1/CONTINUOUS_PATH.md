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

## 라이브 primer 에서 드러난 결함 세 가지

고정 primer 로는 보이지 않다가 연주 입력을 넣자 드러났다.

| 결함 | 증상 | 수정 |
|---|---|---|
| 토큰 예산이 절대값 | primer 가 길어지면 생성 여유가 줄어 duration underfill (1130~1640ms vs 1880ms) | `target_length = len(primer) + generation_tokens` |
| velocity bin 0 샘플링 | 블록 중간에 무음 노트 | grammar mask 에 bin 0 추가 |
| 절단이 velocity 상태 파괴 | primer velocity bins 가 빈 리스트 → carry 할 것이 없어 무음 | `truncate_tokens_preserving_velocity` |

세 번째가 가장 조용한 버그였다. 일정한 세기로 연주하면 velocity 토큰이
맨 앞에 하나만 나오는데, tail 절단이 그것을 통째로 버린다.

| | model 마디 | 오류 |
|---|---|---|
| 수정 전 | 4/8 | 4 |
| 수정 후 | **8/8** | **0** |

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

## 왕복 측정 (가상 키보드 → 모델 → 가상 포트 → 독립 캡처)

별도 프로세스가 CoreMIDI 가상 소스를 열어 250ms 간격으로 연주하고,
세션이 `--input-port` 로 그것을 받아 다음 마디 primer 로 쓴다.

| | |
|---|---|
| 입력 이벤트 수신 | **128** (실제 CoreMIDI 입력 포트 경유) |
| live primer 사용 마디 | **7/8** (마디 0 은 아직 입력 전) |
| model 마디 | **8/8** (fallback 0, 늦은 폐기 0, 오류 0) |
| 생성 | p50 **318.7ms**, 최대 **805.1ms** |
| `input_to_ready_ms` | p50 **413.8ms**, 최대 840.9ms |
| `input_to_bar_start_ms` | p50 **3799.4ms** — 대부분 마디 그리드 대기다 |
| 캡처 | **134/134**, 손실 0, 중복 0, 순서 오류 0 |
| `scheduled_to_capture_ms` | p50 **3.49ms** |
| underrun / 데드라인 미스 | 0 / 0 |

`input_to_bar_start_ms` 가 3.8 초인 것은 모델이 느려서가 아니다. 입력을
반영한 마디는 **다음 마디 경계**에 재생되므로 마디 길이(1875ms)의 배수가
그대로 들어간다. 반응 속도로 읽어야 할 값은 `input_to_ready_ms` 다.

## 아직 아닌 것

- **실물 키보드 미검증.** 위 측정의 "키보드" 는 같은 머신에서 띄운
  CoreMIDI 가상 소스다. 외부 하드웨어로 시험하지 않았다.
- **FL Studio / DAW 오디오 미검증.** 가상 포트로만 냈다.
- **음악적 품질 미검증.** 청취 리뷰 없음. chord conditioning 없음
  (`model_chord_conditioning: false`).
- **장시간 soak 미실시.** 8~16 마디 단발만 돌렸다.
- **입력 반영은 마디 경계 단위다.** 마디 중간에 들어온 연주는 그 마디에
  반영되지 않는다. 턴테이킹에 가깝고 동시 반응이 아니다.
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

# 연주 입력을 받아 다음 마디 primer 로 반영
FORCE_CPU=1 uv run python scripts/run_continuous_jazz.py ... \
    --input-port "<키보드 포트 이름>" --capture

# 모델 없이 fallback 경로만
uv run python scripts/run_continuous_jazz.py --fallback-only --bars 8 \
    --output-dir outputs/continuous/fallback
```

산출물: `continuous_report.json`, `played.mid`
(`played.mid` 는 생성된 블록이 아니라 **스케줄러가 실제로 디스패치한 기록**으로
만든다. fallback 으로 나간 마디도 나간 그대로 남는다.)
