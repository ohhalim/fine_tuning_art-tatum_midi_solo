# Phase 1 — 구간별 지연 측정

측정일 2026-09-20. CLAUDE.md §6 질문 1 "왕복 루프의 경계를 어디로 자를 것인가"
에 대한 **부분적** 답이다. 왕복 지연 자체는 아직 재지 못했다.

`quality_claimed: false` — 능력 측정이고 음악적 품질 주장이 아니다.

## ⚠️ 이 표는 종단 지연이 아니다

아래 값들은 **서로 다른 시점에 독립적으로 돌린 프로세스**에서 나왔다.
같은 이벤트가 각 구간을 통과한 기록이 아니므로 **더하면 안 된다.**
합계는 종단 실측값이 아니고, 단일 요청이 실제로 겪는 지연도 아니다.

구간을 나란히 놓는 목적은 **자릿수 비교**뿐이다.

| 구간 | p50 | p99 | 최대 | 출처 (별개 실행) |
|---|---|---|---|---|
| **모델 생성** (CPU) | **58ms** | — | **545ms** | `run_jazz_mvp.py`, 32마디 |
| 스케줄러 디스패치 지각 | 0.014ms | 0.59ms | 1.31ms | `run_internal_scheduler_probe.py` |
| 디스패치 → 캡처 | 0.47ms | 4.39ms | 4.40ms | 같은 실행 |
| MIDI 전송 → 캡처 (독립) | 0.31ms | 1.60ms | 2.22ms | `run_coremidi_virtual_loopback_probe.py` |
| MIDI 입력 수신 | **미측정** | | | |
| DAW 수신 / 오디오 | **미측정** | | | |

생성 구간 상세는 [GENERATION_LATENCY.md](GENERATION_LATENCY.md).

## 관측

1. **자릿수가 크게 다르다.** 생성은 수십~수백 ms, 스케줄러·전송은 1ms 안팎
   이다. 두 자릿수 이상 차이 난다.

   따라서 **현재 측정한 조건에서는** 지연을 줄이려 할 때 생성 구간이 가장
   큰 항이다. 다만 이것을 "다른 최적화는 무의미하다" 로 읽지 말 것 —
   미측정 구간(입력 수신, DAW 수신)이 두 개 남아 있고, 부하 상태와 연속
   루프에서는 스케줄러·전송 구간도 달라질 수 있다.

2. **무결성은 두 프로브 모두 깨끗했다.**

| | 이벤트 | 손실 | 중복 | 순서 오류 | stuck note | 데드라인 미스 |
|---|---|---|---|---|---|---|
| 스케줄러 (8마디 @128BPM) | 112/112 | 0 | 0 | 0 | 0 | 0 |
| CoreMIDI 루프백 (3s @50Hz) | 148/148 | 0 | 0 | 0 | 0 | — |

   스케줄러 큐 underrun 0, 최대 큐 깊이 1, catch-up 디스패치 0.

## 이 수치가 **아닌** 것

- **게이트 통과가 아니다.** 두 프로브 모두 `passed_*_gate: false` 를 반환한다
  (`wall_clock_soak_completed: false` — 짧은 duration 으로 돌렸다).
  능력 측정이지 게이트 판정이 아니다.
- **왕복이 아니다.** 입력(키보드 → 프로세스)과 DAW 수신이 빠져 있다.
- **종단 합계가 아니다.** 위 경고 참고.
- **연속 루프가 아니다.** 생성 수치는 마디별 독립 생성을 일괄 실행한 값이다.
- **부하 상태가 아니다.** DAW·신스가 같이 도는 상태에서 재측정해야 한다.
- **실물 키보드 / FL Studio / 오디오 검증 없음.**
  `fl_studio_audio_observed: false`.

## 다음에 재야 할 것

1. MIDI 입력 수신 구간 (외부 키보드 → 프로세스 타임스탬프)
2. 연속 루프에서 생성·재생 동시 실행 시 재측정
3. 같은 이벤트를 끝까지 따라가는 **종단** 측정 (구간 합이 아니라)
4. DAW 수신 확인

## 재현

```
uv run python scripts/run_internal_scheduler_probe.py \
    --run_id phase1_scheduler --bpm 128 --duration_seconds 15 \
    --output_root outputs/phase1
uv run python scripts/run_coremidi_virtual_loopback_probe.py \
    --run_id phase1_transport --duration_seconds 3.0 --rate_hz 50 \
    --output_root outputs/phase1
FORCE_CPU=1 uv run python scripts/run_jazz_mvp.py \
    --checkpoint <armD_lora/ckpt/checkpoint_epoch8.pt> \
    --conditioning-midi <data/roles/lead/000002/conditioning.mid> \
    --bars 8 --bpm 128 --seed 42 --output-dir outputs/jazz_mvp/cpu_seed_42
```

## 정정 이력

- 초판은 구간 p50/최대를 더해 "측정된 출력 경로 합 p50 ≈ 59ms, 최악 ≈ 551ms"
  라고 적었다. **서로 다른 실행의 값이라 더할 수 없다.** 삭제했다.
- 초판은 "최적화 대상은 생성 하나뿐이고 스케줄러나 전송을 손대는 것은 의미가
  없다" 고 단정했다. 미측정 구간이 남아 있으므로 단정을 철회한다.
- 초판은 "마디 예산 위반을 토큰 예산으로 결정론적으로 통제할 수 있다" 고
  적었다. OS 지연은 그렇게 보장되지 않는다. 철회한다.
