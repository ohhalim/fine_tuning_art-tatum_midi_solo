# Phase 1 — 구간별 지연 예산

측정일 2026-09-20. CLAUDE.md §6 질문 1 "왕복 루프의 경계를 어디로 자를 것인가 —
구간별 지연 예산을 먼저 정한다" 에 대한 **측정 기반** 답이다.

`quality_claimed: false` — 능력 측정이고 음악적 품질 주장이 아니다.

## 요약

**예산의 거의 전부가 생성이다. 전송과 스케줄링은 반올림 오차 수준이다.**

| 구간 | p50 | p99 | 최대 | 출처 |
|---|---|---|---|---|
| **모델 생성** (CPU) | **58ms** | — | **545ms** | `run_jazz_mvp.py`, 32마디 |
| 스케줄러 디스패치 지각 | 0.014ms | 0.59ms | 1.31ms | `run_internal_scheduler_probe.py` |
| 디스패치 → 캡처 | 0.47ms | 4.39ms | 4.40ms | 같음 |
| MIDI 전송 → 캡처 (독립) | 0.31ms | 1.60ms | 2.22ms | `run_coremidi_virtual_loopback_probe.py` |
| MIDI 입력 수신 | **미측정** | | | |
| DAW 수신 / 오디오 | **미측정** | | | |

측정된 출력 경로 합: p50 ≈ **59ms**, 최악 관측 ≈ **551ms**.
생성이 나머지 전부보다 약 **100배** 크다.

## 함의

1. **최적화 대상은 생성 하나뿐이다.** 스케줄러나 전송을 손대는 것은 의미가 없다.
   CLAUDE.md §5 층 1 이 "4스레드 구조, KV 캐싱, 양자화를 베낀다" 라고 했지만,
   우리 규모에서 그 최적화들이 겨냥하는 구간은 이미 무시할 수 있다.

2. **생성 지연은 토큰 수에 선형이다.** 따라서 마디 예산 위반은 확률적 사고가
   아니라 **토큰 예산으로 결정론적으로 통제할 수 있다.** 자세한 값은
   [GENERATION_LATENCY.md](GENERATION_LATENCY.md).

3. **여유는 마디의 절반 이상이다.** 128 BPM 마디 1875ms 기준, CPU 에서 토큰
   예산 80 을 다 써도 859ms(46%). 남는 예산을 입력 수신과 DAW 왕복에 쓸 수 있다.

## 무결성 (지연과 별개)

두 프로브 모두 이벤트 손실 없음:

| | 이벤트 | 손실 | 중복 | 순서 오류 | stuck note | 데드라인 미스 |
|---|---|---|---|---|---|---|
| 스케줄러 (8마디 @128BPM) | 112/112 | 0 | 0 | 0 | 0 | 0 |
| CoreMIDI 루프백 (3s @50Hz) | 148/148 | 0 | 0 | 0 | 0 | — |

스케줄러 큐 underrun 0, 최대 큐 깊이 1, catch-up 디스패치 0.

## 이 수치가 **아닌** 것

- **게이트 통과가 아니다.** 두 프로브 모두 `passed_*_gate: false` 를 반환한다.
  짧은 duration 으로 돌려 `wall_clock_soak_completed: false` 이기 때문이다.
  능력 측정이지 게이트 판정이 아니다.
- **왕복이 아니다.** 입력(키보드 → 프로세스)과 DAW 수신이 빠져 있다.
  이 둘을 재야 Phase 1 게이트에 답할 수 있다.
- **연속 루프가 아니다.** 생성 수치는 마디별 독립 생성을 일괄 실행한 값이다.
  롤링 루프에서 생성과 재생이 겹칠 때의 경합은 반영돼 있지 않다.
- **부하 상태가 아니다.** DAW·신스가 같이 도는 상태에서 재측정해야 한다.

## 다음에 재야 할 것

1. MIDI 입력 수신 구간 (외부 키보드 → 프로세스 타임스탬프)
2. 롤링 루프에서 생성·재생 동시 실행 시 생성 지연 재측정
3. DAW 수신 확인 — 현재 `fl_studio_audio_observed: false`

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
