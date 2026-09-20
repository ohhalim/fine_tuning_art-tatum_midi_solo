# 재개 문서 — Phase 1 실시간 MIDI 런타임

작성 2026-09-20. 이 문서 하나만 읽으면 재개할 수 있게 쓴다.
`quality_claimed: false` — 아래 수치는 전부 능력 측정이고 음악 품질 주장이 아니다.

## 1. 완료 범위

| 항목 | 상태 |
|---|---|
| 일괄 생성 후 재생 (`run_jazz_mvp.py`) | 보존. 회귀 없음 |
| **연속 경로** (`run_continuous_jazz.py`) | 재생 중 다음 마디 생성 |
| 연주 입력 → 다음 마디 primer | 동작. 가상 CoreMIDI 키보드로 검증 |
| 독립 CoreMIDI 캡처 | 동작 (`--capture`) |
| 한 줄 데모 | `run_mvp_demo.sh`, `run_continuous_demo.sh` |
| Phase 1 지연 측정 | 생성 구간 + 스케줄러 + 전송. **왕복 종단 측정은 부분적** |

**하지 않은 것**: 멜다우 적응, 새 학습, 장시간 soak, chord conditioning,
D6 번호 사용.

## 2. 실행 명령

```bash
# 한 줄 데모 (모델 없으면 fallback 경로)
scripts/run_continuous_demo.sh

CK=<...>/outputs/d1_experiment/armD_lora/ckpt/checkpoint_epoch8.pt
PR=<...>/data/roles/lead/000002/conditioning.mid

# 모델로 연속 재생 + 독립 캡처
CHECKPOINT=$CK PRIMER=$PR scripts/run_continuous_demo.sh

# 연주 입력까지 (INPUT_PORT 는 실제 포트 이름)
CHECKPOINT=$CK PRIMER=$PR INPUT_PORT="내 키보드" scripts/run_continuous_demo.sh

# 직접 실행
FORCE_CPU=1 uv run python scripts/run_continuous_jazz.py \
    --checkpoint $CK --conditioning-midi $PR \
    --bars 8 --bpm 128 --seed 42 --capture \
    --input-port "내 키보드" --output-dir outputs/continuous/run

# 테스트
PYTHON_BIN=<venv python> bash scripts/agent_harness.sh quick   # 138 tests
```

`FORCE_CPU=1` 은 권장값이다. 13.7M 모델에서 CPU 가 MPS 보다 빨랐다
(§4). 산출물: `continuous_report.json`, `played.mid`.

## 3. 실제 측정값

### 왕복 4회 연속 (8마디씩, 가상 키보드 → 모델 → 가상 포트 → 독립 캡처)

| run | 완주 | model | 오류 | 생성 p50 | 생성 최대 | 캡처 | 손실 | 데드라인 미스 | 지각 최대 |
|---|---|---|---|---|---|---|---|---|---|
| 10 | ✅ | 8/8 | 0 | 360ms | 501ms | 102/102 | 0 | 0 | 14.7ms |
| 11 | ✅ | 8/8 | 0 | 341ms | 631ms | 114/114 | 0 | 0 | 6.4ms |
| 12 | ✅ | 8/8 | 0 | 482ms | 919ms | 172/172 | 0 | 0 | 12.9ms |
| 13 | ✅ | 8/8 | 0 | 456ms | 829ms | 144/144 | 0 | 0 | 18.5ms |

합산 32마디: **오류 0**, 생성 p50 337ms / 최대 919ms, **마디(1875ms) 초과 0/32**.

이전 왕복 실행: 입력 128 수신, live primer 7/8, `input_to_ready_ms` p50 **413.8ms**,
`scheduled_to_capture_ms` p50 **3.49ms**.

### 지연 항목은 섞지 말 것

| 필드 | 의미 |
|---|---|
| `generation_ms` | 모델 시간만 |
| `input_to_ready_ms` | 입력 도달 → 생성 완료. **반응 속도로 읽을 값** |
| `input_to_bar_start_ms` | 위 + 마디 그리드 대기. p50 3.8초인데 **모델 속도가 아니다** |
| `scheduled_to_capture_ms` | 예정 재생 시각 → 독립 입력 관측. 출력 경로만 |

구간 값은 서로 다른 실행에서 나왔으므로 **더하면 종단 지연이 되지 않는다.**

## 4. 실패한 실행 (숨기지 않고 기록)

| 실행 | 증상 | 원인 | 처리 |
|---|---|---|---|
| rep3 | model 7/8, bar1 늦어 fallback | bar0+bar1 생성 합 1036ms > start_delay 1.0s | start_delay 2.5s |
| **rep4** | **run_completed False, 2마디에서 중단** | 25.0ms 지각 1회 + `abort_on_first_miss` 정책 | 연속 경로 기본값을 `record_and_continue` 로 |
| rep4 | `bar1: fallback_not_started` | **내 첫 수정이 무효였다.** `max_lead_bars=1` 에서 bar1 은 `get(0)`(=`run()` 진입) 전에는 생성을 시작조차 못 한다. 기다려도 헛돈다 | `max_lead_bars=2` |
| rep8 | model 6/8 | bar0 노트 0개(온마디 쉼표), bar5 밀집 마디 50ms 부족 | 쉼표는 `allow_rest_bar` 로 연주 가능하게. 밀집 부족은 남음 |
| 라이브 primer 초기 | model 4/8, underfill 4 | `target_length=80` 절대값이라 primer 가 길수록 생성 여유 감소 | `len(primer) + generation_tokens` |
| 라이브 primer | 무음 노트 | 절단이 carried velocity 파괴 (primer velocity bins = []) | `truncate_tokens_preserving_velocity` |

### 기각/정정한 가설

- **KV 캐시**: "효과 없음 입증" 이 아니라 **"현재 설정에서 기대 이득이 작아
  우선순위 낮춤"**. 단일 forward 비용이 길이 8~80 에서 평평(16~21ms)하지만,
  마이크로벤치는 실제 생성 경로를 포함하지 않는다.
- **grammar mask velocity bin 0**: 한 번 "효과 없음" 으로 기각했다가 **되살렸다.**
  측정 자체는 맞았지만 일반화가 틀렸다. 고정 primer 에서는 조건이 발생하지
  않았을 뿐, 연주 입력 primer 에서는 실제로 나온다.
- **지연 외삽 137%/46%**: 틀렸다. `target_seq_length=80` 은 primer 포함이라
  신규 forward 상한은 48 이고 이미 포화였다. 실측 최대로 교체
  (MPS 2237ms=119%, CPU 545ms=29%).

## 5. 미검증 (주장하지 않는 것)

- **실물 하드웨어 키보드.** 위 "키보드" 는 같은 머신의 CoreMIDI 가상 소스다.
- **FL Studio / DAW 오디오.** `fl_studio_audio_observed: false`. WAV 는
  사운드폰트를 못 찾아 `pretty_midi` 사인 합성으로 렌더링했다.
- **음악적 품질.** 청취 리뷰 없음. model 마디 29개(노트 111개)의 chord-tone
  비율 **0.3153 vs 무작위 기대 0.3333, z=−0.40 — 우연과 구별 불가.**
- **chord conditioning.** `model_chord_conditioning: false`. 코드 진행은
  fallback 과 백킹에만 쓰이고 모델 입력에 들어가지 않는다.
- **장시간 soak.** 8~16 마디 단발만. 프로브들은 `wall_clock_soak_completed: false`
  라 게이트 통과가 아니다.
- **부하 상태.** DAW·신스 동시 구동 조건 미측정.

## 6. 다음 한 작업

**코드 진행을 모델 입력에 넣는다 (chord conditioning).**

지금 chord-tone 이 우연 수준인 이유가 이것이다. `build_live_primer` 가 이미
조건 토큰을 만드는 자리이므로, `control_prefix_tokens` 에 코드 심볼을 더하는
형태가 자연스럽다. 판정 기준을 **실험 전에** 등록할 것:
코드 진행을 바꿨을 때 chord-tone 비율이 우연(0.3333) 위로 유의하게 올라가는가.

## 7. 미해결 설계 제한

1. **입력 반영이 마디 경계 단위다.** 마디 중간 연주는 그 마디에 반영되지
   않는다. 턴테이킹에 가깝고 동시 반응이 아니다.
2. **밀집 마디 토큰 부족.** 29노트 마디가 1830/1880ms 로 끝났다.
   `--generation-tokens` 를 올리면 줄지만 지연이 함께 오른다.
3. **선행 깊이 2 가 입력 신선도를 깎는다.** bar N+1 을 미리 만들수록 반영되는
   입력이 오래된 것이 된다. 지연과 반응성의 맞바꿈이다.
4. **기기 간 재현성 없음.** CPU/MPS 부동소수점 차이로 같은 seed 에서도
   결과가 갈린다(model 마디 CPU 28/32 vs MPS 29/32).
5. **`run_jazz_mvp.py` 와 `run_continuous_jazz.py` 가 일부 중복.**
   의도적으로 분리했다(기존 경로 보존). 코드 진행 작업 후 합칠지 판단할 것.
6. **온마디 쉼표는 opt-in 이다.** 기본 경로는 여전히 거부한다. 게이트 걸린
   컴포넌트라 기본값을 바꾸지 않았다.
