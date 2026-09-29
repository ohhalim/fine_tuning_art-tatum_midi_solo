# 연주 중 어댑터 선택 — Program Change / CC (사전 등록)

작성 2026-09-29. 이슈 #1525. `style_verified: false` · `musical_quality_verified: false` · 가상 CoreMIDI 포트, 실물 키보드 없음

**이 절은 측정 전에 작성했다.** 구현은 측정 전에 끝냈다.

## 목적
어댑터 스왑(#1519)은 고정 스케줄로만 바꿀 수 있었다. 연주자가 키보드에서 바꿀 수 있어야 "내가 실제로 연주에 쓴다"는 목표에 맞는다. 대부분의 키보드는 프리셋 버튼으로 Program Change를 보내므로 이를 선택 입력으로 쓴다. CC도 지원한다.

## 설계
- `scripts/adapter_bank.py` `LiveAdapterSelector`
  - producer가 이미 잡은 입력 스냅샷에서 선택 메시지를 찾는다. 가장 최근 메시지를 적용한다
  - Program Change p는 p번째 어댑터를 고른다(`--checkpoint`가 0번, 이후 `--swap-adapter` 순서). 어댑터 수를 넘는 p는 무시한다
  - `cc:N`은 CC N의 0–127을 어댑터 수로 균등 분할한다
  - 선택은 다음 선택 메시지가 올 때까지 유지된다. 입력 창(4초)이 메시지를 잊어도 유지된다
- `run_continuous_jazz.py --adapter-control program|cc:N`(`--input-port` 필요, `--adapter-schedule`과 동시 사용 불가)
  - 선택은 **다음에 생성하는 마디의 시작**에 적용된다
  - 리포트 `adapter_swap.control_events`에 메시지마다 세션 시작 기준 도착 시각, 값, 선택한 어댑터를 남긴다
- 예상 지연
  - producer는 한 마디 앞서 만든다. 마디 r 중에 도착한 메시지는, 마디 r+1 생성이 이미 시작됐으면 r+2에, 아니면 r+1에 적용된다
  - 따라서 **1–2마디**다(128 BPM에서 1.9–3.8초)
- `scripts/run_live_select_probe.py`: 가상 포트를 열고 런타임을 자식 프로세스로 띄운 뒤, 정해진 시각에 Program Change를 보낸다. 리포트에서 "도착 마디 → 새 어댑터가 처음 연주된 마디"를 계산한다

## 조건
- 쌍 S: Tatum 완성(`--checkpoint`, 0번) + 공통 base 멜다우 u128(`--swap-adapter mehldau=…`, 1번)
- 128 BPM, 16마디, 코드 primer 2 sub-block, seed 42/43/44
- 전송: 실행 후 12초 PC1, 20초 PC0, 28초 PC1(세션은 실행 후 약 6–8초에 시작해 30초 동안 이어진다)

## 판정 (모두 충족하면 합친다)
1. 단위 테스트: PC/CC 매핑, 유지, 중복 무시, 범위 밖 무시
2. 세 실행 모두에서 보낸 메시지가 전부 기록되고, 세션 안에 도착한 각 메시지의 **적용 지연이 1–2마디**다
3. 선택하지 않은 어댑터로 연주된 마디가 없다. 마디별 어댑터가 "직전 적용 메시지"와 일치해야 한다
4. fallback 0
5. 연주된 마디가 같은 seed 단독 세션의 같은 마디와 동일하다. 입력에 음이 없으므로 primer가 같기 때문이다

---

## 1차 결과 (현재 fetch 방식) — 기준 2 미달, 원인 확인
원시값: `docs/experiments/live_select/probe_downbeat_seed{42,43,44}.json`.

| seed | 보낸 PC | 기록된 PC | (도착 마디 → 적용 마디, 지연) | fallback | 미스 | 단독과 같은 마디 | 마디별 어댑터 |
|---|---|---|---|---|---|---|---|
| 42 | 3 | 2 | (4→8, **4**), (8→12, **4**) | 0 | 0 | 16/16 | TTTTTTTT MMMM TTTT |
| 43 | 3 | 2 | (4→8, 4), (8→12, 4) | 0 | 0 | 16/16 | 같음 |
| 44 | 3 | 2 | (4→8, 4), (8→12, 4) | 0 | 0 | 16/16 | 같음 |

- 기준 1, 3, 4, 5는 충족했다. 선택은 정확히 따라갔고 마디는 단독 세션과 같다
- 기준 2는 **미달**이다. 지연이 1–2마디가 아니라 **4마디**(128 BPM에서 약 7.5초)다. 세 번째 PC(13마디째 도착)는 **적용되지 않았다.** 남은 마디의 생성이 이미 끝난 뒤였기 때문이다
- **원인(코드와 타임라인으로 확인):** 나는 producer가 1마디 앞서 생성한다고 가정했다. 실제로는 이렇다
  - 스케줄러가 마디 k의 다운비트에서 마디 k+1을 미리 가져간다(consumed watermark = k+1)
  - producer의 선행 한도가 2다(`max_lead_bars=2`, 초기 워밍업용)
  - 그래서 마디 k 다운비트에 마디 k+3의 생성이 시작된다. 입력 스냅샷도 그때 찍힌다
  - 리포트의 `input_to_bar_start_ms`가 6.1–8.4초였다. **연주자의 입력 음도 같은 지연을 겪는다.** 어댑터 선택만의 문제가 아니다
- 사전 등록대로 이 상태로는 합치지 않는다. 생성 선행을 줄이는 문제는 한 질문으로 따로 다룬다(#1526, `docs/experiments/GENERATION_LEAD.md`). 그 결과로 기준 2를 다시 잰다

## 2차 결과 (늦은 fetch 기본값, #1526) — **기준 1–5 모두 충족, 합친다**
원시값: `docs/experiments/live_select/probe_latefetch_seed{42,43,44}.json`.
- 기준 2: 세 실행 모두 PC 3개가 **전부 기록**됐다. 적용 지연은 전부 **2마디**다(4→6, 8→10, 13→15)
- 기준 3: 마디별 어댑터가 직전에 적용된 메시지와 일치한다(TTTTTT MMMM TTTTT M)
- 기준 4: fallback 0, 미스 0
- 기준 5: 연주 마디가 단독 실행과 **48/48** 같다
- 사용 예
  ```bash
  python scripts/run_continuous_jazz.py --input-port "<키보드 포트>" \
    --checkpoint outputs/final_tatum/export/checkpoint_update518.pt --adapter-name tatum \
    --swap-adapter mehldau=outputs/clean_base/c2_export/checkpoint_update128.pt --allow-different-bases \
    --adapter-control program --conditioning-midi outputs/chord_ab/ii_V_I.mid --chord-primer --chord-blocks-per-bar 2
  ```
  이 설정에서 키보드 프리셋 버튼(PC 0 = Tatum, PC 1 = 멜다우)으로 전환한다. 실물 키보드로는 검증하지 않았다(`external_keyboard_verified: false`)

## 리뷰 후 교정 (#1558)
- 알려진 probe 결함(Astra M3): 이미 선택된 어댑터를 다시 요청하면 현재 마디를 적용 마디로 잡아 음수 지연이 나온다. `per_bar`는 생성 시점 선택이므로 fallback도 적용으로 셀 수 있다
- 이 문서와 #1526·#1530·#1532의 지연 수치는 **영향이 낮을 것으로 예상한다.** PC가 매번 어댑터를 바꿨고(1/0/1…, no-op 없음) 모든 probe 실행이 fallback 0이었기 때문이다. 확정은 아니다. probe를 고친 뒤 새 매핑(제어 이벤트 → 소비 블록 → 실제 채택)을 기존 artifact에 적용해 검산할 수 있는 범위에서 검산한다

## probe 매핑 수정 (#1564, Astra M3)
- **결함:** probe가 "도착 뒤 처음 그 어댑터로 연주한 마디"를 적용으로 잡았다. 그래서 이미 선택된 어댑터를 다시 요청하면 현재 마디가 잡혀 음수 지연이 나왔다. `per_bar`는 생성 시점 선택이라 fallback으로 대체된 블록도 적용으로 셀 수 있었다
- **수정**
  - 선택기(`LiveAdapterSelector.update`)가 메시지마다 다음을 기록한다: 소비한 블록(`consumed_block`), 직전 선택(`previous`), `noop`(이미 활성인 어댑터 요청), `superseded`(같은 블록에서 뒤 메시지가 덮음)
  - 리포트에 `adopted_blocks`(get()이 모델 블록을 실제 반환한 블록, #1562)를 남긴다
  - probe의 상태는 `applied` / `applied_after_fallback` / `not_adopted` / `noop` / `superseded` / `ignored`다. 지연은 "도착 → 그 선택을 연주한 첫 **채택** 블록의 다운비트"다. noop·superseded·ignored에는 지연을 매기지 않는다
- **기존 artifact 검산**(`docs/experiments/live_select/m3_recheck.json`)
  - 대상: #1525, #1526, #1530(두 arm), #1532, #1536의 probe 18개, PC 75건
  - 새 매핑에서 75건 모두 `applied`였고, 지연값은 이전 보고와 **전부 같다**
  - 단서: 이 옛 리포트에는 소비 블록과 채택 기록이 없다. 채택은 "완주 + 모든 바가 모델 블록, fallback 없음"으로 **재구성**했다. get()만 미준비 바를 fallback으로 표시하기 때문이다. 소비 블록은 "noop이 없을 때 도착 뒤 처음 그 어댑터로 생성한 블록"으로 재구성했다. 기록값이 아니라 재구성값이다
- **새 코드 확인(사전 등록, 실행 전):** 반 마디 블록, seed 42, PC를 11.0초 1 / 13.0초 1(noop) / 15.40초 0 / 15.45초 1(앞의 0을 덮을 가능성) / 19.9초 0으로 보낸다. 판정 기준은 다음과 같다
  - 13.0초 메시지가 `noop`이고 지연이 없다
  - 적용된 메시지의 지연이 모두 0 이상이다
  - 채택 근거가 `recorded`다
  - 15.40초 메시지는 같은 블록에서 덮이면 `superseded`, 아니면 `applied`다. 어느 쪽인지 보고한다
- **새 코드 확인 결과** (반 마디 블록, seed 42, fallback 0, 미스 0)

  | 도착 ms | PC | 1차 상태 | 수정 후 상태 | 소비 블록 | 지연 ms |
  |---|---|---|---|---|---|
  | 7,402 | 1 | applied | applied | 9 | 1,035 |
  | 9,399 | 1 | noop | noop | 11 | — |
  | 11,803 | 0 | superseded | superseded | 14 | — |
  | 11,853 | 1 | **applied (1,287)** | **noop** | 14 | — |
  | 16,304 | 0 | applied | applied | 18 | 571 |

  - 사전 기준을 모두 충족했다: 13.0초 noop, 적용 지연 모두 0 이상, 채택 근거 `recorded`. 15.40초 메시지는 같은 블록에서 덮여 `superseded`였다
  - **1차 실행에서 발견한 의미 오류:** 15.45초 메시지(→ 멜다우)는 블록 직전에도 멜다우였다. 순효과가 없는데 중간 상태(0으로 바뀐 직후) 기준으로 판정돼 `applied`로 잡혔다. noop을 "직전 블록의 어댑터" 기준으로 바꿨다. 같은 블록의 앞 메시지는 `superseded`, 마지막 메시지만 적용 또는 noop이다. 수정 후 재실행에서 `noop`으로 잡혔다
