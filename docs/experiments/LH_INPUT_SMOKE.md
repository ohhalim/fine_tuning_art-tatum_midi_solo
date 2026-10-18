# 가상 MIDI 왼손 입력 → producer opt-in smoke (사전 등록)

작성 2026-10-18 15:0x KST. 이슈 #1797. 아스트라 범위(msg_8805130fea29). 새 학습·새 음악 탐색·기본 경로 변경·청취 없음. `quality_claimed: false`.

## 1. 질문 (하나)

가상 MIDI로 재생한 왼손이, 그 블록 시작 전에 도착한 것만 #1773 계약대로 문맥에 들어가고, 앞부분 재사용(#1795 규칙)으로 생성된 오른손이 1박 마감·안전 정책대로 송출되는가?

- 성공 의미는 여기까지입니다. 왼손이 더 좋은 솔로를 만든다는 검증이나 사용자 청취를 대신하지 않습니다

## 2. 구성 (opt-in, 기본 경로 그대로)

| 프로세스 | venv | 하는 일 |
|---|---|---|
| `rt_lh_feeder.py` | main | 고정 왼손 음 목록을 가상 출력 포트 `LH_IN`으로 세션 시계에 맞춰 재생 |
| `rt_lh_producer.py` | t3_aria(MLX) | 수신 스레드가 `LH_IN`을 열어 받은 메시지를 시각과 함께 큐에 넣기만 함(계산 없음). 본 루프가 블록마다 snapshot → #1773 b 문맥 → 재사용 → 생성 → stdout JSON |
| `rt_sender.py` | main | #1747 그대로 + `abort` 메시지를 비정상 종료와 같게 처리 |
| `rt_capture.py` | main | #1747 그대로, 모델 출력 포트만 캡처 |

- 사람 입력과 모델 출력은 포트로 구분합니다(`LH_IN` 대 모델 출력 포트). 시험 입력은 G3 아래 음으로 제한하지만, G3를 제품의 역할 정의로 확정하지 않습니다
- 모델 계산은 producer 본 루프에서만 합니다. 송출·수신 스레드는 계산하지 않습니다

## 3. 입력 규칙 (실행 전 고정)

- 세션 시계: producer의 `start`(음악 기준 t0, 벽시계 W)를 feeder가 파일로 받아, 음악 시각 m의 음을 벽시계 W + (m − t0)에 보냅니다
- 수신 시각 → 음악 시각: m = t0 + (수신 ns − W) / 1e6
- 블록 k 시작 s_k = t0 + k·B에서 snapshot: **onset < s_k**인 음만. off는 snapshot 전에 실제로 받은 것만 쓰고, 아직 안 뗀 음은 s_k에서 가상 종료(#1773 held 규칙)
- 같은 음높이를 다시 치면(앞 음이 안 떼진 채) 앞 음을 새 onset에서 끝냅니다. velocity 0 note_on은 note_off입니다. 받은 적 없는 음의 off는 무시하고 기록합니다
- CC64(서스테인)는 지원하지 않습니다. 받으면 무시하고 개수를 기록합니다
- 입력 끝: feeder가 목록을 다 보내면 끝. 안 뗀 음은 snapshot 규칙대로 가상 종료
- 관측 창 7.5초, crop, 병합 순서는 #1773 그대로(`lh_context.window_notes`, `merge`)

## 4. 생성 규칙 (실행 전 고정)

- 블록 0(cold)은 왼손 없이 #1747 producer와 같게 만듭니다
- 블록 k ≥ 1: 오른손 추측분을 버리고(`rollback_to_committed`), 확정 오른손 + snapshot 왼손으로 N을 만듭니다. P_reuse = min(공통 앞부분(KV에 있는 모델 입력, N), KV 유효 끝), crop이 바뀌면 0, S = 0이면 마지막 1토큰. N[P_reuse:]를 계산한 뒤 오른손을 지평(h_k = s_k + B + 1박)까지 생성하고 추측분 한 블록을 더 생성합니다(`fill_block`, 블록당 최대 64토큰)
- 생성 길이는 음악 지평 도달로 정합니다. n개로 자르지 않습니다
- 샘플러·형식 마스크·seed는 #1747 producer와 같습니다(#1726 샘플러, `<D>`·`<E>` 차단)

## 5. 마감과 안전 (실행 전 고정)

- 마감 = W + (k·B + 1박). 블록 k의 확정·예약(on_ready)이 마감 뒤면 그 결과를 송출하지 않고 `abort`(이유 `deadline_miss`)를 보낸 뒤 실험을 멈춥니다
- 송출기는 `abort`를 비정상 종료로 처리합니다: 미송출 note-on 취소, 울리는 소유음 note-off, 포트 닫기(#1747 종료 경로)
- 늦은 결과를 몰아서 내거나, 조용히 추측분으로 바꾸거나, 다음 블록으로 넘기지 않습니다. crop 비용도 같은 마감으로 셉니다

## 6. 실행 (실행 전 고정)

| 시나리오 | 내용 | 기대 |
|---|---|---|
| A 정상 | 128 BPM 4/4, 8마디(16블록 + cold), 기본 crop 설정 | 모든 블록이 마감 안이면 끝까지, 아니면 첫 초과에서 abort(그대로 기록) |
| B 강제 crop | crop 용량 400·남길 길이 200(prompt 102토큰이라 실행 중 crop이 여러 번 일어남) | crop 블록이 마감 안이면 계속, 넘으면 abort |
| C 느린 producer | 블록 6에서 suffix 계산 전에 600 ms를 더 기다림 | 블록 6에서 deadline_miss로 abort, 안전 종료 |

- prompt: `/Users/ohhalim/git_box/mehldau_aria/data.json` 첫 prompt(#1747과 같음). seed #1747과 같음
- 왼손 입력: dev 곡 44의 사람 음(g3 대리, velocity 복구본) 중 미리 정한 15초 구간을 음악 시각 t0 + B부터 재생. 구간과 음 목록 sha256을 보고서에 적습니다
- B와 C는 회귀 동작(abort·crop 처리) 확인이고, A의 실시간 통과와 따로 표시합니다
- 실패하면 그 실행 기록으로 끝냅니다. seed 반복·문턱 완화·연속 확대 없음

## 7. 계측과 검사

- 블록별(요청 ID = 블록 번호): input_received(받은 메시지 수·마지막 시각), snapshot, context_ready, suffix_ready, generation_ready(on_ready), enqueued(stdout 쓰기), sent·captured(송출기·캡처 기록과 블록 번호로 결합), 마감, crop 여부, P_reuse, V, S, 생성 토큰 수, 중단 이유. 서로 다른 실행의 구간을 더하지 않습니다
- 검사
  - A1–A3(#1747): 캡처열 = 송출기 예상열, 음높이별 on/off 교대·끝에 모두 off, 논리 짝
  - B1 입력 무결성: producer가 받은 왼손 메시지 = feeder가 보낸 목록(개수·순서·음높이)
  - B2 인과: 블록 k 문맥의 모든 왼손 onset < s_k, held 끝 ≤ s_k(#1773 L1)
  - B3 재사용 동등성 표본: A에서 블록 4·8·12의 N을 전체 재계산과 비교(top-1 일치, 마지막 |차| ≤ 1e-3)
  - C1 마감: 블록별 ready 여유 분포, 초과 블록. 통과율로 일반화하지 않음
  - C2 안전 종료(B·C): abort 뒤 미송출 note-on 0, 모든 소유음 off
- fixture 시험(실행 전): 입력 규칙(재타격, velocity 0, 모르는 off, CC64 무시, onset < s 필터, 미래 off 불변), 같은 시각 정렬, crop 시 전체 reset, abort 판정, 송출기 abort 처리

## 8. 재현

- `scripts/rt_lh_smoke_run.py <t3_aria python> <aria repo> <ckpt> <data json> <lh notes json> <out dir> <report json>`
- 시험 `tests/test_rt_lh_input.py`
- 보고 `docs/experiments/lh_input_smoke/report.json`, 원자료는 저장소 밖 `/Users/ohhalim/git_box/lh_input_smoke_run1/`

## 9. 이탈

1. **A 1차 실행은 분석 단계에서 멈췄습니다.** producer 보고에 #1747의 A3 검사(확정 구간 불변, 확정 토큰 형식 유효, 블록마다 지평 도달)를 넣지 않아, 재사용한 분석 함수가 `checks` 키를 찾지 못했습니다. A3는 실행 중 상태로만 확인할 수 있어 사후에 채울 수 없습니다. 1차 원자료는 `A_normal_run0_analysis_crash/`로 보존했고, producer에 A3 검사를 넣은 뒤 세 시나리오를 처음부터 다시 실행합니다(seed·입력·설정 같음). 1차 기록: 블록 15에서 deadline_miss(여유 −58 ms)로 abort, 왼손 23쌍 수신, 동등성 표본 블록 4·8·12 통과
2. **재실행 전 1차 원자료로 분석을 미리 돌리다 측정 기록 문제 두 개를 찾아 고쳤습니다.** 판정 기준 값은 바꾸지 않았습니다.
   - **B2 기록 단위:** producer가 왼손 onset·끝을 정수 ms로 반올림해 기록했습니다. 그래서 s_k = 16307.5인 블록의 held 끝(=s_k)이 16308로 남아 B2가 거짓으로 나왔습니다. #1773 L1은 반올림 전 음악 시각으로 정의돼 있으므로, 반올림 전 값(`raw_on`·`raw_off`)을 기록하도록 바꿨습니다.
   - **B1 범위:** feeder는 왼손 15초를 끝까지 보냅니다. 마지막 블록(s_16 = t0 + 15000 ms) 뒤나 abort 뒤에 보낸 메시지는 어떤 블록 문맥에도 들어갈 수 없는데, 원 정의(받은 것 = 보낸 전체)대로면 정상 실행도 실패합니다. 그래서 producer가 입력 포트를 닫기 직전에 큐를 한 번 더 비우고(`final_drain_ns`), B1을 다음 두 조건으로 좁혔습니다. ① 받은 메시지는 보낸 목록의 앞부분과 순서·음높이가 같다. ② `final_drain_ns − 2 ms` 이전에 보낸 메시지는 하나도 빠지지 않았다. 그 뒤에 보낸 개수는 따로 보고합니다
