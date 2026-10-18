# 왼손 입력 smoke 재실행 (사전 등록)

작성 2026-10-18 16시대 KST. 이슈 #1807. 새 학습, 새 음악 탐색, 기본 경로 변경, 청취 없음. `quality_claimed: false`.

## 1. 질문

#1798, #1802, #1803 수정 후 같은 왼손 입력에서 확정 지평 앞 음과 지각 송출이 사라지고, crop과 생성 지연이 실제로 일어나는 조건에서 각각 정해진 안전 경로로 처리되는가.

## 2. 배경

- #1797(`LH_INPUT_SMOKE.md`): 세 실행 모두 블록 4에서 마감 초과로 중단. crop(용량 400)과 블록 6 지연 주입은 미실행
- 사후 발견: 확정 지평 앞 음(첫 실행 5/95, 재실행 1/24), 최대 394 ms 지각 송출
- 수정: 확정 지평 하한과 접수 지각 거부(#1804), 지평 미도달 블록 중단(#1805), crop 위치 병합 길이 선택과 forward 전 용량 검사(#1806)

## 3. 실행 (실행 전 고정)

입력, prompt, seed, 실행기, 블록 수(cold + 16)는 #1797과 같음. 왼손 음 목록 sha256 `c8b42bdc…`.

| 시나리오 | 설정 | 의도 |
|---|---|---|
| A 정상 | 기본 용량 | 수정 후 정상 경로. 마감 결과는 기록만 |
| B 이른 crop | 용량 260, 남길 길이 120 | 블록 2에서 crop 발생 |
| C 블록 2 지연 | 블록 2 suffix 전 600 ms 대기 | 블록 2에서 마감 초과 중단 |

- B 설정 근거: #1797 기록의 prompt(102토큰, `<T>` 1개)와 블록 0, 1 확정 음으로 블록 2 요청을 재구성해 계산. 병합 148 > 260 − 예약 131로 crop, prompt의 `<T>`에서 잘려 병합 93. 실제 생성이 달라지면 crop 블록도 달라질 수 있음
- 용량 260은 5초 segment 하나의 토큰 수보다 작아서, 이후 블록에서 자를 위치가 없으면 `capacity_failure`로 멈출 수 있음. 이것은 안전 경로의 하나로 봄
- C 설정 근거: #1797 블록 2 여유는 +210~+232 ms, 600 ms 지연이면 마감 초과
- 재사용 동등성 표본: A의 블록 2, 3
- 실행 명령: `rt_lh_smoke_run.py <t3 python> <aria> <ckpt> <data> <lh notes> /Users/ohhalim/git_box/lh_input_smoke_run2 docs/experiments/lh_input_smoke_rerun/report.json --rerun`

## 4. 판정 기준 (실행 전 고정)

모든 시나리오:

1. 확정 지평 앞 음 0 (`posthoc_before_horizon`의 `notes_before_previous_horizon`)
2. 접수 지각 거부 0 (`rejected_late_on`)
3. forward한 모든 블록이 `len_N + 예약 131 ≤ 용량`
4. 정지는 `end` 또는 기록된 abort(`deadline_miss`, `horizon_underfill`, `capacity_failure`)
5. 안전 종료: 미송출 note-on 0, 포트 닫힘, note-off 실패 0
6. 기존 검사(송출과 캡처 대조, 음높이별 교대, 확정 구간 불변, 입력 무결성, 인과) 통과

시나리오별:

- B: crop 1회 이상
- C: abort가 정확히 `deadline_miss`, 블록 2
- A: 마감 결과(초과 블록, 여유 분포)는 기록만. 합격 조건 아님. 촘촘한 블록의 마감 초과는 생성 비용과 반응 예산의 원인 분리, 대안 비교가 필요한 제품 판단으로 미결정

1과 2는 #1797 원자료에 같은 판정 함수를 돌려 거짓이 나오는 것을 확인(양성 대조, A 재실행 기록에서 1이 거짓).

## 5. 하지 않는 것

- 실패 시 seed 반복, 기준 완화, 설정 변경 후 재실행
- 결과를 음악 품질이나 실사용 가능 판단으로 해석
- 촘촘한 블록 대응(2박 지연, 밀도 제한, 추측분 유지) 채택

## 6. 재현

- 시험: `tests/test_rt_lh_input.py`, `tests/test_rt_sender.py`, `tests/test_stream_contract.py`
- 원자료: 저장소 밖 `/Users/ohhalim/git_box/lh_input_smoke_run2/`
- 보고서: `docs/experiments/lh_input_smoke_rerun/report.json` (`--analyze-only --rerun`으로 저장 파일만 읽어 재생성 가능)
