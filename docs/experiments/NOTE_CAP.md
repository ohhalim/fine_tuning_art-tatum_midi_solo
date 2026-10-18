# 블록 창 음 수 상한과 마감 여유 (사전 등록)

작성 2026-10-18 23시대 KST. 이슈 #1819. 학습, 다운로드 없음. 실시간 기본값 변경 없음. 음악 변화는 청취 전 품질 주장 없음(`quality_claimed: false`).

## 1. 질문

- 블록 창(2박)당 음 수 상한 12가 q8 실시간 경로에서 지평 전 샘플 수와 마감 여유를 얼마나 바꾸고, 얼마나 자주 발동하는가

## 2. 배경

- #1813 v2: q8 루프 샘플당 시간 fp32의 0.557배. q8 블록 생성 최대 342 ms(지평 전 샘플 51), suffix 최대 131 ms
- 마감 초과 2건 모두 지평 전 샘플 40개 이상: v2 fp32 A 블록 14(42), #1813 첫 실행 q8 B 블록 15(51, 16음). 같은 q8 B 블록 15가 실행에 따라 342 ms와 438 ms
- 지평 전 샘플 수는 블록 창 음 수의 약 3배(음당 토큰 3개)
- #1809 원하는 릭 선율 대리의 2박 창 최대 11음. 출력 전체 음 수(동시음 포함)와 같은 단위 아님. 상한 12는 이 값을 직접 환산한 근거가 아닌 사전 선택

## 3. 변경 (실행 전 고정)

- `stream_contract.floor_parts(floor_ms, st)`: 확정 하한 계산을 함수로 분리(동작 동일). `TokenStream.onsets_since_commit()`: 확정 뒤 임시 음의 절대 onset(증분 스캔)
- `rt_lh_input.cap_floor`: 창 [t0 + b + i·B, t0 + b + (i+1)·B)(블록 확정 창). 현재 시각이 속한 창의 임시 음이 K개면 다음 창 시작을 하한으로 반환
- producer `--note-cap K`(기본 0 = 끔): 샘플마다 확정 하한과 상한 하한 중 큰 값으로 문법 마스크 하한. 확정 하한의 append 검사는 그대로(상한은 마스크만). `--seed N`. 블록 행 `capped_to_ready`, `capped_samples`
- 오케스트레이터 `--note-cap`, `--seed`, `--scenarios`, 시나리오마다 `density`

## 4. 실행 (실행 전 고정)

- q8, #1807 입력(data `8501d2ac…`, 왼손 음 목록 `c8b42bdc…`, checkpoint `5e6b07ee…`), `--rerun` 시나리오 중 A 정상과 B 이른 crop(C 지연 주입은 이 질문과 무관해 제외)
- 팔 2개: 상한 없음(K 0), 상한 12. seed 1, 2, 3, 4. seed마다 두 팔 실행, 팔 순서는 seed 홀수면 상한 없음 먼저, 짝수면 상한 먼저
- 2팔 × 4 seed × 2 시나리오 = 16 실행. 팔·seed 실행 전후 열 압력 기록
- 원자료 `/Users/ohhalim/git_box/note_cap_run1/{nocap,cap12}_s{seed}/`, 보고 `docs/experiments/note_cap/{nocap,cap12}_s{seed}.json`

## 5. 판정 기준 (실행 전 고정)

- K1 계약: 16 실행 모두 fail-closed `rerun_checks` 전부 참, B는 crop 1회 이상
- D1 상한 동작: 상한 팔의 블록 지평 전 샘플 수 최대 ≤ 3 × 12 + 5 = 41(창 안 음 12개 토큰 36 + `<T>` 최대 2 + 지평을 넘는 첫 음 3). 넘으면 구현 결함으로 분리
- D2 마감: 상한 팔 마감 초과 블록 수 ≤ 상한 없음 팔. 두 팔의 최소 여유, 여유 하위 5%, 지평 전 샘플 수 최대·p95 기록
- D3 발동 빈도: 상한 팔에서 지평 전 상한 발동 블록 비율과 블록 창 음 수 분포 기록. 25% 초과면 음악 변화가 잦다고 표시
- 루프 ms/sample 중앙값 두 팔 기록(상한 계산 비용 확인)
- 결론 규칙
  - K1, D1 충족, 상한 팔 마감 초과 0: 상한 12를 마감 보호 후보로 기록, 기본값 전환 전 사용자 청취 필요
  - 상한 팔에도 마감 초과: 상한만으로 부족, 남은 원인(suffix 재계산, 샘플당 시간 변동) 분리
  - D1 미충족: 구현 결함 분리
- 한 입력·4 seed 범위. 마감 보장이나 음악 품질 판단 아님

## 6. 범위 밖

- suffix 재계산(왼손 만료로 공통 앞부분이 짧아짐) 축, fp32, 다른 K 값, 청취

## 7. 재현

- `rt_lh_smoke_run.py <t3 python> <aria repo> <ckpt> <data json> <lh notes json> <out dir> <report json> --rerun --precision q8 --note-cap K --seed N --scenarios A_normal,B_crop_early`
- 시험 `tests/test_stream_contract.py`(`OnsetScanTest`), `tests/test_rt_lh_input.py`(`NoteCapTest`)

## 8. 판정 해석 보완 (아스트라 M1·M2·LOW, 실행 중 추가)

- 추가 시점: 16 실행 중 8개 완료 시점, 결과 집계 전. 기존 지표·임계값·원자료 변경 없음, 해석 규칙만 강화
- D2 해석: 첫 실패 뒤 실행이 끝나므로 마감 초과 절대 개수만 비교하지 않음. 팔별 예정·시도·송출 완료 블록 수, 16블록 완주 실행 수, 중단 사유 전부 병기. underfill·capacity_failure 등으로 먼저 끝난 실행은 마감 초과 0이어도 성공이 아니라 불완전으로 분리. 같은 seed·시나리오 짝의 첫 실패 블록과 완주 여부 중심으로 해석
- D1 범위: 3K + 5는 warm 블록에서 rollback 뒤 확정 창 하나를 채우는 샘플 수(창 안 음 토큰, segment `<T>`, 지평을 넘는 첫 음) 가정. cold 블록, 블록 총 상한 64, 추측 생성, forward 횟수·시간 상한과 다름. K1·D1 통과는 음악적 적합성 근거 아님
- `capped_to_ready`·`capped_samples`는 하한 활성 샘플 수. dur 자리에서는 마스크가 하한을 무시하므로 실제 마스크 개입·선택 변화 수가 아님. D3 발동 블록은 "지평 전 하한 활성 블록"으로 명명. 실제 선택 변화 측정은 이 단위 범위 밖
- 결과 설명에 상한 팔에서 onset이 실제로 다음 창으로 밀렸는지, 쉼·동시음 밀도 변화를 포함한 뒤 청취 대상으로 판단
- 결합 회귀 시험 추가: 실제 TokenStream + GrammarMask + cap_floor, 가장 이른 onset만 고르는 밀집 표본기로 5 s segment 경계와 commit → 추측 → rollback에서 창당 K 이하(`NoteCapStreamTest`)
