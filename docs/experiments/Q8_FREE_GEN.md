# fp32 대 q8 Aria 자유 생성 출력 분포 (사전 등록)

작성 2026-10-18 23시대 KST. 이슈 #1823. 오프라인 생성만. 실시간 실행, 기본값 전환, 학습, 다운로드 없음. `quality_claimed: false`.

## 1. 질문

- fp32와 q8 Aria의 자유 생성(샘플링) 출력 분포 차이 여부

## 2. 배경

- #1812: teacher forcing 기준 q8 대 fp32 KL 평균 0.0009, TV 평균 0.0139. 문법 마스크 없는 test 10창 범위, 자유 생성·음악성 증명 아님(아스트라 리뷰)
- #1817: 같은 정밀도 안 캐시 suffix 동등성 계약 fp32·q8 모두 `contract_ok`
- #1818: q8 실시간 관측 루프 샘플당 시간 fp32의 0.557배. 기본값 전환은 별도 단위, 전환 전 출력 차이 근거 필요

## 3. 생성 (실행 전 고정)

- 모델: `t3_aria/ckpt/model-gen.safetensors`(checkpoint `5e6b07ee…`), `lh_kv_reuse.load_mlx(…, precision)`. fp32(현행) 대 q8(bf16 + `model.model` 8비트 그룹 32, #1811 방법)
- 프롬프트: `mehldau_aria/data.json` prompts 0~3(102, 180, 69, 159토큰). 실시간 경로는 prompts 0
- seed 1~32. 프롬프트 4 × seed 32 = 짝 128, 정밀도마다 128 실행
- 샘플러: 실시간 producer와 같은 것. 온도 1, 단계마다 Exp(1) 잡음 하나(`masked_pick`), 형식 마스크(`GrammarMask`), `<D>`·`<E>` 제외. 확정 하한은 커밋이 없어 비활성
- 실시간과 다른 점: 벽시계, 왼손 입력, 블록 rollback·추측 생성 없음. 프롬프트 끝(t0)부터 음악 시각 30초까지 KV 캐시로 한 토큰씩 이어 생성, 토큰 2,400개 상한
- 같은 (프롬프트, seed)는 두 정밀도가 같은 잡음열을 씀. 첫 다른 선택까지 같은 토큰, 그 뒤는 서로 다른 연주. 짝 비교는 같은 조건의 두 표본 비교이지 같은 연주의 두 녹음 비교가 아님
- GPU 잠금 `mkdir /Users/ohhalim/git_box/.gpu_lock`, `.timing_lock` 있으면 대기. 잠금 구간 15분 이내로 나눠 실행, 구간 전후 열 압력 기록

## 4. 실행 단위 통계 (실행 전 고정)

t0 ≤ onset < t0 + 30 s 음 기준.

| 통계 | 정의 |
|---|---|
| notes_per_s | 음 수 ÷ 30 |
| pitch_median, pitch_range | 음높이 중앙값, 최고 − 최저 |
| chord_share | onset 30 ms 묶음 중 2음 이상 묶음에 속한 음 비율 |
| ioi_median_s | 묶음 시작 간격 중앙값 |
| dur_median_s | 음 길이 중앙값 |
| vel_median | 세기 중앙값 |
| rest_share | 30초 중 울리는 음이 없는 시간 비율 |
| win_max, win_median | 2박 창(937.5 ms, 128 BPM) 32개의 음 수 최대·중앙값 |

- 형식 유효성: `aria_token_validity.check`(무효 또는 끝이 음 중간이면 무효). 종료 사유(30초 도달, 토큰 상한, 허용 질량 0) 기록
- 짝별 첫 분기 토큰 위치 기록

## 5. 판정 기준 (실행 전 고정)

- 주 검정: 통계마다 짝(프롬프트, seed) 차이 q8 − fp32에 Wilcoxon signed-rank(0 차이 제외, 정규 근사, 동순위 보정, 연속성 보정 없음). 효과 크기는 matched-pairs rank-biserial r
- 차이 검출: p < 0.005(10개 통계 Bonferroni, 가족 0.05)이고 |r| ≥ 0.3
- 판정: 검출 통계 0개이고 두 팔 모든 실행 형식 유효면 "이 n에서 검출된 차이 없음". 하나라도 검출이면 "차이 검출"과 해당 통계. 형식 무효 실행이 있으면 함께 표시
- 검출력 한계(사전 계산): 짝 128에서 p < 0.005에 필요한 |r|은 약 0.29(정규 근사). |r|이 이보다 작은 차이는 검출 못할 수 있음. "검출 안 됨"은 "같음"이 아님
- 보조(판정에 안 씀): 짝 차이 중앙값과 부트스트랩 95% 구간(2,000회, seed 0), 합쳐서 본 Cliff's delta, 프롬프트별 중앙값
- 결과를 본 뒤 기준·통계 목록·n 변경 없음

## 6. 청취 자료 (실행 전 고정)

- 짝 4개: (프롬프트 0, seed 1), (0, 2), (3, 1), (3, 2). 생성 부분 첫 20초를 MIDI·m4a로 렌더
- A/B 순서는 seed 1823 난수로 섞고 정답은 보고서에만 기록(페이지에는 숨김). 답은 "A / B / 차이 모름"과 메모를 복사해 받음
- 원자료·페이지 `/Users/ohhalim/git_box/q8_free_gen_run1/`

## 7. 범위 밖

- 실시간 경로의 블록 rollback·왼손 조건 아래 분포, 마감, 기본값 전환, 음악 품질 판단

## 8. 재현

- `scripts/q8_free_gen.py --generate <aria repo> <ckpt> <data json> fp32|q8 <out jsonl> [--prompts …] [--seeds 1-32]`
- `scripts/q8_free_gen.py --compare <fp32 jsonl> <q8 jsonl> <report json>`, `--render <fp32 jsonl> <q8 jsonl> <out dir> <report json>`
- 시험 `tests/test_q8_free_gen.py`
- 보고 `docs/experiments/q8_free_gen/report.json`
