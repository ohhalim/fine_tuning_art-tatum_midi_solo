# 런타임 --candidates 2 feasibility (사전 등록, 사후 선택)

작성 2026-10-03. 이슈 #1646. 선행: `RUNTIME_CANDIDATES.md`(N=3 미달). 아스트라와 합의했다(공동 리뷰 4). 판정 규칙은 실행 전에 고정했다. `musical_quality_verified: false`.

## 성격
- **사후에 고른 N이다.** N=3이 예산을 넘은 것을 본 뒤에 N=2를 골랐다. 그래서 확증 실험이 아니라 feasibility 실험이다
- "단일 p99 × 2 = 410 ms라 예산 안"이라는 근거는 쓰지 않는다. 분위수는 더해서 보장할 수 없다
- 고정된 한 번의 계획만 실행한다
- **중단 조건:** 미달이면 이 경로에서 N이나 start budget을 더 탐색하지 않는다. 다음은 배치 생성(구조 개선)을 따로 등록하는 것이다
- N=3 + start budget 0.9는 입력 응답 지연과 바꾸는 별도 tradeoff라 이번에 섞지 않는다

## 실행
- `#1639`와 같은 설정이다: bebop, `--solo-line`, holdout 진행 3개 × seed 42·43, 16마디, 128 BPM
- 팔 두 개를 교대 실행한다: `--candidates 1` / `--candidates 2`(N=1도 같은 시간대에 다시 돌린다)
- 판정기: `runtime_candidates_check.py --candidate-arm 2 --budget-ms 419`(집합, 완주, 설정 검증 포함)
- 실제 예산 = `start_budget_bars` 0.5 × 블록(469 ms) − fetch margin 50 ms ≈ 419 ms

## 판정 (실행 전 고정, N=2 기준)
1. 대체 패턴 0(6회)
2. block-ready p99 ≤ 419 ms
3. deadline miss ≤ N=1 합 + 2
4. 재생된 선율의 박 위 clash < N=1
5. 초당 음 수가 데이터의 0.67–1.5배
- 보고만: block-ready p50/p95, 예산 대비 여유(419 − 블록별 생성 시간)의 최솟값
- 통과하면 `fl_live`에 `--candidates` 선택지를 넣는다(기본값은 그대로). 청취 렌더는 이 결과와 상관없이 컴핑 개선 후의 설정으로 만든다
