# 런타임 후보 선택 예산 검증 (사전 등록)

작성 2026-10-03. 이슈 #1639. 선행: `CANDIDATE_SELECT.md`(오프라인 통과). 판정 규칙은 실행 전에 고정했다. `musical_quality_verified: false`.

## 변경
- `--candidates N`(기본 1 = 기존과 같다). 반마디 블록마다 같은 primer에서 후보 N개를 만든다(seed + k × 1000)
- `inference/control/candidate_rank.py`의 고정 ranker로 하나를 고른다. 오프라인 ranker와 점수·선택이 같은지 무작위 50건으로 동등성 테스트를 했다
- `--pattern-cache`와 함께 쓸 수 없다. sub-block 경로(`--chord-blocks-per-bar 2`)가 필요하다
- 보고서: `candidates`, `candidate_stats`(순위를 매긴 블록, 후보 0 유지, 자격 있는 후보 없음)
- 스모크(8마디, 측정 아님): 생성 p50 301 ms, 최대 429 ms, 대체 패턴 0

## 실행
- `--preset bebop -- --solo-line`. 팔 두 개(`--candidates 1` / `--candidates 3`)를 같은 seed로 교대 실행한다
- 진행(holdout): Gm7,C7,Fmaj7,Fmaj7 / Bbmaj7,G7,Cm7,F7 / Bm7b5,E7,Am7,Am7. 각 16마디, 128 BPM, seed 42·43. 모두 12회
- comp, breath, history는 끈다(효과 분리)
- 스크립트: `scripts/runtime_candidates_check.py`. 실행 집합 검증을 통과하지 못하면 판정을 거부한다

## 판정 (실행 전 고정, N=3 기준)
1. 대체 패턴 0(6회 합)
2. block-ready(generation_ms) p99 ≤ 0.6 × 블록(562 ms). 블록을 모두 합쳐서 잰다
3. deadline miss ≤ N=1 합 + 2
4. 실제 재생된 맨 위 선율의 박 위 clash가 N=1보다 낮다(오프라인 결과와 같은 방향인지 확인)
5. 초당 음 수가 데이터의 0.67–1.5배(2.49–5.57)
- 통과하면 `fl_live.py`에 `--candidates` 선택지를 추가한다. 청취용 렌더는 컴핑 개선 단위 뒤로 미룬다. 기계적 컴핑이 그대로면 사용자 지시상 당연한 결함을 넘기는 것이기 때문이다
- 미달이면 기록한다. N이나 ranker는 다시 고르지 않는다
- 한계: CPU 부하가 큰 상황과 콜드 스타트는 따로 재지 않는다. 이 머신의 평소 부하에서 잰다
