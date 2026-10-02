# 앞 구절 잇기(G2) 런타임 설정 — 멜다우 (사전 등록)

작성 2026-10-02. 이슈 #1612. 판정 규칙은 실행 전에 고정했다. 독립 리뷰는 사용자 지시로 나중에 일괄로 받는다. `musical_quality_verified: false`.

## 질문 (하나)
Tatum에서 동기 재사용을 실제 수준 가까이 올린 G2 설정(#1610)이, 바꾸지 않고 멜다우 모델에도 같은 효과를 내는가.

## 설계
- 멜다우 프리셋(멜다우 #1497, 반 마디 블록, 코드 primer)에 다음 옵션을 더한다
  - `--context-carry-tokens 256 --context-history --context-carry-position before --max-sequence 512`
  - `--temperature 0.6 --pattern-cache --start-budget-bars 0.9 --generation-tokens 128`
- 진행 3개 × seed 42·43 = 6회. 기준은 같은 진행·seed의 멜다우 기본 런타임 실행(seed 42 쇼케이스, seed 43 #1572 A)이다
- 판정은 `scripts/runtime_coherence_check.py --model mehldau`로 한다

## 판정 (실행 전 고정)
1. 동기 재사용 ≥ 0.5 × 실제 멜다우(0.094) = 0.047
2. 코드톤 짝 평균 차이 ≥ −0.03
3. 6회 모두 fallback 0(미스는 보고만)
- 미달이면 기록하고 이 단위에서 설정을 바꿔 다시 탐색하지 않는다. 기본값은 바꾸지 않는다
