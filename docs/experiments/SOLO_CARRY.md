# 블록 경계 연결: solo-only carry A/B (사전 등록)

작성 2026-10-03. 이슈 #1652. 아스트라에게 분석과 제안을 보냈고 회신을 기다리는 중에 등록했다(이견이 오면 반영해서 기록한다). 판정 규칙은 실행 전에 고정했다. `musical_quality_verified: false`.

## 근거 (탐색, combined 6회 `outputs/comp_provenance`, 태깅된 솔로)
- 연속 음정(쉼 > 0.3초 제외)
  - 블록 안: p50 3, p90 7
  - **블록 경계: p50 9, p90 25**
  - 실제 비밥 RH(train 60곡, 같은 정의): p50 2, p90 9, ≥ 9반음 11%
- 블록 첫 음 − 블록 중앙값: p50 −3.5, p10 −13. 첫 음 pitch p10은 55(G3 바닥)다
  - 블록마다 guide 근처 낮은 음에서 시작해 올라가고, 다음 블록에서 다시 떨어진다
  - 기본 런타임은 블록마다 history 없이 guide만 보고 새로 생성한다
- 블록 끝에서 30 ms 미만으로 잘린 솔로 음: 경계 192개 중 16개(별도 과제)
- 사용자: "그냥 막 치는 것 같다." 이 결함과 연결될 수 있다(원인 확정은 아님)
- 옥타브 정렬 초안(블록 전체를 옥타브 단위로 옮겨 경계 도약을 최소화)은 철회했다. 스모크에서 블록 16개 중 8–12개가 +1옥타브로 옮겨져 선율이 모델 음역보다 위에 고정됐다. 증상만 처리하는 방식이다(브랜치 `exp/register-continuity` WIP)

## 변경
- `--comp`가 켜져 있으면 context carry로 넘기는 토큰을 컴핑을 뺀 솔로 렌더(`trace["solo_tokens"]`)로 바꾼다. 컴핑을 끈 경로와 기존 carry 실험에는 영향이 없다
- 사용하는 옵션은 기존 `--context-carry-tokens K --context-carry-position after`다. primer = [guide][직전 블록 솔로 tail]

## 실행
- 공통: `--preset bebop -- --solo-line --comp --comp-style varied --phrase-breath 24`, `--candidates 1`
- 팔 두 개: **carry0** / **carry48**(`--context-carry-tokens 48 --context-carry-position after`)
  - K = 48은 사전에 고정했다. 실제 비밥 솔로 창의 토큰 중앙값이 0.94초당 17개라 약 2–3블록 분량이다
- holdout 진행 3개 × seed 42·43, 16마디, 128 BPM. 같은 seed로 교대 실행한다
- 스크립트: `scripts/boundary_check.py`. 집합과 설정을 검증하고 거부할 수 있다

## 판정 (실행 전 고정, carry48 기준)
1. 경계 음정 ≥ 9 비율 ≤ 실제 × 2(0.22)
2. 경계 음정 p50 ≤ 4
3. 블록 안 음정 p90 ≤ carry0 + 2
4. 박 위 clash ≤ carry0 + 0.03
5. 같은 음 반복 비율 ≤ carry0 + 0.05, 직전 블록과 음 순서가 똑같은 블록 비율 ≤ 0.05(복사 방지)
6. 초당 음 수가 데이터의 0.67–1.5배, 쉼 ≥ 데이터의 0.5배
7. 대체 패턴 0, miss ≤ 2, invalid 0, block-ready p99 ≤ 419 ms
- 보고만: 첫 음 − 블록 중앙값 p50, 경계 p90
- 미달이면 기록한다. K나 위치는 다시 고르지 않는다
- 한계: carry 상태는 생성 순서를 따른다. 폐기 → fallback 뒤에는 실제로 재생된 직전 블록과 다를 수 있다
