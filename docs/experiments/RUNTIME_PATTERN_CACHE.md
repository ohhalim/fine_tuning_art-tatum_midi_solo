# 런타임 opt-in: 이력 문맥 + 낮은 온도 + 패턴 캐시 (사전 등록)

작성 2026-10-02. 이슈 #1608. 판정 규칙은 실행 전에 고정했다. 독립 리뷰는 사용자 지시로 나중에 일괄로 받는다. `musical_quality_verified: false`.

## 근거
- #1606: 오프라인 자연 문맥에서 T 0.6 + 패턴 캐시를 쓰면 Tatum의 음정 재등장 IR이 기준을 넘었다(0.039, 실제 0.052)
- 런타임은 입력이 다르다. 반 마디마다 새로 생성하고 코드 primer를 쓴다(#1597)
  - 자연 문맥에 가장 가까운 런타임 입력은 이력 모드다(`--context-history`, #1572 E). 채택·예약된 블록의 이력 256토큰을 코드 진술 앞에 넣는다

## 변경
- `--pattern-cache`(opt-in, 기본 off): 반 마디 블록 생성에 패턴 캐시 바이어스(ln 3)를 쓴다. 캐시는 primer(이력 포함)와 생성 중인 블록을 함께 본다
- 리포트 `pattern_cache_stats`: 샘플링 단계 수와 바이어스가 작동한 단계 수

## 측정
- 후보 R: Tatum 프리셋(반 마디 블록, 코드 primer)에 다음 옵션을 더한다
  - `--context-carry-tokens 256 --context-history --context-carry-position before --max-sequence 512`
  - `--temperature 0.6 --pattern-cache`
- 진행 3개(ii–V–I C 16마디 128 BPM, F 블루스 12마디 120 BPM, 마이너 ii–V–i C 16마디 128 BPM) × seed 42, 43 = 6회. 입력 없음, 가상 포트, `--capture`
- 기준 A: 같은 진행·seed의 기본 런타임 실행(seed 42 쇼케이스, seed 43 #1572 A 실행). 새로 실행하지 않는다
- 지표(`scripts/runtime_coherence_check.py`, `coherence_metrics`와 같은 정의)
  - 동기 재사용(8초 음정 3-gram 재등장)
  - 경계/내부 비율
  - 코드톤(진행·seed로 짝)
  - fallback, 미스, 캐시 작동 비율

## 판정 (실행 전 고정, COHERENCE_GOAL 목표 계승)
- **통과:** 모두 충족
  1. R 6회 합산 동기 재사용 ≥ 0.5 × 실제 Tatum(0.073) = 0.0365
  2. 코드톤 짝 평균 차이(R − A) ≥ −0.03
  3. R 6회 모두 fallback 0(미스는 보고만)
- 경계/내부 비율은 보고만 한다(실제 1.277)
- 통과하면 사용자 청취를 받는다. 기본값은 청취 결과로 별도 결정한다
- 미달이면 원인을 기록하고, 온도·바이어스를 이 단위에서 다시 탐색하지 않는다
- #1572 2단계에서 낮은 온도는 fallback을 늘렸다. 조건 3이 가장 위험하다
