# 컴핑 출처 보존과 combined 설정 검산 (사전 등록)

작성 2026-10-03. 이슈 #1648. 아스트라 공동 리뷰 5에서 합의했다. 음악 파라미터는 바꾸지 않는다. 판정 규칙은 실행 전에 고정했다. `musical_quality_verified: false`.

## 근거
- `COMPING.md`의 코드 정렬 1.0은 `comp_half`가 만든, 솔로와 **합치기 전** 음을 기준으로 잰 값이다
- 런타임 렌더(`solo_with_comp_tokens`)는 같은 음이 시간상 겹치면 컴핑 음을 뺀다. 그래서 3음이나 7음이 빠질 수 있다
- 이전 회귀의 전달률 104–106%는 솔로 저음이 섞여서 검증값으로 쓸 수 없다

## 변경
- `solo_with_comp_tokens` / `render_block`의 `trace`: planned, emitted, dropped(이유별), clipped, outcome
  - drop 이유: same_pitch_overlap, past_block_end, reencode_fallback
  - outcome: rendered, raw_invalid, rendered_invalid. raw_invalid와 rendered_invalid이면 emitted = []
- 런타임 보고서의 `comp_trace`: 생성된 반마디 블록마다 기록한다(채택 여부와 무관)
- `comping_eval.py`: 솔로 실행 집합을 정확히 검증한다(N=1 6회, 완주, 설정). 기존 판정은 같다
- fixture 테스트: 정상, 솔로가 컴핑 음과 같은 음을 겹쳐 침, 블록 끝을 넘김과 clip, raw invalid. reencode_fallback은 일부러 일으키기 어려워 테스트하지 않았다

## 실행
- 청취물 F1/F2와 같은 combined 설정: `--preset bebop -- --solo-line --comp --comp-style varied --phrase-breath 24 --candidates 2`
- holdout 진행 3개 × seed 42·43, 16마디, 128 BPM
- `scripts/comp_source_check.py`: 재생된 음과 그 블록의 emitted 컴핑을 조인한다(같은 음, 시작 차이 12 ms 이내) → 출처 태깅
  - 지표는 모델이 재생한 블록에서만 잰다
  - 솔로 지표는 태깅된 솔로 음으로만 잰다

## 판정 (실행 전 고정, 6회 합)
1. 대체 패턴 0
2. rendered_invalid와 raw_invalid 0
3. deadline miss ≤ 2
4. block-ready p99 ≤ 419 ms
5. emitted 기준 코드 정렬 ≥ 0.95. 코드가 바뀐 블록에서 첫 박 안 같은 onset에 3음과 7음이 함께 나가야 한다
6. 전달률(재생된 컴핑 / emitted 컴핑) ≥ 0.98
- 보고만: drop 이유별 수, 솔로 초당 음, 쉼, 프레이즈 중앙값
- 한계
  - 컴핑과 숨 쉬기 상태는 생성 순서를 따른다. 폐기 → fallback 경로는 이번에 일부러 일으키지 않는다(후속)
  - 출처 태깅은 음과 시간으로 조인하는 방식이다. 같은 음이 12 ms 안에 겹치면 잘못 붙을 수 있다
