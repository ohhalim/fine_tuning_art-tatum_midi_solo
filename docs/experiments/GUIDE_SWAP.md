# 런타임 guide 구 vs 계약 guide (사전 등록)

작성 2026-10-03. 이슈 #1635. 아스트라와 합의한 다음 단위다(#1633 리뷰 3 후속). 판정 규칙은 실행 전에 고정했다. `musical_quality_verified: false`.

## 근거 (직접 확인)
기존 런타임 guide(`chord_guide_notes_for_duration` → `_voicing_for_chord`)는 C3 위의 코드톤을 아래부터 3개만 고른다. 그래서 입력 정보가 빠진다.

| 코드 | 구 guide | 계약 guide |
|---|---|---|
| G7 | 43 50 53 55 (**Gm7과 같음**, 3음 B 없음) | 43 50 53 59 |
| Gm7 | 43 50 53 55 | 43 50 53 58 |
| F7 | 41 48 51 53 (3음 A 없음) | 41 48 51 57 |
| Dm7 | 38 48 50 53 (5음 A 없음) | 38 48 53 57 |
| Dm7b5 | 38 48 50 53 (**Dm7과 같음**, b5 없음) | 38 48 53 56 |
| Cmaj7 | 48 52 55 59 | **36** 52 55 59 |
| C7 / Cm7 | 48 … / 48 … | **36** … / **36** … |

구와 신은 빠진 음만 다른 것이 아니다. C 계열은 근음이 한 옥타브 내려간다. 그래서 이 대조의 결과는 **guide 구현 전체를 교체한 효과**로만 읽는다. 구 guide의 최대 음은 59다. 음역을 ≤ 54로 바꾸는 것은 이번에 섞지 않는다.

## 실행 (`scripts/guide_swap_eval.py`)
- 체크포인트: bebop(`outputs/bebop_rh/export/checkpoint_update516.pt`), 병합해서 CPU에서 돌린다
- 런타임 기본 생성 호출 한 블록(반마디 0.9375초, 128 BPM)
  - generation tokens 96, max sequence 192, T 1.0, top-k 32, top-p 0.95, grammar mask, KV cache
  - primer = guide만 준다. 입력, history, comp, breath는 모두 끈다
- 코드 9개(Dm7 G7 Cmaj7 F7 Bb7 Gm7 C7 Dm7b5 Cm7) × guide 2종 × seed 40개. 같은 seed를 쓴다
- 저장: 표본마다 원시 토큰, 입력 guide, 유효성, 솔로(G3 이상 최고음), 뺀 저음 수(`samples.json`)

## 지표
- **기술통계(게이트 아님):** guide 종류별로 각 코드의 pc를 기준으로 잰다
  - fit, clash, 3음 비중(솔로 길이 중 그 코드 3음의 비율), 초당 음 수
  - invalid 비율, 빈 블록 비율, 블록당 뺀 저음 수
- **2x2 반응(실행 전에 고정한 쌍):** Dm7/Dm7b5, G7/Gm7(이 둘은 구 guide에서 입력이 같다), Cmaj7/Cm7, C7/Cmaj7
  - 같은 seed로 y_A(guide A 뒤)와 y_B(guide B 뒤)를 생성한다
  - 대각 반응 = [fit(y_A,A) − fit(y_A,B) + fit(y_B,B) − fit(y_B,A)] / 2
  - seed bootstrap 95% CI와 같은 출력 비율을 함께 기록한다
  - fit proxy 기준의 반응일 뿐이다. 음악 품질 인증이 아니다

## 판정 (실행 전 고정): "런타임 기본 guide를 계약 guide로 교체"
1. 계약 guide가 9개 코드 모두에서 그 코드의 pc를 전부 담는다(단위 테스트, 결정론)
2. 구 guide에서 입력이 같던 두 쌍(Dm7/Dm7b5, G7/Gm7)에서 계약 guide의 대각 반응 CI 하한 > 0이다
   - 구 guide는 입력이 같아서 반응이 정의상 0이다. 같은 출력 비율 1.0으로 확인한다
3. 구 guide에서도 입력이 다르던 두 쌍에서 계약 guide 대각 반응 평균 ≥ 구 평균 − 0.02다(악화 없음)
4. invalid 비율(신) ≤ 구 + 0.02, 빈 블록 비율(신) ≤ 구 + 0.05
- 모두 충족하면 다음 PR에서 런타임 기본 guide를 계약 guide로 바꾸고 런타임 회귀 검사(대체 패턴, 지연)를 한다
- 통과해도 "틀린 음이 줄었다"거나 "듣기 좋아졌다"고 주장하지 않는다. 입력 정보를 보존하는 수정의 가치와 음악적 개선은 따로 판단한다
- 미달이면 기록한다. 음을 걸러 내거나 다시 튜닝해서 코드톤을 올리지 않는다
