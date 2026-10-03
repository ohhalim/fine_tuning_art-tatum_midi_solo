# 런타임 --comp-style varied 회귀 검사 (사전 등록)

작성 2026-10-03. 이슈 #1644. 선행: `COMPING.md`(통과). 판정 규칙은 실행 전에 고정했다. `musical_quality_verified: false`.

## 변경
- `run_continuous_jazz.py --comp-style {shell, varied}`(기본 shell = 기존과 같다), `fl_live.py --comp-style`
- varied는 블록마다 `comp_half`를 부르고 상태(figure, 보이싱, 코드)를 이어 간다
- 한계: 상태는 생성 순서를 따른다(숨 쉬기와 같다). 블록이 폐기되거나 fallback이 나면 실제 재생과 어긋날 수 있다

## 실행
- `--preset bebop -- --solo-line --comp --comp-style varied`
- holdout 진행 3개(Gm7,C7,Fmaj7,Fmaj7 / Bbmaj7,G7,Cm7,F7 / Bm7b5,E7,Am7,Am7), 16마디, 128 BPM, seed 42

## 판정 (실행 전 고정, 3회 모두)
1. `solo_line_render.rendered_invalid` = 0, `raw_invalid` = 0
2. 대체 패턴 0
3. deadline miss ≤ 1(회당)
4. 재생된 G3 미만 음(컴핑) 수 ≥ 같은 진행·seed로 `comp_half`가 설계한 G3 미만 음 수의 90%
   - 솔로와 같은 음이 시간상 겹치면 빠지는 규칙이 있으므로 100%는 요구하지 않는다
- 통과하면 `fl_live`의 기본값을 varied로 바꾼다. 미달이면 기록하고 shell을 유지한다

---

## 결과 (2026-10-03): **통과**
원시값은 `outputs/runtime_comp_varied/`에 있다. 사전 등록은 실행 전에 push했다(723313ba).

| 진행 | 렌더 | 대체 패턴 | miss | 재생된 G3 미만 음 / 설계한 수 |
|---|---|---|---|---|
| Gm7 C7 Fmaj7 | rendered 32, invalid 0 | 0 | 0 | 56 / 53 |
| Bbmaj7 G7 Cm7 F7 | rendered 32, invalid 0 | 0 | 0 | 39 / 37 |
| Bm7b5 E7 Am7 | rendered 32, invalid 0 | 0 | 0 | 55 / 53 |

- 판정 1–4 모두 충족 → **통과.** `fl_live.py`의 기본값을 `--comp-style varied`로 바꿨다(`--comp-style shell`로 되돌릴 수 있다)
- 주의: 재생 수가 설계 수보다 3–6% 많다. 솔로 맨 위 선율에도 G3 아래 음이 일부 있기 때문이다. 이 비율은 컴핑 전달률을 조금 과대평가한다. 컴핑 음이 빠진 경우를 정확히 세려면 채널이나 음 출처 표시가 필요하다
