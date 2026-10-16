# 왼손 조건 학습 입력: 블록 요청 단위 계약과 train-only 내보내기 (#1756) — 사전 등록

작성 2026-10-15. 학습과 생성은 하지 않는다. 범위는 아스트라가 #1755 결과 리뷰에서 제안했다. 사용자가 자는 동안 Claude가 아스트라 리뷰 없이 준비하고 실행한다. **사후 리뷰 대상이다.** 실제 자료는 이 문서를 커밋한 뒤 한 번만 연다.

## 질문 (하나)

`g3` 왼손 조건을 Aria 학습 입력에 블록 요청 단위로 붙일 수 있나? 조건은 두 가지를 지켜야 한다.
- 모든 입력 위치의 조건이 실시간에서 그 위치가 가질 값과 같다
- 정답 onset이 조건을 고르지 않는다

이 두 가지를 지키는 계약과 내보내기를 만들 수 있는지 묻는다.

## 기각한 안

- **음 n의 토큰에 `snapshot(onset_n)` 조건을 주는 안**(Claude가 #1756 본문에 제안)
  - 아스트라 Q2로 기각했다. pitch를 고르는 시점에 아직 모르는 onset이 조건을 고르게 된다
  - 예(t0 = 0): 같은 prefix에서 다음 onset이 2000이면 snapshot은 937.5, 2400이면 1875다. 인간 음 on = 1200은 뒤의 경우에만 조건에 들어간다
  - #1754 감사의 onset별 분류는 감사로는 유효하다. 하지만 학습 입력 선택기로 쓰지 않는다

## 계약 — `scripts/block_request.py`

예제 하나 = 런타임 블록 요청 R_k 하나다(`rt_producer.py`, 제안 블록 계약과 되돌리기 규칙 포함). 토큰은 Aria abs 토큰이다.

| 항목 | 정의 |
|---|---|
| 요청 | k, `s_k = block_start(k, t0)`, 확정 창 `[h_(k−1), h_k)`, `h_k = horizon(k, t0)`, `h_(−1) = t0` |
| 문맥 | 모델이 이미 채택한 이력이다. 타깃 토큰열에 `commit_until(h_(k−1))`까지 적용해 확정된 토큰이다. primer(onset < t0인 음까지)에서 시작한다 |
| continuation | window(R_k가 확정하는 토큰) + boundary(확정되지 않은 `<T>`와 창 뒤 첫 음 하나). `<E>`는 넣지 않는다. 곡의 마지막 요청은 boundary가 없을 수 있다 |
| 예측 위치의 조건 | `payload(s_k)` 하나. 마지막 문맥 위치 p0 = ctx_end − 1부터 끝까지 모두 같은 값이다. 정답 onset과 무관하다 |
| 문맥 위치의 조건 | 위치 q는 토큰 q+1을 확정한 요청의 payload를 갖는다. primer 위치는 cold다 |
| payload | 타깃 이력이 아니라 **인간 음**(pitch < 55)으로 만든다. s_k의 g3 sounding을 12-PC multihot으로 나타내고, `available`을 붙인다 |

- **boundary를 넣는 이유:** boundary가 없으면 빈 window에는 label이 없다. 그러면 모델이 "이 조건에서는 쉰다"를 배우지 못한다
  - boundary 음은 다음 요청에서 window나 boundary로 다시 나올 수 있고, 그때 조건이 다를 수 있다. 실시간에서 투기 생성한 뒤, 조건이 바뀌면 다시 생성하는 구조와 같다
  - `loss_class`로 window와 boundary를 나눠 내보낸다. 어느 쪽에 loss를 걸지는 학습 설계에서 고른다
- **문맥 위치의 조건:** 되돌리기 규칙을 지키면, R_k가 시작할 때 KV에 남아 있는 값이 이것이다. 이 값들은 과거에 확정된 사실이라 누출이 아니다
  - 두 번째 구현(`simulate_kv`)으로 대조한다. 이 구현은 요청을 순서대로 재연하고, 조건이 바뀌면 `committed_end − 1`부터 KV를 다시 쓴다
- **KV 원칙(아스트라 Q3):** 조건이 바뀐 요청은, 조건이 바뀐 가장 이른 입력 위치(p0 = committed_end − 1)부터 다시 쓴다
  - 기존 adapter는 마지막 층 입력에 조건을 더한다(`aria_cond_smoke.py:158-163`). 그래서 그 위치의 마지막 층만 바뀔 것으로 보이지만, 보장하지 않는다
  - 주입을 구현할 때, 같은 prefix·같은 조건의 full recompute와 cache/rollback 경로의 첫 pitch logits를 비교해 확인한다
- **payload(아스트라 Q1):**
  - cold(블록 0)와 primer는 관측 불가다(`available` = 0). prebuffer는 인간 입력을 읽기 전에 돈다
  - warm인데 울리는 음이 없으면 "관측했고 비어 있음"이다(0 열두 개, `available` = 1)
  - 나이, velocity, 옥타브, s_k 뒤의 떼기는 넣지 않는다. 이 값은 저음 PC의 대리 정보이고, 코드나 왼손의 정답이 아니다
- 인간 음은 토큰열에 들어가지 않는다. 모델 이력과 인간 이력은 서로 다른 입력이다

## 내보내기 — `scripts/lh_request_export.py`

- `split`(본체 venv): `data/bebop_rh/manifest.json`의 train 189곡만 연다(SHA1 대조, val/test 미개봉)
  - 감사와 같은 로더(pretty_midi, tempo map 적용)를 쓴다
  - g3로 나눈다. pitch ≥ 55는 타깃 MIDI(원 velocity, 페달 CC 없음)로 쓴다. 인간 음과 타깃 음의 원래 ms는 `NNN.notes.json`에 남긴다(이탈 2 뒤)
- `build`(Aria venv): 실제 `AbsTokenizer`로 타깃 MIDI를 토큰화하고, 요청 표와 payload를 붙인다. 토큰화 인자는 `remove_preceding_silence=False, add_dim_tok=False`다(이탈 2 뒤)
  - t0 = PHASES[곡 번호 % 4]이고, 위상은 #1754와 같다
  - 곡마다 `requests/NNN.json.gz`(토큰, 토큰 ID, 요청 표)를 쓴다. 위치별 조건은 `position_requests` 규칙으로 다시 만든다
- 산출물은 repo 밖 `/Users/ohhalim/git_box/lh_request_export/`, 보고는 `docs/experiments/lh_block_request/report.json`이다

## 검사와 판정 (실행 전 고정)

| 검사 | 내용 | 통과 |
|---|---|---|
| C1 | 모든 요청의 문맥 위치 payload가 `simulate_kv` 재연과 같다 | 불일치 0 |
| C2 | 요청의 문맥 끝까지만 남긴 토큰열(뒤에 `<E>`)도 문맥의 확정 요청 배정이 같다. 타깃 미래가 문맥 조건을 바꾸지 못한다 | 불일치 0 |
| C3 | s_k까지 관측한 인간 음만으로 만든 payload가, 인간 음 전체로 만든 payload와 같다. 뒤의 note-on은 빼고, s_k 뒤의 떼기는 모르는 것으로 둔다 | 불일치 0 |
| C4 | 토큰열에 pitch < 55인 음이 없다 | 0 |
| C5 (이탈 2로 추가) | 토큰 음마다 절대 onset이 같은 pitch의 원래 타깃 음과 10 ms 안에서 맞는다. 토큰 시간과 인간 시간의 원점이 같다는 검사다 | 어긋난 음 0 |
| C6 (이탈 2로 추가) | `<D>` 토큰이 없다 | 0 |

- **입력 계약 구현 완료** = 189곡 모두 처리 + C1–C6 모두 0
- 하나라도 0이 아니면 검사기나 계약의 버그로 보고 조사한다. 고친 뒤 다시 돌리면 이탈로 기록한다. 계약과 payload 정의는 바꾸지 않는다

## 함께 보고 (판정과 무관)

- 요청 수, cold/warm 수, warm payload가 비어 있지 않은 비율, 서로 다른 PC 집합 수와 상위 5개
- 빈 window, boundary가 없는 요청, window·boundary 토큰 분포
- 문맥 길이 분포, 4096 토큰(`rt_producer.py` 캐시 용량)을 넘는 요청 수
- window + boundary가 블록당 64 토큰(`CAP`)을 넘는 요청 수
- 토큰 음 수 대 타깃 음 수(토큰화기가 합치거나 버린 음)

## 검사기 검증 (실행 전에 마침)

- `tests/test_block_request.py` 11개 통과. 아스트라 필수 fixture 다섯 가지를 모두 넣었다
  - 다음 target onset만 바꿔도 요청의 모든 조건이 같다. 아스트라 예(2000 대 2400, 인간 음 1200)를 그대로 넣었고, 기각한 규칙이면 두 경우가 달라진다는 것도 같이 확인한다. 무작위 suffix로도 확인한다
  - s_k 뒤의 인간 on/off는 보이지 않는다. s_k 전의 변화는 보인다(대조군)
  - 경계에서 첫 label과 그 예측 위치가 맞는다. p0가 `<S>`(BOS), 이전 dur, `<T>`인 세 경우를 모두 본다
  - cold, warm 빈 조건, 빈 window
  - 문맥이 정확히 확정된 모델 토큰열이다. 모든 토큰이 window 하나에 한 번씩 순서대로 들어가고, 인간 음은 들어가지 않는다
- 두 구현 대조: 무작위 곡 4개 × 위상 4개에서 문맥 위치 payload가 `simulate_kv`와 같다
- **변형 4/4 검출:** 기각한 onset 기준 규칙, p0 한 칸 어긋남, s_k 뒤 300 ms 관측, KV를 committed_end부터 무효화(한 칸 늦음)
- 실제 토큰화기 dry run: 실제 크기의 합성 곡 한 곡(약 300초, 타깃 2,400음)으로 돌렸다. 검사 넷 모두 0, 4.6초가 걸렸다

## 실행

```
/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/.venv/bin/python scripts/lh_request_export.py split data/bebop_rh/manifest.json /Users/ohhalim/git_box/lh_request_export
/Users/ohhalim/git_box/t3_aria/.venv/bin/python scripts/lh_request_export.py build /Users/ohhalim/git_box/lh_request_export /Users/ohhalim/git_box/t3_aria/aria docs/experiments/lh_block_request/report.json
```

- 한 번만 돌린다(약 15분 예상). 열 압력을 실행 전후로 확인한다

## 알려진 한계와 말하지 않는 것

- **teacher forcing:** 요청 안의 정답 오른손 토큰에는, s_k 뒤 인간 입력에 대한 연주자의 반응이 담겨 있을 수 있다. 조건 누출은 아니지만, 실시간에는 없는 정보를 입력 토큰으로 보는 노출 편향이다
- CAP(블록당 64 토큰)과 4096 용량은 모델링하지 않고 수만 보고한다. 문맥을 다듬는 방식은 학습 설계에서 정한다
- 타깃 음의 dur는 원 note 길이다(페달 미적용). #1750의 sounding 정의와 같다
- 되돌리기 규칙과 조건 주입은 구현하지 않았다. 이 내보내기는 그것이 구현됐을 때 학습이 맞춰야 할 입력을 만든다
- 화성 정답, 손 정확도, 스타일, 음악성, 학습 효과는 다루지 않는다

## 다음 (이번 범위 밖)

- 가장 작은 학습 실행 설계: 문맥 다듬기, loss_class 선택, adapter 크기, 같은 곡·같은 primer의 (a) 조건 없음 (b) 실제 조건 (c) 다른 곡 조건 비교, Aria 무조건 능력 보존

## 실행 기록 — 이탈 1건

- **1회차(92556c2e, 650초, 열 압력 0):** 189곡을 모두 처리했다. C1·C3·C4는 0이었고, **C2가 666건 불일치**였다
- **원인(실제 자료로 확인):** 666건 모두 문맥이 `<T>`로 끝나는 요청이었다. 다른 위치는 모두 그 끝의 `<T>` 하나이고, prefix 쪽 배정이 `None`이었다
  - C2는 문맥 끝에서 자른 토큰열로 요청 표를 다시 만든다. 그런데 `request_table`이 "모든 음이 확정되면" 멈춰서, 자른 토큰열 끝의 `<T>`를 확정하지 않았다
  - 전체 곡 토큰열에서는 마지막 음 뒤에 `<T>`가 0개라서 드러나지 않았다
  - 계약 산식(창, 조건, 문맥 배정)의 오류가 아니라 요청 표의 종료 조건 버그다
- **수정:** `request_table`과 `simulate_kv`가 `<E>` 앞의 모든 토큰이 확정될 때까지 돈다. 자른 토큰열 시험을 추가했고, 옛 종료 조건이면 그 시험이 실패한다. 시험은 12개다
  - 전체 곡에는 끝 `<T>`가 없으므로, 내보낸 요청 파일은 1회차와 바이트까지 같아야 한다. 2회차에서 곡별 파일 sha256으로 확인한다
- 1회차 보고는 repo 밖 `/Users/ohhalim/git_box/lh_request_export_run1_report.json`에 남겼다. `split` 산출물은 다시 만들지 않고 그대로 쓴다

## 실행 기록 — 이탈 2건째: 토큰화기 기본값이 시간 원점을 옮기고 무작위 `<D>`를 넣었다

- **2회차(55e6c98e, 2430초, 열 압력 0):** C1–C4가 모두 0이었다. 요청 수와 모든 분포가 1회차와 같았다
  - 그런데 곡별 요청 파일이 1회차와 바이트까지 같은 곡은 189곡 중 27곡뿐이었다. 바이트까지 같아야 한다는 Claude의 예측이 틀렸다
  - 1회차 파일은 같은 폴더에 덮어써져 직접 비교하지 못했다
- **원인 1 — 재현성(실제로 확인):** `AbsTokenizer._format()`이 150토큰이 넘는 곡마다 `<D>`(곡이 곧 끝남) 토큰을 넣는다. 위치는 끝에서 `−100 + random.randint(−10, 10)`이다
  - 같은 `PYTHONHASHSEED`에서도 프로세스마다 토큰열이 달랐다
  - `<D>`는 끝없는 실시간 생성에 맞지 않는 "곧 끝남" 신호다. 이 신호가 학습 window에 들어가면 안 된다
- **원인 2 — 시간 정렬(원인 1을 찾다가 발견, 실제로 확인):** `tokenize()`의 기본값 `remove_preceding_silence=True`는 모든 onset을 첫 타깃 음 시각만큼 당긴다. 인간 음과 payload의 s_k는 원래 ms다
  - 그래서 1·2회차의 요청 창과 payload는 곡마다 첫 타깃 음 시각만큼 어긋나 있었다. 189곡 모두 해당하고, 어긋남은 최소 95 ms, 중앙값 609 ms, 최대 8.6초였다. 134곡은 500 ms를 넘었다
  - **1·2회차 내보내기는 쓰지 않는다.** C1–C4는 내부 일관성만 보고 토큰 시간과 인간 시간의 원점은 보지 않아서 이것을 잡지 못했다. dry run의 합성 곡은 첫 음이 거의 0 ms라서 드러나지 않았다
- **수정(f 커밋 참조):**
  - 토큰화 인자를 `remove_preceding_silence=False, add_dim_tok=False`로 바꿨다
  - `split`이 타깃 음의 원래 ms도 남긴다(`NNN.notes.json`)
  - 검사 C5(시간 원점 정렬)와 C6(`<D>` 없음)을 더했다. 시간을 옮긴 토큰열, 빠진 음, `<D>`를 각각 잡는지 시험한다(시험 13개)
  - 계약, payload 정의, 판정 기준은 바꾸지 않았다. 판정 범위만 C1–C4에서 C1–C6으로 넓혔다
- **실제 곡으로 확인(실행 전):** 다섯 곡을 별도 프로세스에서 두 번 토큰화했다. 출력이 같았고 `<D>`는 0개였다. 첫 토큰 음 시각은 원래 시각과 5 ms 안에서 맞았다
- 3회차는 아스트라 검토 뒤 `split`부터 다시 돌린다
