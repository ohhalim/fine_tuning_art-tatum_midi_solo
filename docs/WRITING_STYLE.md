# 커밋·PR·이슈 작성 규칙

반년 뒤의 나와 리뷰어가 읽는 글. 무엇을 바꿨는지는 diff에 이미 있음. diff에 없는 것,
즉 왜 바꿨는지와 무엇을 확인했는지를 적음.

근거는 아래 자료. 한국어 어미와 줄 길이는 공인된 기준이 없어 이 저장소에서 정함.

- 리눅스 커널 [submitting-patches](https://www.kernel.org/doc/html/latest/process/submitting-patches.html) "Describe your changes"
- Git [SubmittingPatches](https://git-scm.com/docs/SubmittingPatches)
- Google [Writing good CL descriptions](https://google.github.io/eng-practices/review/developer/cl-descriptions.html), [How to write code review comments](https://google.github.io/eng-practices/review/reviewer/comments.html)
- Go [Contribution Guide](https://go.dev/doc/contribute) 커밋 메시지 절
- Mozilla [Bug writing guidelines](https://bugzilla.mozilla.org/page.cgi?id=bug-writing.html)
- Claude Code 팀 [How the Claude Code team does code review](https://claude.dev/blog/solving-code-review-at-anthropic/) (2026-10-15)

## 공통

문장 끝은 명사형으로 맺음. "리뷰 권고 단위", "실행 전 커밋", "~ 확인", "~ 없음", "~함",
"~됨"처럼 씀. "~입니다", "~했습니다", "~한다", "~했다"로 끝내지 않음.

문장으로 씀. 가운뎃점(·)은 "커밋·PR"처럼 짧은 명사 두세 개를 묶을 때만 쓰고, 문장을
줄이는 데 쓰지 않음. "~의 ~의"로 명사를 쌓지 않음. 화살표, 등호, 빗금을 문장 대신 쓰지
않음. 수식과 변수 이름은 백틱으로 감쌈.

숫자는 주장을 뒷받침할 때만, 비교 기준과 함께 씀. 제목에는 넣지 않음.

리뷰 메시지 ID나 문서 안의 검사 코드(C1, B2 같은 것)는 쓰지 않음. 필요하면 "입력 무결성
검사"처럼 풀어 씀.

해당 없는 항목은 비워 두지 말고 지움. "학습 없음", "청취 없음" 같은 문구를 매번 붙이지
않음. 범위를 오해할 만할 때만 한 줄.

굵은 글씨 라벨 목록(`**라벨:** 설명`), 이모지, 하위 제목만 담은 제목은 쓰지 않음.

본 것과 짐작한 것을 구분. 확인하지 않은 것은 "확인하지 않음"으로 적음.

## 커밋

제목은 `type(scope): 명사구`, 한글 30자 안팎. "이 커밋을 적용하면 ___"의 빈칸에 들어가는
말. 마침표 없음.

빈 줄 다음에 본문. 순서는 다음과 같음.

1. 지금 무엇이 문제인지
2. 이 커밋이 무엇을 바꾸는지
3. 왜 이 방식인지, 버린 대안이 있으면 그것까지

본문은 짧은 문단, 한글 36자(72열) 근처에서 줄바꿈, Markdown 없음.

실험 결과 수치는 본문에 나열하지 않음. 결과는 실험 문서에 두고 본문에는 결론 한 문장.
본문이 길어지면 커밋을 나눌 신호.

꼬리말은 `Refs: #이슈`, 버그 수정이면 `Fixes: <커밋 해시> ("<제목>")`.

예전:

```
exp(realtime): 실제 cache 상태 연속 요청 재생의 사전 등록·스크립트·시험 추가

#1794 리뷰 권고 단위입니다. 실행 전 커밋입니다.

- 재사용 규칙: P_reuse = min(공통 앞부분, KV 유효 끝), crop이면 0, S=0이면 마지막 1토큰 재계산
- 사례: 긴 P 4쌍, crop 1쌍, S=0 1쌍 × fed/unfed. 매 재생 전 KV를 다른 곡 토큰으로 오염
- 시험 2개. 계약 변경·생성·학습·청취 없음
```

지금:

```
exp(realtime): 실제 KV 상태로 앞부분 재사용 재생 준비

#1794 리뷰 권고 단위. 실행 전 커밋.

앞부분 재사용과 전체 재계산의 동등성은 앞 요청 입력을 KV에 전부
넣은 상태에서만 확인. 실제 루프에서는 마지막으로 뽑은 토큰이 아직
KV에 없을 수 있어 같은 결과라는 보장 없음.

실제 루프와 같은 KV 상태에서 연속 요청을 재생해 전체 재계산과
비교하는 스크립트 추가. 잘못된 재사용을 일부러 넣은 두 경우가
실패로 잡히는지도 확인. 판정 기준은 실행 전에 문서로 먼저 등록.

Refs: #1795
```

## PR

제목은 `type(scope): 명사구 (#이슈)`.

본문 첫머리는 섹션 없이 두세 문단. 왜 바꾸는지, 핵심 결정, 리뷰어가 특히 볼 곳.
파일 목록을 다시 읽어 주지 않음.

그 아래에 한두 줄씩.

- 테스트: 실제로 돌린 명령과 결과
- 위험: 깨질 수 있는 것
- 되돌리기: merge revert로 끝나지 않을 때만

마지막 줄에 `Closes #이슈`.

리뷰 지적에는 빠짐없이 답함. 고쳤으면 커밋, 안 고치면 이유 한 줄. 반영 요약은 "지적 3건
중 2건 수정, 1건은 의도된 동작"처럼 개수로 시작.

## 이슈

제목은 열 단어 안팎으로 문제를 적음. 해결책을 적지 않음.

버그 이슈 본문 순서는 다음과 같음.

1. 무엇을 했고 무엇을 봤는지(재현 방법, 정확한 수치, 출력)
2. 무엇을 기대했는지
3. 원인(확인한 것과 짐작한 것 구분)
4. 할 일

실험 이슈 본문에는 다음을 적음.

- 답할 질문 하나
- 실행 전에 정한 판정 기준
- 범위

할 일이 둘 이상이면 체크박스.
