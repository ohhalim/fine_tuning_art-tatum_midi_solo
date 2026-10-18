# 커밋·PR·이슈 작성 규칙

반년 뒤의 나와 리뷰어가 읽는 협업용 작업 문서. 포트폴리오 설명문 아님.

## 근거와 우선순위

1. 사용자 지시: `~/.codex/AGENTS.md` Writing Style Rules, 2026-10-18 명사형 종결 요청
2. 이 문서
3. 외부 지침
   - 리눅스 커널 [submitting-patches](https://www.kernel.org/doc/html/latest/process/submitting-patches.html) "Describe your changes"
   - Git [SubmittingPatches](https://git-scm.com/docs/SubmittingPatches)
   - Google [Writing good CL descriptions](https://google.github.io/eng-practices/review/developer/cl-descriptions.html), [How to write code review comments](https://google.github.io/eng-practices/review/reviewer/comments.html)
   - Go [Contribution Guide](https://go.dev/doc/contribute) 커밋 메시지 절
   - Mozilla [Bug writing guidelines](https://bugzilla.mozilla.org/page.cgi?id=bug-writing.html)
   - Claude Code 팀 [How the Claude Code team does code review](https://claude.dev/blog/solving-code-review-at-anthropic/) (2026-10-15)

외부 지침의 공통점: 무엇을 바꿨는지는 diff에 있으므로 짧게, 왜 바꿨는지와 무엇을 검증했는지를 기록. 한국어 어미는 외부 지침에 기준이 없어 사용자 지시를 따름.

## 문장

- 업무 항목형, 명사형 종결. 예: "리뷰 권고 단위", "실행 전 커밋", "병목 후보 제외", "~ 확인", "~ 없음"
- "~입니다", "~했습니다", "~한다", "~했다", "~하는 게 맞다" 같은 설명형 종결 최소화
- 구어체("발생했다", "쌓였다", "보인다")보다 관측, 증가, 제외, 분리, 검토, 정의, 기록
- 감상, 설득 문장, 도메인 일반론 제외
- 확실하지 않은 원인은 "후보", "가능성", "추가 검증 대상"으로 표시. 본 것과 짐작한 것 구분
- 수식과 변수 이름은 백틱. 화살표나 등호를 설명 대신 쓰지 않음
- 리뷰 메시지 ID, 문서 안의 검사 코드(C1, B2 등), 도구 이름은 본문에 쓰지 않음. 필요하면 "입력 무결성 검사"처럼 풀어 씀
- 해당 없는 항목은 지움. "학습 없음", "청취 없음" 같은 면책 문구를 매번 붙이지 않음. 범위를 오해할 만할 때만 한 줄
- 굵은 글씨 라벨 목록, 이모지, Markdown 제목(`##`) 남발 없음. 항목 묶음 이름은 평문 한 줄(예: `배경`)

## 수치

- 관측값과 판단 근거는 수치 기반
- 판단에 필요한 핵심 수치, 측정 조건, 비교 기준(before/after)은 본문에 기록. 전체 표와 상세는 실험 문서에 두고 링크
- 성능 변경은 before/after 수치, 테스트 조건, 제외된 병목 후보, 남은 병목을 함께 기록
- 제목에는 결과 수치 없음. 이슈 번호, 버전, 버그 재현값은 예외

## 배경 순서

1. 이전 작업 결과
2. 주요 측정값
3. 제외된 후보
4. 현재 구조의 문제 지점
5. 이번 작업의 검토 대상

## 커밋

- 제목: `type(scope): 명사구`, 한글 30자 안팎, 마침표 없음. "이 커밋을 적용하면 ___"의 빈칸에 들어가는 말
- 빈 줄 다음 본문. 문제 지점, 변경, 이유(버린 대안 포함) 순서의 짧은 항목. 한글 36자(72열) 근처 줄바꿈
- 한 커밋은 한 가지 일. 본문이 길어지면 커밋 분리 신호
- 꼬리말: `Refs: #이슈`, 버그 수정이면 `Fixes: <커밋 해시> ("<제목>")`

예전:

```
exp(realtime): 실제 cache 상태 연속 요청 재생의 사전 등록·스크립트·시험 추가

#1794 리뷰 권고 단위입니다. 실행 전 커밋입니다.

- 재사용 규칙: P_reuse = min(공통 앞부분, KV 유효 끝), crop이면 0, S=0이면 마지막 1토큰 재계산
- 시험 2개. 계약 변경·생성·학습·청취 없음
```

지금:

```
exp(realtime): 실제 KV 상태로 앞부분 재사용 재생 준비

#1794 리뷰 권고 단위. 실행 전 커밋.

- 기존 동등성 확인은 앞 요청 입력 전체를 KV에 넣은 상태 기준
- 실제 루프는 마지막 샘플 토큰이 KV에 없는 상태 가능, 이 조건의
  동등성 미검증
- 실제 루프와 같은 KV 상태에서 연속 요청을 재생해 전체 재계산과
  비교하는 스크립트 추가
- 잘못된 재사용 2종이 실패로 잡히는지 함께 확인
- 판정 기준은 실행 전 문서로 등록

Refs: #1795
```

## PR

- 제목: `type(scope): 명사구 (#이슈)`
- 첫 줄: 이 PR이 무엇인지 한 문장. 실패 기록이면 실패 기록이라고 명시
- 항목 묶음: `배경`, `변경`, `결과`(실험일 때), `검증`, `남은 리스크`. 해당 없는 묶음은 지움
- 리뷰어가 특히 볼 파일이나 함수가 있으면 `변경`에 표시
- `검증`에는 실제로 돌린 명령과 결과
- 되돌리기가 merge revert로 끝나지 않으면 `되돌리기` 추가
- 마지막 줄: `Closes #이슈`

리뷰 지적에는 빠짐없이 답변. 수정이면 커밋, 미수정이면 이유 한 줄. 반영 요약은 "지적 3건 중 2건 수정, 1건 의도된 동작"처럼 개수로 시작.

## 이슈

- 제목: 해결책이 아니라 문제, 열 단어 안팎
- 버그: 관측(재현 방법, 정확한 수치, 출력), 기대, 원인 후보(확인과 짐작 구분), 할 일
- 실험: 답할 질문 하나, 실행 전 판정 기준, 범위
- 할 일이 둘 이상이면 체크박스
