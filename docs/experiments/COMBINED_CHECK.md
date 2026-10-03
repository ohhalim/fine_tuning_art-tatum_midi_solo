# 청취 설정 결합 검사 (사전 등록)

작성 2026-10-03. 이슈 #1654. 아스트라는 사용량이 끝났고 Claude 혼자 진행한다. 판정 규칙은 실행 전에 고정했다. `musical_quality_verified: false`.

## 설정
- `--preset bebop -- --solo-line --comp --comp-style varied --phrase-breath 24 --candidates 2 --context-carry-tokens 48 --context-carry-position after`
- 요소별로 앞서 통과한 단위
  - 후보 2개: #1646
  - carry48: #1652
  - 숨 쉬기: #1626
  - 새 컴핑: #1644, #1650
- **결합은 처음이다.** 후보 선택이 carry가 이어 준 선율을 깰 수 있다. 쉼이 과해질 수 있다(carry48에서 17.6%). 예산도 다시 봐야 한다
- holdout 진행 3개 × seed 42·43, 16마디, 128 BPM. 스크립트는 `scripts/combined_check.py`이고 집합과 설정을 검증한다

## 판정 (실행 전 고정)
1. 대체 패턴 0, miss ≤ 2, invalid 0, block-ready p99 ≤ 419 ms
2. 경계 음정 ≥ 9 비율 ≤ 0.22, 경계 p50 ≤ 4
3. 박 위 clash ≤ .397(carry48 단독 값). 후보 선택이 이 값을 낮추거나 적어도 유지해야 한다
4. 초당 음 수가 데이터의 0.67–1.5배, 쉼 ≥ 데이터의 0.5배
5. 복사 블록 ≤ 0.05
6. 컴핑 전달률 ≥ 0.99, composite 정렬 ≥ 0.95
- 보고만: 쉼의 상한 쪽(데이터 대비), 블록 안 음정, 컴핑만 본 정렬, unknown
- 통과하면 이 설정으로 청취물을 만들고 `fl_live`에 carry 선택지를 추가한다. 미달이면 어느 요소가 깨는지 기록하고, 청취물은 마지막으로 통과한 결합으로 만든다
