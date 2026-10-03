# avoid 벌점 + 직전 음 반복 벌점 (사전 등록)

작성 2026-10-03. 이슈 #1658. Claude 혼자 진행한다. 판정 규칙은 실행 전에 고정했다. `musical_quality_verified: false`.

## 근거
- `HARMONY_BIAS.md`(#1656, 미달): avoid 벌점 2.0으로 avoid 음 시간 .161 → .026, 동시 충돌 .122 → .082가 됐다. 하지만 같은 음 반복이 .109 → .175로 늘었다
- 실제 비밥 RH 같은 음 반복은 .055다. 기준선(.109)부터 이미 높다

## 변경
- `--repeat-penalty R`(`--harmony-bias`와 함께 쓴다): 지금까지의 시퀀스(primer + 생성)에서 가장 최근 note_on 음의 logit에서 R을 뺀다
- **R = 1.5로 고정한다**(avoid 벌점 2.0보다 약하게. 반복음을 금지하지 않는다). 결과를 보고 다시 고르지 않는다
- 스모크(8마디, 측정 아님): 오류 0, 대체 0, 생성 최대 229 ms

## 실행
- 새 팔 `bias2rep1.5`: 청취 설정 전체 + `--harmony-bias 2 --repeat-penalty 1.5`, holdout 진행 3개 × seed 42·43
- 기준선은 #1656의 bias0 6회를 그대로 쓴다. 그 뒤로 코드 경로가 바뀌지 않았다(pattern-cache 통계 수정은 bias 경로에만 해당한다)
- `dissonance_check.py --label bias2rep1.5 --bias 2 --repeat 1.5`

## 판정 (실행 전 고정, 새 팔 vs bias0)
1. avoid 음 시간 ≤ bias0의 0.5배
2. 컴핑과 동시 충돌 시간 ≤ bias0의 0.7배
3. 박 위 clash ≤ bias0 + 0.01
4. 경계 연결: ≥ 9 비율 ≤ 0.22, p50 ≤ 4
5. **같은 음 반복 ≤ bias0(.109)**, 시작 음정 고유 비율 ≥ bias0의 0.9배, 복사 블록 ≤ 0.05
6. 초당 음 수와 쉼이 데이터 범위 안
7. 대체 0, miss ≤ 2, invalid 0, p99 ≤ 419 ms
- 통과하면 이 설정으로 청취물을 다시 만들고 `fl_live`에 선택지를 넣는다. 미달이면 기록하고 R과 S는 다시 고르지 않는다
