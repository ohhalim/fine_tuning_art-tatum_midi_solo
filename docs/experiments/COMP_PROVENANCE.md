# 컴핑 격자 반올림과 정확한 출처 조인 (사전 등록)

작성 2026-10-03. 이슈 #1650. 선행: `COMP_SOURCE.md`(#1649, 미달 보존). 아스트라와 합의했다. 판정 규칙은 실행 전에 고정했다. `musical_quality_verified: false`.

## 근거 (확인함)
- `scripts/generate.py encode_notes_simple`은 이벤트 사이 delta를 반올림하면서 `cur_time`은 원시 시각으로 갱신한다
- 모델 출력은 10 ms 격자 위에 있어서 그것만 있으면 오차가 0이다
- 박의 분수에 있는 컴핑(예: 128 BPM 1.5박 = 0.703125초)이 섞이면 그 뒤의 솔로까지 밀린다
- 무작위 시험(격자 밖 14음, 200회)에서 encode → decode onset 오차는 중앙값 9.1 ms, p95 19.4 ms, 최대 28.4 ms였다
- #1649에서 "13 ms 일찍" 나온 36개는 이 현상 때문이다. 측정만의 문제가 아니라 **실제 솔로 타이밍 결함**이다

## 변경
- `solo_with_comp_tokens`: 병합하기 전에 컴핑 시각을 10 ms 격자로 반올림한다. 컴핑만 최대 5 ms 움직이고 솔로는 움직이지 않는다
- 불변성 테스트(무작위 60블록)
  1. 최종 디코딩된 솔로의 pitch, onset, offset, velocity가 컴핑 없는 렌더와 같다
  2. emitted 컴핑이 디코딩 결과에 onset과 offset까지 1:1로 있다
  - 반올림을 끄면 실패한다(확인함)
- `comp_source_check.py`
  - onset과 offset이 정확히 같아야 조인한다(허용 6 ms, float 오차용)
  - 가까운 시각에 같은 음이 있지만 정확히 맞지 않으면 unknown으로 분류한다
  - fallback 블록의 음은 따로 센다
  - 솔로 지표에 끝쉼을 넣는다(`runtime_rh_check`와 맞춘다)
  - 첫 박 합성음(컴핑 + 솔로)에서 3음과 7음이 울리는지(composite)를 컴핑만 본 정렬과 따로 보고한다
- fixture: 합성 보고서로 fallback 블록 처리, 정확한 조인, 솔로가 컴핑에서 빠진 음을 대신 치는 경우, 끝쉼을 확인한다
- 한계
  - 폐기 → fallback 경로를 producer에서 실제로 일으키는 결정론 테스트는 아직 없다. 부분 송신도 마찬가지다
  - 컴핑과 숨 쉬기 상태는 생성 순서를 따른다

## 실행
- #1648과 같은 combined 설정과 진행·seed로 6회를 새로 돌린다(`outputs/comp_provenance`)

## 판정 (실행 전 고정)
1. 대체 패턴 0, invalid 0, miss ≤ 2, block-ready p99 ≤ 419 ms
2. 전달률(onset·offset 정확 조인) ≥ 0.99
3. unknown 음 = 0
- 보고만: 컴핑만 본 정렬, composite 정렬, drop 이유, 솔로 지표
- 정렬을 높이는 보이싱 변경은 다음 단위다. 이번 단위에서는 정렬을 판정에 쓰지 않는다
