# 연주 중 어댑터 선택 — Program Change / CC (사전 등록)

작성 2026-09-29. 이슈 #1525. `style_verified: false` · `musical_quality_verified: false` · 가상 CoreMIDI 포트, 실물 키보드 없음

**이 절은 측정 전에 작성했다.** 구현은 측정 전에 끝냈다.

## 목적
어댑터 스왑(#1519)은 고정 스케줄로만 바꿀 수 있었다. 연주자가 키보드에서 바꿀 수 있어야 "내가 실제로 연주에 쓴다"는 목표에 맞는다. 대부분의 키보드는 프리셋 버튼으로 Program Change를 보내므로 이를 선택 입력으로 쓴다. CC도 지원한다.

## 설계
- `scripts/adapter_bank.py` `LiveAdapterSelector`
  - producer가 이미 잡은 입력 스냅샷에서 선택 메시지를 찾는다. 가장 최근 메시지를 적용한다
  - Program Change p는 p번째 어댑터를 고른다(`--checkpoint`가 0번, 이후 `--swap-adapter` 순서). 어댑터 수를 넘는 p는 무시한다
  - `cc:N`은 CC N의 0–127을 어댑터 수로 균등 분할한다
  - 선택은 다음 선택 메시지가 올 때까지 유지된다. 입력 창(4초)이 메시지를 잊어도 유지된다
- `run_continuous_jazz.py --adapter-control program|cc:N`(`--input-port` 필요, `--adapter-schedule`과 동시 사용 불가)
  - 선택은 **다음에 생성하는 마디의 시작**에 적용된다
  - 리포트 `adapter_swap.control_events`에 메시지마다 세션 시작 기준 도착 시각, 값, 선택한 어댑터를 남긴다
- 예상 지연
  - producer는 한 마디 앞서 만든다. 마디 r 중에 도착한 메시지는, 마디 r+1 생성이 이미 시작됐으면 r+2에, 아니면 r+1에 적용된다
  - 따라서 **1–2마디**다(128 BPM에서 1.9–3.8초)
- `scripts/run_live_select_probe.py`: 가상 포트를 열고 런타임을 자식 프로세스로 띄운 뒤, 정해진 시각에 Program Change를 보낸다. 리포트에서 "도착 마디 → 새 어댑터가 처음 연주된 마디"를 계산한다

## 조건
- 쌍 S: Tatum 완성(`--checkpoint`, 0번) + 공통 base 멜다우 u128(`--swap-adapter mehldau=…`, 1번)
- 128 BPM, 16마디, 코드 primer 2 sub-block, seed 42/43/44
- 전송: 실행 후 12초 PC1, 20초 PC0, 28초 PC1(세션은 실행 후 약 6–8초에 시작해 30초 동안 이어진다)

## 판정 (모두 충족하면 합친다)
1. 단위 테스트: PC/CC 매핑, 유지, 중복 무시, 범위 밖 무시
2. 세 실행 모두에서 보낸 메시지가 전부 기록되고, 세션 안에 도착한 각 메시지의 **적용 지연이 1–2마디**다
3. 선택하지 않은 어댑터로 연주된 마디가 없다. 마디별 어댑터가 "직전 적용 메시지"와 일치해야 한다
4. fallback 0
5. 연주된 마디가 같은 seed 단독 세션의 같은 마디와 동일하다. 입력에 음이 없으므로 primer가 같기 때문이다
