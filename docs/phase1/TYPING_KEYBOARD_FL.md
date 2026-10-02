# FL Studio에서 치고 세럼으로 듣기 (#1581, 갱신)

작성 2026-10-01. `musical_quality_verified: false`. 사용자 실연주 전 문서다.

## 권장 경로: FL 자판 → MIDI Out → 런타임 → 세럼 (추가 도구 없음)

### 다시 켜는 법 (2026-10-01 설정 저장됨)
1. FL에서 iCloud `mvp/mvp.flp`를 연다(MIDI Out 채널 Port 5, Serum #2 입력 Port 7, Serum 입력 Port 6이 저장돼 있다)
2. 채널 랙에서 **MIDI Out**을 선택한다
3. 터미널:
   ```
   cd /Users/ohhalim/orca/workspaces/fine_tuning_art-tatum_midi_solo/즉흥연주재-설계
   /Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/.venv/bin/python scripts/fl_live.py --bars 64
   ```
   - 64마디 = 128 BPM에서 2분. Ctrl-C로 멈추면 모든 음을 끈다. `--preset mehldau`, `--no-follow`(코드 고정)
   - `--preset bebop`(#1620): 비밥 피아니스트 오른손으로 학습한 어댑터. 솔로는 맨 위 선율만, 컴핑은 근음·3음·7음을 1박과 3박 &에 짧게 친다(지금 코드만 알려 주는 용도)
- macOS IAC(포트 "AI In"과 두 번째 버스)와 FL MIDI 설정(출력 IAC AI In Port 5, 입력 IAC 두 번째 버스 Port 7)은 프로젝트 밖 설정이라 그대로 유지된다
- 2026-10-01 실연결: 입력이 런타임에 들어오고(150–176번 블록 입력 3음), AI 음이 Serum #2로 나왔다. 사용자 청취 소감은 "무작위적"이다. 연결은 되지만 내용은 아직 쓸 만하지 않다
사용자 지적: FL Studio는 자판 입력을 이미 받는다. 그래서 별도 자판 도구 없이 FL에서 MIDI를 내보내고 다시 받으면 된다.

```
FL(자판 → MIDI Out 채널, port 1) ──IAC Bus 1──▶ 런타임 ──IAC Bus 2──▶ FL(세럼 채널, input port 2) → 소리
```

1. macOS(한 번만): Audio MIDI 설정 → 윈도우 → MIDI 스튜디오 보기 → IAC Driver → "기기가 온라인 상태임", 버스 2개(Bus 1, Bus 2)
2. FL Options → MIDI settings
   - Output: `IAC Driver Bus 1`, Port **1**
   - Input: `IAC Driver Bus 2`만 Enable, Port **2**. Bus 1은 Input에서 켜지 않는다(자기 출력을 다시 받아 반복됨)
3. 채널
   - MIDI Out 채널 추가, Port **1**. 이 채널을 선택하고 자판으로 친다
   - 세럼: 플러그인 톱니바퀴 → MIDI → Input port **2**
4. 런타임
   ```
   cd /Users/ohhalim/orca/workspaces/fine_tuning_art-tatum_midi_solo/즉흥연주재-설계
   PY=/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/.venv/bin/python
   FORCE_CPU=1 $PY scripts/play_personalized.py --preset tatum --bars 128 --bpm 128 --chords Cmaj7 \
     --input-port "IAC Driver Bus 1" -- --port "IAC Driver Bus 2" --live-chords follow --chord-split 128
   ```
   - `--chord-split 128`: 높이와 무관하게 눌린 음 전부로 코드를 판단한다(FL 자판 옥타브는 확인 전)
- 한계: 템포는 FL과 맞추지 않는다(런타임 자체 128 BPM). FL 메뉴 이름은 버전마다 다를 수 있다. FL 경로 자체는 Claude가 GUI를 조작할 수 없어 검증하지 못했다

### FL 대역 시뮬레이션 (2026-10-01, 1회, 탐색)
`outputs/sim_fl/sim_fl.py`(git 미추적): 가상 출력 포트가 FL MIDI Out 역할로 코드를 보내고, 가상 입력 포트가 세럼 역할로 출력을 받는다. 위 런타임 명령을 포트 이름만 바꿔 그대로 썼다.
- 처음에는 `--chord-split 128`이 범위(1–127) 밖이라 런타임이 거부했다. 안내 명령이 틀렸던 것이다. 128(전부)을 허용하도록 고쳤다
- 결과: 코드 4개(C-E-G → Cmaj7, Dm7, G7, Abmaj7) 모두 인식, 반영 지연 약 1.39초, fallback 0, 미스 0, 세럼 역할 포트에 음 415개 도착
- 각 코드를 누르고 1.5초 뒤부터 받은 음의 코드톤 비율: Dm7 0.63(정적 Cmaj7 기준 0.18), G7 0.54(0.35), Abmaj7 0.43(0.26)

---

# (참고) 터미널 자판 도구
FL을 쓰지 않을 때만 필요하다.

## 구성
```
컴퓨터 자판 ──(터미널 1: typing_keyboard.py)──▶ 가상 MIDI "TypingKeyboard"
                                                      │ --input-port
                                         (터미널 2: 런타임, Tatum/멜다우)
                                                      │ 가상 MIDI "ContinuousJazz"
                                                      ▼
                                               FL Studio 악기 → 소리
```

## 자판 배열 (`scripts/typing_keyboard.py`, 추가 패키지 없음)
| 줄 | 키 | 음 |
|---|---|---|
| 코드(왼손) | Z S X D C V G B H N J M | C3 C#3 D3 D#3 E3 F3 F#3 G3 G#3 A3 A#3 B3 |
| 멜로디(오른손) | Q 2 W 3 E R 5 T 6 Y 7 U I 9 O 0 P | C4부터 E5까지 |

- 코드: 0.15초 안에 같이 누른 키를 한 코드로 묶는다. 다음 코드를 누르거나 **스페이스**를 칠 때까지 계속 눌린 것으로 친다
  - 예: Dm7 = X V N(D F A) 또는 X V N + Z(C3, 근음 D가 베이스가 아니어도 인식)
  - 예: G7 = B M V(G B F), Cmaj7 = Z C B M(C E G B)
- 멜로디: 누르면 0.25초짜리 음이 나간다. `[` `]`로 옥타브를 옮긴다. Esc로 끝낸다
- 한계: 터미널은 키를 뗀 순간을 모른다. 그래서 멜로디 길이는 고정이고, 코드는 "다음 코드까지 유지"로 처리한다
- 노트북 자판은 동시에 4–5키 이상 누르면 일부가 안 잡힐 수 있다(자판 하드웨어 한계)

## 실행 순서
1. **터미널 1** (자판 도구를 먼저 켜야 런타임이 포트를 찾는다)
   ```
   cd /Users/ohhalim/orca/workspaces/fine_tuning_art-tatum_midi_solo/즉흥연주재-설계
   PY=/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/.venv/bin/python
   $PY scripts/typing_keyboard.py
   ```
   - 체크포인트와 최신 코드는 이 워크트리에 있다. 본체 저장소(`git_box/...`)는 오래된 main이고 체크포인트가 없다. 파이썬만 본체의 venv를 쓴다
2. **터미널 2** (런타임. 128마디 = 128 BPM에서 4분)
   ```
   cd /Users/ohhalim/orca/workspaces/fine_tuning_art-tatum_midi_solo/즉흥연주재-설계
   PY=/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/.venv/bin/python
   FORCE_CPU=1 $PY scripts/play_personalized.py --preset tatum --bars 128 --bpm 128 \
     --chords Dm7,G7,Cmaj7,Cmaj7 --input-port TypingKeyboard -- --live-chords follow --live-metrics
   ```
   - `--live-chords follow`: 왼손 코드를 다음 블록 코드로 쓴다(약 1–1.3초 뒤 반영). 빼면 `--chords` 진행 그대로다
   - 멜다우는 `--preset mehldau`
3. **FL Studio**
   - Options → MIDI settings → Input 목록에서 **ContinuousJazz**를 켠다(Enable). 런타임이 실행 중일 때만 보인다. 안 보이면 Rescan devices
   - 피아노 계열 채널을 선택해 두면 그 채널로 소리가 난다
   - 내가 치는 음도 듣고 싶으면 **TypingKeyboard**도 켠다
4. **터미널 1에 포커스를 두고** 친다(자판 입력은 포커스된 터미널로만 간다)

## 확인할 것 (청취 메모용)
- 왼손 코드를 바꾸면 1–2블록(약 1–2초) 안에 반주가 따라오는가
- 끊기거나 이상한 블록이 귀에 걸리는가. 터미널 2의 `block … chord-tone …` 줄과 끝난 뒤 `continuous_report.json`의 `fallback_bar_count`로 대조할 수 있다
- 지연이 연주에 방해가 되는가

---

## 긴 실행 확인 (사전 등록, 실행 전)
- 마디 상한을 16 → 256으로 늘렸다. 16은 첫 구현(ace0787f)의 범위 축소였고 기술적 근거가 기록돼 있지 않다
- 측정: Tatum 프리셋, 128마디(반 마디 블록 256개), 128 BPM, 입력 없음, seed 42, `--capture`, 1회
- 판정
  1. 128마디를 끝까지 연주하고 fallback 0(미스는 보고만 한다)
  2. 보고만: 실행 시작 → 첫 블록까지 시간, 블록 생성 시간 p50·최대, 끝까지 생성 시간이 늘어나는 추세가 있는지
- 미달이면 원인을 기록한다. 상한 확대는 유지하되 체크리스트에 주의를 적는다

## 긴 실행 결과 (2026-10-01): 끝까지 연주, 추세 없음. fallback 3/256으로 판정 1 미달
원시값: `outputs/long_run/tatum_128/`, 로그 `outputs/long_run_tatum_128.log`. 실행 중 부하 평균 6.3–7.8(다른 작업과 동시).
- 128마디(256블록)를 끝까지 연주했다. 실행 전체 245초(연주 240초 + 시작 약 5초)
- fallback 3블록(1.2%): 검증 실패 2, 준비 시간 초과 1 ✗. 입력이 없어도 생긴다. 미스 0
- 블록 생성 ms p50 184, 최대 504. 4분위 구간별 중앙값 185 / 180 / 180 / 184 → 길게 돌려도 느려지지 않는다
- 캡처: 보낸 음 이벤트 5,954개를 모두 같은 순서로 받았다(손실·중복 0)
- 결론: 상한 확대는 유지한다. 긴 연주에서는 1분에 1블록(0.9초) 정도 fallback 패턴이 끼어들 수 있다. 소리가 끊기지는 않는다

## 자판 도구 동작 확인
- 가상 터미널로 X V N(D F A) → Q → 스페이스 → Esc를 보냈다
- 받은 MIDI: D3 F3 A3 on(20 ms 간격), C4 on → 0.26초 뒤 off, 스페이스에서 D3 F3 A3 off, Esc로 정상 종료
- 첫 시도에서 키를 cbreak 전환 전에 보내서 버려졌다. 준비 문구를 전환 뒤에 출력하도록 고쳤다. 문구가 보인 뒤에 치면 된다

## 되돌아오는 AI 음 거르기 (`--ignore-echo-ms`, 2026-10-01)
- FL에서 직접 확인: IAC 버스 2(포트 7)로 보낸 음이 Serum #2(입력 포트 7)뿐 아니라 **선택된 채널에도** 들어갔다. 선택 채널의 입력 포트를 6으로 바꿔도 같았다
- 그래서 연주용으로 MIDI Out 채널을 선택해 두면 AI 음이 MIDI Out → IAC AI In → 런타임 입력으로 되돌아온다
- FL 대역 시뮬레이션(받은 AI 음을 그대로 되돌림)
  - 거르지 않음: 코드 인식이 Ebm7b5, Am7, Bbm7b5로 틀어지고 입력 이벤트 806개, fallback 1, 미스 2
  - `--ignore-echo-ms 30`: 되돌아온 음 829개를 버리고 사용자 코드 이벤트 30개만 통과. Cmaj7, Dm7, G7, Abmaj7 정확 인식, fallback 0, 미스 0
- 방식: 런타임이 보낸 음을 기억하고, 같은 종류(on/off)·같은 음높이의 입력이 N ms 안에 오면 한 번 버린다. 사용자가 AI와 같은 음을 30 ms 안에 치면 그 한 번은 버려질 수 있다
