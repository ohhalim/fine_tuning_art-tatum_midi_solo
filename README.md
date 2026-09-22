# 실시간 재즈 MIDI 즉흥연주

연주 입력 MIDI를 문맥으로 받아 다음 마디를 생성하고 MIDI 포트로 재생하는 로컬 실험 프로젝트입니다.
Music Transformer와 LoRA를 사용하며, 코드 진행 유도와 **Brad Mehldau 개인화 학습까지 시도**했습니다.

**현재 상태: 연속 생성·전송은 가상 CoreMIDI 환경에서 검증. 음악적 품질과 실제 FL Studio 협연은 미검증.**

## 어디까지 구현했나

| 항목 | 결과 |
|---|---|
| 연속 생성·재생 | 현재 마디 재생 중 다음 마디 생성, 최근 입력을 primer에 반영 |
| 실패 처리 | 늦거나 무효인 생성 결과 대신 준비된 fallback 사용, 종료 시 reset |
| MIDI 검증 | 가상 입력 → 모델 → 출력 → 독립 캡처. 기록된 4회 실행에서 32마디 완주, 532/532 이벤트 캡처 |
| 코드 진행 유도 | 기존 음표 토큰으로 코드 primer 구성. 한 마디 안에서 2회 화성 제시하는 opt-in 경로 |
| 멜다우 개인화 | 18곡으로 두 출발점의 LoRA 학습 및 비교, 생성·런타임 로드 확인. **스타일 적응 성공은 미입증** |

전송 성공, 제시간 생성, 좋은 음악은 서로 다른 기준으로 평가합니다.
위 짧은 실행 결과는 장시간 안정성이나 공연 준비 완료를 의미하지 않습니다.

## 빠르게 실행하기

Python 환경과 의존성은 `uv`로 준비합니다. CoreMIDI 검증은 macOS 기준입니다.

```bash
# 모델 없이 연속 재생 배선 확인: deterministic fallback + 가상 포트 캡처
bash scripts/run_continuous_demo.sh

# 실제 모델: 로컬 체크포인트와 primer MIDI 경로 지정
CHECKPOINT=/path/to/checkpoint.pt PRIMER=/path/to/conditioning.mid \
  bash scripts/run_continuous_demo.sh

# 실제 MIDI 장비: 기존 포트 이름 지정
CHECKPOINT=/path/to/checkpoint.pt PRIMER=/path/to/conditioning.mid \
  INPUT_PORT="keyboard port" OUTPUT_PORT="DAW port" \
  bash scripts/run_continuous_demo.sh
```

산출물은 `outputs/continuous/demo/continuous_report.json`, `played.mid`입니다.
가상 포트 검증만으로 소리가 나지는 않습니다. MIDI를 DAW 악기에 연결하거나 `played.mid`를 가져와 청취하세요.
체크포인트·원본 데이터는 저장소에 포함하지 않습니다.

### 코드 진행 유도 켜기

```bash
FORCE_CPU=1 uv run --with-requirements requirements.txt python scripts/run_continuous_jazz.py \
  --checkpoint /path/to/checkpoint.pt --conditioning-midi /path/to/conditioning.mid \
  --bars 8 --bpm 128 --capture --chord-primer --chord-blocks-per-bar 2 \
  --output-dir outputs/continuous/chord_demo
```

이는 **음표 기반 유도**입니다. 별도로 학습한 코드 조건 모델이 아니며, 화성 지표 개선이 음악적 품질을 보장하지 않습니다.
CPU 사용은 이 모델·기기에서의 측정 결과에 따른 선택입니다.

## 멜다우 개인화 시도

- 데이터: 18곡, 적응 학습용 train 16곡 / validation 2곡.
- 비교: 일반 재즈 base와 기존 Tatum-adapted 모델 각각에 멜다우 LoRA 적용.
- 학습: LoRA r=16, 8 epoch, seed 42. 기존 체크포인트 보존.
- 관측: 평가 loss 변화 약 −0.007 / −0.006(각 출발점 대비). 생성 특징에서 뚜렷한 개인화 효과를 확인하지 못함.
- 평가 제한: 18곡 모두 base 데이터와 토큰 시퀀스 일치, validation 2곡도 base train에 포함. **새 멜다우 곡에 대한 일반화 평가 불가.**
- 해석: 데이터 중복은 평가의 한계이며, 변화가 작은 원인을 단독으로 증명하지 않음. 학습 설정·업데이트 크기·평가 민감도도 후속 확인 대상.
- 사용자 초기 청취 의견: “느낌은 비슷하고 베이스 음 뒤 솔로 같은 선율이 반복”. 블라인드 스타일 검증 결과는 아님.

로컬 비교 파일: `outputs/mehldau_eval/{base,tatum,mehldau_from_base,mehldau_from_tatum}.wav`.
WAV는 사인 합성 참고음이며 실제 피아노 음색 평가는 별도입니다.
상세 설정과 수치는 [멜다우 개인화 실험](docs/experiments/MEHLDAU_PERSONALIZATION.md)을 참고하세요.

## 아직 남은 것

1. 실제 키보드·FL Studio에서 입력 반영 지연과 청취 품질 확인.
2. 마디 경계에서 음을 닫는 제약, 입력 신선도와 선행 생성 깊이의 균형 확인.
3. 개인화 평가는 base와 겹치지 않는 검증곡 및 더 명확한 스타일 기준 확보 후 재실험.

현재 코드는 공연용으로 검증된 완제품이 아닙니다. 자동 지표만으로 특정 연주자 스타일이나 음악적 우수성을 주장하지 않습니다.

## 검증

```bash
uv run --with-requirements requirements.txt bash scripts/agent_harness.sh quick
uv run --with-requirements requirements.txt bash scripts/agent_harness.sh demo
```

`demo`의 기본 경로는 fallback-only입니다. 실제 모델 실험과 별도로 해석해야 합니다.

## 문서 안내

- [재개 위치·실행 방법·알려진 제한](docs/phase1/RESUME_HANDOFF.md)
- [연속 런타임 구조](docs/phase1/CONTINUOUS_PATH.md)
- [코드 primer 비교 실험](docs/experiments/CHORD_PRIMER_AB.md)
- [멜다우 개인화 실험](docs/experiments/MEHLDAU_PERSONALIZATION.md)
- [기존 D0–D4 연구 기록](docs/RESEARCH_SUMMARY.md)

과거 Stage B 실험은 `archive/`와 연구 문서에 보존합니다. 현재 실행 진입점은 위 연속 런타임입니다.
