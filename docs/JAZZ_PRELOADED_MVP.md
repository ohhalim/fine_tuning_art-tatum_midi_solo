# 8마디 생성·재생 MVP

## 범위

- 기존 체크포인트 1회 로드 → 마디별 생성·검증 → 무효 블록 fallback → MIDI 저장
- 128 BPM 4/4 실제 마디 길이 1,875ms에 음표 경계 crop; 내부 1,880ms 양자화 오차 제거
- 정상적인 중간·마지막 쉼표 유지, 각 마디 끝에서 음 종료
- 선택한 MIDI 출력으로 lead만 재생; 20ms 초과 dispatch 지연은 기존 scheduler 정책대로 중단
- 정상 종료·예외·Ctrl-C에 reset/panic 수행

## 빠른 실행

```bash
# 체크포인트 없이 배선 확인: deterministic fallback 8마디
uv run --with-requirements requirements.txt bash scripts/run_mvp_demo.sh

# 실제 모델: 경로를 로컬 파일로 지정
uv run --with-requirements requirements.txt python scripts/run_jazz_mvp.py \
  --checkpoint /path/to/checkpoint.pt \
  --conditioning-midi /path/to/conditioning.mid \
  --bars 8 --bpm 128 --chords Dm7,G7,Cmaj7,A7 \
  --output-dir outputs/jazz_mvp/my_run
```

출력:

- `lead.mid`: 생성 솔로
- `with_chords.mid`: 같은 솔로 + 별도 트랙의 코드 참고 반주
- `report.json`: 마디별 model/fallback 출처, 준비시간, 검증 실패 원인

FL Studio에서 `with_chords.mid`를 가져와 피아노 악기에 연결해 청취.
직접 MIDI 출력은 위 명령에 `--port '정확한 MIDI 출력 이름'` 추가.
포트는 미리 준비되어 있어야 하며 lead만 출력. 기존 FL 악기 연결과 오디오 출력은 사용자 환경에서 확인 필요.

## 검증 및 제한

- D1 epoch8 체크포인트, seed 42..49, 128 BPM, 8마디 실행: model 8/8, fallback 0/8, 15초 시간창
- 전체 quick + demo: 73 tests 통과; demo 기본 경로는 fallback-only
- 생성 완료 후 재생하는 preloaded MVP. 실시간 생성 producer나 사람 MIDI 입력 반응 미구현
- 각 마디는 같은 32-token primer 사용. 마디 간 프레이즈 기억 미구현
- `--chords`는 참고 반주와 fallback에만 반영. 모델의 코드 조건 학습/준수 주장이 아님
- 샘플 8개 준비시간으로 R2 p99, 장시간 안정성, 음악 품질을 주장하지 않음
- MIDI 포트 send 성공과 실제 capture/audio 검증 분리
- 기존 미커밋 generated-block 변환기와 scheduler 경계 검증 변경 위에 연결
