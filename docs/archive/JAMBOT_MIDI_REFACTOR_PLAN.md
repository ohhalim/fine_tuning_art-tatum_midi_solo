# JAM_BOT MIDI Realtime Refactor Plan

실시간 MIDI 즉흥 연주 시스템(FL Studio + MCP + Transformer)을 위한 실행 문서입니다.
핵심 원칙은 다음 2가지입니다.

1. 실시간성(지연/안정성)과 스타일 학습(Brad Mehldau-like)을 분리해서 설계한다.
2. Stage A(MVP) -> Stage B(제어 강화) -> Stage C(품질 확장) 순서로 진행한다.

---

## 1) 최종 목표

- 입력: 키보드로 친 chord/gesture MIDI
- 처리: 조건부 심볼릭 모델이 실시간으로 솔로 MIDI 생성
- 출력: FL Studio에서 즉시 파형 재생
- 라이브 환경 기준 안정 동작

---

## 2) 현재 문제 정의

- 오디오 기반(Magenta RT) 문서와 심볼릭 기반(Music Transformer) 코드가 혼재되어 경로가 분산됨
- 현 파이프라인은 오프라인 생성 중심이며 실시간 스케줄러 구조가 없음
- 스타일 충실도와 지연(특히 jitter) 지표가 명확히 측정되지 않음

---

## 3) 방향성 결정

- 제품 메인 라인: Symbolic MIDI Transformer + LoRA + Role-conditioned dataset
- 리서치 보조 라인: Magenta RT(오디오)는 별도 실험 폴더로 유지
- 학습: RunPod
- 실시간 추론: 로컬 GPU

---

## 4) KPI (숫자로 고정)

- Stage A 게이트
- E2E TTFN <= 120ms
- Dead-air(출력 공백) 임계치: gap >= 180ms 이벤트 최소화
- 코드 변화 반응 지연: 실연주에서 체감 가능한 수준으로 안정화
- 10분 연속 재생 중 크래시/멈춤 없음

- 최종 목표
- E2E TTFN < 80ms
- Jitter 5~10ms 범위
- 항상 1~2 bar ahead 스케줄 유지

---

## 5) 구현 단계

## Phase 0: 문서/경로 정리 (0.5일)

- README를 실제 실행 흐름 기준으로 재작성
- 본 문서를 단일 기준 문서로 사용
- MAGENTA_RT 문서는 "audio research only"로 역할 명확화

완료 기준
- 신규 기여자가 README만 보고 Stage A 학습/생성 실행 가능

## Phase 1: Role 데이터셋 파이프라인 (2일)

- 신규 스크립트: `scripts/prepare_role_dataset.py`
- 출력 구조
- `data/roles/lead/<sample_id>/conditioning.mid`
- `data/roles/lead/<sample_id>/target.mid`
- `data/roles/lead/<sample_id>/meta.json`

핵심 기능
- lower-register 기반 conditioning 추출
- transpose 증강(최소 ±6, 옵션으로 all-keys)
- 깨진 MIDI/빈 트랙 제거

완료 기준
- `data/roles/lead` 샘플셋 생성 성공
- 샘플 무결성(읽기/길이) 검사 통과

## Phase 2: 학습 파이프라인 전환 (2일)

- `scripts/train_qlora.py`를 role-conditioned 학습 모드로 확장 또는 분리
- 권장 신규 엔트리포인트: `scripts/train_role_lora.py`
- 최소 컨트롤 토큰 도입
- `ROLE_LEAD`
- `TEMPO_*`
- `COND_SEP`

완료 기준
- RunPod에서 `lead` 역할 학습 1회 완주
- 체크포인트/로그/샘플 출력 일관성 확보

## Phase 3: 생성 + 오프라인 평가 (1일)

- `scripts/generate.py`를 conditioning 입력 기반 생성으로 전환
- 신규 스크립트: `scripts/eval_offline_metrics.py`

평가 항목
- chord-tone ratio
- repetition rate
- note density
- dead-air proxy

완료 기준
- 학습 종료 후 자동으로 샘플 생성 + 리포트 저장

## Phase 4: 실시간 런타임 골격 (3일)

- 신규 패키지: `realtime/`
- `clock.py`
- `midi_input.py`
- `prompt_builder.py`
- `generation_worker.py`
- `scheduler.py`
- `main.py`

설계 원칙
- 입력/처리/생성/스케줄 분리
- chord change 또는 트리거 이벤트 시 재프롬프트
- 생성 결과는 항상 미래 시점에 스케줄

완료 기준
- 로컬에서 MIDI in -> MIDI out 루프 안정 동작
- 저 BPM 환경에서 10분 무중단

## Phase 5: RunPod 자동화 (1일)

- 신규 스크립트: `scripts/runpod_train_stage_a.sh`
- 실행 단계 자동화
- 데이터 준비
- 학습
- 샘플 생성
- 오프라인 평가

완료 기준
- 단일 명령으로 Stage A 학습 파이프라인 재현 가능

## Phase 6: 성능 최적화 (2일)

- KV cache 적용
- context window 256~512로 축소
- 필요 시 ONNX + int8 양자화

완료 기준
- Stage A KPI 안정 달성

---

## 6) Stage A 권장 하이퍼파라미터

- n_layers: 6~8
- d_model: 256~384
- n_heads: 4~8
- context/max_length: 256~512
- primer_max_tokens: 64 (dead-air 스윕 베스트)
- batch_size: GPU 상황 기준 최적화
- LoRA rank: 8~16 (초기 권장)

---

## 7) Dead-Air 스윕 결과 (2026-02-20)

결과 요약:
- Best candidate: `p64`
- 고정 모델: `checkpoints/jazz_lora_stage_a`
- 고정 생성 파라미터: `--primer_max_tokens 64`

핵심 지표:
- baseline(기존 stage_a): `avg_dead_air_ratio = 0.534149`
- p64(20샘플): `avg_dead_air_ratio = 0.454905`
- p96(20샘플): `avg_dead_air_ratio = 0.482452`

결정:
1. Stage A 기본 생성값은 `primer_max_tokens=64`로 고정한다.
2. split_pitch 재학습 계열(`sp52/55/...`)은 현재 기준으로 기본값 채택하지 않는다.
3. Stage B 전까지는 base LoRA(`jazz_lora_stage_a`) + `p64` 조합을 기준선으로 사용한다.

---

## 8) 실행 순서 (커밋 단위)

1. Commit 1
- `prepare_role_dataset.py`
- 데이터 구조 생성 및 검증

2. Commit 2
- `train_role_lora.py` 또는 `train_qlora.py` role 모드 확장
- role-conditioned 학습 1회 완주

3. Commit 3
- `generate.py` conditioning 입력 전환
- `eval_offline_metrics.py` 추가

4. Commit 4
- `realtime/` 런타임 골격 추가
- MIDI 입출력 및 스케줄러 연동

5. Commit 5
- `runpod_train_stage_a.sh` 자동화
- 재현성 점검

6. Commit 6
- KV cache + 성능 튜닝
- 필요 시 ONNX/int8 실험

---

## 9) 리스크 및 대응

- 리스크: 스타일은 붙었지만 실시간성이 깨짐
- 대응: 모델 크기/컨텍스트를 먼저 줄이고 선생성 길이를 고정

- 리스크: 실시간성은 되는데 스타일이 약함
- 대응: conditioning/target 데이터 정합성 개선, role 분리 강화

- 리스크: 라이브 중 큐 언더런(출력 공백)
- 대응: fallback phrase 즉시 전환 규칙 적용

---

## 10) 최종 산출물 정의

- RunPod 학습 자동화 스크립트
- role-conditioned 데이터 파이프라인
- 로컬 실시간 MIDI 런타임
- KPI 리포트(TTFN, dead-air, chord-response)
- 60초 데모 MIDI/영상(포트폴리오용)
