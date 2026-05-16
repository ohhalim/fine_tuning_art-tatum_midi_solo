# Current Status and Revised Plan

작성일: 2026-05-16

이 문서는 현재 프로젝트의 기준 문서다. 기존 `docs/JAMBOT_MIDI_REFACTOR_PLAN.md`는 초기 리팩터링 계획과 2026-02-20 dead-air 실험 결과 기록으로 유지하고, 앞으로의 실행 순서는 이 문서를 기준으로 갱신한다.

MVP 구현을 위한 세부 문서는 `docs/README.md`에서 시작한다.

## 1. 현재 결정 사항

- 메인 라인: Symbolic MIDI 기반 `Music Transformer + LoRA`.
- 보조 리서치 라인: `magenta-realtime/` 및 Magenta RT 문서는 오디오 기반 실험 자료로만 유지.
- 학습 위치: RunPod GPU.
- 생성/실시간 추론 목표 위치: 로컬 머신 + FL Studio/MCP 연동.
- Stage A 기본 생성값: `--primer_max_tokens 64`.

## 2. 현재 구현 상태

현재 브랜치는 `feature/magenta-rt-jazz-finetuning`이다.

완료된 축:

- `scripts/prepare_role_dataset.py`
  - `conditioning.mid`, `target.mid`, `meta.json` 생성.
  - `conditioning + TOKEN_END + target + TOKEN_END` 형식으로 tokenized train/val 생성.
- `scripts/train_qlora.py`
  - Music Transformer에 LoRA를 붙여 학습.
  - best validation 기준으로 `lora_weights.pt` 저장.
- `scripts/generate.py`
  - LoRA checkpoint와 conditioning MIDI를 받아 MIDI 샘플 생성.
  - Stage A 기본 primer 길이는 64 token.
- `scripts/eval_offline_metrics.py`
  - note density, dead-air proxy, 4-gram repetition 평가.
- `scripts/runpod_train_stage_a.sh`
  - prepare/train/generate/eval 단일 실행 파이프라인.
- `scripts/run_dead_air_sweep.sh`
  - primer sweep, split_pitch 재학습 sweep, best candidate 선택, 재검증, archive 생성.

주의할 점:

- `music_transformer/generate.py`는 원본 Music Transformer 계열의 legacy generator다.
- 현재 Stage A에서 사용해야 하는 생성 엔트리포인트는 `scripts/generate.py`다.
- 로컬 워크트리에는 추적되지 않은 데이터, 샘플, 문서가 많다. 문서/코드 작업 시 기존 산출물을 정리하거나 삭제하지 않는다.

## 3. 현재 로컬 산출물

데이터:

- `data/roles/lead/dataset_summary.json`
  - samples: 18
  - train: 16
  - val: 2
  - transpose_all_keys: false
- `data/roles_sp60/lead/dataset_summary.json`
  - samples: 216
  - train: 194
  - val: 22
  - transpose_all_keys: true

체크포인트:

- `checkpoints/jazz_lora_stage_a/lora_weights.pt`
- `checkpoints/jazz_lora_sp60/lora_weights.pt`

실험 기록:

- README와 `docs/JAMBOT_MIDI_REFACTOR_PLAN.md`에는 2026-02-20 기준 full dead-air sweep 결과가 기록되어 있다.
- 현재 로컬 `samples/dead_air_sweep_smoke/`는 smoke run 결과이며, full sweep 결과와 동일한 기준 데이터로 보지 않는다.

## 4. Stage A 재현 명령

가장 빠른 전체 실행:

```bash
bash scripts/runpod_train_stage_a.sh \
  --mode all \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --transpose_all_keys \
  --overwrite \
  --install_deps
```

기존 checkpoint로 생성/평가만 확인:

```bash
python scripts/generate.py \
  --lora_path ./checkpoints/jazz_lora_stage_a \
  --conditioning_midi ./data/roles/lead/000000/conditioning.mid \
  --primer_max_tokens 64 \
  --num_samples 5 \
  --length 512 \
  --max_sequence 512 \
  --output ./samples/stage_a

python scripts/eval_offline_metrics.py \
  --input ./samples/stage_a \
  --dead_air_threshold_ms 180 \
  --output_json ./samples/stage_a/metrics.json
```

## 5. 다시 잡은 작업 계획

### Phase 0. 문서와 기준선 정리

목표:

- 현재 기준 문서를 만든다.
- README에서 최신 기준 문서로 연결한다.
- legacy generator와 Stage A generator의 역할을 명확히 분리한다.

완료 기준:

- `docs/CURRENT_STATUS_AND_PLAN.md` 존재.
- README의 참고 문서에 현재 기준 문서가 표시됨.

### Phase 1. Stage A 재현성 고정

목표:

- 기존 checkpoint로 `generate -> eval`을 다시 실행해 현재 환경에서 결과가 나오는지 확인한다.
- `samples/stage_a/metrics.json`을 기준 출력으로 재생성한다.
- smoke run과 full run 결과를 구분해 기록한다.

완료 기준:

- `samples/stage_a/*.mid` 생성.
- `samples/stage_a/metrics.json` 생성.
- empty/undecodable MIDI가 있으면 원인 기록.

### Phase 2. 생성 품질 안정화

목표:

- 빈 MIDI 또는 note density 0 결과를 실패로 처리한다.
- `scripts/generate.py`에 sampling 제어값을 추가한다.
- 최소 후보:
  - temperature
  - top-k
  - top-p
  - retry-on-empty
- 평가 스크립트에 fail gate를 추가한다.

완료 기준:

- 생성 결과 중 빈 MIDI 비율이 리포트에 표시됨.
- dead-air, repetition, note density가 모두 gating 기준에 들어감.
- 실패 샘플은 파일명 또는 별도 JSON에 명확히 표시됨.

### Phase 3. Conditioning 의미 강화

목표:

- 현재 `TOKEN_END`를 separator처럼 쓰는 구조를 명시적인 control token 구조로 바꾼다.
- 후보 토큰:
  - `ROLE_LEAD`
  - `TEMPO_*`
  - `COND_SEP`
  - `BAR`
- 학습 스크립트를 `train_role_lora.py`로 분리할지, 기존 `train_qlora.py`에 role mode를 둘지 결정한다.

완료 기준:

- tokenized sequence format이 문서화됨.
- 새 포맷으로 작은 데이터셋 학습 1회 성공.
- 기존 Stage A 포맷과 새 포맷이 혼동되지 않음.

### Phase 4. Realtime 런타임 골격

목표:

- 새 패키지 `realtime/` 추가.
- 최소 모듈:
  - `clock.py`
  - `midi_input.py`
  - `prompt_builder.py`
  - `generation_worker.py`
  - `scheduler.py`
  - `main.py`
- 입력, 프롬프트 구성, 생성, 출력 스케줄링을 분리한다.

완료 기준:

- 로컬에서 MIDI in -> generated MIDI out 루프가 돌아감.
- 10분 smoke run 동안 크래시 없음.
- TTFN, queue underrun, generated notes 수가 로그에 남음.

### Phase 5. 실시간 KPI 튜닝

목표:

- TTFN과 jitter를 계측한다.
- 항상 1~2 bar ahead queue를 유지한다.
- queue underrun 시 fallback phrase를 사용한다.

완료 기준:

- Stage A runtime gate:
  - TTFN <= 120ms
  - 10분 연속 재생 크래시 없음
  - dead-air threshold 180ms 기준 악화 없음
- 이후 목표:
  - TTFN < 80ms
  - jitter 5~10ms 범위

### Phase 6. 포트폴리오 산출물

목표:

- 60초 데모 MIDI 생성.
- FL Studio/MCP 연결 흐름 캡처.
- 모델/데이터/결과 지표를 한 페이지로 요약.

완료 기준:

- `samples/demo_60s/` 산출물.
- 최종 README에 실행 명령, 결과 지표, 한계점 반영.

## 6. 다음 실행 순서

가장 먼저 할 일:

1. 기존 checkpoint로 `scripts/generate.py`를 실행해 현재 로컬에서 Stage A 샘플이 정상 생성되는지 확인한다.
2. `scripts/eval_offline_metrics.py`로 metrics를 재생성한다.
3. 빈 MIDI 또는 density 0 샘플이 나오면 생성 실패 처리와 retry 정책부터 넣는다.
4. 그다음 control token 기반 Conditioning 포맷으로 넘어간다.

즉, 바로 realtime으로 가지 않는다. 먼저 현재 생성 파이프라인의 실패 조건을 명확히 잡고, 그다음 실시간 런타임으로 연결한다.
