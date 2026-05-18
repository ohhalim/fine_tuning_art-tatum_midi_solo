# One Month Roadmap

작성일: 2026-05-16

목표: 한 달 안에 `musical request -> model generation -> repaired valid MIDI output -> metrics`가 동작하는 모델 MVP를 만든다.

## Week 1. Generation Baseline 고정

목표:

- 현재 레포의 Stage A 생성 경로를 확인한다.
- valid MIDI를 만드는 최소 CLI를 확정한다.
- 실패 output을 감지한다.

작업:

- `scripts/generate.py` 실행 확인.
- `scripts/eval_offline_metrics.py` 실행 확인.
- `outputs/` 또는 `samples/stage_a/` 기준 출력 위치 결정.
- empty MIDI, density 0, decode failure를 실패로 분류.
- fallback phrase generator 초안 작성.
- CLI 입력을 MVP request field와 맞춘다.

산출물:

- `generated.mid`
- `metrics.json`
- generation failure case 기록.
- README 실행 명령 초안.

완료 기준:

- 한 명령으로 MIDI와 metrics가 생성된다.
- 실패 시 실패 이유가 JSON에 남는다.

## Week 2. Model Output Repair

목표:

- 기존 Stage A 모델 output을 fallback 전에 최대한 usable MIDI로 repair한다.

작업:

- `inference/app/postprocess.py`
- pitch range octave mapping.
- phrase start alignment.
- requested bars 기준 trim.
- model output gate 재검증.
- fallback 전환 조건 기록.

산출물:

- `model_repaired=true` 결과 JSON.
- fallback 없이 통과하는 repaired model MIDI.

완료 기준:

- 기존 Stage A 모델 output이 최소 1개 이상 fallback 없이 gate를 통과한다.

## Week 3. Model Quality Loop

목표:

- fallback 비율을 낮추고 모델 출력 품질을 개선한다.

작업:

- seed/chord/density sweep.
- repaired output metrics summary.
- dead-air와 note density 기준 튜닝.
- 필요 시 `scripts/generate.py` sampling parameter 추가.
- 필요 시 tokenization/data format 재검토.

산출물:

- sweep metrics JSON.
- sweep summary Markdown.
- best generation settings.
- 실패 유형 목록.

완료 기준:

- model-first path가 반복 실행에서 일정 비율 이상 gate를 통과한다.

## Week 4. Quality Gate와 Portfolio Polish

목표:

- MVP를 설명 가능한 프로젝트로 정리한다.

작업:

- metrics table 정리.
- sample MIDI 3~5개 생성.
- failure handling 정리.
- README 업데이트.
- architecture diagram 문서화.
- demo script 작성.
- 수동 QA 체크리스트 통과.

산출물:

- 최종 README.
- sample request/response.
- sample MIDI outputs.
- metrics summary.
- 한계점과 next steps.

완료 기준:

- 처음 보는 사람이 README만 보고 실행할 수 있다.
- 면접에서 architecture와 tradeoff를 설명할 수 있다.

## Cut Line

시간이 부족하면 반드시 남길 것:

1. Python CLI valid MIDI generation.
2. model output repair.
3. fallback generator.
4. metrics JSON.
5. README.

시간이 부족하면 미룰 것:

1. realtime DAW routing.
2. control token retraining.
3. Spring Boot backend.
4. UI.
5. advanced jazz theory metric.
