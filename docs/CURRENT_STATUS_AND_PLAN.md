# Current Status and Plan

작성일: 2026-05-28

## Current Focus

현재 이 저장소의 우선순위는 전체 jazz piano MIDI corpus를 audit하고, generic jazz pianist base를 만든 뒤 Brad Mehldau style adaptation으로 좁힐 수 있는지 검증하는 것이다.

현재 브랜치:

- 기준 브랜치: `main`

현재 active issue:

- latest functional result: Issue #246, Stage B margin-recovered proxy keep consolidation
- 다음 권장 이슈: `Stage B margin-recovered proxy keep focused package`

현재 범위가 아닌 것:

- Spring Boot backend
- API server MVP
- ERD/PostgreSQL job system
- realtime DAW/plugin integration
- SaaS/UI/product polish

위 문서들은 `docs/archive/`로 이동했다.

## Current Decision

Stage A는 아직 실사용 가능한 jazz solo model이 아니다.

이전에 생성된 MIDI는 `.mid` 파일로는 존재했지만, 실제 piano roll에서는 다음 문제가 있었다.

- note count가 너무 적음
- 긴 sustain block
- chord block처럼 보이는 출력
- solo-line으로 볼 수 없는 구조
- sparse/medium 일부에서 chord-tone 반응이 약함

따라서 지금의 목표는 "그럴듯한 제품 MVP"가 아니라, 전체 dataset 품질과 작은 probe를 통해 model training path를 검증하는 것이다.

## Current MVP Audit Result

Issue #220은 현재 작업이 core인지, MVP 완료로 볼 수 있는지를 검증한 audit이다.

판정:

- 이 작업은 model-core 작업이 맞다.
- pipeline MVP는 조건부 완료로 볼 수 있다.
- unconstrained trained-model MVP는 아직 완료가 아니다.

근거:

- `stage-b-window-prepare`: samples `70`, train/val `63/7`, vocab size `547`, fits vocab `true`
- `stage-b-generation-probe`: training path 실행, raw generated sample gate 실패, valid `0/1`, failure `note count too low: 3 < 6`
- `stage-b-overlap-gate`: constrained generation strict review gate 통과, valid `1/1`, strict `1/1`, grammar `1/1`
- `stage-b-rhythm-phrase-variation`: selected modes gate 통과, objective-clean candidates `6/6`, duplicate note sequences `0`

해석:

- dataset -> tokenization -> training -> generation -> decode -> objective review 연결은 검증됐다.
- raw trained model generation은 Stage B grammar를 안정적으로 유지하지 못했다.
- 지금 이력서/README claim은 "완성된 개인화 재즈 모델"이 아니라 "symbolic MIDI 생성 검증 파이프라인"이어야 한다.

Docs:

- `docs/STAGE_B_MODEL_CORE_MVP_COMPLETION_AUDIT_2026-05-28.md`

## Current Raw Generation Gate Result

Issue #222는 Issue #220에서 실패한 raw generation gate를 repair한 작업이다.

변경:

- `stage-b-generation-probe`를 50 epoch tiny-overfit, 5 samples, top_k `4`, overlap postprocess 조건으로 조정
- note group / valid / strict gate 실패 시 harness 실패 처리

검증:

- `RUN_ID=issue_222_stage_b_generation_probe bash scripts/agent_harness.sh stage-b-generation-probe`

결과:

- best validation loss: `1.6905`
- valid sample count: `5/5`
- strict valid sample count: `5/5`
- grammar gate sample count: `5/5`
- complete note groups: `21-22`
- invalid token count: `0`
- postprocess note count: `13-18`
- unique pitch count: `4-6`
- max simultaneous notes: `2`
- phrase coverage ratio: `0.8125-1.0`

해석:

- model-core MVP는 로컬 tiny-overfit/review gate 기준 완료로 볼 수 있다.
- 단, broad model quality, human listening preference, Brad style adaptation은 아직 미검증이다.
- postprocess 없이 solo-line polyphony gate를 통과한다고 주장하지 않는다.

Docs:

- `docs/STAGE_B_RAW_GENERATION_GATE_REPAIR_2026-05-28.md`

## Current Repeatability Sweep Result

Issue #224는 raw generation gate를 2-file/3-seed 조건으로 넓혀 검증한 작업이다.

검증:

- `RUN_ID=issue_224_stage_b_raw_generation_repeatability_final2 bash scripts/agent_harness.sh stage-b-raw-generation-repeatability`

조건:

- source files: `2`
- seeds: `17, 23, 31`
- samples: `9`
- epochs: `50`
- top_k: `4`
- overlap postprocess: enabled

결과:

- valid sample count: `8/9`
- strict valid sample count: `8/9`
- grammar gate sample count: `9/9`
- strict pass-rate: `0.889`
- max postprocess removal ratio: `0.429`
- seed `31`에서 dead-air ratio outlier `1`개 발생

해석:

- model-core MVP는 2-file/3-seed local repeatability gate까지 통과했다.
- broad model quality, human preference, Brad style adaptation은 아직 미검증이다.
- 다음 작업은 dead-air outlier 원인 분리다.

Docs:

- `docs/STAGE_B_RAW_GENERATION_REPEATABILITY_SWEEP_2026-05-28.md`

## Current Dead-Air Outlier Diagnostics Result

Issue #226은 Issue #224의 seed `31` dead-air outlier 원인을 분리한 작업이다.

검증:

- `REPORT_PATH=outputs/stage_b_generation_probe/issue_224_stage_b_raw_generation_repeatability_final2_seed31_files2/report.json RUN_ID=issue_226_stage_b_dead_air_diagnostics bash scripts/agent_harness.sh stage-b-dead-air-diagnostics`

결과:

- outlier sample: `1`
- dead-air ratio: `0.857`
- dead-air gate: `0.800`
- phrase coverage: `0.469`
- onset coverage: `0.250`
- sustained coverage: `0.375`
- tail empty steps: `11`
- longest sustained empty run: `10`
- dead-air start gaps: `6/7`
- postprocess removal ratio: `0.273`
- collapse warning: false

해석:

- 실패 원인은 collapse나 postprocess 과다 제거가 아니다.
- phrase span은 일부 확보했지만 onset/sustained coverage가 낮아 180ms 이상 start gap 비율이 과도한 후보다.
- 다음 작업은 raw 후보 중 dead-air가 낮은 strict-valid candidate를 우선 선택하는 gate다.

Docs:

- `docs/STAGE_B_DEAD_AIR_OUTLIER_DIAGNOSTICS_2026-05-28.md`

## Current Dead-Air-Aware Candidate Gate Result

Issue #228은 repeatability sweep에 dead-air outlier 집계와 strict-valid 후보 선택 기준을 추가한 작업이다.

검증:

- `ISSUE_NUMBER=228 RUN_ID=issue_228_stage_b_dead_air_candidate_gate bash scripts/agent_harness.sh stage-b-raw-generation-repeatability`

결과:

- total samples: `9`
- strict valid sample count: `8/9`
- grammar gate sample count: `9/9`
- dead-air outlier count: `1`
- dead-air outlier rate: `0.111`
- max allowed outlier rate: `0.250`
- selected best candidate: seed `17`, sample `3`
- selected best dead-air ratio: `0.333`
- seed `31` best strict candidate: sample `2`, dead-air `0.750`

해석:

- outlier를 숨기지 않고 별도 rate로 기록한다.
- outlier가 있는 seed에서도 strict-valid 대체 후보를 선택할 수 있다.
- 다음 작업은 source file 수를 늘려 candidate selection gate가 유지되는지 확인하는 것이다.

Docs:

- `docs/STAGE_B_DEAD_AIR_AWARE_CANDIDATE_GATE_2026-05-28.md`

## Current Broader Source Candidate Gate Result

Issue #230은 source file 수를 `3`으로 늘린 상태에서 candidate gate가 유지되는지 검증한 작업이다.

변경:

- `stage-b-raw-generation-repeatability` harness에서 `MAX_FILES`, `SEEDS`, `EPOCHS`, `NUM_SAMPLES`, `TOP_K`, `TEMPERATURE`, `MAX_DEAD_AIR_OUTLIER_RATE`를 env로 조정 가능하게 변경

검증:

- `ISSUE_NUMBER=230 MAX_FILES=3 MIN_SOURCE_FILES=3 RUN_ID=issue_230_stage_b_broader_source_candidate_gate bash scripts/agent_harness.sh stage-b-raw-generation-repeatability`

결과:

- source files: `3`
- total samples: `9`
- strict valid sample count: `7/9`
- grammar gate sample count: `9/9`
- dead-air outlier count: `2`
- dead-air outlier rate: `0.222`
- max allowed outlier rate: `0.250`
- selected best candidate: seed `17`, sample `3`
- selected best dead-air ratio: `0.222`
- seed별 strict-valid best candidate 존재: true

해석:

- 3-file 조건에서도 candidate gate는 통과했다.
- strict pass-rate는 2-file `8/9`에서 3-file `7/9`로 낮아졌다.
- 다음 작업은 source file 수를 더 늘릴 때 outlier rate가 gate를 넘는 경계를 확인하는 것이다.

Docs:

- `docs/STAGE_B_BROADER_SOURCE_CANDIDATE_GATE_2026-05-28.md`

## Current Larger Source Risk Boundary Result

Issue #232는 source file 수를 `4`, `5`, `6`까지 늘렸을 때 repeatability gate가 유지되는지 확인한 작업이다.

검증:

- `ISSUE_NUMBER=232 MAX_FILES=4 MIN_SOURCE_FILES=4 RUN_ID=issue_232_stage_b_larger_source_risk_boundary_files4 bash scripts/agent_harness.sh stage-b-raw-generation-repeatability`
- `ISSUE_NUMBER=232 MAX_FILES=5 MIN_SOURCE_FILES=5 RUN_ID=issue_232_stage_b_larger_source_risk_boundary_files5 bash scripts/agent_harness.sh stage-b-raw-generation-repeatability`
- `ISSUE_NUMBER=232 MAX_FILES=6 MIN_SOURCE_FILES=6 RUN_ID=issue_232_stage_b_larger_source_risk_boundary_files6 bash scripts/agent_harness.sh stage-b-raw-generation-repeatability`

결과:

| 항목 | 4 files | 5 files | 6 files |
|---|---:|---:|---:|
| repeatability gate | 통과 | 통과 | 통과 |
| strict valid samples | `8/9` | `7/9` | `7/9` |
| grammar gate samples | `9/9` | `9/9` | `9/9` |
| dead-air outliers | `1` | `2` | `1` |
| dead-air outlier rate | `0.111` | `0.222` | `0.111` |
| selected best dead-air | `0.438` | `0.467` | `0.375` |

해석:

- 4/5/6-file 조건 모두 hard gate는 통과했다.
- grammar gate는 모든 조건에서 `9/9`로 유지됐다.
- 6-file 조건에서 seed `17`은 strict `1/3`까지 내려갔고 `unique pitch count too low` failure가 새로 발생했다.
- 현재 boundary는 hard failure가 아니라 seed-level strict margin 감소다.
- broad quality나 Brad style adaptation이 증명된 것은 아니다.

Docs:

- `docs/STAGE_B_LARGER_SOURCE_RISK_BOUNDARY_2026-05-28.md`

## Current Seed Strict Margin Diagnostics Result

Issue #234는 Issue #232의 6-file run에서 seed `17`만 strict `1/3`까지 내려간 원인을 sample 단위로 분리한 작업이다.

변경:

- repeatability summary와 seed별 generation report를 읽는 진단 스크립트 추가
- seed별 strict margin warning, dead-air sample, unique-pitch sample, overlap sample 분류
- `stage-b-seed-strict-margin-diagnostics` harness mode 추가

검증:

- `bash scripts/agent_harness.sh stage-b-seed-strict-margin-diagnostics`

결과:

- source run: `issue_232_stage_b_larger_source_risk_boundary_files6`
- hard min strict per seed: `1`
- warning min strict per seed: `2`
- margin warning seeds: `17`
- dead-air + unique-pitch overlap seeds: 없음
- dead-air + unique-pitch separate seeds: `17`

seed `17` sample breakdown:

| sample | strict | issue |
|---:|:---:|---|
| `1` | false | dead-air ratio `0.857 >= 0.800` |
| `2` | false | unique pitch count `2 < 3` |
| `3` | true | strict-valid best candidate |

해석:

- 6-file 조건의 hard gate는 아직 유지된다.
- seed `17`도 strict-valid 후보가 하나 남는다.
- dead-air failure와 unique-pitch failure는 같은 후보에 겹친 collapse가 아니다.
- aggregate strict pass-rate만 보면 seed별 margin 감소를 놓칠 수 있다.
- 다음 작업은 per-seed strict margin을 hard gate와 분리된 warning/soft gate로 repeatability summary에 포함하는 것이다.

Docs:

- `docs/STAGE_B_SEED_STRICT_MARGIN_DIAGNOSTICS_2026-05-28.md`

## Current Seed Strict Margin Warning Gate Result

Issue #236은 repeatability summary에 seed별 strict margin warning을 추가한 작업이다.

변경:

- `--warning_min_strict_samples_per_seed` 인자 추가
- `strict_margin_warning_seed_count`, `strict_margin_warning_seeds`, `strict_margin_warning_rows` summary 추가
- summary markdown의 seed table에 `margin warning` column 추가
- harness에서 `WARNING_MIN_STRICT_SAMPLES_PER_SEED` 환경 변수 연결

검증:

- `bash scripts/agent_harness.sh stage-b-raw-generation-repeatability`
- `ISSUE_NUMBER=236 MAX_FILES=6 MIN_SOURCE_FILES=6 RUN_ID=issue_236_stage_b_seed_strict_margin_warning_gate bash scripts/agent_harness.sh stage-b-raw-generation-repeatability`

6-file 결과:

- repeatability gate: 통과
- strict valid samples: `7/9`
- grammar gate samples: `9/9`
- dead-air outlier rate: `0.111`
- hard min strict per seed: `1`
- warning min strict per seed: `2`
- strict margin warning seeds: `17`
- selected best candidate: seed `23`, sample `1`, dead-air `0.375`

해석:

- hard gate와 soft warning이 분리됐다.
- 6-file run은 기존 hard gate를 계속 통과한다.
- seed `17`의 strict margin risk가 aggregate pass-rate에 묻히지 않고 summary에 직접 드러난다.
- warning은 실패 처리가 아니라 다음 실험 우선순위 신호다.
- 다음 작업은 samples per seed를 늘렸을 때 seed `17` margin warning이 줄어드는지 확인하는 것이다.

Docs:

- `docs/STAGE_B_SEED_STRICT_MARGIN_WARNING_GATE_2026-05-28.md`

## Current Candidate Count Margin Recovery Result

Issue #238은 6-file 조건에서 samples per seed를 `3`에서 `5`로 늘렸을 때 seed `17`의 strict margin warning이 회복되는지 확인한 작업이다.

검증:

- `ISSUE_NUMBER=238 MAX_FILES=6 MIN_SOURCE_FILES=6 NUM_SAMPLES=5 RUN_ID=issue_238_stage_b_candidate_count_margin_recovery bash scripts/agent_harness.sh stage-b-raw-generation-repeatability`

결과:

| 항목 | 3 samples/seed | 5 samples/seed |
|---|---:|---:|
| total samples | `9` | `15` |
| strict valid samples | `7/9` | `12/15` |
| strict pass-rate | `0.778` | `0.800` |
| grammar gate samples | `9/9` | `15/15` |
| dead-air outlier count | `1` | `2` |
| dead-air outlier rate | `0.111` | `0.133` |
| strict margin warning seeds | `17` | 없음 |
| selected best candidate | seed `23`, sample `1` | seed `23`, sample `1` |

해석:

- candidate count를 `5`로 늘리면 seed strict margin warning이 사라진다.
- seed `17`은 strict `1/3`에서 `3/5`로 회복했다.
- hard gate는 계속 통과한다.
- dead-air outlier count는 `1`에서 `2`로 늘었지만 rate `0.133`은 gate `0.250` 안에 있다.
- 후보 수 증가는 selection 안정성은 높였지만 failure mode 자체를 제거하지는 않았다.

Docs:

- `docs/STAGE_B_CANDIDATE_COUNT_MARGIN_RECOVERY_2026-05-28.md`

## Current Margin-Recovered Candidate Review Export Result

Issue #240은 Issue #238의 6-file / 5-sample repeatability 결과에서 seed별 best candidate 3개를 objective review table로 추출한 작업이다.

변경:

- repeatability summary를 읽는 candidate review export script 추가
- selected best와 seed별 best candidate를 review rank로 정리
- objective metric markdown/json export 생성
- generated MIDI와 output artifact는 commit하지 않음

검증:

- `bash scripts/agent_harness.sh stage-b-margin-recovered-review-export`

결과:

| rank | selected | seed | sample | seed strict | outliers | dead-air | notes | pitches | phrase | onset | sustained | removal |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | true | `23` | `1` | `4/5` | `1` | `0.375` | `9` | `4` | `0.437` | `0.312` | `0.438` | `0.357` |
| `2` | false | `31` | `5` | `5/5` | `0` | `0.444` | `19` | `4` | `0.937` | `0.500` | `0.719` | `0.095` |
| `3` | false | `17` | `3` | `3/5` | `1` | `0.500` | `17` | `4` | `1.000` | `0.594` | `0.844` | `0.227` |

해석:

- selected best는 dead-air 기준으로 seed `23`, sample `1`이다.
- seed `31`, sample `5`는 selected best보다 dead-air는 높지만 note count와 coverage가 더 높아 listening 비교 가치가 있다.
- seed `17`, sample `3`은 strict-valid지만 seed 내부 failure가 남아 있어 안정성은 가장 낮다.
- 다음 작업은 rank `1`, `2`, `3` 후보를 listening review note template로 정리하는 것이다.

Docs:

- `docs/STAGE_B_MARGIN_RECOVERED_CANDIDATE_REVIEW_EXPORT_2026-05-28.md`

## Current Margin-Recovered Listening Review Notes Result

Issue #242는 Issue #240의 margin-recovered review export를 기반으로 listening review notes template을 생성한 작업이다.

변경:

- margin-recovered candidate review export를 notes template으로 변환하는 script 추가
- selected best 후보가 정확히 1개인지 검증
- 후보별 metric, seed/sample/rank, MIDI path를 notes에 보존
- listening fields는 실제 review 전까지 `pending` 유지

검증:

- `bash scripts/agent_harness.sh stage-b-margin-recovered-listening-notes`

결과:

- candidate count: `3`
- selected best count: `1`
- reviewed count: `0`
- pending count: `3`
- decision counts: pending `3`

후보:

| candidate | selected | dead-air | notes | phrase | onset | sustained | decision |
|---|:---:|---:|---:|---:|---:|---:|---|
| `margin_recovered_rank_1_seed_23_sample_1` | true | `0.375` | `9` | `0.437` | `0.312` | `0.438` | pending |
| `margin_recovered_rank_2_seed_31_sample_5` | false | `0.444` | `19` | `0.937` | `0.500` | `0.719` | pending |
| `margin_recovered_rank_3_seed_17_sample_3` | false | `0.500` | `17` | `1.000` | `0.594` | `0.844` | pending |

해석:

- 이 단계는 review 준비이며 청감 품질 판정이 아니다.
- rank `1`이 실제 listening preference에서도 best라고 확정한 것은 아니다.
- 다음 작업은 MIDI note/context evidence 기준으로 pending fields를 채우는 것이다.

Docs:

- `docs/STAGE_B_MARGIN_RECOVERED_LISTENING_REVIEW_NOTES_2026-05-28.md`

## Current Margin-Recovered MIDI Proxy Review Fill Result

Issue #244는 Issue #242의 pending notes를 MIDI metric 기반 proxy review로 채운 작업이다.

변경:

- margin-recovered listening notes를 proxy review로 채우는 script 추가
- dead-air, note count, phrase/onset/sustained coverage, postprocess removal, seed failure 상태를 기반으로 proxy score 계산
- 실제 청감 review가 아님을 output에 명시

검증:

- `bash scripts/agent_harness.sh stage-b-margin-recovered-proxy-review-fill`

결과:

| candidate | selected by dead-air | score | timing | phrase | vocabulary | decision |
|---|:---:|---:|---|---|---|---|
| `margin_recovered_rank_1_seed_23_sample_1` | true | `0.251` | stiff | weak | thin | needs_followup |
| `margin_recovered_rank_2_seed_31_sample_5` | false | `0.698` | acceptable | strong | acceptable | keep |
| `margin_recovered_rank_3_seed_17_sample_3` | false | `0.564` | acceptable | strong | acceptable | needs_followup |

해석:

- dead-air만으로 selected best를 고르면 phrase richness가 낮은 후보를 선택할 수 있다.
- 6-file 5-sample run에서 proxy 기준 가장 나은 후보는 rank `2` seed `31` sample `5`다.
- rank `2`는 seed 내부 failure가 없고, note count와 coverage가 가장 균형적이다.
- 이 결과는 MIDI metric proxy review이며 실제 청감 review가 아니다.

Docs:

- `docs/STAGE_B_MARGIN_RECOVERED_PROXY_REVIEW_FILL_2026-05-28.md`

## Current Margin-Recovered Proxy Keep Consolidation Result

Issue #246은 Issue #244의 MIDI metric proxy review 결과를 README와 상태 문서에서 사용할 수 있는 claim boundary로 정리한 작업이다.

변경:

- rank `2` seed `31` sample `5`가 proxy keep으로 선택된 이유 문서화
- rank `1` seed `23` sample `1`이 dead-air 기준 selected best였지만 needs_followup으로 내려간 이유 문서화
- proxy keep이 human listening preference나 broad trained-model quality claim이 아님을 명시
- README의 구현 범위 요약에 문제, 해결 방식, 검증 결과, 주장 경계를 추가

검증:

- `bash scripts/agent_harness.sh quick`

결과:

| 항목 | 결과 |
|---|---|
| proxy keep candidate | `margin_recovered_rank_2_seed_31_sample_5` |
| proxy score | `0.698` |
| decision | keep |
| timing | acceptable |
| phrase | strong |
| vocabulary | acceptable |
| rejected selected-best assumption | rank `1` dead-air best만으로 최종 선택 불가 |
| claim boundary | MIDI metric proxy review, not human listening proof |

해석:

- dead-air gate는 필요하지만 단독 ranking 기준으로는 부족하다.
- 현재 주장 가능한 것은 reviewable MIDI 후보를 검증하고 좁히는 pipeline이다.
- 다음 작업은 rank `2` 후보를 focused solo/context review package로 격리하는 것이다.

Docs:

- `docs/STAGE_B_MARGIN_RECOVERED_PROXY_KEEP_CONSOLIDATION_2026-05-28.md`

## Latest README Footer Section Removal Result

Issue #218은 README 하단의 참조성 섹션을 제거한 작업이다.

Updated file:

- `README.md`

Result:

- `주요 파일` 섹션 제거
- `현재 상태` 섹션 제거
- `문서` 섹션 제거
- README 흐름을 구현 내용, 문제 해결, 검증 결과, 실행 방법 중심으로 유지

Next recommended issue:

- `이력서 프로젝트 bullet 정리`

## Previous README Implementation Focus Result

Issue #216은 README를 구현 내용과 문제 해결 흐름 중심으로 재정리한 작업이다.

Updated file:

- `README.md`

Result:

- `구현한 것` 섹션을 추가해 dataset audit, tokenization, generation probe, MIDI decode, objective review, focused review, harness, documentation 범위를 명시했다.
- `문제와 해결` 표를 추가해 문제, 원인/관찰, 해결, 결과를 한눈에 볼 수 있게 했다.
- 평가형 섹션을 제거했다.
- current best focused review candidate evidence와 conservative claim boundary는 유지했다.

Next recommended issue:

- `이력서 프로젝트 bullet 정리`

## Previous README Business Style Result

Issue #214는 README 문체를 서술형 설명에서 명사형, 사무형 구조로 정리한 작업이다.

Updated file:

- `README.md`

Result:

- README 전체를 표, bullet, 명사형 항목 중심으로 재구성했다.
- 문제 정의, 접근 방식, 검증 기준, 핵심 결과, 한계, 현재 상태를 빠르게 스캔 가능한 구조로 정리했다.
- current best focused review candidate evidence는 유지했다.
- 새 실험 결과나 모델 품질 주장은 추가하지 않았다.

Next recommended issue:

- `이력서 프로젝트 bullet 정리`

## Previous Portfolio README Polish Result

Issue #212는 Issue #210 consolidation evidence를 바탕으로 README를 지원용 포트폴리오 문서처럼 재구성한 작업이다.

Updated file:

- `README.md`

Result:

- README 상단에 문제 정의, 접근 방식, 핵심 성과를 배치했다.
- current best focused review candidate evidence를 표로 요약했다.
- "증명한 것"과 "아직 증명하지 않은 것"을 분리했다.
- 실행 방법, 주요 파일, 한계, 다음 작업을 포트폴리오 독자가 읽기 쉬운 순서로 정리했다.
- broad model quality, human/audio preference, style adaptation, product readiness를 주장하지 않는다.

Next recommended issue:

- `이력서 프로젝트 bullet 정리`

## Previous Focused Timing Vocabulary Keep Candidate Consolidation Result

Issue #210은 Issue #208에서 `keep`으로 남은 focused review candidate의 의미를 model-core MVP 관점에서 정리한 consolidation이다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_KEEP_CANDIDATE_CONSOLIDATION_2026-05-27.md`

Consolidated candidate:

- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- consolidated decision: current best focused review candidate
- note count: `64`
- unique pitch count: `19`
- range: `G3-G5`
- phrase span: `32.0` beats
- final landing: `D5` over `Ebmaj7`
- max interval: `4`
- objective flags: `[]`
- adjacent pitch repeats: `0`
- duplicated 3/4/8-note pitch-class chunks: `0`
- timing: `acceptable`
- chord fit: `strong`
- phrase continuation: `acceptable`
- landing: `strong`
- jazz vocabulary: `acceptable`

Consolidated interpretation:

- This is a reviewable symbolic MIDI solo-line outcome under the current constrained Stage B pipeline.
- This is not broad model-quality proof, human/audio listening proof, or pianist style adaptation proof.
- The next issue should turn this evidence into a portfolio README narrative without overstating model quality.

## Previous Focused Timing Vocabulary Follow-up Focused Listening Fill Result

Issue #208은 Issue #206 focused listening review notes template의 pending fields를 MIDI note/context evidence 기준으로 채운 작업이다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_FOLLOWUP_FOCUSED_LISTENING_FILL_2026-05-27.md`

Filled notes:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_focused_listening_notes/focused_listening_review_notes_filled.json`

Filled result:

- candidate count: `1`
- reviewed count: `1`
- pending count: `0`
- keep: `1`
- needs followup: `0`
- reject: `0`

Focused listening fields:

- timing: `acceptable`
- chord fit: `strong`
- phrase continuation: `acceptable`
- landing: `strong`
- jazz vocabulary: `acceptable`
- decision: `keep`

Supporting evidence:

- note count: `64`
- unique pitch count: `19`
- range: `G3-G5`
- final landing: `D5` over `Ebmaj7`
- max interval: `4`
- objective flags: `[]`
- adjacent pitch repeats: `0`
- duplicated 3/4/8-note pitch-class chunks: `0`
- objective outside ratio: `0.016`

Decision:

- Keep the candidate as the current best focused review candidate.
- Do not claim broad training readiness or style adaptation success from one focused keep.
- The next issue should consolidate what is proven and what remains proxy-only before choosing broader repeatability or portfolio README polish.

## Previous Focused Timing Vocabulary Follow-up Focused Listening Review Notes Result

Issue #206은 Issue #204에서 `keep_for_focused_listening`으로 남은 단일 후보를 focused listening review notes template으로 분리한 작업이다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_FOLLOWUP_FOCUSED_LISTENING_REVIEW_NOTES_2026-05-27.md`

Generated artifacts:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_focused_listening_notes/focused_listening_review_notes_template.json`
- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_followup_focused_listening_notes/focused_listening_review_notes_summary.json`

Result:

- candidate count: `1`
- reviewed count: `0`
- pending count: `1`
- decision pending: `1`
- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- focused context decision: `keep_for_focused_listening`
- proxy decision: `keep`
- proxy issue: `too_mechanical`
- objective bucket: `clean`
- objective flags: `[]`

Pending real-listening fields:

- timing
- chord fit
- phrase continuation
- landing
- jazz vocabulary
- decision

Decision:

- The focused listening notes template exists for the single focused-context keep candidate.
- No generation rule should be changed from this pending template alone.
- The next issue should fill the focused listening review notes using solo/context MIDI evidence.

## Previous Focused Timing Vocabulary Follow-up Focused Context Decision Result

Issue #204는 Issue #200 focused package의 단일 proxy `keep` 후보를 solo/context MIDI note 기준으로 다시 판단한 focused context decision이다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_FOLLOWUP_FOCUSED_CONTEXT_DECISION_2026-05-27.md`

Focused candidate:

- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- prior proxy decision: `keep`
- focused context decision: `keep_for_focused_listening`
- note count: `64`
- unique pitch count: `19`
- pitch range: `G3-G5`
- final landing: `D5`
- final chord: `Ebmaj7`
- final role: `guide/chord tone`
- max interval: `4`
- adjacent pitch repeats: `0`
- adjacent pitch-class repeats: `0`
- duplicated 3-note pitch-class chunks: `0`
- duplicated 4-note pitch-class chunks: `0`
- duplicated 8-note pitch-class chunks: `0`
- objective flags: `[]`

Context track check:

- chord guide exists: `32` notes, range `C3-G#4`
- bass root guide exists: `8` notes, range `C2-A#2`
- solo track exists: `64` notes, range `G3-G5`

Decision:

- The focused package does not show a register, cadence, repeated-cell, or context-track blocker.
- The candidate is good enough to move into a focused listening review artifact.
- This is not proof of final musical quality because timing and vocabulary naturalness still require review.

## Previous Focused Timing Vocabulary Follow-up Proxy Keep Package Result

Issue #200은 Issue #198 proxy keep 후보를 focused context review package로 분리한 작업이다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_FOLLOWUP_PROXY_KEEP_PACKAGE_2026-05-27.md`

Generated package:

- `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/focused_review_package.json`
- `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_followup_proxy_keep_package/focused_review_package.md`

Result:

- candidate count: `1`
- decision filter: `keep`
- copied MIDI files: `2`
- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- mode: `data_motif_rhythm_phrase_variation`
- sample seed: `18`
- valid: `true`
- strict valid: `true`

Candidate metrics:

- note count: `64`
- unique pitch count: `19`
- source syncopated onset ratio: `0.719`
- source most-common IOI ratio: `0.397`
- source tension ratio: `0.344`
- objective bucket: `clean`
- objective flags: `[]`

Decision:

- The proxy keep candidate is now isolated as a one-candidate focused package.
- This remains a focused-context review input, not final musical quality.
- The next issue should create focused listening review notes for the Issue #204 keep candidate.

## Previous Focused Timing Vocabulary Follow-up Proxy Review Result

Issue #198은 Issue #196 focused listening follow-up repair 후보를 MIDI-note/context 기준으로 다시 판단한 proxy review다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_FOLLOWUP_PROXY_REVIEW_2026-05-27.md`

Result:

- candidate count: `6`
- reviewed count: `6`
- pending count: `0`
- decisions:
  - `keep`: `1`
  - `needs_followup`: `3`
  - `reject`: `2`
- phrase quality: `phrase=3`, `fragment=2`, `exercise=1`
- timing: `acceptable=3`, `too_stiff=3`
- chord fit: `fits=6`
- objective bucket: `clean=6`
- objective flags: `{}`

Proxy keep:

- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- adjacent repeated pitch count: `0`
- duplicated 3-note pitch-class chunks: `0`
- duplicated 4-note pitch-class chunks: `0`
- duplicated 8-note pitch-class chunks: `0`
- unique pitch count: `19`
- source tension ratio: `0.344`
- objective tension ratio: `0.469`
- final landing: `D5`
- max interval: `4`

Decision:

- Rank 2 is the only repaired variation candidate with adjacent repeat and 3/4/8-note duplicated cells all at `0`.
- This is a proxy keep for focused context review only.
- Next issue should isolate that candidate into a focused package before any final quality claim.

## Previous Focused Timing Vocabulary Listening Follow-up Repair Result

Issue #196은 Issue #194 focused listening fill에서 남은 `timing=stiff`, `jazz_vocabulary=thin` 병목을 generation rule 쪽에서 좁게 본 작업이다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_LISTENING_FOLLOWUP_REPAIR_2026-05-27.md`

Implementation:

- safe alternative가 있을 때 직전 pitch-class를 후보에서 제외했다.
- repeat fallback 직전에 tension/recovery/next-guide pitch-class 대체 후보를 시도했다.
- fallback 후보는 최근에 쓰지 않은 실제 pitch를 우선 선택한다.

Result:

- `data_motif_rhythm_phrase_variation` valid: `3/3`
- strict: `3/3`
- final landing resolved: `3/3`
- max interval: `4`
- objective MIDI flags: `{}`
- avg syncopated onset ratio: `0.703`
- avg duration diversity ratio: `0.089`
- avg most-common IOI ratio: `0.397`
- avg source tension ratio: `0.307`
- avg root-tone ratio: `0.036`

Pitch-cell effect:

- rank 1 adjacent repeats: `2 -> 0`
- rank 2 adjacent repeats: `4 -> 0`
- rank 3 adjacent repeats: `2 -> 0`
- rank 2 duplicated 3-note cells: `7 -> 0`
- rank 2 duplicated 4-note cells: `3 -> 0`

Decision:

- Objective-clean/register/cadence guardrails are preserved.
- Rank 2 has the clearest repair signal.
- Source tension fell and rank 1/3 short-cell repeats increased, so this is a fresh proxy review target, not final keep.

## Previous Focused Timing Vocabulary Focused Listening Fill Result

Issue #194는 Issue #192 focused listening review notes template을 MIDI-note/context evidence 기준으로 채운 작업이다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_FOCUSED_LISTENING_FILL_2026-05-27.md`

Filled notes:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_focused_listening_notes/focused_listening_review_notes_filled.json`

Filled result:

- candidate count: `1`
- reviewed count: `1`
- pending count: `0`
- decision:
  - `keep`: `0`
  - `needs_followup`: `1`
  - `reject`: `0`
- timing: `stiff`
- chord fit: `acceptable`
- phrase continuation: `acceptable`
- landing: `strong`
- jazz vocabulary: `thin`

Decision:

- The candidate survives focused context register/cadence checks and has a strong guide landing.
- It does not survive as final keep because timing remains grid-derived and vocabulary reads thin/mechanical.
- Next repair should target adjacent repeats, duplicated 3-note cells, and chord-color/tension while preserving objective-clean/register/cadence guardrails.

## Previous Focused Timing Vocabulary Focused Listening Review Notes Result

Issue #192는 Issue #190에서 `keep_for_focused_listening`으로 남은 단일 후보를 focused listening review notes template으로 만든 작업이다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_FOCUSED_LISTENING_REVIEW_NOTES_2026-05-27.md`

Generated artifact:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_timing_vocab_focused_listening_notes/focused_listening_review_notes_template.json`

Result:

- candidate count: `1`
- reviewed count: `0`
- pending count: `1`
- decision pending: `1`
- candidate: `data_motif_rhythm_phrase_variation_rank_3_sample_3`
- proxy decision: `keep`
- objective bucket: `clean`
- objective flags: `[]`

Pending real-listening fields:

- timing
- chord fit
- phrase continuation
- landing
- jazz vocabulary
- decision

Decision:

- The template exists for the single focused-context keep candidate.
- No generation rule should change from this pending template alone.
- The next issue should fill the focused listening review notes before another repair.

## Previous Focused Timing Vocabulary Focused Context Decision Result

Issue #190은 Issue #188 focused package의 단일 proxy `keep` 후보를 solo/context MIDI note 기준으로 다시 판단한 focused context decision이다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_FOCUSED_CONTEXT_DECISION_2026-05-27.md`

Focused candidate:

- candidate: `data_motif_rhythm_phrase_variation_rank_3_sample_3`
- prior proxy decision: `keep`
- focused context decision: `keep_for_focused_listening`
- note count: `64`
- unique pitch count: `20`
- pitch range: `G3-G5`
- final landing: `D5`
- final chord: `Ebmaj7`
- final role: `guide`
- max interval: `4`
- objective flags: `[]`
- duplicated 3-note pitch-class chunks: `2`
- duplicated 4-note pitch-class chunks: `0`
- duplicated 8-note pitch-class chunks: `0`

Context track check:

- chord guide: `32` notes, range `C3-G#4`
- bass root guide: `8` notes, range `C2-A#2`
- solo track: `64` notes, range `G3-G5`

Decision:

- The candidate survives focused-context register, cadence, and context-track checks.
- This is still not final musical quality.
- The next issue should create focused listening review notes before changing generation rules again.

## Previous Focused Timing Vocabulary Proxy Keep Focused Package Result

Issue #188은 Issue #186 proxy keep 후보를 focused context review package로 분리한 작업이다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_PROXY_KEEP_FOCUSED_PACKAGE_2026-05-27.md`

Generated package:

- `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_proxy_keep_focused_package/focused_review_package.json`
- `outputs/stage_b_focused_review_package/harness_stage_b_focused_timing_vocab_proxy_keep_focused_package/focused_review_package.md`

Result:

- candidate count: `1`
- decision filter: `keep`
- copied MIDI files: `2`
- candidate: `data_motif_rhythm_phrase_variation_rank_3_sample_3`
- mode: `data_motif_rhythm_phrase_variation`
- sample seed: `19`
- valid: `true`
- strict valid: `true`

Candidate metrics:

- note count: `64`
- unique pitch count: `20`
- source syncopated onset ratio: `0.703`
- source most-common IOI ratio: `0.397`
- source tension ratio: `0.297`
- objective chord-tone ratio: `0.547`
- objective tension ratio: `0.453`
- objective stepwise interval ratio: `0.460`
- objective unresolved large leap ratio: `0.000`

Decision:

- The proxy keep candidate is now isolated as a one-candidate focused package.
- This is still only a focused-context review input, not final musical quality.
- The next issue should decide whether the candidate survives focused context MIDI-note review.

## Previous Focused Timing Vocabulary Proxy Review Result

Issue #186은 Issue #184 focused timing/vocabulary follow-up repair 후보를 MIDI-note/context evidence 기준으로 다시 채운 proxy review다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_PROXY_REVIEW_2026-05-27.md`

Generated artifacts:

- `outputs/stage_b_listening_review_notes/harness_stage_b_focused_timing_vocab_proxy_review/focused_timing_vocab_repaired_review_notes.json`
- `outputs/stage_b_listening_review_aggregate/harness_stage_b_focused_timing_vocab_proxy_review/listening_review_aggregate.json`

Filled result:

- candidate count: `6`
- reviewed count: `6`
- pending count: `0`
- decisions:
  - `keep`: `1`
  - `needs_followup`: `3`
  - `reject`: `2`
- phrase quality: `phrase=3`, `fragment=2`, `exercise=1`
- timing: `acceptable=3`, `too_stiff=3`
- chord fit: `fits=6`
- objective bucket: `clean=6`
- objective flags: `{}`

Proxy keep:

- candidate: `data_motif_rhythm_phrase_variation_rank_3_sample_3`
- note count: `64`
- unique pitch count: `20`
- max interval: `4`
- final landing: `guide`
- source syncopated onset ratio: `0.703`
- source most-common IOI ratio: `0.397`
- objective stepwise interval ratio: `0.460`
- objective tension ratio: `0.453`

Decision:

- Exactly one repaired candidate is promoted to proxy `keep`.
- This is still only a focused-context review input, not final musical quality.
- The next issue should isolate the proxy keep candidate into a focused context review package.
- Broad training and Brad style adaptation remain premature.

## Previous Focused Timing Vocabulary Follow-up Repair Result

Issue #184는 Issue #182 focused listening fill에서 드러난 `timing=stiff`, `jazz_vocabulary=thin` 병목을 generation rule 쪽에서 좁게 본 작업이다.

Docs:

- `docs/STAGE_B_FOCUSED_TIMING_VOCABULARY_REPAIR_2026-05-27.md`

Implementation:

- replayed 3-note/4-note pitch-class cell을 이어 만드는 candidate pitch-class를 safe alternative가 있을 때 제외했다.
- repeated cell penalty를 강화했다.
- max interval 후보가 없을 때 넓은 leap 대신 repeat fallback을 사용해 max interval guardrail을 보존했다.

Result:

- `data_motif_rhythm_phrase_variation` valid: `3/3`
- strict: `3/3`
- final landing resolved: `3/3`
- max interval: `4`
- objective MIDI flags: `{}`
- avg syncopated onset ratio: `0.703`
- avg duration diversity ratio: `0.089`
- avg most-common IOI ratio: `0.397`
- avg tension ratio: `0.323`
- avg root-tone ratio: `0.031`
- repaired variation unique pitch count: `19-20`
- repaired variation stepwise interval ratio: `0.460`

Decision:

- Objective-clean/register/cadence guardrails are preserved.
- Rank 1/3 reduce 4-note pitch-class cell repetition.
- Rank 2 introduces more adjacent pitch repeat and short-cell repetition.
- This is a tradeoff repair, so the next step is fresh proxy review, not final keep.

## Latest Phrase Vocabulary/Motif Focused Listening Fill Result

Issue #182는 Issue #180 focused listening review notes template을 MIDI-note/context evidence 기준으로 채운 작업이다.

Docs:

- `docs/STAGE_B_PHRASE_VOCAB_MOTIF_FOCUSED_LISTENING_FILL_2026-05-27.md`

Filled result:

- candidate count: `1`
- reviewed count: `1`
- pending count: `0`
- decision:
  - `keep`: `0`
  - `needs_followup`: `1`
  - `reject`: `0`
- timing: `stiff`
- chord fit: `acceptable`
- phrase continuation: `acceptable`
- landing: `acceptable`
- jazz vocabulary: `thin`

Decision:

- The candidate survives focused context register and cadence checks.
- It does not survive as final keep because timing remains grid-derived and vocabulary reads thin/mechanical.
- Next repair should target timing-grid stiffness and repeated short pitch-class cells while preserving objective-clean/register/cadence guardrails.

## Latest Phrase Vocabulary/Motif Focused Listening Review Notes Result

Issue #180은 Issue #178에서 `keep_for_focused_listening`으로 남은 단일 후보를 focused listening review notes template으로 만든 작업이다.

Docs:

- `docs/STAGE_B_PHRASE_VOCAB_MOTIF_FOCUSED_LISTENING_REVIEW_NOTES_2026-05-27.md`

Generated artifact:

- `outputs/stage_b_focused_listening_review_notes/harness_stage_b_phrase_vocab_motif_focused_listening_notes/focused_listening_review_notes_template.json`

Result:

- candidate count: `1`
- reviewed count: `0`
- pending count: `1`
- decision pending: `1`
- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- proxy decision: `keep`
- objective bucket: `clean`
- objective flags: `[]`

Pending real-listening fields:

- timing
- chord fit
- phrase continuation
- landing
- jazz vocabulary
- decision

Decision:

- The focused listening review notes template is ready.
- No generation rule changes should come from this pending template alone.
- Next step is to fill the focused listening review fields for this one candidate.

## Latest Phrase Vocabulary/Motif Focused Context Decision Result

Issue #178은 Issue #176 focused package의 단일 proxy `keep` 후보를 solo/context MIDI note 기준으로 다시 판단한 focused context decision이다.

Docs:

- `docs/STAGE_B_PHRASE_VOCAB_MOTIF_FOCUSED_CONTEXT_DECISION_2026-05-27.md`

Candidate:

- `data_motif_rhythm_phrase_variation_rank_2_sample_2`

Positive evidence:

- objective bucket: `clean`
- objective flags: `[]`
- note count: `64`
- unique pitch count: `18`
- pitch range: `G3-G5`
- final landing: `G4`
- max interval: `4`
- unresolved large leap ratio: `0.000`
- adjacent repeated pitch count: `0`
- duplicated 8-note pitch-class chunks: `0`
- context chord guide: `32` notes, range `C3-G#4`
- context bass root guide: `8` notes, range `C2-A#2`

Remaining risk:

- duplicated 3-note pitch-class chunks: `5`
- duplicated 4-note pitch-class chunks: `2`
- timing remains grid-derived
- source tension ratio: `0.344`

Decision:

- focused context decision: `keep_for_focused_listening`
- ready for broad training: `no`
- ready for style adaptation claim: `no`
- next step: focused listening review notes template for this single candidate

## Latest Phrase Vocabulary/Motif Proxy Keep Focused Package Result

Issue #176은 Issue #174 proxy keep 후보를 focused context review package로 분리한 작업이다.

Docs:

- `docs/STAGE_B_PHRASE_VOCAB_MOTIF_PROXY_KEEP_FOCUSED_PACKAGE_2026-05-27.md`

Generated package:

- `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/focused_review_package.json`
- `outputs/stage_b_focused_review_package/harness_stage_b_phrase_vocab_motif_proxy_keep_focused_package/focused_review_package.md`

Result:

- candidate count: `1`
- decision filter: `keep`
- copied MIDI files: `2`
- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- valid: `true`
- strict valid: `true`
- note count: `64`
- unique pitch count: `18`
- source syncopated onset ratio: `0.719`
- source most-common IOI ratio: `0.397`
- objective chord-tone ratio: `0.531`
- objective tension ratio: `0.469`
- objective stepwise interval ratio: `0.460`
- objective unresolved large leap ratio: `0.000`
- objective flags: `[]`

Decision:

- Issue #176 successfully isolates the proxy keep candidate into a focused package.
- This is still not final musical quality.
- Next step is focused context decision using the copied solo/context MIDI pair.

## Latest Phrase Vocabulary/Motif Variation Proxy Review Result

Issue #174는 Issue #172 phrase vocabulary/motif variation repair 후보를 MIDI-note/context 기준으로 채운 proxy review다.

Docs:

- `docs/STAGE_B_PHRASE_VOCAB_MOTIF_VARIATION_PROXY_REVIEW_2026-05-27.md`

Result:

- reviewed candidates: `6`
- pending candidates: `0`
- decisions:
  - `keep`: `1`
  - `needs_followup`: `3`
  - `reject`: `2`
- timing:
  - `acceptable`: `3`
  - `too_stiff`: `3`
- phrase quality:
  - `phrase`: `3`
  - `fragment`: `2`
  - `exercise`: `1`
- chord fit: `fits=6`
- objective bucket: `clean=6`
- objective flags: `{}`

Proxy keep:

- candidate: `data_motif_rhythm_phrase_variation_rank_2_sample_2`
- note count: `64`
- unique pitch count: `18`
- source syncopated onset ratio: `0.719`
- source duration diversity ratio: `0.094`
- source most-common IOI ratio: `0.397`
- objective stepwise interval ratio: `0.460`
- objective tension ratio: `0.469`
- final landing: `guide`
- max interval: `4`

Aggregate follow-up signals:

- `improve_phrase_vocabulary`: `13`
- `fix_timing_grid`: `6`
- `increase_motif_variation`: `3`

Decision:

- Issue #172 repair produced one proxy keep candidate.
- This is not final musical quality; it only justifies a focused context package.
- Next step is to isolate the proxy keep candidate and run focused context review before any further generation repair.

## Latest Phrase Vocabulary/Motif Variation Repair Result

Issue #172는 Issue #170 proxy review에서 남은 small-cell mechanical contour와 motif variation 병목을 generation rule 쪽에서 좁게 본 작업이다.

Docs:

- `docs/STAGE_B_PHRASE_VOCAB_MOTIF_VARIATION_REPAIR_2026-05-27.md`

Implementation:

- `phrase_level_duration_ioi_bar_positions()`의 8-note/bar pattern을 균형화했다.
- `bounded_phrase_pitch_for_pitch_classes()`에 최근 exact pitch와 pitch-class 재사용 penalty를 추가했다.
- normal phrase pitch 선택에서 3/4-step motif-sized interval을 선호하되, 최근 pitch 재사용 회피를 interval 선호보다 먼저 보도록 했다.

Result:

- `data_motif_rhythm_phrase_variation` valid: `3/3`
- strict: `3/3`
- final landing resolved: `3/3`
- max interval: `4`
- objective MIDI flags: `{}`
- objective bucket: `clean=6`
- repaired variation unique pitch count: `18-20`
- avg syncopated onset ratio: `0.703`
- avg duration diversity ratio: `0.089`
- avg most-common duration ratio: `0.406`
- avg IOI diversity ratio: `0.095`
- avg most-common IOI ratio: `0.397`
- avg tension ratio: `0.318`

Decision:

- Issue #172는 Issue #168의 most-common IOI 집중 악화를 `0.481 -> 0.397`로 줄였다.
- duration diversity도 `0.078 -> 0.089`로 개선됐다.
- 대신 IOI diversity는 `0.111 -> 0.095`, source tension ratio는 `0.375 -> 0.318`로 내려갔다.
- 따라서 musical keep으로 승격하지 않고, next step은 repaired 후보의 fresh proxy review다.

## Latest Duration/IOI Repaired Proxy Review Result

Issue #170은 Issue #168 duration/IOI objective repair 이후의 후보를 MIDI-note/context 기준으로 다시 채운 proxy review다.

Docs:

- `docs/STAGE_B_DURATION_IOI_REPAIRED_PROXY_REVIEW_2026-05-27.md`

Result:

- reviewed candidates: `6`
- pending candidates: `0`
- decisions:
  - `keep`: `0`
  - `needs_followup`: `4`
  - `reject`: `2`
- timing:
  - `acceptable`: `2`
  - `too_stiff`: `4`
- phrase quality:
  - `phrase`: `2`
  - `fragment`: `3`
  - `exercise`: `1`
- chord fit: `fits=6`
- objective bucket: `clean=6`
- objective flags: `{}`

Aggregate follow-up signals:

- `improve_phrase_vocabulary`: `12`
- `fix_timing_grid`: `8`
- `increase_motif_variation`: `4`

Decision:

- Issue #168은 objective duration/IOI diversity repair로 유지한다.
- 그러나 proxy `keep` 후보는 아직 없다.
- 다음은 small-cell mechanical contour를 줄이는 phrase vocabulary/motif variation repair다.

## Previous Duration/IOI Objective Repair Result

Issue #168은 Issue #164 이후 남은 phrase-level duration/IOI 병목을 generation rule 쪽에서 좁게 본 작업이다.

Docs:

- `docs/STAGE_B_DURATION_IOI_OBJECTIVE_REPAIR_2026-05-27.md`

Implementation:

- `data_motif_rhythm_phrase_variation`의 8-note/bar 경로에 phrase-level duration/IOI bar-position plan을 추가했다.
- review candidate sort key에서 IOI diversity와 most-common IOI를 duration diversity보다 먼저 반영했다.

Result:

- `data_motif_rhythm_phrase_variation` valid: `3/3`
- strict: `3/3`
- final landing resolved: `3/3`
- max interval: `4`
- objective MIDI flags: `{}`
- duplicate note sequences: `0`
- avg syncopated onset ratio: `0.682`
- avg duration diversity ratio: `0.078`
- avg IOI diversity ratio: `0.111`
- avg most-common IOI ratio: `0.481`
- avg tension ratio: `0.375`

Decision:

- Duration/IOI diversity objective는 개선됐다.
- 그러나 most-common IOI ratio가 악화됐기 때문에 musical keep으로 승격하지 않는다.
- 다음은 repaired 후보를 MIDI-note/context 기준으로 proxy review해 이 tradeoff가 실제로 덜 mechanical한지 확인한다.

## Previous Data-Derived Timing Phrase Proxy Review Result

Issue #164는 Issue #162 data-derived timing phrase vocabulary repair 이후의 후보를 MIDI-note/context 기준으로 다시 채운 proxy review다.

Docs:

- `docs/STAGE_B_DATA_DERIVED_TIMING_PHRASE_PROXY_REVIEW_2026-05-27.md`

Result:

- reviewed candidates: `6`
- pending candidates: `0`
- decisions:
  - `keep`: `0`
  - `needs_followup`: `5`
  - `reject`: `1`
- timing:
  - `acceptable`: `2`
  - `too_stiff`: `4`
- chord fit: `fits=6`
- objective bucket: `clean=6`
- objective flags: `{}`

Aggregate follow-up signals:

- `improve_phrase_vocabulary`: `16`
- `fix_timing_grid`: `8`
- `increase_motif_variation`: `3`

Decision:

- Issue #162는 reviewable timing/tension tradeoff로 유지할 수 있다.
- 그러나 proxy `keep` 후보는 아직 없다.
- 다음은 row selection이 아니라 duration/IOI objective를 직접 개선하는 단계다.

## Latest Data-Derived Timing Phrase Repair Result

Issue #162는 Issue #160 proxy review에서 확인한 timing stiffness와 phrase vocabulary blocker를 generation rule 쪽에서 다시 좁힌 작업이다.

Docs:

- `docs/STAGE_B_DATA_DERIVED_TIMING_PHRASE_REPAIR_2026-05-27.md`

Implementation:

- `top_full_templates`에서 phrase-like timing rows를 우선 선택한다.
- long-span/long-sustain full templates는 review-safe phrase slot에 맞지 않아 제외한다.
- position/duration shaping은 기존 review-safe path를 유지한다.

Result:

- `data_motif_rhythm_phrase_variation` valid: `3/3`
- strict: `3/3`
- final landing resolved: `3/3`
- max interval: `4`
- objective MIDI flags: `{}`
- avg syncopated onset ratio: `0.693`
- avg unique bar-position pattern ratio: `0.958`
- avg duration diversity ratio: `0.073`
- avg IOI diversity ratio: `0.079`
- avg most-common IOI ratio: `0.392`
- avg tension ratio: `0.375`

Decision:

- strict/objective-clean guardrail은 유지됐다.
- syncopation과 tension은 개선됐지만, duration/IOI diversity는 조금 후퇴했다.
- 다음은 fresh proxy review로 이 tradeoff가 실제로 덜 mechanical한지 판단한다.

## Latest Timing Motif Repaired Proxy Review Result

Issue #160은 Issue #158 register-safe timing motif follow-up repair 이후의 후보를 MIDI-note/context 기준으로 다시 채운 proxy review다.

Docs:

- `docs/STAGE_B_REGISTER_SAFE_TIMING_MOTIF_REPAIRED_PROXY_REVIEW_2026-05-27.md`

Result:

- reviewed candidates: `6`
- pending candidates: `0`
- decisions:
  - `keep`: `0`
  - `needs_followup`: `5`
  - `reject`: `1`
- timing: `too_stiff=6`
- chord fit: `fits=6`
- objective bucket: `clean=6`
- objective flags: `{}`

Aggregate follow-up signals:

- `improve_phrase_vocabulary`: `16`
- `fix_timing_grid`: `12`
- `increase_motif_variation`: `3`

Decision:

- Issue #158의 register-safe phrase-cell penalty guard는 partial safety improvement로 유지할 수 있다.
- 그러나 repaired candidate를 proxy `keep`으로 올리지는 않는다.
- 다음은 같은 penalty를 더 누적하는 것이 아니라 data-derived timing/phrase vocabulary repair다.

## Latest Timing Motif Repair Result

Issue #158은 Issue #156 focused listening fill에서 나온 timing stiffness, repeated pitch-class cell, thin vocabulary blocker를 generation rule 쪽에서 좁게 다시 본 작업이다.

Docs:

- `docs/STAGE_B_REGISTER_SAFE_TIMING_MOTIF_REPAIR_2026-05-27.md`

변경:

- recent phrase memory를 최근 `6`음에서 `8`음으로 확장했다.
- repeated pitch-class cell penalty lookback을 `18`에서 `32`로 확장했다.
- repeated 3-note/4-note pitch-class cell과 exact 4-note cell penalty를 강화했다.
- asymmetric timing-position variation은 지표가 악화되어 최종 변경에서 제외했다.

Result:

- `data_motif_rhythm_phrase_variation` valid: `3/3`
- strict: `3/3`
- final landing resolved: `3/3`
- max interval: `4`
- objective MIDI flags: `{}`
- avg syncopated onset ratio: `0.684`
- avg unique bar-position pattern ratio: `0.958`
- avg IOI diversity ratio: `0.091`
- avg most-common IOI ratio: `0.385`
- avg tension ratio: `0.358`
- avg root-tone ratio: `0.021`

Decision:

- register-safe final cadence guardrail은 유지됐다.
- phrase-cell repetition guard는 부분 개선으로 남긴다.
- timing stiffness는 해결됐다고 보지 않는다.
- 다음은 repaired candidate set을 fresh proxy review로 다시 판단한다.

## Latest Focused Listening Fill Result

Issue #156은 Issue #154 focused listening review notes template을 MIDI-focused proxy review 기준으로 채운 작업이다.

Docs:

- `docs/STAGE_B_REGISTER_SAFE_FOCUSED_LISTENING_FILL_2026-05-27.md`

중요한 전제:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note sequence, solo/context MIDI path, chord/bass guide context, objective metrics를 근거로 채운 focused review다.
- broad training이나 style adaptation claim으로 해석하지 않는다.

Result:

- candidate count: `1`
- reviewed count: `1`
- pending count: `0`
- decision:
  - `keep`: `0`
  - `needs_followup`: `1`
  - `reject`: `0`

Candidate:

- `data_motif_rhythm_phrase_variation_rank_1_sample_3`
- timing: `stiff`
- chord fit: `acceptable`
- phrase continuation: `weak`
- landing: `acceptable`
- jazz vocabulary: `thin`
- decision: `needs_followup`

Decision:

- Do not promote the candidate to final `keep`.
- Keep the register-safe repair and final cadence guardrail.
- Next work should target timing stiffness, motif variation, and phrase vocabulary without reopening the C6-to-G3 register blocker.

## Previous Focused Listening Notes Result

Issue #154는 Issue #152에서 `keep_for_focused_listening`으로 판단한 단일 후보를 실제 청취용 review notes template으로 만든 작업이다.

Docs:

- `docs/STAGE_B_REGISTER_SAFE_FOCUSED_LISTENING_REVIEW_NOTES_2026-05-27.md`

Result:

- candidate count: `1`
- reviewed count: `0`
- pending count: `1`
- generated template:
  - `outputs/stage_b_focused_listening_review_notes/harness_stage_b_focused_listening_review_notes/focused_listening_review_notes_template.json`

Candidate:

- `data_motif_rhythm_phrase_variation_rank_1_sample_3`
- proxy decision: `keep`
- proxy timing: `acceptable`
- note count: `63`
- unique pitch count: `18`
- source tension ratio: `0.349`
- objective flags: `[]`

Decision:

- The one-candidate focused listening review template is ready.
- The candidate remains pending until a real listening pass is filled.
- No generation repair should start from this artifact alone.

## Previous Focused Context Decision

Issue #152는 Issue #150 focused package의 단일 proxy `keep` 후보를 solo/context MIDI note 기준으로 다시 판단한 focused context decision이다.

Docs:

- `docs/STAGE_B_REGISTER_SAFE_FOCUSED_CONTEXT_DECISION_2026-05-27.md`

중요한 전제:

- 실제 오디오 청취 리뷰가 아니다.
- `keep`은 focused listening review로 넘길 수 있다는 뜻이다.
- broad training이나 style adaptation claim으로 해석하지 않는다.

Result:

- prior proxy decision: `keep`
- focused context decision: `keep_for_focused_listening`
- keep as diagnostic seed: `yes`
- ready for broad training: `no`
- ready for style adaptation claim: `no`

Positive evidence:

- candidate: `data_motif_rhythm_phrase_variation_rank_1_sample_3`
- objective flags: `[]`
- note count: `63`
- unique pitch count: `18`
- pitch range: `G3-G5`
- final landing: `G4`
- max interval: `4`
- max active notes: `1`
- off-sixteenth-grid count: `0`

Remaining risk:

- repeated 3-note pitch-class cells and one repeated 4-note pitch-class cell remain.
- timing is still grid-derived.
- chromatic color handling needs real context listening.

Decision:

- The prior C6-to-G3 focused-context blocker is no longer present.
- The next boundary should create focused listening review notes for this one candidate.
- Do not change generation rules again until that focused listening artifact is filled.

## Previous Package Result

Issue #150은 Issue #148에서 복구된 proxy `keep` 후보만 분리해 focused context review용 package로 묶은 작업이다.

Docs:

- `docs/STAGE_B_REGISTER_SAFE_PROXY_KEEP_FOCUSED_PACKAGE_2026-05-27.md`

중요한 전제:

- `keep`은 MIDI-note proxy 기준이다.
- 실제 오디오 청취 승인이나 최종 musical-quality claim이 아니다.
- broad training이나 Brad style adaptation으로 바로 확장하지 않는다.

Result:

- decision filter: `keep`
- package candidate count: `1`
- copied solo MIDI files: `1`
- copied context MIDI files: `1`

Selected candidate:

- `data_motif_rhythm_phrase_variation_rank_1_sample_3`
- phrase: `phrase`
- timing: `acceptable`
- chord fit: `fits`
- notes: `63`
- unique pitch count: `18`
- source tension ratio: `0.349`
- objective flags: `[]`

Decision:

- Issue #148 proxy keep candidate is now isolated as a single focused context review artifact.
- The next decision should come from reviewing this one solo/context MIDI pair.
- This still does not prove broad model quality.

## Previous Proxy Review Result

Issue #148은 Issue #146 register-safe phrase vocabulary repair 이후의 후보를 MIDI-note/context 기준으로 다시 채운 proxy review다.

Docs:

- `docs/STAGE_B_REGISTER_SAFE_PHRASE_VOCAB_PROXY_REVIEW_2026-05-27.md`

중요한 전제:

- 실제 오디오 청취 리뷰가 아니다.
- `keep`은 focused context review로 넘길 proxy 후보라는 뜻이다.
- broad training이나 style adaptation claim으로 해석하지 않는다.

Result:

- reviewed candidates: `6`
- pending candidates: `0`
- decisions:
  - `keep`: `1`
  - `needs_followup`: `4`
  - `reject`: `1`
- timing:
  - `acceptable`: `2`
  - `too_stiff`: `4`
- chord fit:
  - `fits`: `6`
- objective MIDI flags: `{}`

Proxy keep candidate:

- `data_motif_rhythm_phrase_variation_rank_1_sample_3`
- note count: `63`
- unique pitch count: `18`
- pitch range: `G3-G5`
- final landing: `G4`
- max interval: `4`
- outside ratio: `0.000`
- max active notes: `1`
- off-sixteenth-grid count: `0`

Aggregate follow-ups:

- `improve_phrase_vocabulary`: `13`
- `fix_timing_grid`: `8`
- `increase_motif_variation`: `3`

Decision:

- Issue #146 register-safe phrase vocabulary repair should be kept.
- The top repaired variation candidate is strong enough to isolate as a focused context review candidate.
- This still does not prove final musical quality.
- The next boundary should package the proxy keep candidate for focused context review.

## Previous Probe Result

Issue #146는 Issue #144 proxy review에서 남은 boxed-in/cell-like phrase blocker를 generation rule 쪽에서 좁게 고친 작업이다.

Docs:

- `docs/STAGE_B_REGISTER_SAFE_PHRASE_VOCAB_REPAIR_2026-05-26.md`

Result:

- variation valid/strict samples: `3/3`
- final landing resolved: `3/3`
- max interval: `4`
- duplicate note sequences: `0`
- objective MIDI flag counts: `{}`
- top repaired candidate: `data_motif_rhythm_phrase_variation_rank_1_sample_3`
- top repaired candidate unique pitch count: `18`
- top repaired candidate pitch range: `G3-G5`
- exact repeated 4-note cells in top repaired solo review MIDI: `0`

Decision:

- Issue #142 register/cadence bounds remain intact.
- Issue #146 reduces exact repeated phrase-cell evidence, but does not prove final musical quality.
- The next boundary should be a fresh proxy/listening review of the register-safe phrase vocabulary repaired candidates.

## Previous Review Result

Issue #144는 Issue #142 register-cadence repair 이후의 후보를 MIDI-note/context 기준으로 다시 채운 focused proxy review다.

Docs:

- `docs/STAGE_B_REGISTER_CADENCE_REPAIRED_PROXY_REVIEW_2026-05-26.md`

중요한 전제:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note, context chord guide, bass root guide, objective metrics 기준의 proxy review다.
- broad training이나 style adaptation claim으로 해석하지 않는다.

Result:

- reviewed candidates: `6`
- pending candidates: `0`
- decisions:
  - `keep`: `0`
  - `needs_followup`: `5`
  - `reject`: `1`
- timing:
  - `acceptable`: `2`
  - `too_stiff`: `4`
- chord fit:
  - `fits`: `6`
- objective MIDI flags: `{}`

Repaired top candidate:

- `data_motif_rhythm_phrase_variation_rank_1_sample_3`
- note count: `63`
- unique pitch count: `18`
- pitch range: `C#4-G5`
- final landing: `G4`
- final bar notes: `F4, G4, A#4, A4, F4, D4, F#4, G4`
- objective MIDI flags: `[]`

Aggregate follow-ups:

- `improve_phrase_vocabulary`: `14`
- `fix_timing_grid`: `8`
- `increase_motif_variation`: `3`

Decision:

- Issue #142 register-cadence repair should be kept.
- The prior `C6` to final `G3` context blocker is fixed.
- No candidate is promoted to `keep`.
- The next repair should re-expand phrase vocabulary and motif development while keeping the new register bounds.

## Previous Probe Result

Issue #142는 Issue #140 focused context decision에서 확인한 C6-to-G3 register/cadence blocker를 generation rule 쪽에서 좁게 고친 작업이다.

Docs:

- `docs/STAGE_B_FOCUSED_CONTEXT_REGISTER_CADENCE_REPAIR_2026-05-25.md`

Result:

- variation strict samples: `3/3`
- final landing resolved: `3/3`
- max interval: `4`
- duplicate note sequences: `0`
- objective MIDI flag counts: `{}`
- repaired top candidate final landing: `G4`

## Previous Decision Result

Issue #140은 Issue #138 focused package의 단일 proxy `keep` 후보를 solo/context MIDI note 기준으로 다시 판단한 focused context decision이다.

Docs:

- `docs/STAGE_B_PROXY_KEEP_FOCUSED_CONTEXT_DECISION_2026-05-25.md`

Result:

- prior proxy decision: `keep`
- focused context decision: `needs_followup`
- keep as diagnostic seed: `yes`
- ready for broad training: `no`
- blocker: register/contour reaches `C6` around bar 4, then drifts down to `G3` by the final bar.

## Previous Package Result

Issue #138은 Issue #136에서 처음 나온 proxy `keep` 후보만 solo/context MIDI와 objective note summary로 묶은 focused review package다.

Docs:

- `docs/STAGE_B_PROXY_KEEP_FOCUSED_REVIEW_PACKAGE_2026-05-25.md`

Result:

- decision filter: `keep`
- package candidate count: `1`
- copied solo MIDI files: `1`
- copied context MIDI files: `1`

## Previous Proxy Review Result

Issue #136은 Issue #134 phrase-shape/tension repaired rhythm 후보를 MIDI-note/context 기준으로 다시 채운 proxy review다.

Docs:

- `docs/STAGE_B_PHRASE_SHAPE_TENSION_PROXY_REVIEW_2026-05-25.md`

Result:

- reviewed candidates: `6`
- decisions:
  - `keep`: `1`
  - `needs_followup`: `5`
  - `reject`: `0`
- timing:
  - `acceptable`: `2`
  - `too_stiff`: `4`
- chord fit:
  - `fits`: `6`
- duplicate note sequences: `0`
- objective MIDI flags: `{}`
- aggregate follow-ups:
  - `improve_phrase_vocabulary`: `10`
  - `fix_timing_grid`: `8`
  - `increase_motif_variation`: `5`
  - `increase_tension_approach_vocabulary`: `0`

## Previous Probe Result

Issue #134는 Issue #132 proxy review에서 남은 no-keep 병목을 generation rule 쪽에서 다시 좁힌 작업이다.

Docs:

- `docs/STAGE_B_RHYTHM_VARIATION_PHRASE_SHAPE_TENSION_REPAIR_2026-05-25.md`

Result:

- variation candidates:
  - strict: `3/3`
  - final landing: `3/3`
  - max interval: `4`
  - average tension ratio: `0.437`
  - average bar-position pattern ratio: `0.958`
  - average IOI diversity ratio: `0.091`
  - average most-common IOI ratio: `0.385`
- duplicate note sequences: `0`
- objective MIDI flags: `{}`
- before/after max simultaneous notes for variation review MIDI: `1/1`

## Previous Sample-Diverse Review Result

Issue #124는 Issue #122에서 sample diversity를 고친 rhythm variation 후보를 MIDI-note proxy review로 다시 채웠다.

중요한 전제:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note timing, pitch contour, objective MIDI metrics, context chord guide track, duplicate note-sequence fields 기준이다.
- `keep` 후보는 만들지 않는다.

Docs:

- `docs/STAGE_B_SAMPLE_DIVERSE_RHYTHM_PROXY_REVIEW_2026-05-25.md`

Result:

- reviewed candidates: `6`
- pending candidates: `0`
- decisions:
  - `needs_followup`: `6`
  - `reject`: `0`
  - `keep`: `0`
- timing:
  - `too_stiff`: `6`
- duplicate note sequences:
  - `0`

Aggregate follow-ups:

- `improve_phrase_vocabulary`: `14`
- `fix_timing_grid`: `12`
- `increase_motif_variation`: `6`

Candidate decisions:

| candidate | phrase | timing | chord_fit | decision |
|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `phrase` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_rhythm_phrase_variation_rank_1_sample_2` | `phrase` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_rhythm_phrase_variation_rank_2_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_rhythm_phrase_variation_rank_3_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` |

Decision:

- sample diversity repair는 유효했고, exact duplicate 문제는 해결됐다.
- 하지만 variation 후보도 아직 timing-stiff이고 phrase vocabulary가 mechanical하다.
- 다음 generation issue는 duplicate repair가 아니라 timing-grid repetition repair다.

## Previous Probe Result

Issue #122는 Issue #120 proxy review에서 확인된 rhythm variation exact duplicate 문제를 고쳤다.

Docs:

- `docs/STAGE_B_RHYTHM_VARIATION_SAMPLE_DIVERSITY_2026-05-25.md`

Implemented:

- `data_motif_rhythm_phrase_variation` seed가 rhythm template row, contour template row, slot boundary, duration variation, pitch-cell selection, approach target에 반영된다.
- review export가 MIDI note/start/end/pitch sequence signature를 기록한다.
- review manifest가 `unique_note_sequence_count`와 `duplicate_note_sequence_count`를 기록한다.
- review candidates markdown에 `duplicate` column이 추가됐다.

Validation result:

- candidate count: `6`
- unique note sequences: `6`
- duplicate note sequences: `0`
- objective MIDI flag counts: `{}`

Variation candidates:

| candidate | notes | pitches | sync | dur-var | ioi-var | ioi-rep | landing | max interval |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `data_motif_rhythm_phrase_variation_rank_1_sample_2` | 62 | 29 | 0.667 | 0.111 | 0.097 | 0.500 | guide | 6 |
| `data_motif_rhythm_phrase_variation_rank_2_sample_3` | 62 | 22 | 0.730 | 0.111 | 0.113 | 0.565 | guide | 6 |
| `data_motif_rhythm_phrase_variation_rank_3_sample_1` | 60 | 21 | 0.694 | 0.097 | 0.115 | 0.426 | guide | 6 |

Decision:

- sample-level duplicate problem은 해결됐다.
- variation candidates는 이제 independent review evidence로 볼 수 있다.
- 다만 IOI repetition이 여전히 높아 timing stiffness risk는 남아 있다.
- 다음은 새 sample-diverse 후보를 MIDI-note proxy review로 채우는 단계다.

## Previous Review Result

Issue #120은 Issue #118 rhythm/phrase variation 후보 3개와 contour repair baseline 3개를 같은 listening review notes schema로 채운 MIDI-note proxy review다.

중요한 전제:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note timing, pitch contour, objective MIDI metrics, context chord guide track, exact note-sequence comparison을 기준으로 한 proxy review다.
- `keep` 후보는 만들지 않는다.

Docs:

- `docs/STAGE_B_RHYTHM_PHRASE_VARIATION_MIDI_PROXY_REVIEW_2026-05-25.md`

Generated outputs:

- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_rhythm_phrase_variation_proxy/rhythm_phrase_variation_review_notes_midi_proxy.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_rhythm_phrase_variation_proxy/listening_review_aggregate.md`

Result:

- reviewed candidates: `6`
- pending candidates: `0`
- decisions:
  - `needs_followup`: `4`
  - `reject`: `2`
  - `keep`: `0`
- timing:
  - `too_stiff`: `6`
- aggregate follow-ups:
  - `fix_timing_grid`: `12`
  - `improve_phrase_vocabulary`: `10`
  - `increase_motif_variation`: `6`

Candidate decisions:

| candidate | phrase | timing | chord_fit | decision |
|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `phrase` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_rhythm_phrase_variation_rank_1_sample_1` | `phrase` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_rhythm_phrase_variation_rank_2_sample_2` | `phrase` | `too_stiff` | `fits` | `reject` |
| `data_motif_rhythm_phrase_variation_rank_3_sample_3` | `phrase` | `too_stiff` | `fits` | `reject` |

Decision:

- rhythm/phrase variation은 register floor, max interval, duration/IOI objective metrics를 개선했다.
- 그러나 `data_motif_rhythm_phrase_variation` rank 1-3 MIDI note/start/duration sequence가 완전히 동일했다.
- rank 1은 representative follow-up candidate로 남기고, rank 2/3은 duplicate review evidence라 reject한다.
- 다음은 variation mode의 sample-level diversity repair다.

## Previous Probe Result

Issue #118은 Issue #116 contour repair MIDI-note proxy review에서 드러난 `too_stiff`, `too_mechanical`, `too_repetitive`, `weak_phrase` 문제를 좁혀서 검증했다.

Docs:

- `docs/STAGE_B_RHYTHM_PHRASE_VARIATION_2026-05-25.md`

Implemented:

- `data_motif_rhythm_phrase_variation` baseline mode
- variable slot boundary rhythm placement
- varied duration fitting before next onset
- penultimate approach pitch class before guide-tone landing
- solo register floor/ceiling: `48-84`
- variation mode pitch interval bound: `6`
- harness:
  - `bash scripts/agent_harness.sh stage-b-rhythm-phrase-variation`

Result:

| mode | samples | strict | landing | max interval | sync | dur-var | ioi-var | objective flags |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `data_motif_contour_landing_repair` | 3 | 3 | 3/3 | 7 | 0.625 | 0.062 | 0.079 | `{}` |
| `data_motif_rhythm_phrase_variation` | 3 | 3 | 3/3 | 6 | 0.694 | 0.097 | 0.115 | `{}` |

Decision:

- rhythm/phrase variation은 objective rhythm target을 개선했다.
- variation 후보는 pitch floor `>=51`, unresolved large leap ratio `0.000`, repeated pitch interval ratio `0.000`이다.
- 하지만 note count가 `60`으로 줄고 tension ratio가 `0.371`로 낮아졌다.
- Issue #120 proxy review 결과, 새 variation 후보 3개가 exact duplicate라 sample-level diversity repair가 필요하다.

## Previous Review Result

Issue #116은 Issue #115 이후 contour/landing repair 후보 3개와 phrase recovery baseline 후보 3개를 같은 listening review notes schema로 채웠다.

중요한 전제:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note timing, pitch contour, objective MIDI metrics, context chord guide track을 읽은 proxy review다.
- `keep` 후보는 만들지 않는다.

Docs:

- `docs/STAGE_B_CONTOUR_REPAIR_MIDI_PROXY_REVIEW_2026-05-25.md`

Generated outputs:

- proxy-filled notes:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_contour_landing_repair_proxy/contour_repair_listening_review_notes_midi_proxy.json`
- aggregate:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_contour_landing_repair_proxy/listening_review_aggregate.md`

Result:

- reviewed candidates: `6`
- pending candidates: `0`
- decisions:
  - `needs_followup`: `5`
  - `reject`: `1`
  - `keep`: `0`
- phrase quality:
  - `phrase`: `1`
  - `fragment`: `4`
  - `exercise`: `1`
- timing:
  - `too_stiff`: `6`
- issues:
  - `bad_timing`: `6`
  - `too_mechanical`: `6`
  - `too_repetitive`: `6`
  - `weak_phrase`: `5`

Candidate decisions:

| candidate | phrase | timing | chord_fit | decision |
|---|---|---|---|---|
| `data_motif_contour_landing_repair_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_contour_landing_repair_rank_2_sample_2` | `phrase` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_contour_landing_repair_rank_3_sample_3` | `fragment` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_phrase_recovery_rank_1_sample_1` | `fragment` | `too_stiff` | `fits` | `needs_followup` |
| `data_motif_phrase_recovery_rank_2_sample_2` | `fragment` | `too_stiff` | `unclear` | `needs_followup` |
| `data_motif_phrase_recovery_rank_3_sample_3` | `exercise` | `too_stiff` | `unclear` | `reject` |

Decision:

- contour/landing repair는 objective target을 개선했지만 아직 musical keep 후보를 만들지 못했다.
- strongest candidate는 `data_motif_contour_landing_repair_rank_2_sample_2`다.
- 다음은 landing repair가 아니라 rhythm/phrase vocabulary variation을 봐야 한다.
- broad training, audio diffusion, backend/UI는 아직 다음 단계가 아니다.

## Previous Probe Result

이번 probe는 2026-05-24 MIDI-note proxy review에서 드러난 contour/landing 문제를 좁혀서 검증했다.

Docs:

- `docs/STAGE_B_CONTOUR_LANDING_REPAIR_2026-05-25.md`

Implemented:

- `data_motif_contour_landing_repair` baseline mode
- data-derived rhythm/contour template 유지
- bar 마지막 note를 guide tone/non-root chord tone으로 landing
- 같은 음 반복을 피하는 bounded pitch-class selector
- contour/landing metrics:
  - `final_landing_resolved`
  - `final_landing_role`
  - `max_abs_interval`
  - `abrupt_register_reset_count`
- harness:
  - `bash scripts/agent_harness.sh stage-b-contour-landing-repair`

Result:

| mode | samples | strict | final landing | max interval | resets | objective flags |
|---|---:|---:|---:|---:|---:|---|
| `data_motif_contour_landing_repair` | 3 | 3 | 3/3 | 7 | 0 | `{}` |
| `data_motif_phrase_recovery` | 3 | 3 | 1/3 | 13 | 0 | `{}` |

Decision:

- contour/landing objective target은 개선됐다.
- new mode는 repeated-pitch objective flag도 만들지 않았다.
- 하지만 rhythm stiffness는 해결되지 않았다.
- listening review notes는 아직 `6`개 모두 pending이다.
- 다음은 generation rule을 더 바꾸기 전에 context MIDI 기준으로 이 6개 후보를 review해야 한다.

## Previous Review Result

Issue #113은 Issue #109 clean review package와 Issue #111 clean context diagnostics 결과를 바탕으로,
objective-clean context MIDI 후보 3개를 같은 schema에서 review할 수 있는 notes template을 추가한 단계다.

그 다음 로컬 follow-up으로 MIDI-note proxy review를 작성했다.

중요한 전제:

- 실제 오디오 청취 리뷰가 아니다.
- MIDI note timing, pitch contour, context chord guide track을 읽은 piano-roll proxy review다.
- `keep` 후보는 아직 없다.
- 이 결과는 다음 generation probe 방향을 정하기 위한 보조 근거다.

Docs:

- `docs/STAGE_B_CLEAN_LISTENING_REVIEW_NOTES_2026-05-23.md`
- `docs/STAGE_B_CLEAN_MIDI_PROXY_REVIEW_2026-05-24.md`

Result:

- reviewed candidates: `3`
- pending candidates: `0`
- decisions:
  - `needs_followup`: `2`
  - `reject`: `1`
  - `keep`: `0`

Candidate decisions:

| candidate | timing | chord_fit | phrase | landing | vocabulary | decision |
|---|---|---|---|---|---|---|
| `data_motif_phrase_recovery_rank_1_sample_1` | `stiff` | `acceptable` | `acceptable` | `acceptable` | `thin` | `needs_followup` |
| `data_motif_phrase_recovery_rank_2_sample_2` | `stiff` | `acceptable` | `weak` | `unresolved` | `thin` | `needs_followup` |
| `data_motif_phrase_recovery_rank_3_sample_3` | `stiff` | `acceptable` | `broken` | `unresolved` | `exercise_like` | `reject` |

Decision:

- 후보 1은 현재 best follow-up baseline으로 남긴다.
- 후보 2는 landing repair와 contour smoothing 테스트용으로만 의미가 있다.
- 후보 3은 negative example로 둔다.
- 다음 probe는 broad training이나 audio pivot이 아니라 contour/cadence/landing repair여야 한다.
- rhythm stiffness와 repeated duration/rest template도 같이 추적해야 한다.

## Previous Probe Result

Issue #111은 Issue #109 clean review package 후보 3개를 context listening review 전에 MIDI note-level로 다시 진단한 단계다.

중요한 전제:

- 새 generation rule을 추가한 것이 아니다.
- objective-clean 후보가 실제 jazz quality를 보장하지 않는다.
- 목적은 "이제 들어볼 후보인지"와 "자동 rule을 더 고칠 문제인지"를 분리하는 것이다.

Implemented:

- `scripts/build_clean_context_diagnostics.py`
- `tests/test_clean_context_diagnostics.py`
- `scripts/agent_harness.sh stage-b-clean-context-diagnostics`
- docs:
  - `docs/STAGE_B_CLEAN_CONTEXT_DIAGNOSTICS_2026-05-23.md`

Result:

- output:
  - `outputs/stage_b_clean_context_diagnostics/harness_stage_b_clean_context_diagnostics/clean_context_diagnostics.md`
- candidate count: `3`
- diagnostic flags: none
- decision hint:
  - `listen_with_context`: `3`
- all candidates:
  - note count: `63`
  - bar coverage: `8/8`
  - off-grid ratio: `0.000`
  - max duration: `1.000` beat
  - context MIDI: exists
  - chord guide: exists
  - bass root guide: exists

Decision:

- 객관 진단상 지금 후보 3개는 더 자동으로 거르기보다 들어볼 단계다.
- Issue #113과 2026-05-24 proxy review에서 notes는 채워졌다.
- proxy review 결과, 후보들은 objective-clean이지만 여전히 timing stiff, contour/landing weakness, thin vocabulary 문제가 있다.
- 다음은 data-derived contour/cadence landing repair probe다.
- LMDM/audio diffusion 구현으로 pivot하지 않는다.

## Previous Probe Result

Issue #109는 Issue #107의 Stage B data-motif phrase recovery 결과에서 objective-clean 후보만 추출해 listening review package로 묶는 단계다.

중요한 전제:

- 새 generation rule을 추가한 것이 아니다.
- objective clean은 subjective jazz quality를 뜻하지 않는다.
- 목적은 "들을 가치가 있는 후보"만 줄여서 review loop를 좁히는 것이다.

Implemented:

- `scripts/build_clean_review_package.py`
- clean 후보 필터:
  - `objective_bucket == clean`
  - `objective_flags == []`
  - `mode == data_motif_phrase_recovery`
- copied solo MIDI/context MIDI review package
- `scripts/agent_harness.sh stage-b-clean-review-package`
- docs:
  - `docs/STAGE_B_CLEAN_REVIEW_PACKAGE_2026-05-23.md`

Result:

- output:
  - `outputs/stage_b_clean_review_package/harness_stage_b_clean_review_package/clean_review_package.md`
- candidate count: `3`
- selected mode:
  - `data_motif_phrase_recovery`
- selected candidates:
  - `data_motif_phrase_recovery_rank_1_sample_1`
  - `data_motif_phrase_recovery_rank_2_sample_2`
  - `data_motif_phrase_recovery_rank_3_sample_3`
- note count: `63`, `63`, `63`
- unique pitch count: `19`, `23`, `22`
- unresolved large leap ratio: `0.000`, `0.000`, `0.045`
- chord-tone ratio: `0.508`, `0.524`, `0.476`
- tension ratio: `0.492`, `0.476`, `0.524`

Decision:

- 이제 "무작위 후보 전체"가 아니라 objective-clean 후보 3개를 context MIDI로 들으면 된다.
- 이 결과는 아직 jazz solo quality 성공이 아니다.
- 다음 판단은 chord context 위에서 phrase continuation, landing, timing, jazz vocabulary가 실제로 들리는지 확인하는 것이다.

## Previous Probe Result

Issue #107은 `phrase_recovery` pitch grammar를 data-derived motif rhythm template과 결합한 단계다.

중요한 전제:

- hand-written grid가 아니라 실제 데이터에서 추출한 rhythm shape를 사용한다.
- objective clean은 subjective jazz quality를 뜻하지 않는다.
- 다음 단계는 듣기 리뷰가 필요하다.

Implemented:

- `data_motif_phrase_recovery` baseline mode
- data-derived rhythm position/duration 유지
- phrase recovery pitch grammar 결합
- `scripts/agent_harness.sh stage-b-data-motif-phrase-recovery-review`
- docs:
  - `docs/STAGE_B_DATA_MOTIF_PHRASE_RECOVERY_2026-05-22.md`

Result:

- candidate count: `9`
- objective bucket counts:
  - clean: `6`
  - warning: `3`
- objective flag counts:
  - unresolved large leaps: `3`
- mode flag counts:
  - `data_motif_guide_tones`: unresolved large leaps `3`
  - `data_motif_phrase_recovery`: no objective flags
  - `phrase_recovery`: no objective flags
- `data_motif_guide_tones` unresolved large leap ratio: `0.583-0.652`
- `data_motif_phrase_recovery` unresolved large leap ratio: `0.000-0.045`
- `data_motif_phrase_recovery` tension ratio: `0.476-0.524`

Decision:

- `data_motif_phrase_recovery`는 data rhythm shape와 phrase recovery를 동시에 만족한다.
- 다음 작업은 review MIDI/context MIDI를 listening review 대상으로 정리하거나, subjective review notes를 채우는 것이다.

## Previous Probe Result

Issue #105는 Issue #103에서 드러난 `unresolved_large_leaps` 문제를 줄이기 위해 phrase recovery baseline을 추가한 단계다.

중요한 전제:

- 큰 도약을 금지하지 않는다.
- 큰 도약 뒤 반대 방향 small recovery를 넣는다.
- objective clean은 subjective jazz quality를 뜻하지 않는다.

Implemented:

- `phrase_recovery` baseline mode
- `recovery_pitch_after_large_leap`
- `scripts/agent_harness.sh stage-b-phrase-recovery-review`
- docs:
  - `docs/STAGE_B_PHRASE_RECOVERY_REVIEW_2026-05-22.md`

Result:

- candidate count: `6`
- objective bucket counts:
  - clean: `3`
  - warning: `3`
- objective flag counts:
  - unresolved large leaps: `3`
- mode flag counts:
  - `phrase_cadence`: unresolved large leaps `3`
  - `phrase_recovery`: no objective flags
- `phrase_cadence` unresolved large leap ratio: `0.750-0.757`
- `phrase_recovery` unresolved large leap ratio: `0.000-0.048`

Decision:

- `phrase_recovery`는 objective phrase naturalness risk를 줄인다.
- 다음 작업은 `phrase_recovery`를 data-derived motif rhythm과 결합하거나, review MIDI/context MIDI 기준으로 listening review package를 만드는 것이다.

## Previous Probe Result

Issue #103은 Issue #101의 phrase/cadence 후보가 scalar/chromatic flag를 줄인 대신 leap-heavy exercise가 되었는지 확인하기 위해 phrase naturalness objective metric을 추가한 단계다.

중요한 전제:

- 이 단계는 subjective jazz quality를 자동 판정하지 않는다.
- "큰 도약 뒤 회복 움직임이 있는가"만 objective risk로 본다.
- 기존 Issue #101 결과를 더 엄격하게 다시 해석한다.

Implemented:

- large leap count
- resolved / unresolved large leap count
- unresolved large leap ratio
- `unresolved_large_leaps` objective flag
- listening review notes metric propagation
- docs:
  - `docs/STAGE_B_PHRASE_NATURALNESS_OBJECTIVES_2026-05-22.md`

Result:

- Issue #101 phrase/cadence review set에 새 metric 적용
- candidate count: `12`
- objective bucket counts:
  - warning: `12`
- objective flag counts:
  - chromatic walk: `1`
  - unresolved large leaps: `12`
- previous Issue #101 had clean `11`, warning `1` before this metric existed.

Decision:

- scalar/chromatic issue는 줄었지만, phrase naturalness risk가 전 후보에서 드러났다.
- 이것은 regression이 아니라 이전 metric이 못 보던 failure mode를 드러낸 것이다.
- 다음 작업은 leap 뒤에 반대 방향 small recovery를 넣는 phrase-shape grammar 또는 data-derived contour resolution pattern이어야 한다.

## Previous Probe Result

Issue #101은 duration collapse 이후 남은 scalar/chromatic exercise 문제를 줄이기 위해 phrase/cadence review baseline을 추가한 단계다.

중요한 전제:

- 이 단계는 objective pitch-contour flag를 줄이는 probe다.
- "재즈답다"를 자동으로 보장하지 않는다.
- overlap-free export와 varied-duration grid는 유지한다.
- subjective listening review는 아직 pending이다.

Implemented:

- `phrase_cadence` baseline mode
- phrase interval을 선호하는 pitch-class/register target
- selected-mode strict gate
- `scripts/agent_harness.sh stage-b-phrase-cadence-review`
- docs:
  - `docs/STAGE_B_PHRASE_CADENCE_REVIEW_2026-05-22.md`

Result:

- candidate count: `12`
- objective reviewable count: `12`
- objective bucket counts:
  - clean: `11`
  - warning: `1`
- objective flag counts:
  - chromatic walk: `1`
  - too stepwise/scalar: `0`
  - duration pattern collapse: `0`
  - overlap/polyphonic: `0`
- previous issue had:
  - chromatic walk: `7`
  - too stepwise/scalar: `6`
- aggregate report:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_phrase_cadence_review/listening_review_aggregate.md`

Decision:

- scalar/chromatic objective flag는 크게 줄었다.
- 이제 남은 핵심 문제는 objective metric보다 subjective phrase quality다.
- 다음 작업은 새 review set을 기준으로 listening notes를 채우거나, subjective review 없이 바꿀 경우에는 phrase naturalness를 측정하는 objective metric을 추가해야 한다.

## Previous Probe Result

Issue #99는 duration collapse를 줄이기 위해 varied-duration review baseline을 추가한 단계다.

중요한 전제:

- 이 단계는 rhythmic 다양성을 조금 추가하는 probe다.
- "재즈답다"를 자동으로 보장하지 않는다.
- overlap-free export는 계속 유지한다.
- subjective listening review는 아직 pending이다.

Implemented:

- `varied_grid` baseline mode
- `varied_guide_tones` baseline mode
- 16th-grid onset을 유지하면서 duration pattern 다양화
- `scripts/agent_harness.sh stage-b-duration-variation-review`
- docs:
  - `docs/STAGE_B_DURATION_VARIATION_REVIEW_2026-05-22.md`

Result:

- candidate count: `15`
- objective reviewable count: `15`
- objective bucket counts:
  - clean: `8`
  - warning: `7`
- objective flag counts:
  - chromatic walk: `7`
  - too stepwise/scalar: `6`
  - duration pattern collapse: `0`
  - overlap/polyphonic: `0`
- previous duration pattern collapse count was `6`
- aggregate report:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_duration_variation_review/listening_review_aggregate.md`

Decision:

- duration collapse는 objective flag 기준으로 제거됐다.
- 이제 남은 핵심 문제는 scalar/chromatic exercise 느낌이다.
- 다음 작업은 pitch contour / phrase vocabulary / cadence target을 개선하는 방향이어야 한다.

## Previous Probe Result

Issue #97은 review export 단계에서 overlap-free solo-line MIDI variant를 만드는 단계다.

중요한 전제:

- 원본 generated sample MIDI는 `midi_path`에 보존한다.
- 사람이 듣는 `review_midi_path`에는 `*_overlap_free.mid` variant를 넣는다.
- 이것은 음악성을 높이는 학습이 아니라, chord block처럼 보이는 overlap artifact를 review 전에 제거하는 export step이다.
- subjective jazz quality를 성공으로 주장하지 않는다.

Implemented:

- `--overlap_free_review_midi`
- overlap-free solo-line MIDI writer
- review manifest `review_variant`
- review manifest `review_postprocess_report`
- listening review notes `review_metadata.review_variant`
- `scripts/agent_harness.sh stage-b-overlap-free-review-export`
- docs:
  - `docs/STAGE_B_OVERLAP_FREE_REVIEW_EXPORT_2026-05-22.md`

Result:

- candidate count: `15`
- objective reviewable count: `15`
- objective bucket counts:
  - clean: `5`
  - warning: `10`
- objective flag counts:
  - chromatic walk: `7`
  - duration pattern collapse: `6`
  - too stepwise/scalar: `4`
  - overlap/polyphonic: `0`
- previous overlap/polyphonic count was `9`
- aggregate report:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_overlap_free_review_export/listening_review_aggregate.md`

Decision:

- overlap/polyphonic 문제는 review export에서 제거됐다.
- 지금 남은 문제는 jazz vocabulary보다 duration collapse, scalar/chromatic motion, phrase quality 쪽이다.
- 다음 generation rule change는 duration/rhythm variation 또는 candidate vocabulary 개선이어야 한다.

## Previous Probe Result

Issue #95는 objective MIDI note-level diagnostics를 listening review notes와 aggregate priority에 연결하는 단계다.

중요한 전제:

- 이 단계는 subjective listening review를 대체하지 않는다.
- "재즈답다"를 자동 판정하지 않는다.
- 사람이 들어야 할 후보를 objective problem/warning priority로 정렬한다.
- overlap/polyphonic, off-grid, duration collapse, scalar/chromatic flag를 review notes 안에 보존한다.

Implemented:

- objective penalty / bucket / reviewable / priority score
- listening review notes에 `objective_review` 첨부
- aggregate report에 objective flag/bucket counts 추가
- objective review priority table
- `scripts/build_listening_review_notes.py --objective_midi_review_report`
- `scripts/agent_harness.sh stage-b-objective-flags-review-flow`
- docs:
  - `docs/STAGE_B_OBJECTIVE_FLAGS_REVIEW_FLOW_2026-05-22.md`

Result:

- candidate count: `15`
- objective reviewable count: `6`
- objective bucket counts:
  - problem: `9`
  - warning: `6`
- objective flag counts:
  - chromatic walk: `7`
  - duration pattern collapse: `9`
  - overlap/polyphonic: `9`
  - too stepwise/scalar: `4`
- aggregate report:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_objective_flags_review_flow/listening_review_aggregate.md`

Decision:

- 지금 후보 중 `problem` 9개는 우선순위 낮은 review target이다.
- `warning` 6개도 좋은 재즈 솔로라는 뜻은 아니며, duration collapse/chromatic exercise 문제를 갖는다.
- 다음 generation rule change는 overlap을 만들지 않는 solo-line export와 duration/rhythm variation 개선을 먼저 봐야 한다.

## Previous Probe Result

Issue #93은 generated review MIDI를 직접 읽어 objective note-level diagnostics를 만드는 단계다.

중요한 전제:

- 이 리포트는 subjective listening review가 아니다.
- "재즈답다"를 판정하지 않는다.
- 사람이 듣기 전에 걸러낼 수 있는 machine-observable 문제를 잡는다.

Implemented:

- objective MIDI note review script
- max active notes / polyphonic tick ratio
- 16th grid alignment
- duration pattern collapse
- stepwise / chromatic walk ratio
- chord-tone / tension / outside / root ratio
- first 16 note preview
- `scripts/review_midi_note_objectives.py`
- `scripts/agent_harness.sh stage-b-objective-midi-review`
- docs:
  - `docs/STAGE_B_OBJECTIVE_MIDI_NOTE_REVIEW_2026-05-22.md`

Result:

- candidate count: `15`
- flag counts:
  - chromatic walk: `7`
  - duration pattern collapse: `9`
  - overlap/polyphonic: `9`
  - too stepwise/scalar: `4`
- report:
  - `outputs/stage_b_objective_midi_review/harness_stage_b_objective_midi_review/objective_midi_note_review.md`

Decision:

- `hand_written_swing`은 실제로 16th grid 밖이라기보다 overlap/polyphonic + scalar/chromatic + duration collapse 문제가 크다.
- `straight_grid`는 timing grid는 맞지만 chromatic/scale exercise 성향이 강하다.
- 다음 rule change는 subjective listening 이전에도 objective flags를 review priority와 gate에 반영해야 한다.

## Previous Probe Result

Issue #91은 review manifest 전체를 listening review notes template으로 변환하는 단계다.

중요한 전제:

- subjective listening result를 임의 작성하지 않는다.
- 기존 generated chord eval report 기반 6개 후보 notes 경로는 유지한다.
- 새 경로는 hand-written swing, straight-grid reference까지 포함한 full review package를 notes로 만든다.

Implemented:

- `scripts/build_listening_review_notes.py --review_manifest`
- full review manifest candidate 변환
- `review_metadata`와 `review_files` 필드 추가
- rhythm/timing metrics를 `source_metrics`에 보존
- `scripts/agent_harness.sh stage-b-full-review-notes`
- docs:
  - `docs/STAGE_B_FULL_REVIEW_MANIFEST_NOTES_2026-05-22.md`

Result:

- candidate count: `15`
- reviewed count: `0`
- pending count: `15`
- first candidate:
  - `data_motif_rank_1_sample_1`
- last candidate:
  - `straight_guide_tones_rank_3_sample_3`
- review notes template:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_full_review_notes/review_notes_template.json`

Decision:

- 이제 사람이 들어야 할 review MIDI/context MIDI path가 notes 안에 직접 남는다.
- 다음 rule change는 이 full notes가 실제로 채워진 뒤 aggregate 결과로 결정한다.

## Previous Probe Result

Issue #89는 사람이 채운 listening review notes를 다음 generation rule 수정 후보로 집계하는 단계다.

중요한 전제:

- subjective listening result를 임의 작성하지 않는다.
- pending-only notes에서는 generation rule 변경을 추천하지 않는다.
- 사람이 채운 `issues`와 `decision`만 다음 실험 분기 기준으로 사용한다.
- real Brad/reference chord label 문제는 아직 해결되지 않았다.

Implemented:

- listening review aggregate script
- decision / phrase quality / timing / chord fit / issue count aggregation
- source metric summary by decision
- pending-only safety follow-up
- `scripts/summarize_listening_review_notes.py`
- `scripts/agent_harness.sh stage-b-listening-review-aggregate`
- docs:
  - `docs/STAGE_B_LISTENING_REVIEW_AGGREGATE_2026-05-22.md`

Result:

- candidate count: `6`
- reviewed count: `0`
- pending count: `6`
- has reviewed candidates: `false`
- recommended follow-up:
  - `collect_listening_reviews`
- aggregate report:
  - `outputs/stage_b_listening_review_aggregate/harness_stage_b_listening_review_aggregate/listening_review_aggregate.json`

Decision:

- 지금 artifact만으로 generation rule을 바꾸면 안 된다.
- 다음 rule change는 사람이 채운 review notes의 issue distribution을 근거로 분기한다.

## Previous Probe Result

Issue #87은 청취 리뷰 결과를 후보별로 구조화해 기록할 notes schema를 만든 단계다.

중요한 전제:

- raw generated MIDI는 `outputs/` artifact로만 둔다.
- subjective listening result를 임의 작성하지 않는다.
- 사람이 들은 결과를 같은 enum/field로 기록하기 위한 양식이다.
- real Brad/reference chord label 문제는 아직 해결되지 않았다.

Implemented:

- listening review notes schema
- review notes template generator
- enum/status validator
- `scripts/build_listening_review_notes.py`
- `scripts/agent_harness.sh stage-b-listening-review-notes`
- docs:
  - `docs/STAGE_B_LISTENING_REVIEW_NOTES_2026-05-22.md`

Result:

- candidate count: `6`
- reviewed count: `0`
- pending count: `6`
- decisions:
  - keep: `0`
  - needs_followup: `0`
  - reject: `0`
  - pending: `6`
- review notes template:
  - `outputs/stage_b_listening_review_notes/harness_stage_b_listening_review_notes/review_notes_template.json`

Decision:

- 이제 실제 청취 리뷰 결과를 `phrase_quality`, `timing`, `chord_fit`, `issues`, `decision`으로 분리해서 기록할 수 있다.
- 다음 작업은 filled review notes를 aggregate해 다음 generation rule을 결정하는 것이다.

## Previous Probe Result

Issue #85는 generated chord eval summary를 기존 review markdown과 결합한 단계다.

중요한 전제:

- raw generated MIDI는 `outputs/` artifact로만 둔다.
- 원본 `review_candidates.md`는 수정하지 않는다.
- combined markdown은 새 artifact로 생성한다.
- real Brad/reference chord label 문제는 아직 해결되지 않았다.

Implemented:

- `scripts/evaluate_generated_candidate_chords.py --review_markdown`
- combined review markdown writer
- chord eval append markdown table
- `scripts/agent_harness.sh stage-b-review-markdown-chord-eval`
- docs:
  - `docs/STAGE_B_REVIEW_MARKDOWN_CHORD_EVAL_2026-05-22.md`

Result:

- evaluated candidate count: `6`
- note count: `192`
- aggregate chord-tone ratio: `0.656`
- aggregate tension ratio: `0.120`
- aggregate outside ratio: `0.000`
- `data_motif` chord-tone ratio: `0.500`
- `data_motif_guide_tones` chord-tone ratio: `0.812`
- approach ratio: `0.224`
- outside ratio: `0.000`
- combined review markdown:
  - `outputs/stage_b_generated_chord_eval/harness_stage_b_review_markdown_chord_eval/review_candidates_with_chord_eval.md`

Decision:

- 이제 review markdown 한 파일에서 MIDI path, context path, rhythm metrics, chord-role metrics를 같이 볼 수 있다.
- 다음 작업은 listening review 결과를 구조화해서 "좋다/나쁘다"를 말로 흘리지 않게 만드는 것이다.

## Previous Probe Result

Issue #83은 Issue #81의 bridge를 실제 `stage-b-data-guide-hybrid` review package에 적용한 단계다.

중요한 전제:

- raw generated MIDI는 `outputs/` artifact로만 둔다.
- `review_manifest.json`의 known chord progression metadata만 사용한다.
- real Brad/reference chord label 문제는 아직 해결되지 않았다.

Implemented:

- `stage-b-data-guide-hybrid` review package 생성
- generated chord eval bridge 적용
- `data_motif` vs `data_motif_guide_tones` chord-role profile 비교
- `scripts/agent_harness.sh stage-b-data-guide-generated-chord-eval`
- docs:
  - `docs/STAGE_B_DATA_GUIDE_GENERATED_CHORD_EVAL_2026-05-22.md`

Result:

- evaluated candidate count: `6`
- note count: `192`
- aggregate chord-tone ratio: `0.656`
- aggregate tension ratio: `0.120`
- aggregate outside ratio: `0.000`
- `data_motif` chord-tone ratio: `0.500`
- `data_motif_guide_tones` chord-tone ratio: `0.812`
- `data_motif` approach ratio: `0.281-0.312`
- `data_motif_guide_tones` approach ratio: `0.156`
- report: `outputs/stage_b_generated_chord_eval/harness_stage_b_data_guide_generated_chord_eval/generated_chord_eval_report.md`

Decision:

- `data_motif_guide_tones`는 실제로 chord-tone 쪽으로 더 안전하게 기울어져 있다.
- 현재 문제가 outside note 폭주는 아니다.
- 더 정확한 문제 후보는 chord-tone safety 과다, 낮은 tension 비율, phrase vocabulary 부족이다.

## Previous Probe Result

Issue #81은 Issue #79의 chord-labeled evaluator를 generated candidate report와 연결한 단계다.

중요한 전제:

- generated candidate report가 chord progression metadata를 알고 있을 때만 평가한다.
- solo-only MIDI에서 chord를 추측하지 않는다.
- real Brad/reference chord label 문제는 아직 해결되지 않았다.

Implemented:

- generated review/candidate report bridge
- `chord_progression`, `chords`, `request.chord_progression`, `source_report.request.chord_progression` fallback
- generated MIDI candidate `review_midi_path` / `midi_path` support
- harness-local tiny generated-candidate fixture
- `scripts/evaluate_generated_candidate_chords.py`
- `scripts/agent_harness.sh stage-b-generated-chord-eval`
- docs:
  - `docs/STAGE_B_GENERATED_CHORD_EVAL_2026-05-22.md`

Result:

- fixture sample count: `1`
- fixture note count: `16`
- chord-tone ratio: `1.000`
- tension ratio: `0.000`
- approach ratio: `0.000`
- outside ratio: `0.000`
- report: `outputs/stage_b_generated_chord_eval/harness_stage_b_generated_chord_eval/generated_chord_eval_report.md`

Decision:

- generated candidate report에 known chord progression metadata가 있으면 pitch-role evaluator로 연결할 수 있다.
- fixture score는 model quality가 아니라 bridge smoke result다.
- 다음 작업은 실제 `stage-b-data-guide-hybrid` review manifest 같은 generated review package에 이 bridge를 적용하는 것이다.

## Previous Probe Result

Issue #79는 Issue #77의 결론을 받아, 작은 chord-labeled evaluation subset contract를 만든 단계다.

중요한 전제:

- 실제 Brad/reference 곡에 코드를 임의로 붙이지 않는다.
- 현재 committed fixture는 `inline_notes` 기반 tiny contract test다.
- 실제 phrase는 chord label이 확실하거나 수동 검증된 경우에만 manifest에 추가한다.

Implemented:

- chord-labeled eval manifest schema
- tiny inline-note fixture:
  - `data/eval/stage_b_chord_labeled_tiny/manifest.json`
- manifest validator
- bar-level chord label 기반 pitch-role summary
- MIDI path sample support without committing raw MIDI
- `scripts/evaluate_chord_labeled_subset.py`
- `scripts/agent_harness.sh stage-b-chord-labeled-eval`
- docs:
  - `docs/STAGE_B_CHORD_LABELED_EVAL_2026-05-22.md`

Result:

- fixture sample count: `2`
- fixture note count: `32`
- chord-tone ratio: `0.844`
- tension ratio: `0.156`
- approach ratio: `0.000`
- outside ratio: `0.000`
- report: `outputs/stage_b_chord_labeled_eval/harness_stage_b_chord_labeled_eval/chord_labeled_eval_report.md`

Decision:

- pitch-role evaluator는 known chord labels를 받으면 정상 동작한다.
- 아직 real reference phrase가 라벨링된 것은 아니다.
- 다음 작업은 코드가 확실한 3-5개 phrase를 manifest에 추가하거나, generated candidate metadata를 이 evaluator와 연결하는 것이다.

## Previous Probe Result

Issue #77은 Issue #75에서 드러난 chord annotation blocker를 실제 dataset 기준으로 확인한 단계다.

Implemented:

- role dataset `meta.json` chord field/string scan
- raw dataset sidecar scan:
  - `.json`
  - `.csv`
  - `.tsv`
  - `.txt`
  - `.lab`
  - `.jams`
  - `.xml`
  - `.musicxml`
  - `.mxl`
- MIDI lyric/text event chord-symbol scan
- `scripts/audit_chord_progression_coverage.py`
- `scripts/agent_harness.sh chord-coverage-audit`
- docs:
  - `docs/STAGE_B_CHORD_COVERAGE_AUDIT_2026-05-22.md`

Result:

- role meta scanned: `2812`
- role meta chord hits: `0`
- role meta unique source MIDI count: `28`
- sidecar files found: `0`
- MIDI files scanned for text events: `120`
- MIDI files with any text event: `0`
- MIDI chord-text candidate files: `0`
- usable chord annotation candidate: `false`
- report: `outputs/chord_coverage_audit/harness_chord_coverage_audit/chord_coverage_audit.md`

Decision:

- 현재 로컬 dataset에는 바로 사용할 수 있는 chord progression annotation이 없다.
- 따라서 generated 후보의 pitch-role 비율을 reference와 비교할 수 없다.
- 다음 논리적 작업은 generator 튜닝이 아니라 작은 chord-labeled evaluation subset 또는 chord inference/lead-sheet alignment다.
- 한 달 MVP 관점에서는 full chord inference보다 3-5개 phrase의 수동 chord-labeled evaluation subset이 더 작고 검증 가능하다.

## Previous Probe Result

Issue #75는 generated 후보를 더 만들기 전에 reference pitch-role 기준을 세우려는 단계였다.

Implemented:

- Stage B embedded chord token에서 bar chord 추출
- note group별 pitch role 분류:
  - `root`
  - `guide`
  - `chord`
  - `tension`
  - `approach`
  - `outside`
  - `unknown_chord`
- strong/eighth/offgrid bucket별 landing 분포
- generated 후보와 reference rhythm/pitch-role delta 비교
- reference chord coverage가 부족하면 pitch-role delta를 생략하는 guard
- `scripts/agent_harness.sh stage-b-reference-pitch-roles`
- docs:
  - `docs/STAGE_B_REFERENCE_PITCH_ROLE_STATS_2026-05-22.md`

Result:

- reference record count: `57`
- reference note group mean: `32.649`
- reference syncopated onset ratio mean: `0.736`
- reference duration diversity ratio mean: `0.379`
- reference IOI diversity ratio mean: `0.341`
- known chord note ratio: `0.000`
- unknown chord ratio: `1.000`
- generated pitch-role deltas: intentionally omitted
- report: `outputs/stage_b_reference_stats/harness_stage_b_reference_pitch_roles/reference_stats_report.md`

Decision:

- 현재 reference tokenized records에는 chord progression annotation이 들어있지 않다.
- 따라서 `data_motif_guide_tones`가 reference보다 chord-tone/tension/approach 비율이 어떤지는 아직 판단할 수 없다.
- 다음 논리적 작업은 generator 수정이 아니라 chord progression coverage audit 또는 chord annotation pipeline이다.

## Previous Probe Result

Issue #73은 Issue #71의 다음 단계다.

수동 리뷰 결론:

- `straight_guide_tones`는 pitch vocabulary 기준선이지만 rhythm이 너무 교과서적일 수 있다.
- `data_motif`는 rhythm variation이 더 낫지만 pitch가 scale/chromatic exercise처럼 들릴 수 있다.
- 따라서 이번 후보는 data-derived rhythm과 guide-tone/cadence pitch 문법을 결합한다.

Implemented:

- `data_motif_guide_tones` baseline mode
- data-derived rhythm/duration template 유지
- contour template은 register 방향 참고로만 사용
- strong beat guide-tone constraint
- `scripts/agent_harness.sh stage-b-data-guide-hybrid`
- docs:
  - `docs/STAGE_B_DATA_GUIDE_HYBRID_2026-05-22.md`

Result:

- compare gate: passed
- `data_motif`: strict `3/3`
- `data_motif_guide_tones`: strict `3/3`
- `data_motif_guide_tones` note count: `63`
- `data_motif_guide_tones` unique pitch count: `23-24`
- `data_motif_guide_tones` chord-tone ratio: `0.797`
- `data_motif_guide_tones` tension ratio: `0.062`
- `data_motif_guide_tones` root-tone ratio: `0.000`
- `data_motif_guide_tones` unique bar-position pattern ratio: `1.000`
- review output: `outputs/stage_b_data_motif_review/harness_stage_b_data_guide_hybrid`

Decision:

- 이번 후보의 목적은 "재즈 솔로 완성"이 아니라 `data_motif`와 `straight_guide_tones`의 장단점을 결합한 listening-review candidate를 만드는 것이다.
- 들어볼 핵심 파일은 `outputs/stage_b_data_motif_review/harness_stage_b_data_guide_hybrid/context_midi/*data_motif_guide_tones*_with_context.mid`다.
- 이 후보도 초급 멜로디처럼 들리면 다음은 코드 추가가 아니라 reference-derived guide-tone landing 통계 추출이다.

## Previous Probe Result

Issue #71은 Issue #69 review listening에서 나온 피드백을 반영한다.

수동 리뷰 결론:

- `hand_written_swing`과 `data_motif`의 swing timing은 MIDI 입력 관점에서 박자가 어긋나게 들릴 수 있다.
- `straight_grid`는 박자는 맞지만 scale/chromatic exercise처럼 들린다.
- 따라서 다음 후보는 swing/humanization보다 straight quantized timing과 guide-tone/cadence pitch grammar를 먼저 검증해야 한다.

Implemented:

- `straight_guide_tones` baseline mode
- strong beat guide-tone constraint
- limited approach-tone cadence cell
- `scripts/agent_harness.sh stage-b-guide-tone-cadence`
- docs:
  - `docs/STAGE_B_GUIDE_TONE_CADENCE_2026-05-22.md`

Result:

- compare gate: passed
- `data_motif`: strict `3/3`
- `hand_written_swing`: strict `3/3`
- `straight_grid`: strict `0/3`, timing reference
- `straight_guide_tones`: strict `0/3`, timing/pitch reference
- `straight_guide_tones` note count: `64`
- `straight_guide_tones` unique pitch count: `26-29`
- `straight_guide_tones` chord-tone ratio: `0.656`
- `straight_guide_tones` tension ratio: `0.172`
- `straight_guide_tones` root-tone ratio: `0.000`
- review output: `outputs/stage_b_data_motif_review/harness_stage_b_guide_tone_cadence`

Decision:

- swing 후보는 현재 MVP default로 밀지 않는다.
- straight timing 후보와 data-derived motif 후보를 chord context 위에서 비교한다.
- `straight_guide_tones`는 모델 성공 후보가 아니라, 박자와 harmonic vocabulary를 분리해 듣기 위한 reference다.
- 다음 단계는 chord/pitch-role annotation 또는 data-motif rhythm + guide-tone strong-beat hybrid다.

## Previous Probe Result

Issue #69는 solo-only MIDI 리뷰의 한계를 해결하는 단계다.

수동 리뷰에서 solo-line만 들으면 chord progression이 들리지 않아 in/out 판단이 어렵고, swing/motif 후보는 박자가 딱 맞지 않는 것처럼 들릴 수 있다는 피드백이 나왔다. 따라서 이번 작업은 chord/bass context와 straight-grid timing reference를 review export에 추가한다.

Implemented:

- `scripts/run_stage_b_data_motif_generation_compare.py` chord/context export
- `tests/test_stage_b_data_motif_generation_compare.py` context export and straight-grid tests
- `scripts/agent_harness.sh stage-b-review-context-grid`
- output files:
  - `data_motif_compare_report.json`
  - `data_motif_compare_report.md`
  - `review_manifest.json`
  - `review_candidates.md`
  - `named_midi/*.mid`
  - `context_midi/*_with_context.mid`
  - `chord_guide.mid`

Result:

- source report: `outputs/stage_b_data_motif_compare/harness_stage_b_review_context_grid/data_motif_compare_report.json`
- setup: `./midi_dataset/midi/studio`, max files `4`, `8`-bar windows, stride `4`, min notes `16`
- `hand_written_swing`: strict `3/3`
- `data_motif`: strict `3/3`
- `straight_grid`: exported as timing reference
- review candidates: `9`
- review output: `outputs/stage_b_data_motif_review/harness_stage_b_review_context_grid`
- data minus hand duration diversity delta: `+0.016`
- data minus hand IOI diversity delta: `+0.016`
- data minus hand bar-pattern delta: `+0.500`
- data minus hand syncopation delta: `-0.125`
- `data_motif` duration repetition ratio: `0.375`
- `hand_written_swing` duration repetition ratio: `0.750`

Decision:

- 이제 solo-only가 아니라 chord/bass guide 위에서 in/out을 판단할 수 있다.
- `straight_grid`는 musical quality candidate가 아니라 timing reference로 본다.
- 다음 판단은 context MIDI를 들어보고 swing을 유지할지, straight quantized output을 기본으로 둘지 결정하는 것이다.

Detail:

- `docs/STAGE_B_CANDIDATE_RANKING_2026-05-20.md`
- `docs/STAGE_B_RANKING_HARMONIC_GATE_2026-05-21.md`
- `docs/STAGE_B_CHORD_AWARE_PITCH_2026-05-21.md`
- `docs/STAGE_B_CANDIDATE_REVIEW_EXPORT_2026-05-21.md`
- `docs/STAGE_B_LONGER_PHRASE_PROBE_2026-05-21.md`
- `docs/STAGE_B_PHRASE_CONTOUR_DIAGNOSTICS_2026-05-21.md`
- `docs/STAGE_B_ROOT_BIAS_DIAGNOSTICS_2026-05-21.md`
- `docs/STAGE_B_PITCH_MODE_COMPARE_2026-05-21.md`
- `docs/STAGE_B_8BAR_APPROACH_PHRASE_2026-05-21.md`
- `docs/STAGE_B_SWING_MOTIF_PHRASE_2026-05-21.md`
- `docs/STAGE_B_REFERENCE_PHRASE_STATS_2026-05-21.md`
- `docs/STAGE_B_MOTIF_TEMPLATE_EXTRACTION_2026-05-21.md`
- `docs/STAGE_B_DATA_MOTIF_GENERATION_2026-05-21.md`
- `docs/STAGE_B_DATA_MOTIF_REVIEW_EXPORT_2026-05-21.md`
- `docs/STAGE_B_REVIEW_CONTEXT_GRID_2026-05-22.md`

## Active Issue #14

Current task:

- define Stage B duration-explicit tokenization
- keep Stage B separate from Stage A `control_v1` until the tokenizer contract is tested
- add unit tests for token ranges, chord parsing, quantized note encoding, and roundtrip decoding

First implementation target:

- `scripts/stage_b_tokens.py`
- `tests/test_stage_b_tokens.py`
- `docs/STAGE_B_TOKENIZATION_SPEC.md`

Do not start broad training in this issue.

## Active Issue #15

Current task:

- wire `stage_b_v1` into `scripts/prepare_role_dataset.py`
- produce tokenized train/val records from existing role dataset preparation
- keep Stage B tokenized records target-only for the first contract
- do not start model training yet

First implementation target:

- `prepare_role_dataset.py --sequence_format stage_b_v1`
- unit test with explicit train/val manifests
- local Brad 2-file dry run under `outputs/`

Current result:

- `stage_b_v1` prepare path implemented
- unit tests pass
- Brad 2-file dry run produced tokenized train/val records
- detail: `docs/STAGE_B_ROLE_DATASET_PREP_2026-05-19.md`

## Active Issue #16

Current task:

- split Stage B target continuations into short fixed-bar phrase windows
- keep windowed records target-only for the first contract
- prove Brad 2-file window dry run produces many short `.npy` records
- do not start model training yet

First implementation target:

- `prepare_role_dataset.py --stage_b_window_bars`
- `prepare_role_dataset.py --stage_b_window_stride_bars`
- `prepare_role_dataset.py --stage_b_min_window_target_notes`
- unit test with manifest train/val windows
- local Brad 2-file window dry run under `outputs/`

Current result:

- 2-bar Stage B window path implemented
- Brad 2-file dry run produced 137 role samples
- token lengths: min `22`, p50 `77`, max `212`, mean `82.94`
- detail: `docs/STAGE_B_PHRASE_WINDOW_DATASET_2026-05-19.md`

## Active Issue #17

Status:

- completed and merged via PR #17

Completed task:

- connect Stage B phrase/window records to the model training path
- move Stage B token ranges into shared model constants
- ensure model `VOCAB_SIZE` covers Stage B tokens
- add a Stage B window tiny-overfit smoke script
- fail fast if the prepared dataset has no token records or token IDs exceed model vocab

First implementation target:

- `music_transformer/utilities/constants.py`
- `scripts/stage_b_tokens.py`
- `scripts/run_stage_b_window_tiny_overfit.py`
- `scripts/agent_harness.sh stage-b-window-prepare`
- tests for Stage B vocab compatibility and empty dataset rejection

Current result:

- one Brad file prepare-only smoke produced 70 Stage B window records
- max Stage B token id: `544`
- model vocab size: `547`
- one-epoch tiny training smoke completed
- train loss: `6.1135`
- val loss: `5.8195`
- detail: `docs/STAGE_B_WINDOW_TINY_OVERFIT_2026-05-19.md`

## Active Issue #18

Status:

- completed and merged via PR #19

Completed task:

- add a Stage B token generation/decode probe
- make `MusicTransformer.generate()` able to sample Stage B token IDs by using `sample_vocab_size=VOCAB_SIZE`
- decode generated Stage B tokens back to MIDI
- run existing MIDI metrics/gates on decoded output
- report invalid outputs honestly instead of treating MIDI file creation as success

First implementation target:

- `music_transformer/model/music_transformer.py`
- `scripts/run_stage_b_generation_probe.py`
- `scripts/agent_harness.sh stage-b-generation-probe`
- tests for Stage B primer, full-vocab sampling, and MIDI decode

Current result:

- one Brad file Stage B window prepare succeeded
- one-epoch tiny training succeeded
- generation sampled with `sample_vocab_size=547`
- decoded MIDI file was created
- generated sample failed the review gate with `generated MIDI has no notes`
- `passed_generation_gate=false`
- detail: `docs/STAGE_B_GENERATION_PROBE_2026-05-19.md`

## Active Issue #20

Status:

- completed and merged via PR #21

Completed task:

- add a grammar-constrained Stage B generation mode
- analyze generated tokens for complete `POSITION + VELOCITY + NOTE_PITCH + NOTE_DURATION` groups
- separate grammar-gate success from full musical review-gate success
- require at least one decoded MIDI note before expanding training scope

First implementation target:

- `scripts/run_stage_b_generation_probe.py`
- `scripts/agent_harness.sh stage-b-constrained-probe`
- `tests/test_stage_b_generation_probe.py`
- `docs/STAGE_B_CONSTRAINED_TINY_OVERFIT_2026-05-19.md`

Current result:

- constrained generation produced 8 complete Stage B note groups
- decoded MIDI note count: `8`
- grammar gate passed: `true`
- full review gate passed: `false`
- full gate failure reason: `too many simultaneous notes: 3 > 2`
- decision: Stage B note grammar can now be forced through the model logits, but musical validity still needs overlap/deduplication control

## Completed Issue #22

Status:

- completed and merged via PR #23

Completed task:

- remove duplicate same-onset/same-pitch notes after Stage B decode
- limit excessive overlapping notes before metrics
- keep decoded MIDI note count non-zero
- get constrained Stage B smoke through the full review gate

First implementation target:

- `scripts/run_stage_b_generation_probe.py`
- `scripts/agent_harness.sh stage-b-overlap-gate`
- `tests/test_stage_b_generation_probe.py`
- `docs/STAGE_B_OVERLAP_GATE_2026-05-19.md`

Current result:

- constrained generation still produced 8 complete Stage B note groups
- postprocess reduced notes from `8` to `6`
- max simultaneous notes reduced from `3` to `2`
- grammar gate passed: `true`
- full review gate passed: `true`
- detail: `docs/STAGE_B_OVERLAP_GATE_2026-05-19.md`

## Completed Issue #24

Status:

- completed and merged via PR #25

Completed task:

- strengthen the Stage B local generation probe from one sample to multiple samples
- keep the probe honest by reporting sample-level failures
- compare deterministic `top_k=1` collapse against `top_k=2`
- do not claim musical quality from a single passing MIDI file

First implementation target:

- `scripts/run_stage_b_generation_probe.py`
- `scripts/agent_harness.sh stage-b-stronger-probe`
- `tests/test_stage_b_generation_probe.py`
- `docs/STAGE_B_STRONGER_MULTISAMPLE_PROBE_2026-05-20.md`

Current result:

- `top_k=2`: `1/3` samples passed the full MIDI review gate
- all `3/3` samples passed the Stage B grammar gate
- `top_k=1` negative control collapsed to `0/3` valid samples
- detail: `docs/STAGE_B_STRONGER_MULTISAMPLE_PROBE_2026-05-20.md`

## Active Issue #29

Current task:

- add collapse diagnostics to Stage B generated token reports
- explain invalid samples with repeated position/pitch metrics
- add a sampling sweep over `top_k` settings
- compare `top_k=1` collapse against `top_k=2`

First implementation target:

- `scripts/run_stage_b_generation_probe.py`
- `scripts/run_stage_b_sampling_sweep.py`
- `scripts/agent_harness.sh stage-b-collapse-sweep`
- `tests/test_stage_b_generation_probe.py`
- `tests/test_stage_b_sampling_sweep.py`
- `docs/STAGE_B_COLLAPSE_SWEEP_2026-05-20.md`

Current result:

- `top_k=1`: valid `0/3`, collapse warning `3/3`, avg repeated position/pitch pair ratio `0.875`
- `top_k=2`: valid `1/3`, collapse warning `1/3`, avg repeated position/pitch pair ratio `0.292`
- best config: `top_k=2`, `temperature=0.9`
- decision: grammar is no longer the immediate bottleneck; note distribution collapse is
- detail: `docs/STAGE_B_COLLAPSE_SWEEP_2026-05-20.md`

## Dataset Strategy

현재 데이터셋은 Brad Mehldau-only fine-tuning보다 generic jazz pianist base 학습에 더 적합해 보인다.

파일 시스템 기준:

| Split | MIDI files |
|---|---:|
| physical MIDI paths under `midi_dataset` | 5554 |
| active audit tree: `midi_dataset/midi` | 2777 |
| duplicate mirror tree: `midi_dataset/midi_kong` | 2777 |
| active studio | 1994 |
| active live | 783 |
| Brad Mehldau studio | 18 |
| Brad Mehldau live | 54 |
| Brad Mehldau total | 72 |

Decision:

- 전체 dataset은 generic jazz piano prior 후보로 본다.
- Brad Mehldau subset은 style adaptation과 holdout evaluation에 사용한다.
- `midi_dataset/midi_kong`는 `midi_dataset/midi`의 duplicate mirror로 보고 active training tree에서 제외한다.
- 전체 dataset을 바로 train에 넣지 않고 audit 후 candidate manifest를 만든다.
- 자세한 기준은 `docs/DATASET_STRATEGY.md`를 따른다.

## Implemented Foundation

- `control_v1` token format
  - `ROLE_LEAD + TEMPO_* + BAR + conditioning + COND_SEP + target + END`
- role-conditioned dataset preparation
  - `conditioning.mid`
  - `target.mid`
  - tokenized train/val records
- control-aware crop for long training sequences
  - random crop이 `ROLE/TEMPO/BAR/COND_SEP` prompt를 날리지 않도록 수정됨
- full-checkpoint/from-scratch training entrypoint
- adapter training entrypoint
- tiny-overfit smoke harness
- model generation and MIDI validity metrics
- fallback/gate contract for invalid MIDI
- Brad Mehldau dataset audit script
- full jazz piano dataset audit script
- audit-based training manifest split builder
- manifest-based role dataset preparation smoke

## Full Jazz Piano Dataset Audit

Audit command:

```bash
python scripts/audit_jazz_piano_dataset.py
```

Fast smoke:

```bash
python scripts/audit_jazz_piano_dataset.py --max_files 100
```

Generated outputs:

```text
outputs/dataset_audit/jazz_piano_dataset_audit.json
outputs/dataset_audit/jazz_piano_dataset_audit.md
```

These outputs are not committed.

Current full audit result for `midi_dataset/midi`:

| Metric | Value |
|---|---:|
| files | 2777 |
| readable | 2777 |
| candidate | 2775 |
| candidate non-Brad | 2703 |
| candidate Brad | 72 |
| review too long | 1 |
| reject too few notes | 1 |
| exact duplicate hash groups | 0 |

## Brad Mehldau Dataset Audit

Audit command:

```bash
python scripts/audit_brad_mehldau_dataset.py
```

Current result:

| Metric | Value |
|---|---:|
| MIDI files | 18 |
| usable files | 18 |
| unusable files | 0 |
| max_sequence | 512 |
| files exceeding max_sequence | 18 |

Token stats:

| Metric | Min | P50 | P90 | Max | Mean |
|---|---:|---:|---:|---:|---:|
| `control_v1_token_count` | 1136 | 3241 | 5663 | 10653 | 3931.39 |
| `conditioning_token_count` | 468 | 1608 | 2843 | 4550 | 1937.22 |
| `target_token_count` | 419 | 1716 | 2894 | 6098 | 1989.17 |
| `note_count` | 266 | 756 | 1286 | 2636 | 942.33 |

Decision:

- Full-song sequences are too long for plain `max_sequence=512` training.
- Control-aware crop is required.
- If results stay musically invalid, build phrase-window data or duration-explicit tokenization.

## Next Execution Plan

### 0. Completed Issue #13 review point

Brad 2-file `control_v1` probe는 완료됐다.

Current conclusion:

- 더 많은 postprocess로 해결할 단계가 아니다.
- 현 `control_v1` full-song continuation은 solo-line generation representation으로 약하다.
- next issue는 Stage B tokenization spec/tests로 잡는다.

### 0.5. Issue #14 Stage B tokenizer contract

Status:

- completed and merged via PR #14

Goal:

- define `stage_b_v1`
- encode note events as explicit `POSITION + VELOCITY + NOTE_PITCH + NOTE_DURATION`
- include bar-level chord context with `CHORD_ROOT + CHORD_QUALITY`
- prove encode/decode roundtrip on tiny deterministic notes

Acceptance:

- Stage B token IDs start after existing Stage A control tokens
- chord symbols such as `Cm7`, `F7`, `Bbmaj7`, `F#m7b5` parse into stable tokens
- generated token sequence contains explicit duration tokens
- roundtrip preserves quantized pitch/start/end on a small example
- `bash scripts/agent_harness.sh quick` passes

### 0.6. Issue #15 Stage B role dataset preparation

Status:

- completed and merged via PR #15

Goal:

- let `prepare_role_dataset.py` accept `--sequence_format stage_b_v1`
- encode target MIDI notes with explicit `POSITION + VELOCITY + NOTE_PITCH + NOTE_DURATION`
- preserve manifest train/val boundaries
- avoid using `COND_SEP` or Stage A NOTE_ON/OFF tokens in Stage B records

Acceptance:

- unit test writes Stage B train/val `.npy` files from tiny MIDI manifests
- a local Brad 2-file dry run writes tokenized Stage B train/val records
- `bash scripts/agent_harness.sh quick` passes

Dry run result:

- output: `outputs/issue15_stage_b_probe2/roles_stage_b_probe2`
- train tokens: `4430`
- val tokens: `6482`

### 0.7. Issue #16 Stage B phrase-window dataset

Status:

- completed and merged via PR #16

Goal:

- split Stage B target MIDI into fixed-bar windows
- normalize note times to the window start
- keep windows only when they have enough target notes
- keep generated token records short enough for tiny-overfit probes

Acceptance:

- unit test writes multiple Stage B windows from one train/val MIDI pair
- local Brad 2-file dry run creates windowed tokenized train/val records
- `bash scripts/agent_harness.sh quick` passes

Dry run result:

- output: `outputs/issue16_stage_b_window_probe2/roles_stage_b_window_probe2`
- role samples: `137`
- tokenized train: `123`
- tokenized val: `14`
- token length p50: `77`
- token length max: `212`

### 0.8. Issue #17 Stage B window tiny-overfit smoke

Status:

- completed and merged via PR #17

Goal:

- prove Stage B phrase/window token records fit the Music Transformer vocabulary
- prepare a small Brad window dataset through the normal dataset entrypoint
- run a minimal full-model tiny training smoke against Stage B records
- avoid treating empty tokenized datasets as successful

Acceptance:

- `STAGE_B_VOCAB_SIZE == VOCAB_SIZE`
- local prepare-only smoke creates non-empty tokenized Stage B windows
- max token id is lower than model `VOCAB_SIZE`
- one-epoch training smoke exits successfully
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-window-prepare` passes

Smoke result:

- prepare-only output: `outputs/stage_b_window_tiny_overfit/harness_stage_b_window_prepare`
- training output: `outputs/stage_b_window_tiny_overfit/harness_stage_b_window_train_e1`
- role samples: `70`
- token length p50: `89`
- token length max: `212`
- max token id: `544`
- vocab size: `547`
- epoch 1 train loss: `6.1135`
- epoch 1 val loss: `5.8195`

### 0.9. Issue #18 Stage B decode/generation probe

Status:

- completed and merged via PR #19

Goal:

- make generation capable of emitting Stage B token IDs above `TOKEN_END`
- decode generated Stage B tokens into MIDI
- apply the same metrics gate used after the Stage A failure
- document whether the first Stage B generation smoke is musically valid

Acceptance:

- `MusicTransformer.generate(..., sample_vocab_size=VOCAB_SIZE)` can sample the full Stage B vocabulary
- generated Stage B tokens can be decoded into a MIDI file
- invalid output is reported as invalid, not as a successful sample
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-generation-probe` passes

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_generation_probe`
- role samples: `70`
- max token id in prepared records: `544`
- sample vocab size: `547`
- epoch 1 train loss: `6.2115`
- epoch 1 val loss: `5.9441`
- generated sample count: `1`
- valid sample count: `0`
- failure reason: `generated MIDI has no notes`
- decision: data/model/decode plumbing works, but generation quality is still not validated

### 0.10. Issue #20 Stage B grammar-constrained tiny-overfit

Status:

- completed and merged via PR #21

Goal:

- constrain generated token families into complete Stage B note groups
- verify decoded MIDI has real notes before broad training
- record grammar success separately from musical review success

Acceptance:

- grammar analyzer counts complete note groups
- constrained generation creates `POSITION + VELOCITY + NOTE_PITCH + NOTE_DURATION` groups
- decoded MIDI has non-zero notes
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-constrained-probe` passes

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_constrained_probe`
- role samples: `70`
- generated sample count: `1`
- complete note groups: `8`
- decoded note count: `8`
- grammar gate sample count: `1`
- valid sample count: `0`
- full gate failure reason: `too many simultaneous notes: 3 > 2`
- decision: next issue should reduce repeated same-position/same-pitch overlaps before broad training

### 0.11. Issue #22 Stage B overlap/dedup gate

Status:

- completed and merged via PR #23

Goal:

- remove duplicate notes at the same onset/pitch
- limit active overlapping notes to `max_simultaneous_notes <= 2`
- verify constrained Stage B decoded MIDI can pass the full review gate

Acceptance:

- overlap/dedup unit tests pass
- constrained smoke keeps decoded note count above zero
- constrained smoke reduces max simultaneous notes to `2`
- constrained smoke passes `validate_metrics`
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-overlap-gate` passes

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_overlap_gate`
- role samples: `70`
- complete note groups: `8`
- before note count: `8`
- after note count: `6`
- before max simultaneous notes: `3`
- after max simultaneous notes: `2`
- valid sample count: `1`
- passed generation gate: `true`
- decision: this is the first Stage B constrained smoke to pass the local review gate, but it is still a constrained/postprocessed diagnostic rather than unconstrained musical generation

### 0.12. Issue #24 Stage B stronger multi-sample probe

Status:

- completed and merged via PR #25

Goal:

- strengthen the single-sample Stage B overlap gate
- record sample-level seeds
- report grammar and full review pass rates
- require all samples to pass grammar gate
- require at least one sample to pass the full MIDI review gate

Acceptance:

- multi-sample summary unit tests pass
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-stronger-probe` passes
- report includes `sample_count`, `valid_sample_rate`, `grammar_gate_sample_rate`, and failure reason counts

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_stronger_probe`
- role samples: `70`
- epoch 3 val loss: `5.0104`
- generated samples: `3`
- grammar gate sample count: `3`
- valid sample count: `1`
- valid sample rate: `0.333`
- grammar gate sample rate: `1.000`
- passed grammar gate: `true`
- passed generation gate: `true`
- negative control: `top_k=1` collapsed to `0/3` valid samples with `note count too low: 2 < 6`
- detail: `docs/STAGE_B_STRONGER_MULTISAMPLE_PROBE_2026-05-20.md`

### 0.13. Issue #29 Stage B collapse diagnostics and sampling sweep

Status:

- completed and merged via PR #30

Goal:

- detect repeated position/pitch collapse before scaling training
- report collapse diagnostics per generated sample
- compare sampling configs by pass-rate, not by one hand-picked MIDI

Acceptance:

- collapse diagnostics unit tests pass
- sampling sweep summary tests pass
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-collapse-sweep` passes
- `report.json` includes collapse warnings and diagnostic failure reasons
- `sweep_report.json` and `sweep_report.md` compare `top_k=1` and `top_k=2`

Smoke result:

- output: `outputs/stage_b_sampling_sweep/harness_stage_b_collapse_sweep`
- `top_k=1`: valid `0/3`, collapse warning `3/3`, avg repeated position/pitch pair ratio `0.875`
- `top_k=2`: valid `1/3`, collapse warning `1/3`, avg repeated position/pitch pair ratio `0.292`
- best config: `top_k=2`, `temperature=0.9`
- decision: grammar is no longer the immediate bottleneck; note distribution collapse is
- detail: `docs/STAGE_B_COLLAPSE_SWEEP_2026-05-20.md`

### 0.14. Issue #31 Stage B stricter collapse-aware review gate

Status:

- completed and merged via PR #32

Goal:

- separate basic MIDI validity from stricter collapse-aware sample validity
- require minimum unique pitch, position, and position/pitch pair diversity
- cap collapse warning sample rate during sampling sweep
- prevent one-note, repeated same-position/pitch, or postprocess-heavy outputs from being counted as strong progress

Strict gate defaults:

- minimum unique pitches: `3`
- minimum unique positions: `3`
- minimum unique position/pitch pairs: `4`
- max repeated position/pitch pair ratio: `0.49`
- max postprocess removal ratio: `0.49`
- max collapse warning sample rate: `0.34`

Acceptance:

- strict collapse gate unit tests pass
- sampling sweep summary distinguishes basic and strict gate pass/fail
- `bash scripts/agent_harness.sh stage-b-collapse-sweep` passes
- `bash scripts/agent_harness.sh quick` passes

Smoke result:

- output: `outputs/stage_b_sampling_sweep/harness_stage_b_collapse_sweep`
- `top_k=1`: basic valid `0/3`, strict valid `0/3`, collapse warning `3/3`
- `top_k=2`: basic valid `1/3`, strict valid `1/3`, collapse warning `1/3`
- best config: `top_k=2`, `temperature=0.9`
- passed strict sweep gate: `true`
- decision: strict gate still preserves one valid candidate, so the next issue can move to a Stage B 2-file Brad probe
- detail: `docs/STAGE_B_STRICT_COLLAPSE_GATE_2026-05-20.md`

### 0.15. Issue #33 Stage B 2-file Brad generation probe

Status:

- completed and merged via PR #34

Goal:

- move from one-file tiny smoke to Brad 2-file Stage B generation probe
- verify whether grammar/basic/strict pass-rate survives a larger train/val setup
- decide whether to move toward generic jazz base training or fix another local generation bottleneck

Probe setup:

- input: `./midi_dataset/midi/studio/Brad Mehldau`
- max files: `2`
- generated Stage B windows: `137`
- train samples: `123`
- val samples: `14`
- max token id: `544`
- vocab size: `547`
- training: 3 epochs, full tiny model path, CPU
- best observed val loss: `4.0892`
- generation: constrained, `top_k=2`, temperature `0.9`, 3 samples

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_2file_brad_probe`
- grammar gate: `3/3`
- basic valid: `0/3`
- strict valid: `0/3`
- collapse warning: `0/3`
- avg repeated position/pitch pair ratio: `0.375`
- avg postprocess removal ratio: `0.208`
- avg onset coverage ratio: `0.167`
- avg sustained coverage ratio: `0.417`
- avg position span ratio: `0.740`
- max longest sustained empty run: `11` steps
- failure reason: all samples failed on dead-air ratio near or above `0.800`

Decision:

- Stage B grammar is now stable in this probe.
- The prior collapse issue is not the immediate failure mode in the 2-file probe.
- The current bottleneck is temporal coverage/dead-air, especially sparse onsets and long empty spans inside the 2-bar phrase.
- Do not start generic jazz base training yet.
- Next issue should add position/dead-air coverage diagnostics and a coverage-aware constrained generation probe.

Detail:

- `docs/STAGE_B_2FILE_BRAD_PROBE_2026-05-20.md`

### 0.16. Issue #35 Stage B temporal coverage diagnostics

Status:

- completed and merged via PR #36

Goal:

- explain why the 2-file Brad probe fails dead-air despite passing grammar and collapse checks
- add token-level temporal coverage diagnostics to each sample report
- aggregate coverage summary fields across samples

Implemented diagnostics:

- unique onset position count
- onset coverage ratio
- sustained coverage ratio
- earliest/latest absolute position
- position span ratio
- head/tail empty steps
- longest onset empty run
- longest sustained empty run
- per-bar unique onset positions
- per-bar onset coverage ratio

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_2file_brad_probe`
- grammar gate: `3/3`
- basic valid: `0/3`
- strict valid: `0/3`
- avg onset coverage ratio: `0.167`
- avg sustained coverage ratio: `0.417`
- avg position span ratio: `0.740`
- max longest sustained empty run: `11` steps

Decision:

- dead-air failure is explained by sparse onset coverage and long empty spans.
- MIDI phrase coverage alone is not enough because notes can span much of the phrase while onsets remain sparse.
- Next issue should test coverage-aware constrained generation, not broad training.

Detail:

- `docs/STAGE_B_TEMPORAL_COVERAGE_DIAGNOSTICS_2026-05-20.md`

### 0.17. Issue #37 Stage B coverage-aware constrained generation

Status:

- implemented on `issue-37-stage-b-coverage-aware-generation`

Goal:

- reduce the Stage B 2-file Brad dead-air failure without broad training
- make only constrained `POSITION` selection coverage-aware
- keep pitch, duration, and velocity sampled from model logits
- compare the result against #35 temporal coverage diagnostics

Implementation:

- added coverage-aware position helper for constrained note groups
- added `--coverage_aware_positions`
- added `--coverage_position_window`
- added harness mode `stage-b-coverage-aware-probe`

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_coverage_aware_probe`
- grammar gate: `3/3`
- basic valid: `3/3`
- strict valid: `3/3`
- collapse warning: `0/3`
- avg onset coverage ratio: `0.250`
- avg sustained coverage ratio: `0.427`
- avg position span ratio: `0.813`
- max longest sustained empty run: `6` steps
- avg postprocess removal ratio: `0.000`

Decision:

- coverage-aware constrained `POSITION` selection is a useful local generation constraint.
- It fixes the current 2-file Brad probe's dead-air gate failure under this harness.
- It still does not prove unconstrained generation or personalized Brad style.
- Next issue should run an A/B sweep: plain constrained vs coverage-aware constrained, with `note_groups_per_bar` variants.

Detail:

- `docs/STAGE_B_COVERAGE_AWARE_GENERATION_2026-05-20.md`

### 0.18. Issue #39 Stage B coverage-aware A/B sweep

Status:

- implemented on `issue-39-stage-b-coverage-ab-sweep`

Goal:

- compare plain constrained generation with coverage-aware constrained generation
- test note group density values `4`, `6`, and `8`
- record pass-rate and temporal coverage tradeoffs in JSON/Markdown reports

Smoke result:

- output: `outputs/stage_b_coverage_ab_sweep/harness_stage_b_coverage_ab_sweep`
- configs: `6`
- all configs grammar gate: `3/3`
- plain strict valid: `0/3`, `1/3`, `2/3` for groups/bar `4`, `6`, `8`
- coverage strict valid: `3/3`, `3/3`, `3/3` for groups/bar `4`, `6`, `8`
- best config: coverage groups/bar `8`
- best avg onset coverage ratio: `0.500`
- best avg sustained coverage ratio: `0.865`
- best max longest sustained empty run: `1` step

Decision:

- coverage-aware `POSITION` selection is consistently better than plain constrained generation for this 2-file Brad setup.
- More note groups improve temporal coverage, but lower chord-tone ratio can become the next musical-quality issue.
- Next issue should rank candidate samples/configs with multiple metrics rather than only strict pass/fail.

Detail:

- `docs/STAGE_B_COVERAGE_AB_SWEEP_2026-05-20.md`

### 0.19. Issue #41 Stage B candidate ranking report

Status:

- implemented on `issue-41-stage-b-candidate-ranking`

Goal:

- rank generated MIDI candidates from A/B sweep reports
- include strict validity, temporal coverage, dead-air, chord-tone ratio, repetition, pitch diversity, and collapse warning in the score
- produce JSON/Markdown reports for listening/review priority

Smoke result:

- output: `outputs/stage_b_candidate_ranking/harness_stage_b_candidate_ranking`
- top candidate: coverage groups/bar `8`, sample `1`
- score: `91.080`
- strict valid: `true`
- note count: `16`
- onset coverage ratio: `0.500`
- sustained coverage ratio: `0.906`
- dead-air ratio: `0.467`
- chord-tone ratio: `0.313`

Decision:

- Use this report to choose which generated MIDI files to inspect by ear and piano roll.
- The score is a review-prioritization heuristic, not a musical-quality claim.
- Next issue depends on listening review:
  - if rhythm/shape is acceptable but harmony is weak, add chord-aware pitch filtering/ranking
  - if the line is still mechanically patterned, revisit generation constraints before broad training

Detail:

- `docs/STAGE_B_CANDIDATE_RANKING_2026-05-20.md`

### 0.20. Issue #43 Stage B ranking harmonic/repetition gate

Status:

- implemented on `issue-43-stage-b-ranking-harmonic-gate`

Goal:

- stop ranking from promoting MIDI that only looks good by temporal coverage
- read each candidate MIDI directly before scoring
- flag low chord-tone, repeated pitch, and repeated bar-template failures

Review trigger:

- Issue #41 ranked coverage groups/bar `8`, sample `1` as top candidate.
- Piano-roll review showed it was not a usable solo-line candidate.
- The sample had too little pitch variety and too much repeated mechanical structure.

Latest harness result:

- candidate count: `18`
- valid candidates: `12`
- strict candidates: `12`
- viable candidates without review flags: `0`
- flagged candidates: `18`

Decision:

- Do not treat any current Stage B candidate as a good listening sample.
- The ranking report is now honest: strict-valid MIDI can still be musically invalid.
- Next issue should fix generation-side pitch/harmony behavior, not just ranking.

Detail:

- `docs/STAGE_B_RANKING_HARMONIC_GATE_2026-05-21.md`

### 0.21. Issue #45 Stage B chord-aware pitch constrained generation

Status:

- implemented on `issue-45-stage-b-chord-aware-pitch`

Goal:

- fix the generation-side pitch/harmony failure revealed by Issue #43
- constrain `NOTE_PITCH` candidates by current bar chord
- preserve coverage-aware `POSITION` generation
- reduce repeated pitch and low bar-level chord-tone failures

Latest harness result:

- command: `bash scripts/agent_harness.sh stage-b-chord-aware-probe`
- candidate count: `27`
- strict candidates: `21`
- viable candidates without review flags: `9`
- flagged candidates: `18`
- best mode: `coverage_chord`
- best candidate score: `96.6964`
- best candidate reviewable: `true`

Best candidate:

- mode: `coverage_chord`
- groups/bar: `4`
- sample index: `2`
- note count: `8`
- unique pitch count: `6`
- chord-tone ratio: `0.750`
- bar chord-tone ratio: `0.875`
- min bar chord-tone ratio: `0.800`
- dominant pitch ratio: `0.375`
- repeated pitch ratio: `0.250`
- MIDI path: `outputs/stage_b_coverage_ab_sweep/harness_stage_b_chord_aware_probe_ab_sweep_coverage_chord_g4_k2_t0p9/samples/stage_b_sample_2.mid`

Decision:

- This is the first Stage B probe where candidate ranking finds unflagged reviewable MIDI candidates.
- Do not call this a personalized jazz model yet.
- Next step is manual listening/piano-roll review of top `coverage_chord` candidates.

Detail:

- `docs/STAGE_B_CHORD_AWARE_PITCH_2026-05-21.md`

### 0.22. Issue #47 Stage B coverage_chord candidate review package

Status:

- implemented on `issue-47-stage-b-review-package`

Goal:

- export top `coverage_chord` ranked candidates for manual listening review
- keep generated MIDI artifacts out of git
- make the next decision about broad training based on actual listening/piano-roll review

Output:

- `outputs/stage_b_review_candidates/harness_stage_b_chord_aware_probe/review_manifest.json`
- `outputs/stage_b_review_candidates/harness_stage_b_chord_aware_probe/review_candidates.md`
- copied MIDI files under `outputs/stage_b_review_candidates/harness_stage_b_chord_aware_probe/midi/`

Selected candidates:

- `rank_01_coverage_chord_g4_s2.mid`
- `rank_02_coverage_chord_g6_s1.mid`
- `rank_03_coverage_chord_g8_s1.mid`
- `rank_04_coverage_chord_g6_s2.mid`
- `rank_05_coverage_chord_g8_s3.mid`
- `rank_06_coverage_chord_g8_s2.mid`

Decision:

- This is now a manual review boundary.
- The next technical issue should wait until these MIDI files are heard or inspected in piano roll.

Detail:

- `docs/STAGE_B_CANDIDATE_REVIEW_EXPORT_2026-05-21.md`

### 0.23. Issue #49 Stage B longer coverage_chord phrase probe

Status:

- implemented on `issue-49-stage-b-longer-phrase-probe`

Review trigger:

- Issue #47 review candidates were valid enough to inspect, but too short.
- Piano-roll review showed fragments that may be melodic, but felt unfinished as phrases.
- The next probe should not ask whether a MIDI file exists; it should ask whether the candidate has enough phrase length to review.

Goal:

- run a `4` bar coverage+chord-aware constrained probe
- increase generated note groups from `8-16` notes to `32` note groups per sample
- export the generated samples directly from the generation probe report
- keep the claim limited to "longer review candidates", not "finished jazz solo model"

Probe setup:

- max files: `2`
- window bars: `4`
- window stride bars: `2`
- minimum window target notes: `8`
- generation bars: `4`
- constrained note groups per bar: `8`
- pitch mode: chord tones
- position mode: coverage-aware
- samples: `3`

Latest local result:

- generated samples: `3`
- strict valid samples: `3`
- grammar valid samples: `3`
- note groups per sample: `32`
- average onset coverage ratio: around `0.500`
- average sustained coverage ratio: around `0.680`
- max longest sustained empty run: `2` steps

Decision:

- This directly addresses the "too short / unfinished word" failure mode.
- The next manual review should open the exported 4-bar MIDI candidates, not the previous 2-bar package.
- If these still feel like fragments, the next issue should move to phrase/motif-level structure rather than simply adding more notes.

Detail:

- `docs/STAGE_B_LONGER_PHRASE_PROBE_2026-05-21.md`

### 0.24. Issue #51 Stage B phrase contour/repeated-pitch diagnostics

Status:

- implemented on `issue-51-stage-b-phrase-contour-diagnostics`

Review trigger:

- Issue #49 fixed the phrase length problem structurally.
- However, exported 4-bar candidates still had repeated pitch ratio around `0.719`.
- That number alone did not say whether the sample was adjacent same-note collapse, motif reuse, or chord-tone set reuse.

Goal:

- add phrase contour diagnostics to each Stage B generated sample
- expose repeated-pitch risk in review export without dropping the candidate
- make the manual review question more precise

Implemented diagnostics:

- adjacent repeated pitch ratio
- direction change ratio
- longest same pitch run
- unique interval count
- stepwise/leap motion ratio
- contour warning reasons
- review export `risk_flags`

Latest local result:

- repeated pitch ratio: around `0.719`
- adjacent repeated pitch ratio: `0.000`
- average direction change ratio: around `0.689`
- max longest same pitch run: `1`
- risk flags: `high_repeated_pitch_ratio`, plus `high_dominant_pitch_ratio` for one sample

Decision:

- The current 4-bar candidates are not adjacent same-note collapse.
- They still reuse a limited pitch set heavily.
- The next listening review should judge whether that reuse sounds like motif/inside playing or like constrained mechanical pitch cycling.

Detail:

- `docs/STAGE_B_PHRASE_CONTOUR_DIAGNOSTICS_2026-05-21.md`

### 0.25. Issue #53 Stage B root bias diagnostics

Status:

- implemented on `issue-53-stage-b-root-bias-diagnostics`

Review trigger:

- Manual review said the 4-bar candidates are melody-like.
- The main concern was that it feels like it keeps hitting root notes.
- Before changing generation, this needs to be measured.

Goal:

- add pitch-role diagnostics to generated sample reports
- expose root tone ratio, non-root chord tone ratio, and tension ratio in review export
- decide whether the perceived root bias is actual root overuse or broader chord-tone-only behavior

Latest local result:

- average root tone ratio: around `0.271`
- top review candidate root tone ratio: around `0.219`
- chord tone ratio: around `0.938-1.000`
- tension ratio: `0.000`
- adjacent repeated pitch ratio: `0.000`
- `tones_tensions` comparison result:
  - root tone ratio: around `0.135`
  - tension ratio: around `0.313`
  - strict valid samples: `3/3`

Decision:

- The current issue is not pure root collapse.
- The stronger diagnosis is "safe chord-tone-only line with no tensions."
- `chord_pitch_mode=tones_tensions` reduces root/no-tension stiffness, but repeated pitch-set behavior remains.
- Next generation comparison should test passing/approach pitch or contour/motif constraints against the current `tones_tensions` baseline.

Detail:

- `docs/STAGE_B_ROOT_BIAS_DIAGNOSTICS_2026-05-21.md`
- `docs/STAGE_B_PITCH_MODE_COMPARE_2026-05-21.md`

### 1. Run full jazz piano dataset audit

```bash
python scripts/audit_jazz_piano_dataset.py
```

Expected result:

- readable/unreadable counts
- candidate/review/reject counts
- artist/source distribution
- Brad/non-Brad candidate counts
- duplicate exact hash groups
- duration/note-count/piano-program/sustain stats

Status:

- completed for `midi_dataset/midi`
- generated outputs are under `outputs/dataset_audit/`

### 2. Prepare 2-file control_v1 probe dataset

Before broad training, build concrete candidate splits:

```bash
python scripts/build_jazz_training_manifests.py
```

This produces generic non-Brad train/val manifests plus Brad adaptation/holdout manifests under `data/manifests/`.

Then prepare a tokenized generic split without reshuffling train/val:

```bash
python scripts/prepare_role_dataset.py \
  --train_manifest ./data/manifests/generic_jazz_train.txt \
  --val_manifest ./data/manifests/generic_jazz_val.txt \
  --output_dir ./data/roles_generic_jazz \
  --role lead \
  --sequence_format control_v1 \
  --overwrite
```

For a small local contract check before broad training:

```bash
bash scripts/agent_harness.sh manifest-dry-run
```

Current smoke result:

- `audit_max_files`: 100
- generated generic manifest split: train 57, val 10
- smoke prepare subset: train 4, val 2
- tokenized output: train 4, val 2

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir ./data/roles_probe2 \
  --role lead \
  --sequence_format control_v1 \
  --max_files 2 \
  --overwrite
```

Status:

- completed in issue #13 under `outputs/issue13_control_v1_brad_probe2/roles_probe2`

### 3. Train 2-file control_v1 probe

```bash
python scripts/train_stage_a_full.py \
  --data_dir ./data/roles_probe2/lead/tokenized \
  --output_dir ./checkpoints/brad_mehldau_control_v1_probe2 \
  --epochs 1 \
  --batch_size 4 \
  --num_workers 0 \
  --max_sequence 512
```

Status:

- completed in issue #13
- e5 and e100 checkpoints generated locally under `outputs/issue13_control_v1_brad_probe2/`
- generated artifacts are not committed

### 4. Generate and inspect samples

Use the trained checkpoint with `scripts/generate.py` or the inference wrapper.

Status:

- completed in issue #13
- all generated samples failed the review gate

The sample is not valid unless it passes:

- non-zero note count
- enough unique pitches
- phrase coverage
- max note duration ratio
- max simultaneous notes
- no one-note/two-note output
- no long sustain block
- no chord block pretending to be a solo line

### 5. Review point

Review completed after the 2-file probe generated MIDI.

Decision:

- do not continue to `max_files=5` on current `control_v1`
- do not run full 18-file Brad probe on current `control_v1`
- do not start broad generic non-Brad training yet
- move to duration-explicit tokenization
- create phrase/window dataset before the next training run

## Active References

- `docs/BRAD_MEHLDAU_FINETUNING_PLAN.md`
- `docs/DATASET_STRATEGY.md`
- `docs/STAGE_A_TOKEN_FORMAT.md`
- `docs/STAGE_A_TRAINING_MODES.md`
- `docs/STAGE_A_TINY_OVERFIT.md`
- `docs/STAGE_A_CODE_REVIEW_2026-05-18.md`
- `docs/STAGE_B_TOKENIZATION_SPEC.md`
- `docs/STAGE_B_ROLE_DATASET_PREP_2026-05-19.md`
- `docs/STAGE_B_PHRASE_WINDOW_DATASET_2026-05-19.md`
- `docs/STAGE_B_WINDOW_TINY_OVERFIT_2026-05-19.md`
- `docs/STAGE_B_GENERATION_PROBE_2026-05-19.md`
- `docs/STAGE_B_CONSTRAINED_TINY_OVERFIT_2026-05-19.md`
- `docs/STAGE_B_OVERLAP_GATE_2026-05-19.md`
- `docs/STAGE_B_STRONGER_MULTISAMPLE_PROBE_2026-05-20.md`
- `docs/STAGE_B_COLLAPSE_SWEEP_2026-05-20.md`
- `docs/STAGE_B_STRICT_COLLAPSE_GATE_2026-05-20.md`
- `docs/STAGE_B_2FILE_BRAD_PROBE_2026-05-20.md`
- `docs/STAGE_B_TEMPORAL_COVERAGE_DIAGNOSTICS_2026-05-20.md`
- `docs/STAGE_B_COVERAGE_AWARE_GENERATION_2026-05-20.md`
- `docs/STAGE_B_COVERAGE_AB_SWEEP_2026-05-20.md`
- `docs/STAGE_B_CANDIDATE_RANKING_2026-05-20.md`
- `docs/STAGE_B_RANKING_HARMONIC_GATE_2026-05-21.md`
- `docs/STAGE_B_CHORD_AWARE_PITCH_2026-05-21.md`
- `docs/STAGE_B_CANDIDATE_REVIEW_EXPORT_2026-05-21.md`
- `docs/STAGE_B_LONGER_PHRASE_PROBE_2026-05-21.md`
- `docs/STAGE_B_PHRASE_CONTOUR_DIAGNOSTICS_2026-05-21.md`
- `docs/STAGE_B_ROOT_BIAS_DIAGNOSTICS_2026-05-21.md`
- `docs/REFERENCES.md`
- `docs/INFERENCE_MODEL_SPEC.md`
- `docs/QA_ACCEPTANCE_PLAN.md`

## Validation

Before committing code or docs:

```bash
bash scripts/agent_harness.sh quick
```

For generation, inference, metrics, or model-loading changes:

```bash
bash scripts/agent_harness.sh demo
```

For training-mode or tiny-overfit changes:

```bash
bash scripts/agent_harness.sh tiny-compare
```

For Stage B window dataset/model-vocab changes:

```bash
bash scripts/agent_harness.sh stage-b-window-prepare
```

For Stage B decode/generation changes:

```bash
bash scripts/agent_harness.sh stage-b-generation-probe
```

For Stage B constrained note-grammar changes:

```bash
bash scripts/agent_harness.sh stage-b-constrained-probe
```

For Stage B overlap/dedup gate changes:

```bash
bash scripts/agent_harness.sh stage-b-overlap-gate
```

For Stage B multi-sample review-gate changes:

```bash
bash scripts/agent_harness.sh stage-b-stronger-probe
```

For Stage B collapse/sampling-sweep changes:

```bash
bash scripts/agent_harness.sh stage-b-collapse-sweep
```

For Stage B 2-file Brad generation changes:

```bash
bash scripts/agent_harness.sh stage-b-2file-brad-probe
```

For Stage B coverage-aware constrained generation changes:

```bash
bash scripts/agent_harness.sh stage-b-coverage-aware-probe
```

For Stage B coverage-aware A/B sweep changes:

```bash
bash scripts/agent_harness.sh stage-b-coverage-ab-sweep
```

For Stage B candidate ranking changes:

```bash
bash scripts/agent_harness.sh stage-b-candidate-ranking
```

For Stage B longer phrase review candidates:

```bash
bash scripts/agent_harness.sh stage-b-longer-phrase-probe
```
