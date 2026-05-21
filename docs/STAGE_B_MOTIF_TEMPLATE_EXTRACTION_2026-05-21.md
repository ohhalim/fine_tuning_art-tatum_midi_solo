# Stage B Motif Template Extraction

작성일: 2026-05-21

## 목적

Issue #61에서 실제 jazz MIDI phrase window 통계를 만든 결과, hand-written `swing_motif_approach`는 syncopation은 reference에 가까워졌지만 bar-position variation, duration diversity, IOI diversity가 여전히 부족했다.

이번 작업의 목적은 hand-written rhythm rule을 더 늘리는 것이 아니다.

목표:

- 실제 Stage B phrase window에서 짧은 note-group motif를 추출한다.
- rhythm template, contour template, full motif template을 분리해서 집계한다.
- 다음 generation probe가 data-derived phrase material을 사용할 수 있게 만든다.
- chord-block 또는 same-onset voicing이 solo-line motif로 섞이지 않도록 기본 필터를 둔다.

## 구현

추가 파일:

- `scripts/run_stage_b_motif_template_extraction.py`
- `tests/test_stage_b_motif_template_extraction.py`

하네스:

```bash
bash scripts/agent_harness.sh stage-b-motif-templates
```

출력:

- `outputs/stage_b_motif_templates/<run_id>/motif_template_report.json`
- `outputs/stage_b_motif_templates/<run_id>/motif_template_report.md`

추출 단위:

- Stage B tokenized record의 decoded note groups
- 기본 motif length: `4`
- 기본 max bar span: `2`
- 기본 same-onset filter: enabled

## Same-Onset Filter

초기 extraction에서는 같은 position에 찍힌 note들이 top motif로 올라올 수 있었다.

이것은 piano voicing 또는 chord block일 수 있고, 지금 목표인 solo-line phrase material과 다르다.

따라서 기본 동작은 onset이 strictly increasing인 motif만 남긴다.

진단 목적으로 chord/block motif까지 보고 싶을 때만 다음 옵션을 쓴다.

```bash
python scripts/run_stage_b_motif_template_extraction.py --allow_same_onset_motifs
```

## Local Result

실행 조건:

```bash
bash scripts/agent_harness.sh stage-b-motif-templates
```

결과:

- source records: `56`
- motif count: `803`
- unique rhythm templates: `520`
- unique contour templates: `328`
- unique full templates: `526`
- top rhythm support: `0.009`
- top contour support: `0.012`
- top full support: `0.002`

Top rhythm templates:

| rank | count | support | template |
|---:|---:|---:|---|
| 1 | 7 | 0.009 | duration `[2, 2, 2, 2]`, position deltas `[0, 1, 2, 3]` |
| 2 | 6 | 0.007 | duration `[1, 1, 1, 1]`, position deltas `[0, 1, 2, 3]` |
| 3 | 2 | 0.002 | duration `[5, 6, 8, 3]`, position deltas `[0, 2, 3, 5]` |

Top contour templates:

| rank | count | support | template |
|---:|---:|---:|---|
| 1 | 10 | 0.012 | pitch intervals `[0, 0, -2, 0]`, melodic intervals `[0, -2, 2]` |
| 2 | 9 | 0.011 | pitch intervals `[0, 0, 0, -9]`, melodic intervals `[0, 0, -9]` |
| 3 | 9 | 0.011 | pitch intervals `[0, -2, 1, 0]`, melodic intervals `[-2, 3, -1]` |

## 해석

이 결과는 "가장 많이 나온 motif 하나를 가져다 쓰면 된다"는 뜻이 아니다.

오히려 반대다.

- top rhythm template support가 `0.009`라서 dominant rhythm template이 없다.
- top full motif support가 `0.002`라서 full motif 단위 복붙은 위험하다.
- rhythm, contour, pitch interval을 분리해서 sampling하거나 clustering해야 한다.
- next generator는 one best motif가 아니라 reference distribution에서 후보를 뽑아야 한다.

따라서 이번 작업의 가치는 생성 품질을 바로 올리는 데 있지 않다.

가치는 다음과 같다.

- "재즈스럽게 만들어라"를 수치화 가능한 motif catalog로 바꿨다.
- hand-written beginner-like pattern에서 data-derived phrase material로 넘어갈 준비가 됐다.
- chord-block/same-onset motif가 solo-line 후보를 오염시키는 문제를 기본 필터로 막았다.

## 다음 작업

다음 issue는 data-derived motif catalog를 generation-side constraint로 연결하는 것이다.

구체적으로:

- extracted rhythm template에서 position/duration 후보를 sampling한다.
- extracted contour template에서 interval movement 후보를 sampling한다.
- current chord/tension pitch set 위에 contour를 transpose한다.
- generated candidate의 rhythm/duration/IOI diversity가 real reference p25-p75 범위에 가까워지는지 본다.
- MIDI review export에서 hand-written `swing_motif_approach`와 data-derived motif baseline을 비교한다.

성공 기준:

- one-note/two-note/chord-block/long-sustain output이 아니다.
- 8-bar phrase가 너무 짧거나 단어처럼 끊기지 않는다.
- `swing_motif_approach`보다 duration diversity와 IOI diversity가 오른다.
- piano roll에서 "코드톤 나열"보다 phrase contour가 더 분명하다.

## Validation

```bash
./.venv/bin/python -m unittest tests.test_stage_b_motif_template_extraction
./.venv/bin/python -m compileall scripts/run_stage_b_motif_template_extraction.py tests/test_stage_b_motif_template_extraction.py
bash scripts/agent_harness.sh stage-b-motif-templates
```
