# Stage B Clean Listening Review Notes (Issue #113)

작성일: 2026-05-23

## 목표

Issue #109 clean review package와 Issue #111 clean context diagnostics 결과를 바탕으로,
objective-clean 후보 3개를 같은 schema에서 subjective listening review 할 수 있는 템플릿을 만든다.

리뷰 기준은 다음 5개다.

- timing
- chord_fit
- phrase_continuation
- landing
- jazz_vocabulary

## 구현

- `scripts/build_clean_listening_review_notes.py`
  - clean package + clean context diagnostics를 입력으로 review notes 템플릿 생성
  - 후보별 context/chord guide 경로와 핵심 objective metric 연결
  - enum 검증 및 summary 출력
- `tests/test_clean_listening_review_notes.py`
  - 템플릿 기본값/검증/enum 실패 케이스 테스트
- `scripts/agent_harness.sh stage-b-clean-listening-review-notes`
  - clean package, diagnostics가 없으면 선행 harness 자동 실행 후 notes 생성

## 실행

```bash
bash scripts/agent_harness.sh stage-b-clean-listening-review-notes
```

## 출력

- `outputs/stage_b_clean_listening_review_notes/harness_stage_b_clean_listening_review_notes/clean_listening_review_notes_template.json`
- `outputs/stage_b_clean_listening_review_notes/harness_stage_b_clean_listening_review_notes/clean_listening_review_notes_summary.json`

## 해석

이 단계는 generation rule을 바꾸지 않는다.

목적은 objective-clean 3개를 실제 청취 관점으로 같은 기준에서 비교할 수 있게 만드는 것이다.
