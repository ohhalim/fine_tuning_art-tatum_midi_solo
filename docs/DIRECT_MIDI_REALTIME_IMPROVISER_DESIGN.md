# Direct MIDI Realtime Improviser Research Design

- 작성일: 2026-09-03
- 상태: 선행연구 반영 설계 초안
- 관련 이슈: [#1472 직접 MIDI 실시간 즉흥연주 런타임 설계](https://github.com/ohhalim/fine_tuning_art-tatum_midi_solo/issues/1472)
- 선행연구 조사: [`RESEARCH.md`](../exa-results/music-ml-improvisation-landscape-2026-09-03/RESEARCH.md)

## 1. Decision

실시간 연주 데이터 경로에서 MCP와 대화형 LLM 호출을 제외한다.

생성 표현은 symbolic MIDI를 우선한다. 이 선택은 audio generation보다 음악성이
우수하다는 주장이 아니라, 현재 목표인 note 단위 편집, 임의 VST 연결, 화성 제약,
연주자 adapter 비교, deadline 측정에 필요한 시스템 경계다.

```text
MIDI keyboard / FL Studio MIDI Out
  -> virtual MIDI input
  -> resident improviser runtime
  -> beat-aligned lookahead queue
  -> virtual MIDI output
  -> FL Studio instrument channel
  -> Serum / piano VST / other instrument
```

MCP는 필수 구성요소가 아니다. 필요한 경우에도 다음 제어 작업으로 제한한다.

- session start / stop
- style adapter 선택
- density / energy / response mode 변경
- WAV file analysis 요청
- runtime status / metric 조회

동일 제어는 CLI, configuration file, MIDI CC로 대체 가능하다.

WAV는 첫 단계의 생성 표현이 아니라 문맥 입력이다. 전체 곡에서 tempo, beat, chord,
section, bars-to-drop, energy trajectory를 추출해 symbolic generator에 공급한다.

## 2. Goal

제품 목표와 연구 목표를 분리한다.

제품 목표:

최종 동작:

1. 사용자가 MIDI keyboard 또는 FL Studio piano roll에서 chord, gesture, short phrase 입력
2. runtime이 최근 MIDI context와 현재 musical clock 수집
3. 선택된 pianist adapter와 harmonic context를 기준으로 다음 1마디 또는 2마디 lead phrase 생성
4. 예정된 beat에 일반 MIDI note event 출력
5. FL Studio의 Serum 또는 다른 VST가 MIDI를 소리로 변환
6. FL Studio record 기능으로 생성 MIDI 보존 가능

연구 목표:

> 동의를 받은 여러 연주자의 소량 MIDI에서 교체 가능한 연주자 adapter를 학습하고,
> DJ 곡의 bars-to-drop 문맥으로 phrase를 제어하며, adapter 소유권과 학습 phrase
> 복제 위험까지 검증할 수 있는지 확인한다.

이 문장은 신규성 확정이 아니라 검증 대상이다. 실시간 개인화 즉흥연주 자체는
Continuator, OMax, Somax2, jam_bot이 선행했다. symbolic editability, deadline 형식화,
style classifier도 선행 방법을 재사용하는 시스템 요건이다. 현재 연구 기여 후보는
다중 adapter, DJ bars-to-drop 조건, consent/ownership의 교집합이다.

WAV 입력 목표:

1. house / dance track의 BPM, beat grid, chord timeline, section, bars-to-drop, energy 사전 추출
2. 추출 context를 동일한 improviser runtime에 공급
3. drop 또는 transition 이전에 solo block 선생성
4. MIDI 출력만 생성하고 음색 합성은 FL Studio에 위임

## 3. Non-goals

- raw audio generation
- FL Studio 내부 audio engine 재구현
- 대화형 LLM을 통한 note-by-note 생성
- MCP request를 통한 실시간 MIDI event transport
- 첫 단계의 live WAV stream analysis
- 첫 단계의 polyphonic two-hand piano generation
- objective metric만으로 특정 pianist style 또는 음악성 입증
- exact artist clone 표현
- 최초의 AI 즉흥연주, 최초의 실시간 jam, 최초의 개인 스타일 학습 주장
- Music Transformer, jam_bot, LMDM과의 모델 규모 경쟁
- audio 모델과 symbolic 모델의 단일 품질 점수 비교
- 고인이 된 연주자의 이름·스타일을 동의 기반 제품 기능으로 간주

## 4. Current Evidence

### 4.1 Model core

- file-based MIDI-to-solo generation 및 review gate 존재
- input: `conditioning_midi` 또는 chord progression / BPM / bars / time signature
- output: MIDI path, metrics, fallback state
- CLI generation의 grammar mask와 candidate gate 존재
- model loading 재사용 경로 존재

상주 경로의 현재 차이:

- `StageAModelRunner.generate_candidates()`는 `generate_once()`에 `grammar_mask`를 전달하지 않음
- `generate_once()` 기본값은 `grammar_mask=False`
- 따라서 기존 in-process 측정을 D2 grammar 표준 적용 결과로 간주 금지
- resident runtime 연결 전에 grammar mask 전달 계약과 회귀 테스트 필요

근거:

- `docs/INFERENCE_MODEL_SPEC.md`
- `scripts/generate.py`
- `music_transformer/model/music_transformer.py`

### 4.2 Measured generation time

| 조건 | 평균 generation time | 판정 |
|---|---:|---|
| 512 token, model candidates 2 | 약 `31.3s/request` | realtime 부적합 |
| 256 token, model candidates 2 | 약 `9.4s-9.8s/request` | realtime 부적합 |

근거: `docs/archive/REVIEW_2026-05-16.md`

현재 generator는 token 생성마다 전체 prefix에 `forward`를 다시 수행한다.

```text
cur_i 증가
  -> forward(gen_seq[..., :cur_i])
  -> next token sampling
  -> 반복
```

잔여 병목 후보:

- KV cache 없음
- live request당 2개 candidate 생성
- block 길이와 token 수 불일치
- offline file decode / write 계약
- scheduling worker 없음
- 생성 종료가 target beat가 아니라 token 수 또는 `TOKEN_END`에 의존

### 4.3 Existing FL Studio bridge

확인된 기능:

- `mido` input / output port open
- external MIDI note callback
- generated note list 전송
- FL Studio controller script의 `channels.midiNoteOn` 사용

재사용 제외:

- special start / count / end note protocol
- event 사이 반복 `10ms` sleep
- note list를 CC와 note packet으로 직렬화하는 방식
- chord note 변경마다 동기 generation 호출
- runtime data path와 preference/RAG 책임 결합

현재 protocol mismatch:

- sender start / stop signal: `3 / 4`
- controller start / stop signal: `1 / 2`
- controller note-count state와 note collection state 미분리

판정:

> 기존 bridge는 virtual MIDI 연결 참고 자료로만 사용. 실시간 runtime은 일반 MIDI event path로 분리.

### 4.4 Dataset and style boundary

- active PiJAMA tree: `midi_dataset/midi`, 2,777 performances, 120 pianists
- alternate transcription: `midi_dataset/midi_kong`, 동일 2,777 performances
- 두 transcription tree를 독립 곡으로 합산 금지
- 주요 target 수량: Art Tatum 122, Brad Mehldau 72, Fred Hersch 66
- PiJAMA source: full-length solo piano automatic transcription
- Serum MVP target: monophonic lead phrase

문제 지점:

- full-piano voicing distribution과 monophonic lead target 불일치
- automatic transcription error 가능성
- chord / phrase / improvisation-section label 부재
- 기존 D1의 voicing diversity는 Serum lead style의 직접 지표가 아님
- Art Tatum 자료는 studio 중심, Brad Mehldau·Fred Hersch는 live/studio 혼합으로 recording condition confound 가능
- 녹음 시대·음질 차이가 performer classifier의 지름길 feature가 될 가능성

D3/D4 제약:

- D3: lead-only primer 생성 mean voicing `1.305`, no-primer `1.618`
- D4: primer voicing `1.23 / 1.43 / 1.83`에 생성 voicing `1.306 / 1.436 / 1.784`
- 동일 base·재학습 없음 조건에서 primer texture가 output texture를 강하게 제어
- explicit chord timeline을 4음 MIDI primer로 렌더링하면 polyphonic output을 유도할 가능성
- Serum용 monophonic output은 model 고유 능력이 아니라 별도 output-role 계약으로 검증 필요

### 4.5 Prior-art boundary

| 계열 | 이미 검증된 범위 | 이 설계에서 남는 검증 |
|---|---|---|
| Continuator / OMax / Somax2 | 사용자 corpus의 실시간 학습·재조합·응답 | neural adapter와 corpus retrieval의 동일 조건 비교 |
| BebopNet / MINGUS | chord-conditioned monophonic jazz solo | 개인 연주자 adapter, DJ section·drop 문맥 |
| ReaLJam / StreamMUSE | online symbolic accompaniment와 frame scheduling | solo role, adapter swap, FL Studio/VST 출력 |
| jam_bot 2025/2026 | 약 360M AMT, 12키 전조 FT, ONNX·8-bit·KV cache, 4 thread, MTC 동기, 미래 note 무효화, 공연, velocity·장치 지연 보상 | 여러 연주자에 재사용 가능한 adapter와 공개 검증 protocol |
| Magenta RealTime 2 | MIDI·text·audio prompt 기반 저지연 streaming audio와 DAW plugin | editable MIDI output, 현재 공개판 사용자별 fine-tuning |
| LMDM | diffusion audio의 block-wise KV cache, post-training, generative delay | note-level symbolic output과 chord·drop deadline 제어 |

연도 경계:

- Music Transformer: 2018
- Anticipatory Music Transformer: 2023 arXiv / 2024 TMLR
- Jordan Rudess jam_bot 공개 공연: 2024-09-21
- jam_bot 원 시스템 논문: ISMIR 2025
- velocity, 3.5시간 call-response, ggml, 지연 보상 확장: NIME 2026

jam_bot 2025 학습 조건:

- Jordan Rudess의 15~45분 MIDI clip
- all twelve keys 전조 증강
- `stanford-crfm/music-medium-800k`, 약 360M parameter
- 2,000 step 학습, 약 300 step부터 validation plateau/overfitting 관측

현재 프로젝트 D1의 full FT + 12키 증강 arm은 pooled unique voicing `80`으로 실패했다.
같은 12키 증강도 학습 step, 모델 규모, 데이터 역할에 따라 결과가 달라지므로
jam_bot 레시피를 이 프로젝트의 검증된 처방으로 간주하지 않는다.

판정:

- `personalized realtime improviser` 자체는 연구 기여로 사용 금지
- LMDM은 audio runtime 비교 자료이며 symbolic generation baseline이 아님
- Magenta RealTime 2의 audio prompt와 live audio streaming input을 같은 기능으로 표기 금지
- jam_bot의 `ms/token`과 keypress-to-sound end-to-end latency를 같은 지표로 비교 금지
- symbolic output과 deadline metric 자체를 연구 기여로 주장 금지

### 4.6 Data consent and identity boundary

PiJAMA의 Art Tatum, Brad Mehldau, Fred Hersch 데이터는 adapter 분리 가능성 검증용
research corpus로 한정한다. 특정 인물의 승인·공식 복제 모델을 의미하지 않는다.

- Art Tatum: 고인. 저작권·실연자권·퍼블리시티권의 관할별 검토가 별도로 필요
- Brad Mehldau·Fred Hersch: 생존 연주자. 공개 데이터 존재를 학습·배포 동의로 해석 금지
- 실명 adapter 결과는 기본적으로 내부 연구 ID로 익명화
- adapter별 공개 등급을 `internal-only / aggregate-paper-result / public-demo`로 기록
- checkpoint, generated MIDI/audio, 시연 영상은 공개 등급 승인 전 외부 업로드 금지

제품 경로의 데이터 계약:

- 연주자가 직접 제공한 MIDI만 개인 adapter 학습에 사용
- base model 학습 포함 여부를 별도 동의 항목으로 분리
- adapter export / delete / retrain 이력 기록
- 원본 MIDI와 파생 checkpoint의 provenance 연결
- artist name 공개와 비공개 개인 profile을 분리
- 탈퇴·동의 철회 시 원본과 파생물 처리 정책 검증

## 5. System Boundary

### 5.1 Realtime data plane

```text
InputPort
  -> EventCapture
  -> ClockSync
  -> RecentContextBuffer
  -> ContextSnapshotQueue
  -> GenerationWorker
  -> GeneratedBlockQueue
  -> BeatScheduler
  -> OutputPort
  -> FL Studio / VST
```

원칙:

- MIDI callback 내부 inference 금지
- model process 재실행 금지
- block마다 checkpoint reload 금지
- block마다 MIDI file write/read 금지
- input callback은 timestamp와 event enqueue만 수행
- generation result는 absolute beat 또는 target monotonic time 포함
- scheduler는 normal `note_on`, `note_off`, velocity 전송
- stop / exception 시 active note 전체 해제
- context마다 단조 증가하는 `context_version` 부여
- chord, section, clock 또는 adapter 변경 시 아직 시작하지 않은 이전 version block 무효화
- 이미 note-on이 전송된 block은 강제 교체하지 않고 note-off 완료 또는 safe all-notes-off 정책 적용

adapter hot-swap 경계:

- base weight는 session 동안 resident 유지
- adapter registry가 검증된 adapter와 metadata 보유
- 다음 block 경계에서만 active adapter를 atomic 교체
- adapter 변경 시 queued block 무효화 후 새 version으로 재생성
- 최초 사용 adapter의 load/warm-up과 이미 warm 상태인 switch time을 분리 측정

clock source 계약:

- `internal`: monotonic clock + 고정 BPM, unit/integration test용
- `midi_clock`: FL Studio transport의 MIDI Clock pulse 기반 beat 추적
- `mtc`: 장기 song-position 동기화가 필요한 경우 검토
- 외부 clock jump를 즉시 적용하지 않고 offset/drift 보정 정책 사용
- clock message stale, transport stop, song-position jump를 별도 상태 전이로 처리

### 5.2 Optional control plane

```text
CLI / MIDI CC / optional MCP
  -> SessionController
  -> configuration update
```

control message 예시:

- `select_style(art_tatum)`
- `set_mode(chord_response)`
- `set_density(medium)`
- `set_lookahead_bars(1)`
- `start()`
- `stop()`

control plane 정지 또는 연결 해제는 MIDI scheduling 중단 원인이 되면 안 된다.

### 5.3 WAV context plane

1차:

```text
WAV file path
  -> offline analyzer
  -> tempo / beat / chord / section timeline
  -> validated context file
  -> RealtimeContextProvider
```

후속:

```text
FL Studio audio route / loopback / plugin
  -> streaming analyzer
  -> confidence-scored context updates
  -> RealtimeContextProvider
```

live audio capture는 direct MIDI runtime 검증 후 별도 이슈로 분리한다.

## 6. Runtime Contracts

### 6.1 Session configuration

```json
{
  "style_adapter": "art_tatum",
  "role": "lead",
  "mode": "chord_response",
  "bpm": 124.0,
  "time_signature": "4/4",
  "lookahead_bars": 1,
  "clock_source": "internal",
  "output_channel": 1,
  "pitch_range": [48, 88],
  "density": "medium",
  "energy": "high"
}
```

현재 `role_control_token()`은 입력을 무시하고 항상 `ROLE_LEAD`를 반환한다. 따라서
첫 runtime 계약에서 `role`은 확장 가능한 제어값이 아니라 `lead` 고정 validation 값이다.
다른 role은 labeled dataset과 token 동작이 구현된 뒤 추가한다.

runtime clock은 연속 BPM을 사용하지만 현재 모델 조건은 `slow / medium / dance / fast`
4개 bucket이다. 예를 들어 120 BPM과 128 BPM은 모두 `DANCE` token이다. 연속 BPM
반응을 모델 능력으로 주장하지 않고 scheduler 정확도와 model conditioning을 분리한다.

### 6.2 Musical context snapshot

```json
{
  "captured_at_ns": 0,
  "context_id": "ctx-00042",
  "context_version": 42,
  "current_beat": 64.0,
  "clock_source": "internal",
  "clock_offset_ms": 0.0,
  "clock_stale": false,
  "target_start_beat": 68.0,
  "target_end_beat": 72.0,
  "chords": [
    {"start_beat": 68.0, "duration_beats": 2.0, "symbol": "Dm7"},
    {"start_beat": 70.0, "duration_beats": 2.0, "symbol": "G7"}
  ],
  "recent_notes": [],
  "section": "pre_drop",
  "energy": 0.85,
  "confidence": 1.0
}
```

### 6.3 Generated block

```json
{
  "block_id": "block-00042",
  "source_context_id": "ctx-00042",
  "context_version": 42,
  "target_start_beat": 68.0,
  "target_end_beat": 72.0,
  "adapter": "art_tatum",
  "fallback_used": false,
  "metrics": {
    "note_count": 1,
    "duration_beats": 0.25,
    "dead_air_ratio": 0.0,
    "chord_tone_ratio": 1.0
  },
  "events": [
    {
      "pitch": 72,
      "velocity": 96,
      "start_beat": 68.5,
      "duration_beats": 0.25,
      "channel": 1
    }
  ]
}
```

필수 validation:

- target beat 범위 밖 event 없음
- pitch / velocity 범위 유효
- 동일 pitch의 비정상 overlap 없음
- note-off 누락 없음
- monophonic mode에서 polyphonic onset은 deterministic lead-selection repair 적용
- repair 이전 max simultaneous notes와 repair된 onset 비율 기록
- repair 이후 동시 note 수 `<= 1`
- block ID와 source context ID 기록
- scheduler 소비 시 active `context_version`과 일치
- dead-air, phrase coverage, chord role, diversity proxy 기록

## 7. Interaction Modes

### 7.1 Chord response — first target

입력:

- 1차: 명시한 chord timeline 또는 FL Studio에서 검증된 chord label 전송
- 후속: 사용자가 누른 chord note set에서 live chord recognition

출력:

- 다음 마디의 monophonic lead phrase

선정 이유:

- 현재 chord progression parsing, 시간 매핑, chord-aware 평가 자산 재사용 가능
- 입력 의미가 명확함
- 동일 chord progression 기반 대조 실험 가능
- beat-aligned lookahead 검증 가능

현재 부재:

- arbitrary MIDI note set을 chord symbol로 인식하는 production 경로
- inversion, rootless voicing, upper structure, sustain note를 포함한 chord recognition
- chord confidence와 이전 chord hysteresis

기존 `parse_chord`와 `chord_for_time`은 주어진 chord label을 해석·시간 매핑하는
도구이며 MIDI-to-chord recognizer가 아니다. live chord 입력은 별도 baseline과
오인식 평가 없이 첫 model integration의 전제로 사용하지 않는다.

### 7.2 Call and response — second target

입력:

- 사용자가 연주한 1마디 또는 2마디 melody phrase

출력:

- 다음 1마디 또는 2마디 response phrase

추가 요구:

- phrase-end detection
- call / response token 또는 role 구분
- motif similarity와 단순 복사 분리
- response 시작 beat 고정

### 7.3 Continuous accompaniment — deferred

연속 note-by-note reaction보다 block response와 lookahead를 우선한다.

즉시 반응 지연과 의도된 musical response delay를 분리한다.

- transport latency: MIDI event 이동 시간
- response delay: 다음 beat / bar까지 의도적으로 대기한 시간
- generation deadline: target beat 이전 생성 완료 여부

## 8. Research Questions and Experiments

### 8.1 Hypotheses

| ID | 가설 | 기각 기준 |
|---|---|---|
| H1 | shared base + performer adapter가 artist별 full FT보다 저장·전환 비용을 줄이면서 style 분리도·다양성 유지 | full FT 대비 style 분리도 열화, base 대비 차이 없음, 또는 고유 n-gram·entropy가 base 대비 사전 등록 허용치 이상 감소 |
| H2 | chord + section + bars-to-drop + energy 조건이 chord-only보다 transition phrase landing 개선 | held-out track에서 landing·구조 적합도 개선 없음 |
| H3 | symbolic MIDI path가 현재 장비에서 목표 deadline과 note 편집 계약 동시 충족 | p99 deadline 또는 MIDI validity gate 실패 |
| H4 | adapter가 corpus retrieval보다 새 chord·tempo 문맥에 일반화하면서 training phrase 복제율 제한 | unseen 문맥 성능 열화 또는 복제율 증가 |
| H5 | 동일 PiJAMA recording의 full-piano 표현에서 추출한 lead가 performer identity signal을 보존 | full-piano classifier 대비 balanced accuracy 유지율이 사전 등록 허용치 미만 또는 chance 수준 |

H1 다양성 초기 gate는 D1의 사전 등록 좌표계를 승계한다.

- pooled unique n-gram: B3 base의 `>= 80%`
- pitch/rhythm entropy: B3 base 대비 감소 `<= 0.15 bit`
- 두 조건 중 하나라도 실패하면 diversity-preserving 주장 기각

이 값은 D1의 강한 성공 기준 `116/145 = 80%`와 entropy `4.85/5.00`에서 가져온
초기 proxy다. R3의 monophonic representation과 표본 수가 달라 결과 확인 뒤 값을 바꾸지
않고, 부적합하면 결론을 보류하고 다음 사전 등록 실험에서만 교정한다.

H5 retention gate 도출:

- target 3명을 제외한 pianist의 paired full-piano / extracted-lead pilot 사용
- metadata-only classifier가 note-domain classifier보다 사전 등록 margin 이상 낮음을 확인한 경우에만 full-piano classifier를 identity-signal 분모로 사용
- 위 margin은 target 3명 결과를 보기 전 non-target pilot에서 고정
- performer-balanced bootstrap으로 balanced-accuracy retention ratio 분포 계산
- pilot 95% interval의 lower bound를 target evaluation 전에 고정
- target 3명 test에서 retention ratio가 이 기준 미만이거나 3-class chance `1/3` 이하면 기각

### 8.2 Required baselines

| ID | Baseline | 비교 목적 |
|---|---|---|
| B0 | chord-scale / guide-tone rule | 화성 규칙만으로 얻는 최소 성능 |
| B1 | nearest-neighbor phrase retrieval | 학습 phrase 재사용의 품질·복제 위험 |
| B2 | Somax2 실행 비교; 환경 불가 시 최소 Factor Oracle fixture | 실시간 corpus 기반 선행 방식 |
| B3 | generic symbolic LM | adapter 없는 base 성능 |
| B4a | artist별 full fine-tuning, 무증강 | adapter 효율과 style 분리도 비교 |
| B4b | artist별 full fine-tuning, 12키 전조 | jam_bot 레시피와 D1 증강 붕괴 위험 비교 |
| B5a | artist별 LoRA / adapter, 무증강 | 주 실험군 |
| B5b | artist별 LoRA / adapter, 12키 전조 | 증강 효과와 collapse 비교 |
| B6 | shuffled artist label adapter | classifier·transcription artifact 통제 |

Composer Vector 계열 inference steering은 재현 가능한 symbolic checkpoint와 코드가
확인될 때 확장 baseline으로 추가한다. jam_bot, Magenta RealTime 2, LMDM은 출력 형식과
공개 데이터가 달라 동일 수치 경쟁 대신 interaction, latency definition, model reuse,
output editability를 시스템 수준에서 비교한다.

B2는 Somax2의 Max 환경과 동일 MIDI corpus를 사용할 수 있을 때 실행 비교한다. 환경이
재현되지 않으면 baseline 전체를 구현하는 대신 고정 fixture의 Factor Oracle 최소 구현으로
sequence recombination 특성만 비교하고, Somax2와 동등하다고 주장하지 않는다.

[Chord-Symbol Time-Series Adaptation](https://arxiv.org/abs/2606.07334)은 LoRA
출력이 target 분포에 가까워지는 동시에 10/11 장르에서 고유 chord 수와 entropy가
감소했다고 보고했다. 해당 연구는 chord-symbol·장르, 이 프로젝트는 note-level·연주자
적응이므로 직접 모순으로 단정하지 않고 H1의 반대 근거로 등록한다.

### 8.3 Evaluation split

- recording / album 단위 split
- 동일 recording의 fragment 교차 split 금지
- target performer를 base pretraining에서 제외한 clean adaptation arm 포함
- unseen song, chord progression, key, tempo를 각각 기록
- artist별 학습 sequence 수·총 note 수·총 duration 상한 통제
- live/studio, 녹음 시대, transcription confidence 기준으로 metadata-matched strata 구성
- metadata만 입력한 classifier와 note-domain classifier를 비교해 recording-condition shortcut 점검
- `midi`와 `midi_kong` transcription을 독립 표본으로 합산 금지
- generation seed와 context set 고정

H5 참조 경계:

- PiJAMA full-piano와 extracted lead를 동일 recording 단위로 paired 비교
- full-piano classifier 성능을 identity-signal 상한으로 사용
- Weimar/Jazzomat와 Charlie Parker Omnibook은 monophonic pipeline sanity check로 사용
- 외부 corpus는 PiJAMA와 연주자·악기가 다르므로 직접 identity-retention 참조군으로 사용 금지

### 8.4 Evaluation domains

- melody: interval, contour, pitch-class, chromatic approach
- harmony: chord role, strong-beat role, phrase landing
- rhythm: onset, duration, syncopation, density, rest pattern
- dynamics: velocity distribution; transcription 신뢰도와 함께 보고
- diversity: pitch/rhythm entropy, distinct n-gram, collapse rate
- memorization: longest matching subsequence, nearest training-window similarity
- realtime: p50/p95/p99, deadline miss, jitter, underrun, 30분 안정성
- human review: blind A/B style identity, track fit, transition usefulness, playability

### R0. Direct MIDI transport baseline

질문:

> 모델과 scheduler 없이 virtual MIDI loopback의 일반 MIDI event 송수신이 보존되는가?

입력:

- fixed note sequence
- keyboard echo sequence
- note-on/off burst
- repeated same-pitch sequence

경로:

- test sender → virtual MIDI input → direct pass-through → capture output
- FL Studio / Serum 연결 제외

측정:

- input event count
- output event count
- event loss count
- duplicate event count
- stuck note count
- 10분 연속 실행 crash count

진입 게이트:

- 10분 event loss: `0`
- 10분 stuck note: `0`
- crash: `0`

실패 시:

- model integration 중단
- virtual port routing, feedback loop, callback blocking 우선 점검

### R1. Clock and one-bar scheduler baseline

질문:

> 생성 모델 없이도 다음 마디 block을 끊김 없이 예약할 수 있는가?

입력:

- fixed BPM: `90 / 120 / 128 / 160`
- 4/4 clock
- deterministic phrase bank

팔:

- internal monotonic clock
- FL Studio master MIDI Clock
- transport stop / resume / song-position jump fault injection

측정:

- queue depth
- schedule lead time
- deadline miss count
- queue underrun count
- bar-boundary start error
- external clock drift and correction count
- stale clock / transport jump recovery time
- fallback count

판정 게이트:

- 각 BPM 10분 deadline miss: `0`
- 각 BPM 10분 queue underrun: `0`
- bar start error p99: `<= 20ms`
- 10분 external clock drift가 bar-boundary gate 이내
- exception 종료 후 active note: `0`

### R2. Resident model runtime feasibility

질문:

> 현재 Music Transformer를 상주시킨 상태에서 1마디 block 생성 deadline을 만족할 수 있는가?

팔:

| Arm | Model path | Candidate count | Cache |
|---|---|---:|---|
| A | current in-process baseline | 1 | 없음 |
| B | shorter generation block | 1 | 없음 |
| C | cached autoregressive generation | 1 | KV cache |
| D | phrase-bank fallback | 1 | 해당 없음 |
| E | cached generation + active scheduler stress | 1 | KV cache |

고정:

- 동일 checkpoint
- 동일 context
- 동일 seed set
- 동일 target phrase length
- checkpoint의 `max_sequence`와 RPR embedding shape 유지
- Arm B는 `target_length`와 `primer_max_tokens`만 변경
- file write 제외
- model warm-up 이후 측정
- grammar mask 활성화
- explicit chord timeline 사용
- E에서는 입력 callback과 scheduler tick 부하를 동일 process/thread 구성으로 실행

Arm C 진입 전제:

- 현재 custom relative-position attention은 square `len_q == len_k` 경로 기준
- incremental `len_q=1, len_k=n`에 맞춘 RPR attention 수식·mask 재구현 필요
- cache off/on 동일 prefix의 next-token logits parity test 선행
- tolerance와 test sequence를 구현 전에 고정
- parity 실패 시 latency sweep에 Arm C 포함 금지

측정:

- context build time
- generation time p50 / p95 / p99
- decode / validation time
- total block-ready time
- memory usage
- deadline miss count
- fallback count
- grammar validity / monophonic repair rate / repair 후 validity
- scheduler send timing error와 callback queue delay
- Python thread와 별도 process 구성 비교가 필요한지 판정

판정식:

```text
generation_p99 + decode_p99 + scheduling_margin
  <= lookahead_duration
```

운영 여유 판정:

```text
block_ready_p99 <= lookahead_duration * 0.5
```

판정:

- 강한 성공: 운영 여유 판정 통과 + 10분 deadline miss `0`
- 부분 성공: deadline 판정 통과, 운영 여유 판정 실패
- 실패: deadline 판정 실패

Arm E에서 scheduler p99 또는 callback delay가 R0/R1 gate를 벗어나면 model과 scheduler를
별도 process로 분리한다. PyTorch가 항상 GIL을 점유한다고 전제하지 않고 동일 부하 측정으로
thread/process 경계를 결정한다.

block 길이 계약:

- 현재 `generate()`는 token 수 또는 `TOKEN_END`로 정지하므로 정확한 1마디 출력을 보장하지 않음
- R2 전에 decoded musical time이 target end에 도달하면 generation을 멈추는 기준 검토
- 초과 note crop, 경계 note-off repair, 부족 구간 fallback fill을 각각 기록
- 위 계약 전에는 `one-bar generation` 대신 `one-bar target window`로 표기

R2 실패 시 다음 후보:

1. output representation당 token 수 축소
2. block 길이 / context 길이 축소
3. smaller resident model 비교
4. phrase-bank hybrid 비율 확대
5. scheduler process 분리

양자화와 ONNX는 위 후보 측정 후 검토한다. jam_bot은 약 360M, 현재 모델은 약
13.4M으로 규모가 다르므로 동일한 최우선 최적화라고 전제하지 않는다.

### R3. Pianist adapter swap

질문:

> 동일한 harmonic input에서 adapter 교체만으로 연주자별 lead-style 차이가 발생하는가?

대상:

- Art Tatum: 122 performances
- Brad Mehldau: 72 performances
- Fred Hersch: 66 performances

데이터 설계:

- active tree `midi_dataset/midi`만 사용
- target 3명 전체를 generic base에서 제외
- artist별 album 단위 train / validation / test 분리
- 동일 performance의 fragment가 서로 다른 split에 들어가는 것 금지
- monophonic lead extraction 규칙 고정
- extraction sample 수동 검토

팔:

| Arm | 방식 |
|---|---|
| B0 | chord-scale / guide-tone rule |
| B1 | artist별 nearest-neighbor phrase retrieval |
| B3 | generic base LM |
| B4a | artist별 full fine-tuning, 무증강 |
| B4b | artist별 full fine-tuning, 12키 전조 |
| B5a | shared base + artist별 LoRA / adapter, 무증강 |
| B5b | shared base + artist별 LoRA / adapter, 12키 전조 |
| B6 | shuffled-artist-label adapter control |

B2는 R3의 모든 학습 arm과 같은 batch로 돌리지 않는다. Somax2 실행 환경이 확보되면
동일 artist corpus·동일 chord context의 별도 system comparison으로 실행하고, 불가하면
고정 fixture의 최소 Factor Oracle 결과를 `B2-proxy`로 분리 표기한다.

입력 통제:

- 동일 chord timeline
- 동일 BPM / density / energy
- 동일 seed set
- 동일 generated block count

측정:

- grammar validity
- chord / strong-beat landing metric
- phrase coverage
- interval / rhythm / chromaticism distribution distance
- held-out real-performance style classifier 결과
- shuffled-label control 대비 차이
- longest matching subsequence와 nearest training-window similarity
- one-bar binary/velocity piano-roll SSIMuse
- unseen chord / key / tempo 성능
- artist당 checkpoint 저장 크기와 adapter switch time
- single-user blind listening identification
- adapter별 generation latency

스타일 판정 게이트:

- generated sample의 intended-artist identification이 shuffled-label control 상회
- confidence interval 기준 3-class chance level 상회
- harmonic / grammar gate의 base 대비 비열화 확인
- retrieval 대비 unseen context 적합도 개선
- generated-to-train 복제 위반율이 base LM 대비 비증가
- train/held-out real-performance overlap 분포로 생성 결과 확인 전에 고정한 절대 상한 통과
- full FT 대비 style 분리도 허용 범위와 저장·전환 비용을 함께 보고
- pooled unique n-gram이 B3의 `>= 80%`, pitch/rhythm entropy 감소가 B3 대비 `<= 0.15 bit`
- blind listening 결과 별도 기록

제한:

- classifier 성공만으로 음악성 주장 금지
- single-user review로 일반 청취자 선호 주장 금지
- PiJAMA transcription artifact 가능성 유지
- 기존 D1의 16-sequence LoRA 결과를 122/72/66 performance regime으로 외삽 금지
- D1 증강 arm의 diversity 재붕괴를 H1 선행 위험으로 기록

후속 robustness:

- 동일 performance의 `midi` / `midi_kong` transcription으로 adapter 재학습
- transcription source 변경 시 style conclusion 유지 여부 비교

복제 임계값은 generated output을 본 뒤 정하지 않는다. train과 held-out real-performance의
longest match, nearest-window similarity, SSIMuse 분포로 먼저 고정한다. 이때 귀무분포는
R3에서 고정한 동일 lead-extraction 규칙을 train과 held-out real recording에 적용해 계산하고
preregistration에 기록한다. retrieval arm은 복제 위험의 상한 대조이며 통과 기준으로
사용하지 않는다. SSIMuse grid와 window parameter는 원문·공개 구현 확인 후 preregistration에서
확정하며 현재 문서의 one-bar 표현에 수치를 그대로 전이하지 않는다.

### R4. DJ transition context ablation

질문:

> chord 조건에 section, bars-to-drop, energy trajectory를 추가하면 transition용 phrase가
> 목표 beat에 맞춰 tension과 landing을 배치하는가?

팔:

| Arm | Context |
|---|---|
| A | chord only |
| B | chord + section |
| C | chord + section + bars-to-drop |
| D | chord + section + bars-to-drop + energy trajectory |
| E | D의 section / drop label shuffle control |

고정:

- 수동 검증된 MIDI 또는 chord timeline 사용
- 동일 adapter, seed, BPM, chord progression, block length
- WAV analyzer 결과를 사용하지 않아 context model과 analyzer 오차 분리

측정:

- target drop 이전 phrase-end alignment
- final landing chord role
- density / register / velocity trajectory correlation
- invalid / fallback / deadline miss
- blind A/B transition fit

판정:

- D가 A와 E 대비 objective transition metric 개선
- grammar, harmonic validity, diversity의 사전 정의 허용 범위 유지
- blind review는 별도 근거로 기록하며 objective 결과와 합산 금지

### R5. Offline WAV context extraction

질문:

> 전체 WAV를 재생 전에 분석해 drop / transition 구간의 generation context를 만들 수 있는가?

1차 출력:

- BPM
- beat timestamps
- downbeat / bar index
- key
- chord root / quality timeline
- section boundary
- energy curve
- field별 confidence

평가 데이터:

- 사용자가 실제 사용할 house / dance track excerpt
- 수동 BPM / downbeat / chord / drop annotation
- 쉬운 harmonic loop와 복잡한 mix 분리

측정:

- BPM relative error
- half / double tempo error count
- beat timestamp error
- downbeat error
- chord root / quality accuracy
- section-boundary error
- low-confidence coverage

fallback:

- confidence 임계치 미달 시 manual BPM / chord timeline 사용
- 분석 실패를 generation failure로 합산하지 않음

제외:

- full-mix audio-to-MIDI 결과를 chord ground truth로 취급
- 첫 구현에서 live audio capture
- WAV analyzer 정확도와 style adapter 성능의 단일 지표 합산

### R6. Integrated performance run

시나리오:

1. house track WAV 사전분석
2. FL Studio project와 virtual MIDI port 준비
3. Serum lead channel arm
4. style adapter 선택
5. 1마디 lookahead generation 시작
6. drop 이전 adapter 교체
7. 생성 MIDI를 FL Studio에 기록

최종 측정:

- 10분 연속 실행
- MIDI event loss / stuck note
- block deadline miss / queue underrun
- first valid input-to-block-ready p50 / p95 / p99
- 출력 note 간 `>= 180ms` dead-air event와 dead-air ratio
- fallback ratio
- adapter switch time
- bar alignment error
- chord-tone ratio, phrase coverage, repetition, pitch/rhythm diversity
- FL Studio audio underrun count
- 생성 MIDI와 audio recording 보존
- 사용자 청음 기록

완료 경계:

- 시스템 동작 증명: 위 runtime gate 통과
- 스타일 차이 증명: R3 gate 통과
- 음악적 만족도: 사용자 청음 기록으로만 별도 판정
- production-ready / general performer cloning 주장 제외

## 9. Thread and State Model

권장 worker 책임:

| Worker | 책임 | 금지 |
|---|---|---|
| input callback | timestamp, event enqueue | inference, file I/O |
| clock worker | internal/external clock offset, drift, transport state | model inference, MIDI note output |
| context worker | chord / phrase buffer, clock snapshot | MIDI output |
| generation worker | resident model block generation | sleep 기반 재생 |
| scheduler | target time 기준 MIDI output | model inference |
| watchdog | deadline, underrun, all-notes-off | musical quality 판정 |

상태:

```text
STOPPED
  -> WARMING
  -> READY
  -> RUNNING
  -> DEGRADED
  -> STOPPING
  -> STOPPED
```

`DEGRADED` 진입 조건:

- block deadline 임박
- generation exception
- invalid MIDI block
- input clock stale

`DEGRADED` 처리:

- chord-aware phrase-bank fallback
- fallback event 기록
- 다음 block에서 model recovery 시도
- 반복 실패 시 safe stop 및 all-notes-off

## 10. Latency Metrics

필수 timestamp:

- `input_received_ns`
- `context_snapshot_ns`
- `generation_started_ns`
- `generation_completed_ns`
- `block_validated_ns`
- `event_target_ns`
- `event_sent_ns`
- `audio_observed_ns` — loopback 측정이 가능한 run에서만

파생 지표:

```text
context_build_ms
generation_ms
validation_ms
input_to_block_ready_ms = block_validated_ns - input_received_ns
schedule_lead_ms = event_target_ns - block_validated_ns
send_error_ms = event_sent_ns - event_target_ns
deadline_miss = block_validated_ns > event_target_ns
audio_output_latency_ms = audio_observed_ns - event_sent_ns
```

구분:

- MIDI transport latency
- model generation latency
- intentional next-bar response delay
- FL Studio audio-buffer latency
- synth plugin processing latency

단일 `latency` 수치로 합산 보고하지 않는다.

기존 `JAMBOT_MIDI_REFACTOR_PLAN.md` KPI 처리:

- dead-air threshold `>= 180ms`: live block metric으로 승계
- E2E TTFN `<= 120ms`: 즉시 응답 mode의 별도 장기 목표로 유지
- next-bar response mode에서는 의도된 musical delay가 포함되므로 TTFN gate로 사용하지 않음
- next-bar mode의 1차 gate: target beat 이전 block-ready와 deadline miss `0`
- 최종 E2E는 MIDI send뿐 아니라 FL Studio audio loopback timestamp로 별도 측정

## 11. Reuse and Replacement Boundary

재사용 후보:

- Stage B conditioning MIDI note/token extraction
- chord progression parsing
- grammar mask
- MIDI decode / validation
- chord-aware objective metrics
- LoRA checkpoint loading
- phrase-bank fallback
- listening review schema와 과거 artifact 형식

재사용 불가 또는 신규 필요:

- arbitrary live MIDI voicing의 chord recognition
- continuous BPM을 직접 표현하는 model token
- lead 이외 role control
- target beat 기반 autoregressive stopping
- archive 이동으로 깨진 active test/script import

현재 확인된 broken import:

- `tests/test_generated_candidate_chord_eval.py` → `archive/scripts/evaluate_generated_candidate_chords.py`
- `scripts/build_stage_b_midi_to_solo_bebop_language_note_review.py` → archive로 이동된 package module

review package 코드를 재사용하기 전에 active/archive 경계를 별도 이슈에서 복구한다.

분리 또는 교체 대상:

- CLI subprocess 기반 per-request generation
- file path 중심 result contract
- multi-candidate offline ranking을 live 기본값으로 사용하는 구조
- FL Studio bridge의 special note serialization
- callback 내부 synchronous generation
- preference/RAG와 MIDI transport 결합

## 12. Implementation Issue Boundaries

선행연구 문서의 학습 실험 순서보다 transport/scheduler를 먼저 둔다. 이는 adapter 품질을
먼저 입증했다는 뜻이 아니라, stuck note·event loss·clock drift를 모델과 분리하고 실제 악기
실행 안전성을 확보하기 위한 제품 경계다. 모델 연구에서는 retrieval/full FT/adapter 비교를
통과한 승자만 최종 runtime에 연결한다.

설계 승인 후 다음 순서로 별도 이슈 분리:

1. archive 이동으로 깨진 active import 복구
2. direct MIDI echo / event integrity harness
3. monotonic clock / one-bar scheduler / watchdog
4. versioned context / queued-block invalidation contract
5. explicit chord timeline input / live chord-recognition baseline 분리
6. resident in-process model adapter와 grammar-mask 회귀
7. target-window stop/crop/fill 계약
8. KV cache parity / short-block / concurrent scheduler latency experiment
9. monophonic lead extraction identity-signal audit
10. retrieval / full FT / adapter style comparison
11. DJ transition context ablation
12. offline WAV context analyzer
13. FL Studio / Serum integrated performance run
14. live audio context capture feasibility

각 이슈는 이전 단계의 판정 게이트 통과 후 시작한다.

## 13. First Deliverable

### 13.1 Transport echo

```text
test MIDI sender
  -> virtual MIDI input
  -> direct pass-through
  -> virtual MIDI capture output
```

완료 조건:

- model, clock, scheduler, FL Studio, Serum 제외
- 단일 10분 loopback fixture
- event loss `0`
- duplicate event `0`
- stuck note `0`
- crash `0`

### 13.2 Scheduler

```text
internal / FL Studio master clock
  -> 다음 마디 target beat 계산
  -> deterministic phrase block
  -> normal MIDI events 예약 전송
  -> virtual MIDI capture output
```

완료 조건:

- 4개 BPM에서 각 10분 연속 실행
- deadline miss `0`
- queue underrun `0`
- event loss / stuck note `0`
- send timing error p99 `<= 20ms`
- 외부 clock 10분 drift가 동일 gate 이내

`20ms`는 선행연구의 보편적 음악성 임계값이 아니라 첫 로컬 engineering gate다.
실제 FL Studio audio buffer와 VST 출력까지 포함한 end-to-end 측정 후 조정한다.

13.1 통과 후 13.2로 이동한다. FL Studio / Serum 연결은 scheduler loopback 통과 후
별도 이슈로 진행한다.

## 14. References

- `docs/CORE_PLAN.md`
- `docs/INFERENCE_MODEL_SPEC.md`
- `docs/archive/REVIEW_2026-05-16.md`
- `docs/archive/JAMBOT_MIDI_REFACTOR_PLAN.md`
- `docs/d1_experiment/D1_DESIGN.md`
- `docs/d1_experiment/D1_RESULTS.md`
- `docs/d3_experiment/D3_RESULTS.md`
- `docs/d4_experiment/D4_RESULTS.md`
- `music_transformer/model/music_transformer.py`
- `scripts/generate.py`
- FL Studio MIDI scripting reference: <https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/midi_scripting.htm>
- PiJAMA dataset article: <https://transactions.ismir.net/articles/10.5334/tismir.162>
- Music Transformer: <https://arxiv.org/abs/1809.04281>
- Anticipatory Music Transformer: <https://openreview.net/forum?id=EBNJ33Fcrl>
- Continuator: <https://doi.org/10.1076/jnmr.32.3.333.16861>
- OMax: <http://articles.ircam.fr/textes/Assayag06d/index.pdf>
- Somax2: <https://doi.org/10.3233/faia230106>
- BebopNet: <https://archives.ismir.net/ismir2020/paper/000132.pdf>
- MINGUS: <https://archives.ismir.net/ismir2021/paper/000051.pdf>
- ReaLJam: <https://arxiv.org/abs/2502.21267>
- jam_bot, ISMIR 2025: <https://zenodo.org/records/17706584>
- jam_bot expressive extension, NIME 2026: <https://nime.org/proceedings/2026/nime2026_73.pdf>
- StreamMUSE: <https://arxiv.org/abs/2606.11886>
- Magenta RealTime 2: <https://magenta.withgoogle.com/magenta-realtime-2>
- Live Music Diffusion Models: <https://arxiv.org/abs/2605.22717>
- Chord-Symbol Time-Series Adaptation: <https://arxiv.org/abs/2606.07334>
- Artistic fingerprints in jazz: <https://www.nature.com/articles/s42256-026-01279-9>
- SSIMuse symbolic replication: <https://arxiv.org/abs/2509.13658>
- Lakh MIDI de-duplication: <https://ismir2025program.ismir.net/poster_188.html>
- Weimar Jazz Database: <https://jazzomat.hfm-weimar.de/>
- Charlie Parker Aligned Digital Omnibook: <https://arxiv.org/abs/2405.16687>
- Live Music Agents design space: <https://arxiv.org/abs/2602.05064>
- Essentia beat extraction: <https://essentia.upf.edu/reference/std_RhythmExtractor2013.html>
- Essentia beat-synchronous chord detection: <https://essentia.upf.edu/reference/std_ChordsDetectionBeats.html>
