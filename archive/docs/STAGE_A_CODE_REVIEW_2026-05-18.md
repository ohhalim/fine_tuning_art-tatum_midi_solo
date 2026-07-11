# Stage A Code Review: 2026-05-18

작성일: 2026-05-18

이 문서는 FL Studio piano roll 확인 이후, Stage A 모델 출력이 왜 solo-line MIDI로 보기 어려운지 코드 기준으로 정리한 리뷰다.

결론부터 말하면 현재 문제는 metric threshold나 postprocess만의 문제가 아니다. Stage A 모델 학습/로드/토큰화/디코딩 구조 자체가 현재 MVP 목표와 맞지 않는다.

## 1. Executive Summary

현재 Stage A는 "LoRA fine-tuned Music Transformer"라고 부르기 어렵다.

가장 큰 문제는 다음 네 가지다.

1. `train_qlora.py`가 pretrained base checkpoint 없이 랜덤 초기화 Music Transformer에 LoRA만 붙여 학습한다.
2. `scripts/generate.py`도 새 랜덤 base model을 만든 뒤 LoRA weight만 로드한다.
3. MIDI decoder의 `_merge_note()`가 `note_off` 처리 후 active note를 삭제하지 않아 긴 sustain note를 인공적으로 만들 수 있다.
4. 학습 conditioning format과 inference request-derived chord conditioning MIDI가 서로 다르다.

따라서 현재 모델 output이 긴 sustain block, chord block, 1-note/2-note phrase, 코드와 무관한 음을 만드는 것은 자연스러운 결과다.

## 2. Evidence

### 2.1 Current Dataset and Checkpoint

현재 local Stage A dataset:

- `data/roles/lead/dataset_summary.json`
- samples: `18`
- train: `16`
- val: `2`
- transpose_all_keys: `false`

현재 checkpoint:

- `checkpoints/jazz_lora_stage_a/checkpoint_epoch1.pt`
- `checkpoints/jazz_lora_stage_a/lora_weights.pt`
- epoch: `1`
- train loss: 약 `6.08`
- val loss: 약 `6.01`

LoRA weight file에는 LoRA matrix만 있다.

- embedding 없음
- output head 없음
- base transformer weights 없음

즉 생성 시 같은 base model이 재현되지 않으면 LoRA만으로는 의미 있는 음악 모델을 복원할 수 없다.

### 2.2 Solo-line Gate Result

최근 sweep:

- source: `outputs/sweeps/solo_line_gate_p256_27case.json`
- total: `27`
- completed: `27`
- model success: `10/27`
- fallback: `17/27`
- dense model success: `6/9`
- medium model success: `1/9`
- sparse model success: `3/9`

주요 fallback candidate failure:

- `note duration too long`: 20 candidate failures
- `note count too low`: 12 candidate failures
- `too many simultaneous notes`: 1 candidate failure
- `too many long notes`: 1 candidate failure

해석:

- Stage A model output은 dense 일부를 제외하면 solo-line MIDI로 보기 어렵다.
- 특히 medium density에서 duration/note-off 문제가 심하다.
- fallback이 늘어난 것은 regression이 아니라, 이전에 통과시키면 안 되는 MIDI를 성공으로 세고 있었기 때문이다.

## 3. Findings

### Finding 1: Critical - LoRA가 랜덤 base model 위에 학습/로드된다

관련 파일:

- `scripts/train_qlora.py`
- `scripts/generate.py`
- `scripts/runpod_train_stage_a.sh`

문제:

- `train_qlora.py`는 `MusicTransformer(...)`를 새로 만든다.
- `runpod_train_stage_a.sh`는 `train_qlora.py`에 `--checkpoint`를 넘기지 않는다.
- `train_qlora.py`는 전체 model parameter를 freeze하고 attention out projection LoRA만 학습한다.
- 저장 시 LoRA weight만 저장한다.
- `scripts/generate.py`는 다시 새 `MusicTransformer(...)`를 만든 뒤 LoRA weight만 `strict=False`로 로드한다.

결과:

- base model weight는 학습된 적도 없고 저장되지도 않는다.
- generation 시 base model은 새 랜덤 초기화 상태다.
- LoRA만으로 MIDI grammar, timing, note_on/note_off 구조를 배울 수 없다.

판단:

이 상태에서 생성 품질을 기대하면 안 된다. Stage A의 가장 큰 구조적 결함이다.

### Finding 2: Critical - MIDI decoder가 긴 sustain note를 만들 수 있다

관련 파일:

- `music_transformer/third_party/midi_processor/processor.py`

문제:

`_merge_note()`는 `note_off` 이벤트를 만나면 `note_on_dict[pitch]`에서 시작 note를 찾는다. 하지만 note를 result에 추가한 뒤 해당 pitch를 active dictionary에서 삭제하지 않는다.

결과:

- 같은 pitch의 stray `note_off`가 나중에 다시 나오면 과거 `note_on`을 재사용할 수 있다.
- 이 경우 start가 아주 이른 note가 phrase 끝까지 이어지는 긴 sustain block이 생긴다.
- FL Studio에서 보인 "전체 phrase를 물고 있는 긴 note"와 맞는 증상이다.

판단:

이건 모델 품질 이전의 decoder correctness 문제다. Stage A를 계속 쓰려면 가장 먼저 고쳐야 한다.

### Finding 3: High - Conditioning 학습 구조가 request-conditioned generation이 아니다

관련 파일:

- `scripts/prepare_role_dataset.py`
- `scripts/train_qlora.py`

현재 tokenized format:

```text
conditioning_tokens + TOKEN_END + target_tokens + TOKEN_END
```

문제:

- `train_qlora.py`는 전체 sequence를 next-token prediction으로 학습한다.
- target 부분에만 loss를 주지 않는다.
- 긴 sequence는 random crop된다.
- random crop이 conditioning/target boundary를 보존한다는 보장이 없다.

결과:

- 모델이 "conditioning을 보고 target solo를 생성"하도록 명확히 학습되지 않는다.
- `TOKEN_END` 하나가 separator와 end-of-sequence 역할을 동시에 한다.

판단:

Stage B에서는 explicit `COND_SEP`, `BAR`, `POSITION`, `CHORD`, `DENSITY`, `ENERGY` token과 target-only loss가 필요하다.

### Finding 4: High - Inference conditioning이 training conditioning과 다르다

관련 파일:

- `scripts/prepare_role_dataset.py`
- `inference/app/conditioning.py`

학습 conditioning:

- pitch split 기반 left-hand / sparse anchor MIDI
- 실제 MIDI phrase에서 추출

inference conditioning:

- request chord progression을 low-register chord block MIDI로 합성

문제:

- 모델은 inference에서 들어오는 synthetic chord block primer를 학습 때 충분히 본 적이 없다.
- chord progression, section, energy, density는 explicit token으로 학습되지 않았다.

결과:

- request chord progression과 생성 결과의 관계가 약하다.
- density/energy는 모델 내부보다 sampling, repair, fallback에서 주로 반영된다.

판단:

현재 request-derived conditioning은 demo contract용 hack에 가깝다. Stage B에서 token-level conditioning으로 바꿔야 한다.

### Finding 5: Medium - Postprocess는 bad output을 solo line으로 바꾸지 못한다

관련 파일:

- `inference/app/postprocess.py`

문제:

- `repair_model_midi()`는 pitch range mapping, phrase shift, trim을 한다.
- 하지만 긴 note duration을 solo-line phrase로 재구성하지 않는다.
- dense gap fill은 일부 케이스를 좋아 보이게 할 수 있지만 모델 자체 문제를 해결하지 않는다.

판단:

postprocess를 더 세게 걸면 fallback/template generator와 다를 바 없어진다. 모델 개선은 token/duration/conditioning 쪽에서 해야 한다.

## 4. What Not To Do Next

하지 말아야 할 일:

- chord-tone gate만 더 강하게 걸기
- postprocess로 모든 긴 note를 잘라서 모델 성공률을 올린 것처럼 보이게 하기
- 현재 Stage A 결과를 "personalized jazz solo model"로 포장하기
- Spring Boot/API 확장으로 넘어가기
- 실시간 MIDI runtime부터 만들기

이 작업들은 현재 문제를 가린다. 먼저 모델/토큰/디코더 구조를 고쳐야 한다.

## 5. Recommended Next Plan

### Step 1. Decoder Correctness Fix

Status: completed.

목표:

- `_merge_note()`에서 `note_off` 처리 후 active note를 삭제한다.
- 같은 pitch 중복 note_on/note_off 처리 정책을 명확히 한다.
- decoder roundtrip test를 추가한다.

완료 기준:

- 동일 pitch note_on/note_off가 반복되는 MIDI에서 긴 sustain ghost note가 생기지 않는다.
- stray note_off는 조용히 제거되지만 과거 note_on을 재사용하지 않는다.

구현 결과:

- `music_transformer/third_party/midi_processor/processor.py`의 `_merge_note()`가 `note_on_dict.pop(pitch)`로 active note를 소비한다.
- `tests/test_midi_processor_decode.py`가 stray `note_off`와 repeated same-pitch note pair를 검증한다.
- decoder fix 이후 targeted medium smoke는 여전히 `3/3` fallback이다. 이는 decoder ghost sustain은 줄였지만, random base + LoRA-only model 구조 문제는 남아 있다는 뜻이다.

### Step 2. Checkpoint Policy Fix

Status: completed.

목표:

- "random base + LoRA only" 구조를 중단한다.
- 둘 중 하나로 결정한다.

옵션 A:

- full model checkpoint를 저장/로드한다.
- small dataset overfit smoke를 먼저 한다.

옵션 B:

- 실제 pretrained base checkpoint를 확보하고 LoRA를 그 위에만 적용한다.

현재 상황에서는 옵션 A가 더 현실적이다.

완료 기준:

- 학습한 `checkpoint_epoch*.pt`의 full `model_state_dict`를 generation에서 로드한다.
- 같은 checkpoint를 로드했을 때 embedding/Wout/base transformer weights가 복원된다.

구현 결과:

- `scripts/generate.py`는 `--lora_path` 아래 최신 `checkpoint_epoch*.pt`를 자동 선택한다.
- `--checkpoint_path`로 explicit full checkpoint를 지정할 수 있다.
- full checkpoint의 `positional_encoding.pe` shape로 model max sequence를 감지해 strict load한다.
- inference CLI, `StageAModelRunner`, FastAPI model runner, generation sweep script가 full checkpoint path를 전달할 수 있다.
- full checkpoint가 없을 때만 legacy `lora_weights.pt` fallback을 사용한다.

### Step 3. Tiny Overfit Test

Status: runnable smoke path added.

목표:

- 1~3개 MIDI sample에 대해 모델이 target을 과적합할 수 있는지 확인한다.

완료 기준:

- teacher-forced loss가 확실히 내려간다.
- generation 결과가 최소한 note_on/note_off grammar를 유지한다.
- FL Studio piano roll에서 sustain block이 아닌 phrase로 보인다.

구현 결과:

- `scripts/run_stage_a_tiny_overfit.py`가 deterministic known-good MIDI phrase 1~3개를 생성한다.
- 같은 MIDI를 existing Music Transformer token stream으로 변환해 tiny train/val dataset을 만든다.
- `scripts/train_qlora.py --train_full_model`로 LoRA wrapper는 유지하되 random base까지 unfreeze할 수 있다.
- 새 full checkpoint에는 `model_config`가 저장되고, `scripts/generate.py`는 이 config로 작은 architecture를 자동 복원한다.
- smoke report는 raw sample metrics와 MVP inference gate의 `fallback_used` 결과를 함께 기록한다.
- 2026-05-18 `dense_overfit_200` local run은 `fallback_used=false`로 통과했다. 이는 current event token path가 full-model tiny training에서는 MIDI grammar를 배울 수 있음을 보여준다.
- 같은 조건의 `--lora_only` run은 best val loss `4.8228`과 decoded raw `note_count=0`으로 실패했다. 따라서 random-base LoRA-only training은 Stage A 학습 전략으로 검증되지 않았다.

### Step 4. Stage B Tokenization Decision

권장 방향:

- event stream에서 raw `NOTE_ON`/`NOTE_OFF` 의존을 줄인다.
- `NOTE(pitch, duration, velocity)` 또는 REMI-style token으로 전환한다.

최소 token:

```text
STYLE
SECTION
ENERGY
DENSITY
BPM
BAR
POSITION
CHORD
NOTE_PITCH
NOTE_DURATION
NOTE_VELOCITY
```

완료 기준:

- output note duration이 decode 단계에서 명시적으로 결정된다.
- note_off 누락으로 sustain block이 생기는 구조를 제거한다.

## 6. Updated Project Positioning

현재 정직한 설명:

> Stage A is a model-serving and MIDI validation prototype. It currently exposes important failure modes in the symbolic generation stack, especially checkpoint handling, note duration decoding, and weak conditioning.

아직 말하면 안 되는 설명:

> Personalized jazz improvisation model that generates usable solo lines.

지금 MVP의 다음 목표:

> Make a tiny symbolic MIDI model produce structurally valid solo-line phrases before optimizing chord awareness or realtime performance.

## 7. Decision

다음 구현은 Stage B로 바로 크게 넘어가기 전에 아래 순서로 한다.

1. Decoder bug fix.
2. Full checkpoint loading path.
3. Tiny overfit smoke.
4. 그 결과를 보고 Stage B tokenization으로 전환.

이 리뷰를 기준으로, 현재 Stage A sweep 결과는 "모델이 어느 정도 된다"가 아니라 "어떤 구조가 실패하는지 드러낸 진단 결과"로 해석한다.
