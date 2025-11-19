# Phase 5: 생성 및 배포 🎼

**목표**: 최종 모델로 다양한 모드로 음악을 생성하고 데모를 준비합니다.

**예상 시간**: 2-3일
**난이도**: ⭐⭐☆☆☆

---

## 📋 체크리스트

- [ ] 4가지 생성 모드 테스트
- [ ] 고품질 샘플 10-20개 생성
- [ ] MIDI → MP3 변환
- [ ] 최고 샘플 선별
- [ ] 데모 준비

---

## 1. 생성 모드

TatumFlow는 4가지 생성 모드를 지원합니다:

### Mode 1: Continuation (이어가기)

**설명**: 주어진 멜로디를 이어서 연주

**사용 예**:
```bash
python scripts/generate_music.py \
  --checkpoint checkpoints/production/best.pt \
  --mode continuation \
  --input data/samples/tea_for_two_intro.mid \
  --output outputs/tea_for_two_continuation.mid \
  --max_length 512
```

**활용**:
- 짧은 모티브 → 전체 곡
- 인트로만 있는 곡 완성
- 작곡 아이디어 확장

### Mode 2: Style Transfer (스타일 변환)

**설명**: 다른 곡을 Art Tatum 스타일로 변환

**사용 예**:
```bash
python scripts/generate_music.py \
  --checkpoint checkpoints/production/best.pt \
  --mode style_transfer \
  --input data/samples/beethoven_moonlight.mid \
  --output outputs/moonlight_tatum_style.mid \
  --style_strength 0.7
```

**활용**:
- 클래식 → 재즈화
- 다른 재즈 피아니스트 → Tatum 스타일
- 팝송 → 재즈 편곡

### Mode 3: Improvise (즉흥 연주)

**설명**: 완전히 새로운 곡 생성

**사용 예**:
```bash
python scripts/generate_music.py \
  --checkpoint checkpoints/production/best.pt \
  --mode improvise \
  --output outputs/new_improvisation.mid \
  --tempo 180 \
  --key "C" \
  --length 1024
```

**활용**:
- 새로운 즉흥 연주
- 배경 음악 생성
- 무한 재즈 BGM

### Mode 4: Theory Editing (음악 이론 조작)

**설명**: 화성/리듬/다이나믹스를 수동으로 제어

**사용 예**:
```bash
python scripts/generate_music.py \
  --checkpoint checkpoints/production/best.pt \
  --mode theory_editing \
  --input data/samples/original.mid \
  --output outputs/edited.mid \
  --harmony_shift 0.5 \  # 화성 변화
  --rhythm_factor 1.2 \  # 리듬 빠르게
  --dynamics_boost 0.3   # 다이나믹스 증가
```

**활용**:
- 화성 재조화 (Reharmonization)
- 리듬 변주
- 감정 조절 (부드럽게/격렬하게)

---

## 2. 고품질 샘플 생성

### 생성 파라미터 튜닝

#### Temperature

**낮음 (0.7)**:
- 안전하고 예측 가능
- 반복 많음
- 기술적으로 정확

**높음 (1.2)**:
- 창의적이고 다양
- 가끔 이상한 음
- 즉흥적

**권장**: 0.9-1.0 (균형)

#### Top-p (Nucleus Sampling)

**작음 (0.8)**:
- 안전한 음만 선택
- 지루할 수 있음

**큼 (0.95)**:
- 다양한 음 허용
- 더 흥미로움

**권장**: 0.9

### 대량 생성

```bash
# 20개 샘플 생성 (다양한 설정)
python scripts/phase5_generate_samples.py \
  --checkpoint checkpoints/production/best.pt \
  --num_samples 20 \
  --output_dir outputs/final_samples \
  --temperature 0.9 \
  --top_p 0.9
```

---

## 3. MIDI → MP3 변환

### FluidSynth 사용

```bash
# SoundFont 다운로드 (피아노 음색)
wget https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SF2-V3+20200602.tar.xz
tar -xf SalamanderGrandPiano-SF2-V3+20200602.tar.xz

# 변환
python scripts/phase5_midi_to_mp3.py \
  --input outputs/final_samples/*.mid \
  --soundfont SalamanderGrandPiano.sf2 \
  --output_dir outputs/final_samples_mp3
```

### 고품질 설정

```python
# scripts/phase5_midi_to_mp3.py
from midi2audio import FluidSynth

fs = FluidSynth(
    sound_font='SalamanderGrandPiano.sf2',
    sample_rate=48000  # CD 품질
)
fs.midi_to_audio('input.mid', 'output.wav')

# WAV → MP3 (고품질)
import subprocess
subprocess.run([
    'ffmpeg', '-i', 'output.wav',
    '-codec:a', 'libmp3lame',
    '-qscale:a', '0',  # 최고 품질
    'output.mp3'
])
```

---

## 4. 샘플 선별

### 자동 필터링

```bash
python scripts/filter_samples.py \
  --input_dir outputs/final_samples \
  --output_dir outputs/best_samples \
  --min_coherence 0.7 \
  --min_musical_score 3.5
```

### 수동 선별 기준

**반드시 포함**:
1. ✅ 가장 Art Tatum 느낌나는 것
2. ✅ 기술적으로 완벽한 것
3. ✅ 음악적으로 흥미로운 것

**제외**:
- ❌ 너무 반복적인 것
- ❌ 이상한 음 있는 것
- ❌ 리듬 불안정한 것

**목표**: 10-15개의 완벽한 샘플

---

## 5. 데모 제작

### A. YouTube 데모 영상

**구성**:
1. 인트로 (10초)
   - "Art Tatum AI - TatumFlow"
   - 프로젝트 소개
2. 샘플 1-3 (각 30초)
   - Continuation, Style Transfer, Improvise
   - 화면: MIDI 시각화 (pianoroll)
3. 기술 설명 (20초)
   - "Latent Diffusion + Music Theory Disentanglement"
   - 아키텍처 다이어그램
4. 아웃트로 (10초)
   - GitHub 링크
   - "Made with TatumFlow"

**도구**:
- **MIDI 시각화**: https://github.com/craffel/pretty-midi#visualization
- **영상 편집**: DaVinci Resolve (무료)
- **음악**: 생성한 MP3

**길이**: 1-2분 (짧게!)

### B. SoundCloud/YouTube Music

고품질 MP3를 업로드:

```bash
# 메타데이터 추가
ffmpeg -i input.mp3 \
  -metadata title="Art Tatum AI - Improvisation #1" \
  -metadata artist="TatumFlow" \
  -metadata album="AI Jazz Collection" \
  output_with_metadata.mp3
```

업로드:
- SoundCloud: https://soundcloud.com
- YouTube Music
- Bandcamp (선택)

### C. Interactive Demo (HuggingFace Spaces)

```python
# app.py (Gradio)
import gradio as gr
from tatumflow import TatumFlowGenerator

generator = TatumFlowGenerator('checkpoints/best.pt')

def generate(mode, midi_file, temperature):
    output = generator.generate(
        mode=mode,
        input_file=midi_file,
        temperature=temperature
    )
    return output

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Dropdown(["continuation", "style_transfer", "improvise"]),
        gr.File(label="Input MIDI (optional)"),
        gr.Slider(0.5, 1.5, value=0.9, label="Temperature")
    ],
    outputs=gr.Audio(label="Generated Music"),
    title="TatumFlow: Art Tatum AI",
    description="Generate jazz piano in the style of Art Tatum"
)

demo.launch()
```

배포:
```bash
git push https://huggingface.co/spaces/YOUR_NAME/tatumflow
```

---

## 🎓 학습 내용

### Sampling 전략

**Greedy Sampling**:
- 항상 확률 최대 선택
- 안전하지만 지루

**Temperature Sampling**:
- 확률 분포 조정
- 다양성 증가

**Top-p (Nucleus)**:
- 상위 p% 확률만 고려
- 극단적 선택 방지

**권장**: Temperature + Top-p 조합

### MIDI vs Audio 생성

**MIDI 장점**:
- ✅ 파일 작음 (KB)
- ✅ 음 하나하나 제어 가능
- ✅ 악기 변경 쉬움

**Audio 장점**:
- ✅ 바로 재생 가능
- ✅ 실제 음색 표현

**TatumFlow**: MIDI 생성 → 고품질 Audio 변환

---

## ✅ Phase 5 완료 체크

- [ ] 4가지 모드 모두 테스트
- [ ] 20+ 샘플 생성
- [ ] 최고 샘플 10-15개 선별
- [ ] 모든 샘플 MP3 변환
- [ ] 데모 준비 (YouTube/HuggingFace)

---

## 다음 단계

**Phase 6: 포트폴리오화**로 이동:
```bash
cat docs/phase6_portfolio.md
```

**축하합니다! 이제 세상에 공유할 준비가 되었습니다! 🌟**
