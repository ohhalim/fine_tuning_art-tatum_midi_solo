# RunPod Music Transformer QLoRA 학습 가이드

## 준비물
- RunPod 계정 (https://runpod.io)
- 만원 정도 크레딧 (~$7-8)

---

## Step 1: RunPod 인스턴스 생성

1. **RunPod 접속** → Pods → Deploy
2. **GPU 선택**: RTX 3090 (~$0.31/hr) 또는 RTX 4090 (~$0.44/hr)
3. **Template**: PyTorch 2.0+ 선택
4. **Disk**: 20GB 이상
5. **Deploy** 클릭

---

## Step 2: 코드 업로드

### 2-1. GitHub에서 클론

```bash
# RunPod 터미널에서
cd /workspace

git clone https://github.com/ohhalim/fine_tuning_art-tatum_midi_solo.git
cd fine_tuning_art-tatum_midi_solo
git checkout feature/music-transformer-qlora
```

### 2-2. MIDI 데이터셋 업로드

**옵션 A: 로컬에서 업로드** (권장)
- RunPod File Browser 사용
- `midi_dataset/` 폴더를 `/workspace/fine_tuning_art-tatum_midi_solo/`에 업로드

**옵션 B: 압축 후 업로드**
```bash
# 로컬에서
cd /Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo
zip -r midi_dataset.zip midi_dataset/
# RunPod File Browser로 업로드 후
unzip midi_dataset.zip
```

---

## Step 3: 학습 실행

```bash
cd /workspace/fine_tuning_art-tatum_midi_solo
bash scripts/runpod_setup.sh
```

### 자동으로 실행되는 작업:
1. 패키지 설치 (~5분)
2. 데이터 전처리 (~10분)
3. LoRA 학습 3 epochs (~2-3시간)
4. **테스트 샘플 5개 생성**

---

## Step 4: 결과 확인

학습 완료 후:
- `./checkpoints/jazz_lora/` - LoRA 가중치
- `./samples/` - 생성된 MIDI 파일 5개

### MIDI 다운로드
1. RunPod File Browser에서 `samples/` 폴더 다운로드
2. 로컬에서 MIDI 플레이어로 재생

### 평가 기준
- 재즈 스타일이 느껴지는가?
- 멜로디가 자연스러운가?
- 하모니가 어울리는가?

---

## 예상 비용

| 단계 | 시간 | 비용 (RTX 3090) |
|------|------|-----------------|
| 설정 + 전처리 | ~15분 | ~$0.10 |
| 학습 3 epochs | ~2-3시간 | ~$0.60-0.90 |
| 샘플 생성 | ~5분 | ~$0.03 |
| **총합** | ~3시간 | **~$1-1.50** |

---

## 문제 해결

### CUDA 메모리 부족
```bash
# batch_size 줄이기
python scripts/train_qlora.py --batch_size 1 ...
```

### 모듈 없음 오류
```bash
pip install -r requirements.txt
```
