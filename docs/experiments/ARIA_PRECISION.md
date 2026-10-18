# Aria 가중치 정밀도별 토큰당 생성 시간과 출력 분포 차이 (사전 등록)

작성 2026-10-18 17시대 KST. 이슈 #1811. 학습, 다운로드, 실시간 재실행, 청취 없음. `quality_claimed: false`.

## 1. 질문

- Aria 가중치를 bf16, 8비트, 4비트로 바꿀 때 현행 float32 대비 토큰당 생성 시간과 다음 토큰 분포 차이

## 2. 배경

- #1807 재실행 A: 블록 마감 여유 최소 35 ms, 중앙값 151 ms. 블록 4 기준 27토큰 생성 273.5 ms, 토큰당 약 10 ms
- #1797 블록 4: 마감 초과, 지평까지 생성 구간이 대부분(토큰 약 50개는 추정)
- #1809: 원하는 릭의 2박 창 최대 11음. 밀도 상한만으로 마감 여유 확보 여부 미확인
- 현행 실시간 경로(`lh_kv_reuse.load_mlx`, `rt_producer.py`): 가중치 float32(2.63 GB, 6.59억 파라미터), KV 캐시 float32
- Aria 공식 MLX 데모(`demo/demo_mlx.py`): `DTYPE = mx.bfloat16`, `--quantize` 시 `nn.quantize(model.model, group_size=32, bits=8)`
- 이 저장소에 정밀도·양자화 측정 기록 없음

## 3. 방법 (실행 전 고정)

- 체크포인트: `t3_aria/ckpt/model-gen.safetensors`(#1807과 같음), MLX 0.25.2, Aria venv
- 팔 4개. 가중치와 KV 캐시 dtype 같이 변경, 양자화 범위는 데모와 같은 `model.model`(임베딩 포함, lm_head 제외)

| 팔 | 가중치 | KV 캐시 | 양자화 |
|---|---|---|---|
| fp32 | float32 | float32 | 없음(현행) |
| bf16 | bfloat16 | bfloat16 | 없음(데모 기본) |
| q8 | bfloat16 | bfloat16 | 8비트, 그룹 32(데모 `--quantize`) |
| q4 | bfloat16 | bfloat16 | 4비트, 그룹 32 |

- 입력: `mehldau_aria/data.json`의 test 녹음 첫 창(각 1,023토큰, 실제 Mehldau 토큰)
- 시간 측정: producer와 같은 forward 호출(모델 호출, `mx.eval`, logits numpy 변환)을 `perf_counter_ns`로 측정. 샘플링·문법 마스크의 Python 비용은 제외
  - 문맥 L = 256, 512, 896. 첫 test 창 앞 L토큰을 미측정 프리필 후, 실제 다음 토큰 32개를 한 개씩 디코드(위치 L..L+31), 이어서 48토큰 suffix 프리필(#1807 블록 S 46 안팎) 1회
  - 팔마다 프리필 1회와 디코드 8스텝 예열. 반복 5회, 반복마다 팔 순서를 한 칸씩 회전, 네 팔을 같은 프로세스에서 교차 측정
  - 반복 시작과 끝 열 압력(`notifyutil`) 기록
- 분포 차이: test 창 전부(10개)를 팔마다 한 번에 forward. 위치마다 fp32 대비 KL(fp32‖팔), TV 거리, top-1 일치, 실제 다음 토큰 NLL
  - 실시간 경로는 온도 1 전체 분포 샘플링(`masked_pick`)이라 top-1보다 KL·TV를 주 지표로 사용

## 4. 판정 기준 (실행 전 고정)

- 무결성(미충족 시 측정 무효)
  - 모든 팔 logits 유한
  - fp32 같은 창 두 번 forward 최대 절대 차이 ≤ 1e-5
  - 열 압력 0이 아닌 반복은 표시하고 시간 판정에서 제외. 남은 반복이 3회 미만이면 시간 판정 보류
- 속도: 문맥 512 디코드 중앙값의 fp32 대비 비율. 0.75 이하면 속도 이득
- 분포 차이: 위치 평균 KL과 평균 TV
  - KL ≤ 0.02 nats, TV ≤ 0.05: 분포 차이 작음
  - KL ≤ 0.1: 청취 비교 필요
  - KL > 0.1: 실시간 후보 제외
  - 임계값은 보정 근거 없는 사전 선택. bf16(데모 기본) 값을 기준점으로 함께 보고
- 다음 단위 후보: 속도 이득이면서 분포 차이 작은 팔 중 가장 빠른 팔. 없으면 속도 이득이면서 청취 비교 필요인 팔 중 가장 빠른 팔을 청취 비교 단위로
- NLL 변화는 기록만, 판정 없음

## 5. 범위 밖

- 실시간 경로 기본값 변경, 실시간 재실행, 청취, 학습
- 정밀도별 KV 재사용 동등성(#1793 캐시 suffix 대 전체 프리필 비교), `mx.compile`, LoRA 어댑터 양자화
- 샘플 출력의 음악 통계 비교. 후보 팔이 나오면 다음 단위

## 6. 재현

- `scripts/aria_precision_probe.py <aria repo> <ckpt> <data json> <report json>`
- 시험 `tests/test_aria_precision_probe.py`
- 보고 `docs/experiments/aria_precision/report.json`
