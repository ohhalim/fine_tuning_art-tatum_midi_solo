# Aria LoRA 1 step smoke (이슈 #1724)

작성 2026-10-11. **자원 확인만.** 학습 효과·음악 품질 주장 없음.

## 요약

**된다.** 토크나이저와 base를 그대로 두고 전 층 LoRA(r16, 2,359,296개)를 붙여 이 Mac(M1 Max 32 GB, MPS)에서 한 step 학습했다. 두 팔 모두 미리 정한 확인 항목 13개를 전부 통과했다.

- 1024토큰 한 step의 관측 최대 MPS 메모리는 gradient checkpoint를 **켜면 4.12 GiB, 끄면 7.12 GiB**다. base 가중치만 올린 상태는 3.01 GiB다
- 두 팔의 loss는 같고(2.9404), 층별 B grad 크기의 상대 차이는 최대 1.3e-06이다. checkpoint는 계산 결과를 사실상 그대로 두고(재계산에 따른 부동소수 차이만) 메모리를 줄였다
- 앞선 추정(3–4 / 6–9 GiB)보다 약간 크다. 이번 값이 실측이다

## 고정한 설정 (실행 전 커밋 `d047cc87`)

- 입력: Tatum train98을 정렬한 첫 곡 "Come Rain Or Come Shine"(fresh12·val12 아님, 같은 제목의 다른 녹음 없음). AbsTokenizer 그대로, global random seed 0, 앞 1024토큰. 학습 step은 앞 1023토큰으로 다음 토큰을 맞힌다
- 모델: `model-gen.safetensors`(sha256 `5e6b07ee2680f004`), config medium, F32, MPS, batch 1
- LoRA: r16, alpha 16, `mixed_qkv`(qkv가 하나로 합쳐진 1536→4608)와 `att_proj_linear` 16층 = 32쌍. A는 seed 0 균등분포, B는 0. 모델 밖에 두고 forward hook으로 더한다. Adam lr 1e-4, 정확히 1 step
- 두 팔: 둘 다 train 모드, **gradient checkpoint만 켬/끔**. 팔마다 별도 프로세스, 같은 LoRA 초기값(sha `fbcb55fe1b9dbe54`)·입력(sha `19c8c6f9d1206cd3`)·seed, 새 optimizer
- 임베딩 출력에 requires_grad를 거는 훅을 **두 팔 모두** 둔다. Aria의 checkpoint는 `use_reentrant=True`다. PyTorch의 reentrant checkpoint는 입력에 requires_grad가 없으면 블록 안 파라미터로 기울기를 보내지 않는 것으로 알려져 있다. 훅 없이 돌려 보는 시험은 하지 않았다. 훅은 임베딩 가중치를 학습시키지 않는다(base grad 전부 None으로 확인)
- dropout은 구조상 0이다(`resid_dropout` 0, SDPA에 dropout 없음). 그래서 train 모드라도 두 팔의 무작위성 차이가 없다
- logits 검산은 eval·no_grad·같은 기기·F32, 허용 오차 0(실행 전 고정)

## 결과

| checkpoint | 결과 | loss | forward (s) | backward (s) | optimizer (s) | 로드 후 MPS (GiB) | step 관측 최대 (GiB) |
|---|---|---|---|---|---|---|---|
| 끔 | 통과 | 2.9404 | 1.4696 | 0.7398 | 0.0373 | 3.01 | **7.12** |
| 켬 | 통과 | 2.9404 | 0.312 | 0.744 | 0.0336 | 3.01 | **4.12** |

| 확인 | 끔 | 켬 |
|---|---|---|
| MPS CPU 대체 꺼짐 | ✓ | ✓ |
| 1024토큰 창 확보(곡 전체 6,856토큰) | ✓ | ✓ |
| dropout 구조상 0 | ✓ | ✓ |
| LoRA 32쌍·2,359,296개 | ✓ | ✓ |
| 초기(B=0) logits = base, 차이 0 | ✓ | ✓ |
| optimizer에 LoRA 64개 텐서만 | ✓ | ✓ |
| 32쌍 각각 B grad 유한·0 아님 | ✓ | ✓ |
| A grad는 전부 0(B=0이라 정상) | ✓ | ✓ |
| base 파라미터 grad 전부 None | ✓ | ✓ |
| step 뒤 32쌍 모두 B 변화 | ✓ | ✓ |
| base 해시 불변 | ✓ | ✓ |
| LoRA 끈 logits가 처음과 동일, 차이 0 | ✓ | ✓ |
| LoRA 저장 → 재로드 logits 차이 0 | ✓ | ✓ |

**층별 B grad** (원자료 `aria_lora_smoke/ckpt_*.json`의 `layers`)
- 32쌍 모두 유한하고 0이 아니다. 가장 작은 쌍은 0번 층 `mixed_qkv`(0.0165), 가장 큰 쌍은 4번 층 `mixed_qkv`(0.5009)다. 전체 norm 하나로 묶지 않고 쌍마다 확인했다
- A grad는 32쌍 모두 정확히 0이다. B가 0에서 시작하므로 첫 step에서는 정상이다. 두 번째 step부터 A가 움직인다(이번에는 보지 않았다)

## 그대로 읽으면 안 되는 점

- **메모리는 관측 최대값이다.** MPS에는 진짜 peak API가 없다(`torch.mps`에는 `current_allocated_memory`, `driver_allocated_memory`, `recommended_max_memory`만 있음). 2 ms마다 `driver_allocated_memory`를 읽은 최대값이라, 그 사이의 순간 최대는 놓쳤을 수 있다
- **시간은 한 번 잰 첫 train step이다.** 끔 팔의 forward 1.47초는 켬 팔(0.31초)보다 오히려 길다. 활성값을 보관할 메모리를 처음 확보하는 비용이 섞였을 가능성이 크다(확인 안 함). 처리량 추정에 쓰지 않는다
- 두 팔이 저장한 LoRA 파일 해시는 다르다. checkpoint의 재계산 때문에 B grad가 1e-6 수준으로 달라서다. 각 팔 안에서의 재로드는 차이 0이다
- 한 곡의 1024토큰 한 창이다. 더 긴 창, batch 2 이상, 여러 step의 메모리는 재지 않았다

## 결론과 다음

- 전 층 LoRA 적응은 이 장비에서 자원상 가능하다. checkpoint를 켜면 base 대비 약 1.1 GiB만 더 쓴다
- 공식 경로에 LoRA가 없어 직접 붙였다. forward hook 방식은 base state_dict를 그대로 두므로 base 해시 비교와 어댑터 교체(층 2의 "base 1개 + 어댑터 스왑")에 맞는다
- 다음 단위(스타일 적응 사전 등록)는 아스트라와 정한다. 이번 결과로 본학습을 자동으로 시작하지 않는다

## 재현

- `scripts/aria_lora_smoke.py <aria repo> <checkpoint> <midi> <out dir> --ckpt on|off` (t3_aria `.venv`, 팔당 약 15초)
- LoRA 파일은 `/Users/ohhalim/git_box/aria_lora_smoke/ckpt_{on,off}/lora_step1_smoke_only.pt`에만 둔다. smoke 전용이며 학습에 다시 쓰지 않는다
