# ReaLJam 로컬 실행과 지연 측정 (#1768)

#1766에서 정한 따라가기 F0입니다. 안나 황 교수 그룹의 ReaLJam을 이 Mac에서 돌리고, 한 번 생성하는 데 걸리는 시간을 잽니다.

## 1. 설치

- 위치: `/Users/ohhalim/git_box/realjam/`
  - 코드: `realchords-pytorch/`, 커밋 `75a63248`(2026-05-11, "Add ReaLJam MLX backend support")
  - 환경: `.venv/`(Python 3.12.9). `requirements_realjam.txt` + `mlx` 0.32.4. torch 2.14.1, onnxruntime 1.21.0, x-transformers 2.11.23
  - 체크포인트: `realchords-pytorch/checkpoints/`, 657 MB. sha256은 `checkpoints_sha256.txt`
- PyPI `realjam` 1.0.1을 쓰지 않은 이유: 체크포인트 주소(GitHub releases `lukewys/realchords` v0.1.0) 5개가 모두 404이고, MLX 백엔드가 들어 있지 않습니다
- 다운로드 승인: 2026-10-16 사용자

## 2. 사전 등록 (실행 전 커밋)

질문: ReaLJam의 실시간 생성 한 번(`generate_live`)이 이 Mac(M1 Max, 32 GB)에서 몇 ms 걸리는가. 우리 템포에서 16분음표마다 새로 고칠 수 있는가.

방법 (`scripts/realjam_live_bench.py`)

- 브라우저 동작을 흉내 냅니다. 매 프레임 `generate_live`를 부르고, 돌려받은 코드 토큰을 세션 목록의 해당 프레임부터 덮어씁니다
- 입력 멜로디: C 장조 음계에서 1~4프레임 길이 음, 쉼표 15%, 시드 0. 음악 품질과 무관한 부하용 입력입니다
- 세션 320프레임(80박). 무음 8박(32프레임). 256프레임을 넘는 마지막 구간에서 문맥 자르기도 지나갑니다
- 모델: GAPT, ReaLchords. 온도 1.0
- 설정(lookahead:commit 프레임): 16:16(화면 기본값 4박:4박), 8:4(2박:1박)
- 백엔드: PyTorch CPU(torch.compile 기본값), MLX, ONNX CPU
- 첫 호출의 오프라인 모델(인트로 코드 채우기)은 한 번만 일어나므로 따로 기록합니다. 나머지 호출에서 처음 3번을 워밍업으로 뺍니다
- 지표: 호출당 시간의 중앙값, p95, 최대

판정 기준

- 우리 128 BPM에서 16분음표는 117.2 ms입니다. ReaLJam 기본 80 BPM에서는 187.5 ms입니다
- **p95 < 117.2 ms**이면 "128 BPM에서 매 16분음표 갱신 가능"으로 적습니다
- 117.2 ≤ p95 < 187.5 ms이면 "80 BPM에서만 매 프레임 갱신 가능"으로 적습니다
- 그 이상이면 "매 프레임 갱신 불가. 갱신 간격 = p95"로 적습니다
- 백엔드끼리 순위를 매기지 않습니다. 기준을 넘는 백엔드가 있는지만 봅니다

하지 않는 것

- 음악 품질 판단. 사용자 직접 연주 소감은 따로 적습니다
- 브라우저·네트워크 지연 측정. 이번에는 모델 호출 시간만 잽니다

## 3. 결과

(실행 후 채움)
