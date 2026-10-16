"""ReaLJam 서버를 127.0.0.1에만 띄운다 (#1768).

공식 server.main()은 0.0.0.0에 바인딩해서 같은 네트워크의 다른 기기에도 열린다.
여기서는 같은 Flask 앱과 Agent를 쓰되 루프백에만 연다.
브라우저 주소는 http://127.0.0.1:<port> 로 연다(호스트명이 localhost가 아니어야
화면 기본값이 논문 설정 80 BPM, lookahead 4박, commit 4박이 된다).

사용 예:
  cd /Users/ohhalim/git_box/realjam/realchords-pytorch
  PYTHONPATH=. WANDB_MODE=disabled ../.venv/bin/python <this> --backend mlx
"""

import argparse

from realchords.realjam import agent_interface, server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "mlx", "onnx"], default="mlx")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    server.agent = agent_interface.Agent(
        onnx=args.backend == "onnx",
        mlx=args.backend == "mlx",
    )
    print(f"ReaLJam: http://127.0.0.1:{args.port}", flush=True)
    server.app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
