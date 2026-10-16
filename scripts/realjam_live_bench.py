"""ReaLJam generate_live 호출 지연을 이 Mac에서 잰다 (#1768).

realchords-pytorch 저장소 루트에서 실행해야 한다(체크포인트 경로가 cwd 기준).
브라우저가 하는 일을 흉내 낸다: 매 프레임 generate_live를 부르고, 돌려받은
코드 토큰을 세션 토큰 목록의 frame 위치부터 덮어쓴다.

사용 예:
  cd /Users/ohhalim/git_box/realjam/realchords-pytorch
  PYTHONPATH=. ../.venv/bin/python <this> --backend torch --out ../results/torch.json
"""

import argparse
import json
import platform
import random
import statistics
import subprocess
import time

import numpy as np
import torch

from realchords.realjam import agent_interface

FPB = 4  # 박당 프레임(16분음표)


def make_melody(n_frames, seed):
    """결정적 가짜 멜로디: C 장조 음계에서 1~4프레임 길이 음과 쉼표."""
    rng = random.Random(seed)
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    events = []
    f = 0
    while f < n_frames:
        dur = rng.choice([1, 2, 2, 4])
        if rng.random() < 0.15:
            f += dur
            continue
        p = rng.choice(scale)
        events.append({"pitch": p, "frame": f, "on": True})
        events.append({"pitch": p, "frame": f + dur, "on": False})
        f += dur
    return events


def notes_until(events, frame):
    """요청 frame 직전까지 브라우저가 알고 있는 이벤트만 남긴다.
    브라우저는 curFrame + 1을 요청 frame으로 보내므로 frame 미만만 안다."""
    return [e for e in events if e["frame"] < frame]


def run_session(agent, model, la, ca, silence, n_frames, temp, events):
    chord_tokens = []
    intro_set = False
    rows = []
    for frame in range(1, n_frames + 1):
        t0 = time.perf_counter()
        new_chords, new_tokens, intro = agent.generate_live(
            model,
            notes_until(events, frame),
            list(chord_tokens),
            frame,
            la,
            ca,
            temp,
            silence,
            intro_set,
        )
        dt = (time.perf_counter() - t0) * 1000
        if not new_tokens and intro is None:
            continue  # 아직 생성 시작 전(무음 구간)
        if intro is not None:
            chord_tokens[: len(intro)] = intro
            intro_set = True
        if len(chord_tokens) < frame:
            chord_tokens.extend([-1] * (frame - len(chord_tokens)))
        for i, tok in enumerate(new_tokens):
            j = frame + i
            if j < len(chord_tokens):
                chord_tokens[j] = tok
            else:
                chord_tokens.append(tok)
        rows.append({"frame": frame, "ms": dt, "intro": intro is not None,
                     "n_new": len(new_tokens)})
    return rows


def summarize(ms):
    a = np.array(ms)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "mean": float(a.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "mlx", "onnx"], required=True)
    ap.add_argument("--models", default="GAPT,ReaLchords")
    ap.add_argument("--settings", default="16:16,8:4",
                    help="lookahead:commit 프레임 쌍 목록")
    ap.add_argument("--silence_beats", type=int, default=8)
    ap.add_argument("--frames", type=int, default=320)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.perf_counter()
    agent = agent_interface.Agent(
        onnx=args.backend == "onnx",
        mlx=args.backend == "mlx",
        compile=not args.no_compile,
        auto_download=False,
    )
    load_s = time.perf_counter() - t0
    events = make_melody(args.frames + 32, args.seed)
    silence = args.silence_beats * FPB

    out = {
        "backend": args.backend,
        "compile": not args.no_compile,
        "load_s": load_s,
        "machine": platform.platform(),
        "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip(),
        "torch": torch.__version__,
        "args": vars(args),
        "runs": [],
    }
    for model in args.models.split(","):
        for s in args.settings.split(","):
            la, ca = (int(x) for x in s.split(":"))
            rows = run_session(agent, model, la, ca, silence, args.frames,
                               args.temperature, events)
            intro = [r for r in rows if r["intro"]]
            steady = [r for r in rows if not r["intro"]][args.warmup:]
            out["runs"].append({
                "model": model, "lookahead": la, "commit": ca,
                "intro_ms": intro[0]["ms"] if intro else None,
                "steady": summarize([r["ms"] for r in steady]),
                "steady_last64": summarize([r["ms"] for r in steady[-64:]]),
                "rows": rows,
            })
            st = out["runs"][-1]["steady"]
            print(f"{args.backend} {model} la={la} ca={ca} "
                  f"median={st['median']:.1f}ms p95={st['p95']:.1f}ms "
                  f"max={st['max']:.1f}ms n={st['n']}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
