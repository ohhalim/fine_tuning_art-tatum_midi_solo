"""
MPS smoke test for the Music Transformer (incl. rpr=True custom attention).

Run this ON YOUR MAC TERMINAL (not inside a sandbox), where Metal is visible:

    cd /Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo
    ./.venv/bin/python scripts/mps_smoke_test.py

It checks, on the MPS device:
  1. model builds and moves to mps
  2. forward pass runs and returns finite logits
  3. loss + backward + optimizer step run (grads are finite)
  4. a short timing comparison vs CPU

If any step raises a "not implemented for MPS" error, that op is the blocker
and we fall back to CPU (or add PYTORCH_ENABLE_MPS_FALLBACK=1). If all pass,
MPS training is safe.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "music_transformer"))
sys.path.insert(0, str(ROOT / "music_transformer" / "third_party"))

from model.music_transformer import MusicTransformer  # noqa: E402
from model.loss import SmoothCrossEntropyLoss  # noqa: E402
from utilities.constants import VOCAB_SIZE, TOKEN_PAD  # noqa: E402
import utilities.device as devmod  # noqa: E402


def build_model(rpr: bool):
    return MusicTransformer(
        n_layers=6, num_heads=8, d_model=512,
        dim_feedforward=1024, max_sequence=512, rpr=rpr,
    )


def one_step(device_str: str, rpr: bool, seq_len: int = 256, batch: int = 2, steps: int = 3):
    """Run `steps` train steps on the given device. Returns (ok, elapsed, msg)."""
    dev = torch.device(device_str)
    torch.manual_seed(0)
    model = build_model(rpr).to(dev)
    model.train()
    loss_fn = SmoothCrossEntropyLoss(0.1, VOCAB_SIZE, ignore_index=TOKEN_PAD)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    x = torch.randint(0, VOCAB_SIZE, (batch, seq_len), device=dev)
    y = torch.randint(0, VOCAB_SIZE, (batch, seq_len), device=dev)

    t0 = time.time()
    last_loss = None
    for _ in range(steps):
        opt.zero_grad()
        out = model(x)                       # forward (calls get_device() internally for mask)
        loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        # finite-grad check
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            return False, time.time() - t0, f"non-finite grads in {len(bad)} params (first: {bad[0]})"
        opt.step()
        last_loss = float(loss.detach().cpu())
    if dev.type == "mps":
        torch.mps.synchronize()
    return True, time.time() - t0, f"final loss={last_loss:.4f}"


def scale_benchmark():
    """Time MPS vs CPU at realistic TRAINING shapes to find the crossover.

    The default smoke shapes (seq 256, batch 2) are tiny — MPS overhead
    dominates there. Real D0-(b) training uses larger shapes where MPS may
    win. This measures steps/sec at a few (batch, seq) points so you can
    pick the device with data, not a guess.
    """
    if not torch.backends.mps.is_available():
        return
    print("\n" + "=" * 64)
    print("SCALE BENCHMARK (rpr=True) — steps/sec, higher is better")
    print(f"{'batch':>5} {'seq':>5} {'MPS s/step':>12} {'CPU s/step':>12} {'winner':>8}")
    print("-" * 64)
    # warm up MPS once (first-ever MPS op pays a one-time compile cost)
    try:
        one_step("mps", True, seq_len=128, batch=1, steps=1)
    except Exception:
        pass
    for batch, seq in [(4, 256), (8, 512), (8, 1024)]:
        row = {}
        for dev in ("mps", "cpu"):
            try:
                if dev == "cpu":
                    devmod.use_cuda(False)
                ok, el, _ = one_step(dev, True, seq_len=seq, batch=batch, steps=4)
                if dev == "cpu":
                    devmod.use_cuda(True)
                row[dev] = el / 4 if ok else None
            except Exception as e:
                if dev == "cpu":
                    devmod.use_cuda(True)
                row[dev] = None
                print(f"  ({dev} raised at batch={batch} seq={seq}: {type(e).__name__})")
        m, c = row.get("mps"), row.get("cpu")
        win = "?" if (m is None or c is None) else ("MPS" if m < c else "CPU")
        ms = f"{m:.3f}" if m else "FAIL"
        cs = f"{c:.3f}" if c else "FAIL"
        print(f"{batch:>5} {seq:>5} {ms:>12} {cs:>12} {win:>8}")
    print("=" * 64)
    print("Use the device that wins at YOUR training shape (batch/seq).")
    print("Rule of thumb: MPS wins as batch*seq grows; CPU wins for tiny probes.")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", action="store_true",
                    help="Also run the MPS-vs-CPU timing sweep at training shapes.")
    cli = ap.parse_args()

    print("=" * 64)
    print("torch", torch.__version__)
    print("mps built    :", torch.backends.mps.is_built())
    print("mps available:", torch.backends.mps.is_available())
    print("get_device() default ->", devmod.get_device())
    print("=" * 64)

    if not torch.backends.mps.is_available():
        print("\nMPS not available in THIS process. If you are running this in a")
        print("normal Mac terminal on macOS 14+/Apple Silicon, check your torch build.")
        print("(Inside the assistant sandbox MPS is intentionally hidden — that is expected.)")
        return 1

    results = {}
    for rpr in (False, True):
        tag = f"rpr={rpr}"
        print(f"\n### {tag} ###")
        # MPS
        try:
            ok, el, msg = one_step("mps", rpr)
            print(f"  MPS : {'OK ' if ok else 'FAIL'}  {el:.2f}s  {msg}")
            results[(tag, "mps")] = (ok, el)
        except Exception as e:
            print(f"  MPS : FAIL  raised: {type(e).__name__}: {e}")
            results[(tag, "mps")] = (False, None)
        # CPU baseline for the same shape
        try:
            devmod.use_cuda(False)               # force get_device()->cpu during CPU run
            ok, el, msg = one_step("cpu", rpr)
            devmod.use_cuda(True)
            print(f"  CPU : {'OK ' if ok else 'FAIL'}  {el:.2f}s  {msg}")
            results[(tag, "cpu")] = (ok, el)
        except Exception as e:
            devmod.use_cuda(True)
            print(f"  CPU : FAIL  raised: {type(e).__name__}: {e}")
            results[(tag, "cpu")] = (False, None)

    # verdict
    print("\n" + "=" * 64)
    mps_ok = all(results.get((f"rpr={r}", "mps"), (False,))[0] for r in (False, True))
    if mps_ok:
        # relative speed at the (tiny) smoke shape — NOT the training shape
        m = results.get(("rpr=True", "mps"), (True, None))[1]
        c = results.get(("rpr=True", "cpu"), (True, None))[1]
        if m and c:
            faster = "MPS" if m < c else "CPU"
            ratio = (c / m) if m < c else (m / c)
            rel = f"At this tiny smoke shape, {faster} is {ratio:.1f}x faster."
        else:
            rel = ""
        print("VERDICT: MPS is COMPATIBLE for rpr=True (no unsupported ops).")
        print(f"         {rel}")
        print("         Small model + small batch => MPS overhead can lose to CPU.")
        print("         Decide per experiment:")
        print("           - tiny probe (D0-a): use --device cpu")
        print("           - full corpus (D0-b): run --scale below, pick the winner")
        print("         Timing at real training shapes:")
        print("           ./.venv/bin/python scripts/mps_smoke_test.py --scale")
    else:
        print("VERDICT: at least one op is not MPS-compatible.")
        print("Options: (a) run on CPU, or (b) retry with env")
        print("         PYTORCH_ENABLE_MPS_FALLBACK=1 ./.venv/bin/python scripts/mps_smoke_test.py")
        print("         (falls unsupported ops back to CPU — slower but works).")
    print("=" * 64)

    if cli.scale and mps_ok:
        scale_benchmark()
    return 0 if mps_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
