"""Conditioned MLX forward on a tiny random Aria model (issue #1760). Runs in the Aria venv (mlx); skipped elsewhere."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
try:
    import mlx.core as mx
    import numpy as np
    from aria.inference.model_mlx import TransformerLM
    from aria.model import ModelConfig
except ImportError:  # not the Aria venv
    mx = None
from aria_mlx_cond import COND_DIM, cond_forward, cond_rows, cond_vector  # noqa: E402


def tiny():
    cfg = ModelConfig(d_model=64, n_heads=4, n_layers=3, ff_mult=2, drop_p=0.0, max_seq_len=128, grad_checkpoint=False, vocab_size=50)
    mx.random.seed(0)
    model = TransformerLM(cfg)
    mx.eval(model.parameters())
    model.eval()
    return model, cfg


class PayloadTest(unittest.TestCase):
    def test_vector_and_rows(self):
        self.assertEqual(cond_vector(((1,) + (0,) * 11, 1)), [1.0] + [0.0] * 11 + [1.0])
        rows = cond_rows([-1, 0, 2], {0: ((0,) * 12, 1), 2: ((0, 1) + (0,) * 10, 1)})
        self.assertEqual([r[-1] for r in rows], [0.0, 1.0, 1.0])               # cold, warm-empty, warm
        self.assertEqual(len(rows[0]), COND_DIM)


@unittest.skipIf(mx is None, "needs the Aria venv (mlx)")
class CondForwardTest(unittest.TestCase):
    def setUp(self):
        self.model, self.cfg = tiny()
        self.ids = [int(x) for x in np.random.default_rng(1).integers(0, 50, 40)]

    def run_(self, ids, start, cond, w):
        pos = mx.arange(start, start + len(ids), dtype=mx.int32)
        out = cond_forward(self.model, mx.array([ids], dtype=mx.int32), pos, start, start + len(ids) - 1,
                           mx.array([cond], dtype=mx.float32), w)
        mx.eval(out)
        return np.array(out[0])

    def test_zero_weight_equals_the_upstream_forward(self):
        self.model.setup_cache(batch_size=1, max_seq_len=128)
        pos = mx.arange(0, len(self.ids), dtype=mx.int32)
        ref = self.model(idxs=mx.array([self.ids], dtype=mx.int32), input_pos=pos, offset=0, max_kv_pos=len(self.ids) - 1)
        mx.eval(ref)
        self.model.setup_cache(batch_size=1, max_seq_len=128)
        cond = [[1.0] * COND_DIM] * len(self.ids)
        got = self.run_(self.ids, 0, cond, mx.zeros((self.cfg.d_model, COND_DIM)))
        self.assertLess(float(np.abs(got - np.array(ref[0])).max()), 1e-5)

    def test_nonzero_weight_changes_logits_and_incremental_equals_full(self):
        w = mx.random.normal((self.cfg.d_model, COND_DIM)) * 0.5
        rng = np.random.default_rng(2)
        cond = [list(map(float, rng.integers(0, 2, COND_DIM))) for _ in self.ids]
        self.model.setup_cache(batch_size=1, max_seq_len=128)
        full = self.run_(self.ids, 0, cond, w)
        self.model.setup_cache(batch_size=1, max_seq_len=128)
        zero = self.run_(self.ids, 0, [[0.0] * COND_DIM] * len(self.ids), w)
        self.assertGreater(float(np.abs(full - zero).max()), 1e-2)           # the injection is live
        self.model.setup_cache(batch_size=1, max_seq_len=128)
        inc = [self.run_(self.ids[:10], 0, cond[:10], w)[-1]]
        for i in range(10, len(self.ids)):
            inc.append(self.run_([self.ids[i]], i, [cond[i]], w)[0])
        self.assertLess(float(np.abs(np.stack(inc) - full[9:]).max()), 1e-4)

    def test_rewriting_a_position_with_a_new_condition_equals_a_fresh_prefill(self):
        w = mx.random.normal((self.cfg.d_model, COND_DIM)) * 0.5
        a = [[0.0] * COND_DIM] * len(self.ids)
        b = list(a)
        b[19:] = [[1.0] * COND_DIM] * (len(self.ids) - 19)                     # the condition changes from position 19 on
        self.model.setup_cache(batch_size=1, max_seq_len=128)
        self.run_(self.ids, 0, a, w)                                           # old condition everywhere, then rewrite 19..
        rolled = self.run_(self.ids[19:], 19, b[19:], w)
        self.model.setup_cache(batch_size=1, max_seq_len=128)
        fresh = self.run_(self.ids, 0, b, w)
        self.assertLess(float(np.abs(rolled - fresh[19:]).max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
