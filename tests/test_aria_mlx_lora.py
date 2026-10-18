"""Adapter parsing, merge math and the delta wrapper of #1834 (numpy; the wrapper tests need mlx and skip without it)."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import aria_mlx_lora as al  # noqa: E402

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover
    mx = None


def sd(n_layers=2, r=4, d=8, rng=np.random.default_rng(0)):
    out = {}
    for i in range(n_layers):
        out[f"{i}_mixed_qkv.A"] = rng.normal(size=(r, d)).astype(np.float32)
        out[f"{i}_mixed_qkv.B"] = rng.normal(size=(3 * d, r)).astype(np.float32)
        out[f"{i}_att_proj_linear.A"] = rng.normal(size=(r, d)).astype(np.float32)
        out[f"{i}_att_proj_linear.B"] = rng.normal(size=(d, r)).astype(np.float32)
    return out


class ParseTest(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(al.parse_key("12_mixed_qkv.A"), (12, "mixed_qkv", "A"))
        self.assertEqual(al.parse_key("0_att_proj_linear.B"), (0, "att_proj_linear", "B"))
        for bad in ("0_ff_up_proj.A", "x_mixed_qkv.A", "0_mixed_qkv.C"):
            with self.assertRaises(ValueError):
                al.parse_key(bad)

    def test_state_dict_and_shape_checks(self):
        ad = al.from_state_dict(sd())
        self.assertEqual(len(ad), 4)
        al.check_against(ad, 2, {"mixed_qkv": (24, 8), "att_proj_linear": (8, 8)})
        with self.assertRaises(ValueError):
            al.check_against(ad, 3, {"mixed_qkv": (24, 8), "att_proj_linear": (8, 8)})
        with self.assertRaises(ValueError):
            al.check_against(ad, 2, {"mixed_qkv": (16, 8), "att_proj_linear": (8, 8)})
        broken = sd()
        del broken["1_att_proj_linear.B"]
        with self.assertRaises(ValueError):
            al.from_state_dict(broken)
        mixed = sd()
        mixed["0_mixed_qkv.A"] = np.zeros((2, 8), np.float32)
        with self.assertRaises(ValueError):
            al.from_state_dict(mixed)

    def test_merge_equals_hook_formula(self):
        rng = np.random.default_rng(1)
        w = rng.normal(size=(24, 8)).astype(np.float32)
        a, b = rng.normal(size=(4, 8)).astype(np.float32), rng.normal(size=(24, 4)).astype(np.float32)
        x = rng.normal(size=(5, 8)).astype(np.float32)
        hook = x @ w.T + (x @ a.T @ b.T) * 0.5
        self.assertTrue(np.allclose(x @ al.merged(w, a, b, 0.5).T, hook, atol=1e-4))


@unittest.skipIf(mx is None, "mlx not installed")
class WrapperTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.w = rng.normal(size=(64, 32)).astype(np.float32)
        self.a = (0.1 * rng.normal(size=(4, 32))).astype(np.float32)
        self.b = (0.1 * rng.normal(size=(64, 4))).astype(np.float32)
        self.x = mx.array(rng.normal(size=(3, 32)).astype(np.float32))
        self.lin = nn.Linear(32, 64, bias=False)
        self.lin.weight = mx.array(self.w)

    def test_zero_b_is_exactly_the_base(self):
        L = al.make_lora_linear()
        wrapped = L(self.lin, self.a, np.zeros_like(self.b), 1.0, mx.float32)
        self.assertTrue(np.array_equal(np.array(wrapped(self.x)), np.array(self.lin(self.x))))

    def test_delta_matches_merge_in_float32(self):
        L = al.make_lora_linear()
        delta = np.array(L(self.lin, self.a, self.b, 1.0, mx.float32)(self.x))
        m = al.build_merged({(0, "mixed_qkv"): self.w}, {(0, "mixed_qkv"): (self.a, self.b)}, 1.0, mx.float32)[(0, "mixed_qkv")]
        self.assertTrue(np.allclose(delta, np.array(m(self.x)), atol=1e-4))

    def test_quantized_delta_and_merge_are_close(self):
        L = al.make_lora_linear()
        q = nn.QuantizedLinear.from_linear(self.lin, group_size=32, bits=8)
        delta = np.array(L(q, self.a, self.b, 1.0, mx.float32)(self.x))
        m = al.build_merged({(0, "t"): self.w}, {(0, "t"): (self.a, self.b)}, 1.0, mx.float32, bits=8)[(0, "t")]
        ref = np.array(self.x) @ al.merged(self.w, self.a, self.b, 1.0).T
        self.assertLess(np.abs(delta - ref).max(), 0.05 * np.abs(ref).max())
        self.assertLess(np.abs(np.array(m(self.x)) - ref).max(), 0.05 * np.abs(ref).max())


if __name__ == "__main__":
    unittest.main()
