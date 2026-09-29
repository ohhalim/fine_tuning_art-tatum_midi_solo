"""AdapterBank: one base, merged adapters swapped in place (docs/experiments/ADAPTER_SWAP.md)."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "music_transformer"))

from utilities.device import use_cuda

use_cuda(False)

from model.music_transformer import MusicTransformer
from scripts.adapter_bank import AdapterBank, LiveAdapterSelector, adapter_for_bar, parse_schedule
from scripts.train_qlora import add_lora_to_model, merge_lora_for_inference

TINY = dict(n_layers=2, num_heads=2, d_model=16, dim_feedforward=32, max_sequence=48, rpr=True)


def base(seed=0):
    torch.manual_seed(seed)
    return MusicTransformer(**TINY).eval()


def adapted(base_model, targets, seed):
    model = copy.deepcopy(base_model)
    model, _ = add_lora_to_model(model, r=2, alpha=4, dropout=0.0, targets=targets)
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.2)
    model.eval()
    reference = copy.deepcopy(model)
    merge_lora_for_inference(model)
    return model, reference


class AdapterBankTest(unittest.TestCase):
    def test_swap_reproduces_each_adapter_exactly(self) -> None:
        b = base()
        a_model, a_ref = adapted(b, ("out_proj", "qkv"), seed=1)
        m_model, m_ref = adapted(b, ("out_proj",), seed=2)
        bank = AdapterBank({"tatum": a_model, "mehldau": m_model})
        x = torch.randint(0, 388, (1, 24))
        with torch.no_grad():
            for name, ref in (("mehldau", m_ref), ("tatum", a_ref), ("mehldau", m_ref)):
                self.assertTrue(bank.select(name))
                self.assertLess((bank.model(x) - ref(x)).abs().max().item(), 1e-5)
        self.assertFalse(bank.select("mehldau"))
        self.assertEqual(len(bank.swap_ms), 3)
        self.assertTrue(bank.shared_base)
        self.assertTrue(all(k.endswith(("in_proj_weight", "out_proj.weight")) for k in bank.swap_keys))

    def test_different_bases_are_refused(self) -> None:
        a_model, _ = adapted(base(0), ("out_proj",), seed=1)
        m_model, _ = adapted(base(5), ("out_proj",), seed=2)
        with self.assertRaisesRegex(ValueError, "do not share a base"):
            AdapterBank({"a": a_model, "b": m_model})

    def test_different_bases_opt_in_swaps_whole_model(self) -> None:
        a_model, a_ref = adapted(base(0), ("out_proj",), seed=1)
        m_model, m_ref = adapted(base(5), ("out_proj", "qkv"), seed=2)
        bank = AdapterBank({"a": a_model, "b": m_model}, allow_different_bases=True)
        self.assertFalse(bank.shared_base)
        x = torch.randint(0, 388, (1, 24))
        with torch.no_grad():
            for name, ref in (("b", m_ref), ("a", a_ref)):
                bank.select(name)
                self.assertLess((bank.model(x) - ref(x)).abs().max().item(), 1e-5)

    def test_unmerged_adapter_is_refused(self) -> None:
        b = base()
        _, a_ref = adapted(b, ("out_proj",), seed=1)
        with self.assertRaisesRegex(ValueError, "not merged"):
            AdapterBank({"a": a_ref})

    def test_schedule(self) -> None:
        sched = parse_schedule("tatum:4,mehldau:2", ["tatum", "mehldau"])
        self.assertEqual([adapter_for_bar(sched, i) for i in range(8)],
                         ["tatum"] * 4 + ["mehldau"] * 2 + ["tatum"] * 2)
        with self.assertRaises(ValueError):
            parse_schedule("bill:4", ["tatum"])


class LiveSelectorTest(unittest.TestCase):
    def _ev(self, ns, msg):
        from inference.realtime.continuous import TimedInputMessage
        return TimedInputMessage(received_ns=ns, message=msg)

    def test_program_change_selects_and_persists(self) -> None:
        import mido
        sel = LiveAdapterSelector(["tatum", "mehldau"], "program")
        notes = [self._ev(1, mido.Message("note_on", note=60, velocity=90))]
        self.assertEqual(sel.update(notes), "tatum")
        window = notes + [self._ev(2, mido.Message("program_change", program=1))]
        self.assertEqual(sel.update(window), "mehldau")
        self.assertEqual(sel.update(window), "mehldau")   # same snapshot again: no double count
        self.assertEqual(sel.update([]), "mehldau")        # window forgot it: choice persists
        self.assertEqual(sel.update([self._ev(3, mido.Message("program_change", program=7))]), "mehldau")
        self.assertEqual(sel.update([self._ev(4, mido.Message("program_change", program=0))]), "tatum")
        self.assertEqual([e["selected"] for e in sel.events], ["mehldau", None, "tatum"])

    def test_records_consumer_noop_and_superseded(self) -> None:
        import mido
        sel = LiveAdapterSelector(["tatum", "mehldau"], "program")
        pc = lambda ns, p: self._ev(ns, mido.Message("program_change", program=p))
        sel.update([pc(1, 1)], block_index=5)
        sel.update([pc(2, 1)], block_index=6)                 # already mehldau -> noop
        sel.update([pc(3, 0), pc(4, 1)], block_index=7)       # 0 overridden by 1 in the same block
        e = sel.events
        self.assertEqual([(x["consumed_block"], x["previous"], x["noop"], x["superseded"]) for x in e],
                         [(5, "tatum", False, False), (6, "mehldau", True, False),
                          (7, "mehldau", False, True), (7, "tatum", False, False)])

    def test_cc_value_bins(self) -> None:
        import mido
        sel = LiveAdapterSelector(["a", "b", "c"], "cc:20")
        pick = lambda ns, cc, v: sel.update([self._ev(ns, mido.Message("control_change", control=cc, value=v))])
        self.assertEqual(pick(1, 20, 0), "a")
        self.assertEqual(pick(2, 20, 50), "b")
        self.assertEqual(pick(3, 21, 127), "b")   # other CC ignored
        self.assertEqual(pick(4, 20, 127), "c")

    def test_bad_control(self) -> None:
        with self.assertRaises(ValueError):
            LiveAdapterSelector(["a"], "pitchbend")


class RuntimeFlagsTest(unittest.TestCase):
    def _error(self, *extra):
        import tempfile
        from scripts.run_continuous_jazz import main
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit) as cm:
            main(["--output-dir", d, "--checkpoint", "x.pt", "--conditioning-midi", "p.mid", *extra])
        return cm.exception.code

    def test_swap_needs_schedule_and_merge(self) -> None:
        self.assertEqual(self._error("--swap-adapter", "m=y.pt"), 2)
        self.assertEqual(self._error("--adapter-schedule", "primary:4"), 2)
        self.assertEqual(self._error("--swap-adapter", "m=y.pt", "--adapter-schedule", "primary:4,m:4",
                                     "--no-merge-lora"), 2)
        self.assertEqual(self._error("--swap-adapter", "primary=y.pt", "--adapter-schedule", "primary:4"), 2)

    def test_live_control_needs_input_port_and_excludes_schedule(self) -> None:
        self.assertEqual(self._error("--swap-adapter", "m=y.pt", "--adapter-control", "program"), 2)
        self.assertEqual(self._error("--swap-adapter", "m=y.pt", "--adapter-control", "program",
                                     "--adapter-schedule", "primary:4,m:4", "--input-port", "X"), 2)
        self.assertEqual(self._error("--adapter-control", "program", "--input-port", "X"), 2)


if __name__ == "__main__":
    unittest.main()
