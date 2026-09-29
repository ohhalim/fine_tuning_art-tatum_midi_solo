"""One base model, several LoRA adapters swapped in place at runtime.

Every adapter checkpoint is loaded, merged (``merge_lora_for_inference``) and
compared with the others. After merging, the adapters may differ only in the
weights LoRA touches (attention in/out projections, FFN linears); any other
difference means they were trained on different bases, and the bank refuses
them unless ``allow_different_bases`` is set, in which case a swap copies every
differing tensor (a whole-model swap, slower). One model instance is kept;
``select`` copies the chosen adapter's merged weights into it. Call ``select``
only between generations, on the thread that generates (the runtime's producer
thread). docs/experiments/ADAPTER_SWAP.md.
"""
from __future__ import annotations

import time

import torch

# Parameter-name suffixes a LoRA target can change once merged.
LORA_TOUCHED_SUFFIXES = (
    "self_attn.in_proj_weight",
    "self_attn.out_proj.weight",
    "linear1.weight",
    "linear2.weight",
)


def _touchable(name: str) -> bool:
    return name.endswith(LORA_TOUCHED_SUFFIXES)


class AdapterBank:
    def __init__(self, models: dict[str, torch.nn.Module], allow_different_bases: bool = False):
        """``models``: name -> merged model (no ``lora_`` tensors). The first is kept."""
        if not models:
            raise ValueError("need at least one adapter")
        names = list(models)
        states = {n: m.state_dict() for n, m in models.items()}
        for n, sd in states.items():
            if any("lora_" in k for k in sd):
                raise ValueError(f"adapter {n!r} is not merged")
        keys = set(states[names[0]])
        for n in names[1:]:
            if set(states[n]) != keys:
                raise ValueError(f"adapter {n!r} has a different parameter layout")
        differing = sorted(k for k in keys
                           if any(not torch.equal(states[names[0]][k], states[n][k]) for n in names[1:]))
        foreign = [k for k in differing if not _touchable(k)]
        if foreign and not allow_different_bases:
            raise ValueError(f"adapters do not share a base: {len(foreign)} non-LoRA tensors differ, "
                             f"e.g. {foreign[:3]}")
        self.model = models[names[0]]
        self.names = names
        self.swap_keys = differing
        self.shared_base = not foreign
        self._weights = {n: {k: states[n][k].detach().clone() for k in differing} for n in names}
        self._params = dict(self.model.named_parameters())
        self._buffers = dict(self.model.named_buffers())
        self.current = names[0]
        self.swap_ms: list[float] = []

    @torch.no_grad()
    def select(self, name: str) -> bool:
        """Make ``name`` the active adapter. Returns True when weights were copied."""
        if name not in self._weights:
            raise KeyError(f"unknown adapter {name!r}; have {self.names}")
        if name == self.current:
            return False
        t0 = time.perf_counter()
        for k, v in self._weights[name].items():
            target = self._params.get(k)
            if target is None:
                target = self._buffers[k]
            target.copy_(v)
        self.current = name
        self.swap_ms.append((time.perf_counter() - t0) * 1000)
        return True


def parse_schedule(text: str, names) -> list[tuple[str, int]]:
    """``"tatum:4,mehldau:4"`` -> [("tatum", 4), ("mehldau", 4)], cycled over bars."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, bars = part.partition(":")
        if name not in names:
            raise ValueError(f"schedule names unknown adapter {name!r}; have {list(names)}")
        n = int(bars or 1)
        if n < 1:
            raise ValueError("bars per schedule entry must be >= 1")
        out.append((name, n))
    if not out:
        raise ValueError("empty adapter schedule")
    return out


def adapter_for_bar(schedule: list[tuple[str, int]], bar_index: int) -> str:
    period = sum(n for _, n in schedule)
    pos = bar_index % period
    for name, n in schedule:
        if pos < n:
            return name
        pos -= n
    raise AssertionError("unreachable")


class LiveAdapterSelector:
    """Choose the adapter from the player's input: Program Change or one CC.

    ``control`` is ``"program"`` (program p selects the p-th adapter; programs
    past the last adapter are ignored) or ``"cc:N"`` (CC number N; the 0-127
    value range is split evenly across the adapters). The choice persists until
    the next selector message, even after the input window forgets it. Runs on
    the producer thread with the snapshot the producer already took.
    """

    def __init__(self, names, control: str, initial: str | None = None):
        self.names = list(names)
        if not self.names:
            raise ValueError("need at least one adapter")
        if control == "program":
            self.kind, self.cc = "program", None
        elif control.startswith("cc:"):
            self.kind, self.cc = "cc", int(control[3:])
            if not 0 <= self.cc <= 127:
                raise ValueError("cc number must be 0..127")
        else:
            raise ValueError('control must be "program" or "cc:N"')
        self.current = initial if initial is not None else self.names[0]
        if self.current not in self.names:
            raise ValueError(f"unknown initial adapter {self.current!r}")
        self._last_ns = -1
        self.events: list[dict] = []

    def _pick(self, message):
        if self.kind == "program" and message.type == "program_change":
            return message.program, (self.names[message.program]
                                     if message.program < len(self.names) else None)
        if self.kind == "cc" and message.type == "control_change" and message.control == self.cc:
            return message.value, self.names[min(message.value * len(self.names) // 128,
                                                 len(self.names) - 1)]
        return None, None

    def update(self, events, block_index: int | None = None) -> str:
        """Apply new selector messages; ``block_index`` is the block whose
        generation consumes them (docs/experiments/ADAPTER_LIVE_SELECT.md).

        Each recorded event says what it asked for (``selected``, None when the
        program is out of range), what was active before (``previous``), whether
        it asked for the adapter already active (``noop``), and whether a later
        message consumed by the same block overrode it (``superseded``).
        """
        before = self.current                 # adapter of the previous block
        batch = []
        for e in events:
            if e.received_ns <= self._last_ns:
                continue
            self._last_ns = e.received_ns
            value, name = self._pick(e.message)
            if value is None:
                continue
            record = {"received_ns": e.received_ns, "value": value, "selected": name,
                      "previous": before, "noop": False, "superseded": False,
                      "consumed_block": block_index}
            self.events.append(record)
            if name is not None:
                self.current = name
                batch.append(record)
        # Judged against the adapter the previous block used, not an intermediate
        # state: the last valid message in the batch is the one this block acts on
        # (noop if it asks for what was already playing); earlier ones are overridden.
        for r in batch[:-1]:
            r["superseded"] = True
        if batch:
            batch[-1]["noop"] = batch[-1]["selected"] == before
        return self.current
