"""Descriptor-histogram distance between token sequences and a reference corpus.

Used to ask whether generation moves toward one corpus (Mehldau) and away from
another (generic jazz). It is a distributional descriptor, not a quality or
style judgement: two very different pieces can share these histograms.

Features are pooled note-level histograms, chosen to be key-invariant except
register:

* ``interval``  consecutive onset-sorted pitch differences, clipped to +-24
* ``ioi``       inter-onset interval, log-spaced bins (0 = simultaneous)
* ``chord``     notes per onset cluster (onsets within 30 ms), 1..8+
* ``register``  pitch // 6
* ``velocity``  velocity // 8
* ``duration``  note length, log-spaced bins

Distance to a reference is the mean Jensen-Shannon divergence (base 2) over
features.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

FEATURES = ("interval", "ioi", "chord", "register", "velocity", "duration")
_BINS = {"interval": 49, "ioi": 12, "chord": 8, "register": 22, "velocity": 16, "duration": 12}
CHORD_WINDOW_S = 0.03
# Event vocabulary of the MIDI tokenizer: note_on, note_off, time_shift, velocity.
MAX_EVENT_TOKEN = 388


def _log_bin(seconds: float, n_bins: int) -> int:
    """0 for < 10 ms, then octave bins starting at 10 ms."""
    if seconds < 0.01:
        return 0
    return min(n_bins - 1, 1 + int(math.log2(seconds / 0.01)))


def tokens_to_notes(tokens: Iterable[int]):
    from midi_processor.processor import decode_midi

    events = [int(t) for t in tokens if 0 <= int(t) < MAX_EVENT_TOKEN]
    if not events:
        return []
    midi = decode_midi(events)
    return sorted((n for inst in midi.instruments for n in inst.notes),
                  key=lambda n: (n.start, n.pitch))


def feature_counts(notes: Sequence) -> dict[str, np.ndarray]:
    counts = {f: np.zeros(_BINS[f]) for f in FEATURES}
    if not notes:
        return counts
    for n in notes:
        counts["register"][min(21, max(0, n.pitch // 6))] += 1
        counts["velocity"][min(15, max(0, n.velocity // 8))] += 1
        counts["duration"][_log_bin(max(0.0, n.end - n.start), _BINS["duration"])] += 1
    for a, b in zip(notes, notes[1:]):
        counts["interval"][max(-24, min(24, b.pitch - a.pitch)) + 24] += 1
        counts["ioi"][_log_bin(max(0.0, b.start - a.start), _BINS["ioi"])] += 1
    cluster = 1
    for a, b in zip(notes, notes[1:]):
        if b.start - a.start < CHORD_WINDOW_S:
            cluster += 1
        else:
            counts["chord"][min(8, cluster) - 1] += 1
            cluster = 1
    counts["chord"][min(8, cluster) - 1] += 1
    return counts


def pool(counts_list: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    total = {f: np.zeros(_BINS[f]) for f in FEATURES}
    for c in counts_list:
        for f in FEATURES:
            total[f] += c[f]
    return total


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def distance(counts: dict[str, np.ndarray], reference: dict[str, np.ndarray],
             features: Sequence[str] = FEATURES) -> float:
    return float(np.mean([js_divergence(counts[f], reference[f]) for f in features]))


def per_feature_distance(counts, reference) -> dict[str, float]:
    return {f: js_divergence(counts[f], reference[f]) for f in FEATURES}


def feature_vector(counts: dict[str, np.ndarray]) -> np.ndarray:
    """Each histogram normalised to sum 1 (zeros stay zeros), concatenated."""
    parts = []
    for f in FEATURES:
        c = np.asarray(counts[f], dtype=float)
        parts.append(c / c.sum() if c.sum() > 0 else c)
    return np.concatenate(parts)


class LogisticModel:
    """L2 logistic regression with class-balanced weights, full-batch gradient descent."""

    def __init__(self, lam: float = 1.0, steps: int = 2000, lr: float = 0.1):
        self.lam, self.steps, self.lr = lam, steps, lr

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6
        Z = (X - self.mean) / self.std
        w_pos = 0.5 / max(1, y.sum())
        w_neg = 0.5 / max(1, (1 - y).sum())
        sw = np.where(y == 1, w_pos, w_neg)
        self.w = np.zeros(Z.shape[1])
        self.b = 0.0
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            g = sw * (p - y)
            self.w -= self.lr * (Z.T @ g + self.lam * self.w / len(y))
            self.b -= self.lr * g.sum()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std
        return 1 / (1 + np.exp(-(Z @ self.w + self.b)))

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "w": self.w.tolist(), "b": float(self.b), "lam": self.lam}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticModel":
        m = cls(lam=d["lam"])
        m.mean, m.std = np.array(d["mean"]), np.array(d["std"])
        m.w, m.b = np.array(d["w"]), float(d["b"])
        return m
