"""
Build training split manifests from the jazz piano dataset audit JSON.

This script does not scan raw MIDI files. The dataset audit JSON is the source
of truth, and only rows marked as candidate are allowed into training splits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


MANIFEST_SCHEMA_VERSION = "jazz_training_manifests_v1"
ROW_KEYS = (
    "path",
    "sha1",
    "source",
    "artist",
    "album",
    "is_brad_mehldau",
    "duration_sec",
    "non_drum_note_count",
    "piano_program_note_ratio",
    "max_note_duration_ratio",
    "recommendation",
)


def parse_group_fields(raw: str) -> list[str]:
    return [field.strip() for field in raw.split(",") if field.strip()]


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in ROW_KEYS if key in row}


def candidate_rows(rows: Sequence[dict[str, Any]], *, brad: bool | None = None) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row.get("recommendation") == "candidate"]
    if brad is None:
        return filtered
    return [row for row in filtered if bool(row.get("is_brad_mehldau")) is brad]


def diagnostic_rows(rows: Sequence[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("recommendation", "")).startswith(prefix)]


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def metadata_group_key(row: dict[str, Any], group_fields: Sequence[str]) -> tuple[str, ...] | None:
    values = tuple(str(row.get(field) or "") for field in group_fields)
    if not values or not any(values):
        return None
    return values


def connected_groups(rows: Sequence[dict[str, Any]], group_fields: Sequence[str]) -> list[list[dict[str, Any]]]:
    dsu = DisjointSet(len(rows))
    seen: dict[tuple[str, tuple[str, ...] | str], int] = {}

    for index, row in enumerate(rows):
        keys: list[tuple[str, tuple[str, ...] | str]] = []
        sha1 = row.get("sha1")
        if sha1:
            keys.append(("sha1", str(sha1)))
        meta_key = metadata_group_key(row, group_fields)
        if meta_key is not None:
            keys.append(("metadata", meta_key))

        for key in keys:
            previous = seen.get(key)
            if previous is None:
                seen[key] = index
            else:
                dsu.union(previous, index)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[dsu.find(index)].append(row)

    return list(grouped.values())


def ratio_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {name: 0 for name in ratios}
    ratio_sum = sum(max(0.0, value) for value in ratios.values())
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be greater than zero")

    raw_targets = {name: total * max(0.0, ratio) / ratio_sum for name, ratio in ratios.items()}
    targets = {name: int(math.floor(value)) for name, value in raw_targets.items()}
    remaining = total - sum(targets.values())
    remainders = sorted(
        ((raw_targets[name] - targets[name], name) for name in ratios),
        key=lambda item: (-item[0], item[1]),
    )
    for _, name in remainders[:remaining]:
        targets[name] += 1
    return targets


def split_grouped_rows(
    rows: Sequence[dict[str, Any]],
    split_ratios: dict[str, float],
    *,
    seed: int,
    group_fields: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    names = list(split_ratios)
    targets = ratio_targets(len(rows), split_ratios)
    buckets = {name: [] for name in names}
    rng = random.Random(seed)

    randomized_groups = [(rng.random(), group) for group in connected_groups(rows, group_fields)]
    randomized_groups.sort(key=lambda item: (-len(item[1]), item[0]))

    for _, group in randomized_groups:
        split_name = max(
            names,
            key=lambda name: (
                targets[name] - len(buckets[name]),
                -len(buckets[name]),
                name,
            ),
        )
        buckets[split_name].extend(group)

    for name in buckets:
        buckets[name] = sorted(buckets[name], key=lambda row: str(row.get("path", "")))
    return buckets


def count_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return {str(name): int(count) for name, count in Counter(row.get(key) for row in rows).most_common()}


def split_counts(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {name: len(rows) for name, rows in splits.items()}


def assert_no_forbidden_rows(splits: dict[str, list[dict[str, Any]]]) -> None:
    for split_name, rows in splits.items():
        for row in rows:
            if row.get("recommendation") != "candidate":
                raise ValueError(f"Non-candidate row in {split_name}: {row.get('path')}")
            is_brad = bool(row.get("is_brad_mehldau"))
            if split_name.startswith("generic_") and is_brad:
                raise ValueError(f"Brad row leaked into {split_name}: {row.get('path')}")
            if split_name.startswith("brad_") and not is_brad:
                raise ValueError(f"Non-Brad row leaked into {split_name}: {row.get('path')}")


def assert_group_boundaries(
    splits: dict[str, list[dict[str, Any]]], group_fields: Sequence[str]
) -> None:
    owners: dict[tuple[str, tuple[str, ...] | str], str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            keys: list[tuple[str, tuple[str, ...] | str]] = []
            sha1 = row.get("sha1")
            if sha1:
                keys.append(("sha1", str(sha1)))
            meta_key = metadata_group_key(row, group_fields)
            if meta_key is not None:
                keys.append(("metadata", meta_key))
            for key in keys:
                owner = owners.get(key)
                if owner is None:
                    owners[key] = split_name
                elif owner != split_name:
                    raise ValueError(f"Split leakage for {key}: {owner} and {split_name}")


def build_manifest_payload(
    audit_payload: dict[str, Any],
    *,
    audit_json: Path,
    seed: int,
    generic_train_ratio: float,
    generic_val_ratio: float,
    brad_train_ratio: float,
    brad_val_ratio: float,
    brad_holdout_ratio: float,
    group_fields: Sequence[str],
) -> dict[str, Any]:
    rows = list(audit_payload.get("files", []))
    generic_splits = split_grouped_rows(
        candidate_rows(rows, brad=False),
        {
            "generic_jazz_train": generic_train_ratio,
            "generic_jazz_val": generic_val_ratio,
        },
        seed=seed,
        group_fields=group_fields,
    )
    brad_splits = split_grouped_rows(
        candidate_rows(rows, brad=True),
        {
            "brad_adaptation_train": brad_train_ratio,
            "brad_adaptation_val": brad_val_ratio,
            "brad_test_holdout": brad_holdout_ratio,
        },
        seed=seed + 1,
        group_fields=group_fields,
    )
    splits = {**generic_splits, **brad_splits}

    assert_no_forbidden_rows(splits)
    assert_group_boundaries(splits, group_fields)

    review = sorted(diagnostic_rows(rows, "review_"), key=lambda row: str(row.get("path", "")))
    rejected = sorted(diagnostic_rows(rows, "reject_"), key=lambda row: str(row.get("path", "")))

    compact_splits = {
        name: [compact_row(row) for row in split_rows]
        for name, split_rows in splits.items()
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "input_audit_json": str(audit_json),
        "seed": int(seed),
        "group_fields": list(group_fields),
        "split_config": {
            "generic": {
                "train_ratio": float(generic_train_ratio),
                "val_ratio": float(generic_val_ratio),
            },
            "brad": {
                "adaptation_train_ratio": float(brad_train_ratio),
                "adaptation_val_ratio": float(brad_val_ratio),
                "test_holdout_ratio": float(brad_holdout_ratio),
            },
        },
        "audit_summary": audit_payload.get("summary", {}),
        "counts": {
            "splits": split_counts(compact_splits),
            "review": len(review),
            "rejected": len(rejected),
            "artist_counts": {
                name: count_by(rows, "artist") for name, rows in compact_splits.items()
            },
            "source_counts": {
                name: count_by(rows, "source") for name, rows in compact_splits.items()
            },
        },
        "splits": compact_splits,
        "diagnostics": {
            "review": [compact_row(row) for row in review],
            "rejected": [compact_row(row) for row in rejected],
        },
    }


def write_text_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(f"{row['path']}\n" for row in rows if row.get("path")),
        encoding="utf-8",
    )


def write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Jazz Training Manifests",
        "",
        f"- input audit: `{payload['input_audit_json']}`",
        f"- seed: `{payload['seed']}`",
        f"- group fields: `{', '.join(payload['group_fields'])}`",
        "",
        "## Split Counts",
        "",
        "| Split | Files |",
        "|---|---:|",
    ]
    for name, count in payload["counts"]["splits"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            f"| `review` | {payload['counts']['review']} |",
            f"| `rejected` | {payload['counts']['rejected']} |",
            "",
            "## Contract",
            "",
            "- Generic splits contain candidate non-Brad rows only.",
            "- Brad splits contain candidate Brad rows only.",
            "- Review and rejected rows are excluded from training splits.",
            "- Rows sharing exact SHA1 or metadata group fields stay in the same split.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "jazz_training_manifests.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for split_name, rows in payload["splits"].items():
        write_text_manifest(output_dir / f"{split_name}.txt", rows)
    for diagnostic_name, rows in payload["diagnostics"].items():
        write_text_manifest(output_dir / f"{diagnostic_name}.txt", rows)
    write_markdown_summary(output_dir / "jazz_training_manifests.md", payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build jazz piano training manifests from audit JSON")
    parser.add_argument("--audit_json", type=str, default="./outputs/dataset_audit/jazz_piano_dataset_audit.json")
    parser.add_argument("--output_dir", type=str, default="./data/manifests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generic_train_ratio", type=float, default=0.90)
    parser.add_argument("--generic_val_ratio", type=float, default=0.10)
    parser.add_argument("--brad_train_ratio", type=float, default=0.70)
    parser.add_argument("--brad_val_ratio", type=float, default=0.15)
    parser.add_argument("--brad_holdout_ratio", type=float, default=0.15)
    parser.add_argument("--group_fields", type=str, default="artist,album")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audit_json = Path(args.audit_json)
    if not audit_json.exists():
        raise FileNotFoundError(f"Audit JSON does not exist: {audit_json}")

    audit_payload = json.loads(audit_json.read_text(encoding="utf-8"))
    payload = build_manifest_payload(
        audit_payload,
        audit_json=audit_json,
        seed=args.seed,
        generic_train_ratio=args.generic_train_ratio,
        generic_val_ratio=args.generic_val_ratio,
        brad_train_ratio=args.brad_train_ratio,
        brad_val_ratio=args.brad_val_ratio,
        brad_holdout_ratio=args.brad_holdout_ratio,
        group_fields=parse_group_fields(args.group_fields),
    )
    write_outputs(Path(args.output_dir), payload)
    print(json.dumps(payload["counts"], ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
