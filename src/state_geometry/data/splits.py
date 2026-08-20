from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from .schema import is_null


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def build_dependency_groups(
    frame: pd.DataFrame,
    fields: Sequence[str],
    observation_column: str = "observation_id",
) -> pd.Series:
    if observation_column not in frame:
        raise ValueError(f"missing {observation_column}")
    observations = frame[observation_column].astype(str)
    if observations.duplicated().any():
        raise ValueError("observation_id must be unique before grouping")
    uf = UnionFind(observations)
    for field in fields:
        if field not in frame:
            raise ValueError(f"missing grouping field: {field}")
        first_seen: dict[str, str] = {}
        for observation, value in zip(observations, frame[field], strict=True):
            if is_null(value) or (isinstance(value, str) and not value.strip()):
                continue
            # Namespace by field: coincident strings in unrelated ID systems are not edges.
            key = f"{field}\x1f{value}"
            if key in first_seen:
                uf.union(observation, first_seen[key])
            else:
                first_seen[key] = observation
    roots = {observation: uf.find(observation) for observation in observations}
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for observation, root in roots.items():
        members_by_root[root].append(observation)
    canonical = {
        root: f"dep_{min(members)}" for root, members in members_by_root.items()
    }
    return observations.map(lambda value: canonical[roots[value]])


def _multilabel(value: object) -> tuple[str, ...]:
    if is_null(value):
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted(str(item) for item in value if str(item)))
    text = str(value).strip()
    if not text:
        return ()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return tuple(sorted(str(item) for item in parsed if str(item)))
        except json.JSONDecodeError:
            pass
    return tuple(sorted(part.strip() for part in text.split(";") if part.strip()))


@dataclass(frozen=True)
class SplitResult:
    mapping: pd.DataFrame
    realized_counts: dict[str, dict[str, int]]


def assign_group_splits(
    frame: pd.DataFrame,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    names: Sequence[str] = ("train", "validation", "test"),
    stratify: Sequence[str] = (),
    multilabel: Sequence[str] = (),
    seed: int = 20260820,
) -> SplitResult:
    if len(ratios) != len(names) or not ratios or any(r <= 0 for r in ratios):
        raise ValueError("split names and positive ratios must have equal length")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to one")
    required = {"observation_id", "dependency_group_id", *stratify, *multilabel}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing split columns: {sorted(missing)}")

    labels_by_row: list[tuple[str, ...]] = []
    for _, row in frame.iterrows():
        labels = [f"{field}={row[field]}" for field in stratify if not is_null(row[field])]
        for field in multilabel:
            labels.extend(f"{field}={value}" for value in _multilabel(row[field]))
        labels_by_row.append(tuple(sorted(labels)))

    work = frame[["observation_id", "dependency_group_id"]].copy()
    work["_labels"] = labels_by_row
    groups: dict[str, dict[str, object]] = {}
    totals: Counter[str] = Counter()
    for group_id, rows in work.groupby("dependency_group_id", sort=True):
        counts: Counter[str] = Counter()
        for labels in rows["_labels"]:
            counts.update(labels)
        totals.update(counts)
        groups[str(group_id)] = {"size": len(rows), "counts": counts}

    total_rows = len(frame)
    target_size = {name: total_rows * ratio for name, ratio in zip(names, ratios, strict=True)}
    target_label = {
        name: {label: count * ratio for label, count in totals.items()}
        for name, ratio in zip(names, ratios, strict=True)
    }
    current_size = Counter({name: 0 for name in names})
    current_label = {name: Counter() for name in names}

    rng = random.Random(seed)
    tie = {group: rng.random() for group in groups}
    ordered = sorted(
        groups,
        key=lambda group: (
            -sum(1.0 / max(totals[label], 1) for label in groups[group]["counts"]),
            -int(groups[group]["size"]),
            tie[group],
            group,
        ),
    )
    assignment: dict[str, str] = {}
    for group in ordered:
        size = int(groups[group]["size"])
        counts = groups[group]["counts"]
        scores: list[tuple[float, float, str]] = []
        for name in names:
            size_after = current_size[name] + size
            size_fill = size_after / max(target_size[name], 1.0)
            overflow = max(size_after - target_size[name], 0.0) / max(target_size[name], 1.0)
            label_fill: list[float] = []
            for label, increment in counts.items():
                target = target_label[name][label]
                after = current_label[name][label] + increment
                label_fill.append(after / max(target, 1.0))
            # Fill every split in proportion to its target. A small label term
            # breaks near-ties without allowing stratification to destroy 70/15/15.
            score = 10.0 * overflow + size_fill + 0.10 * (
                sum(label_fill) / len(label_fill) if label_fill else 0.0
            )
            scores.append((score, current_size[name] / max(target_size[name], 1.0), name))
        selected = min(scores)[2]
        assignment[group] = selected
        current_size[selected] += size
        current_label[selected].update(counts)

    mapping = frame[["observation_id", "dependency_group_id"]].copy()
    mapping["split"] = mapping["dependency_group_id"].astype(str).map(assignment)
    if mapping.groupby("dependency_group_id")["split"].nunique().max() != 1:
        raise AssertionError("dependency group crossed splits")
    realized = {
        name: {"observations": int(current_size[name]), "groups": sum(v == name for v in assignment.values())}
        for name in names
    }
    return SplitResult(mapping=mapping, realized_counts=realized)
