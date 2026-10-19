#!/usr/bin/env python3
"""Freeze deterministic MOSAIC v2 splits; legacy Test supplies SHA protection only.

V2 metadata and verified image bytes are authoritative. Legacy annotations are
audited, never used to relabel v2. Existing freezes are verified, not overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CELLS = [(d, y) for d in ("Cartoon", "Chart", "Photo_and_poster") for y in (0, 1)]
TARGETS = {"train": 700, "dev": 100, "test": 200}
EXPECTED_POOL_SIZE = 6718
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_FIELDS = ("source_dataset_claim", "source_revision_claim", "source_image_id_claim", "group_id_claim")
MANIFEST_COLUMNS = ("sample_id", "image_path", "domain", "label", "label_name", "sha256",
                    "candidate_id", "split", "group_id", "group_evidence", *SOURCE_FIELDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing/duplicate CSV headers: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV record: {path}")
    return rows


def normalize_domain(raw: str) -> str:
    mapping = {"cartoon": "Cartoon", "chart": "Chart", "photo": "Photo_and_poster",
               "photo_and_poster": "Photo_and_poster"}
    value = str(raw).strip().lower()
    if value not in mapping:
        raise ValueError(f"Unexpected domain: {raw!r}")
    return mapping[value]


def normalize_label(raw: str) -> int:
    value = str(raw).strip().lower()
    if value in ("0", "0_real", "real"):
        return 0
    if value in ("1", "1_fake", "fake"):
        return 1
    raise ValueError(f"Unexpected label: {raw!r}")


def stable_seed(seed: int, *fields: object) -> int:
    text = ":".join(str(v) for v in (seed, *fields)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def legacy_protection(path: Path) -> tuple[set[str], dict]:
    """Read actual legacy file identities; annotation conflicts are audit events."""
    rows = read_csv(path)
    annotations = defaultdict(list)
    cache = {}
    for index, row in enumerate(rows, start=2):
        image = Path(row["image_path"]).expanduser()
        image = (image if image.is_absolute() else path.parent / image).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Legacy Test image missing at row {index}: {image}")
        if image not in cache:
            cache[image] = sha256(image)
        digest = cache[image]
        # These labels are only audited, never used for v2 identity or labels.
        try:
            domain = normalize_domain(row.get("domain", ""))
        except ValueError:
            domain = str(row.get("domain", ""))
        try:
            label = str(normalize_label(row.get("label", "")))
        except ValueError:
            label = str(row.get("label", ""))
        annotations[digest].append({"row": index, "domain": domain, "label": label})
    conflicts = []
    for digest, records in sorted(annotations.items()):
        if len({(r["domain"], r["label"]) for r in records}) > 1:
            conflicts.append({"sha256": digest, "records": records})
    return set(annotations), {
        "legacy_test_records": len(rows),
        "legacy_test_unique_sha256": len(annotations),
        "legacy_test_duplicate_records": len(rows) - len(annotations),
        "legacy_test_conflicting_sha256": len(conflicts),
        "legacy_test_domain_conflicting_sha256": sum(len({r["domain"] for r in c["records"]}) > 1 for c in conflicts),
        "legacy_test_label_conflicting_sha256": sum(len({r["label"] for r in c["records"]}) > 1 for c in conflicts),
        "legacy_test_conflicts": conflicts,
        "legacy_annotations_used_for_v2_labels": False,
    }


def _group_tokens(row: dict[str, str]) -> list[tuple[str, ...]]:
    tokens = []
    if row.get("group_id", "").strip():
        tokens.append(("provided_group", row["group_id"].strip()))
    dataset = row.get("source_dataset_claim", "").strip()
    for field in ("group_id_claim", "source_image_id_claim"):
        value = row.get(field, "").strip()
        if dataset and value:
            tokens.append((field, dataset, value))
    return tokens


def build_groups(rows):
    """Respect supplied source links conservatively, without certifying claims."""
    parents = {r["sha256"]: r["sha256"] for r in rows}
    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    token_owner = {}
    tokens = {r["sha256"]: _group_tokens(r) for r in rows}
    for row in rows:
        key = row["sha256"]
        for token in tokens[key]:
            if token in token_owner:
                a, b = sorted((root(key), root(token_owner[token])))
                parents[b] = a
            else:
                token_owner[token] = key
    grouped = defaultdict(list)
    for row in rows:
        grouped[root(row["sha256"])].append(row)
    groups = []
    for members in grouped.values():
        evidence = sorted({token for row in members for token in tokens[row["sha256"]]})
        group_id = "mosaic_source_" + hashlib.sha256(json.dumps(evidence).encode()).hexdigest() if evidence else ""
        for row in members:
            row["group_id"] = group_id
            row["group_evidence"] = "provided_links_claims_not_independently_verified" if evidence else "NOT_AVAILABLE"
        groups.append(sorted(members, key=lambda r: r["sha256"]))
    return groups, {
        "rows_with_group_or_source_links": sum(bool(r["group_id"]) for r in rows),
        "rows_without_group_evidence": sum(not r["group_id"] for r in rows),
        "linked_components": sum(bool(g[0]["group_id"]) for g in groups),
        "multi_image_components": sum(len(g) > 1 for g in groups),
        "largest_component": max(map(len, groups)),
        "claim_authenticity": "NOT_INDEPENDENTLY_VERIFIED",
        "near_duplicate_isolation": "NOT_VERIFIED",
        "source_batch_used_as_image_group": False,
    }


def _take_groups(groups, count, *, exact=True):
    """Bounded subset-sum: a linked component is never broken across roles."""
    previous = {0: None}
    for index, group in enumerate(groups):
        for current in sorted(previous, reverse=True):
            following = current + len(group)
            if following <= count and following not in previous:
                previous[following] = (current, index)
        if count in previous:
            break
    chosen_count = count if exact else max(previous)
    if chosen_count not in previous:
        raise ValueError(f"Group-aware quota infeasible: need {count}, available {sum(map(len, groups))}")
    chosen_indices = set()
    while chosen_count:
        chosen_count, index = previous[chosen_count]
        chosen_indices.add(index)
    return ([g for i, g in enumerate(groups) if i in chosen_indices],
            [g for i, g in enumerate(groups) if i not in chosen_indices])


def partition(rows, protected, seed):
    groups, group_report = build_groups(rows)
    by_cell = defaultdict(list)
    held_out = []
    # Current multi-image links all stay within a cell. Future cross-cell groups
    # are held outside the experiment whole; fail if exact quotas then cannot fit.
    for group in groups:
        cells = {(r["domain"], int(r["label"])) for r in group}
        if len(cells) != 1:
            held_out.extend(group)
        else:
            by_cell[next(iter(cells))].append(group)
    result = {key: [] for key in TARGETS}
    leftovers = list(held_out)
    required = sum(TARGETS.values())
    group_report["cross_cell_groups_held_out_whole"] = len({r["group_id"] for r in held_out})
    for domain, label in CELLS:
        cell = (domain, label)
        candidates = sorted(by_cell[cell], key=lambda g: g[0]["sha256"])
        random.Random(stable_seed(seed, domain, label)).shuffle(candidates)
        reserved = [g for g in candidates if any(r["sha256"] in protected for r in g)]
        clean = [g for g in candidates if not any(r["sha256"] in protected for r in g)]
        if sum(map(len, candidates)) < required:
            raise ValueError(f"Insufficient candidates in {cell}: need {required}")
        if sum(map(len, clean)) < TARGETS["train"] + TARGETS["dev"]:
            raise ValueError(f"Insufficient protected-free Train/Dev candidates in {cell}")
        old_test, reserved_left = _take_groups(reserved, TARGETS["test"], exact=False)
        test_more, clean = _take_groups(clean, TARGETS["test"] - sum(map(len, old_test)))
        dev, clean = _take_groups(clean, TARGETS["dev"])
        train, clean = _take_groups(clean, TARGETS["train"])
        for split, selected in (("train", train), ("dev", dev), ("test", old_test + test_more)):
            result[split].extend(r for group in selected for r in group)
        leftovers.extend(r for group in reserved_left + clean for r in group)
    for split, items in result.items():
        random.Random(stable_seed(seed, split)).shuffle(items)
        for index, row in enumerate(items):
            row.update(sample_id=f"mosaic_v2_{split}_{index:06d}", split=split)
    for index, row in enumerate(sorted(leftovers, key=lambda r: r["sha256"])):
        row.update(sample_id=f"mosaic_v2_unused_{index:06d}", split="unused")
    validate_partition(result, protected)
    used = {r["sha256"] for items in result.values() for r in items}
    unused = {r["sha256"] for r in leftovers}
    if used & unused or len(used) + len(unused) != len(rows):
        raise AssertionError("Selected and remaining identities do not partition the pool")
    return result, sorted(leftovers, key=lambda r: r["sha256"]), group_report


def validate_partition(splits, protected):
    identities, paths, group_roles = set(), set(), {}
    for split, items in splits.items():
        counts = Counter((r["domain"], int(r["label"])) for r in items)
        if counts != {key: TARGETS[split] for key in CELLS}:
            raise ValueError(f"Bad {split} counts: {dict(counts)}")
        for row in items:
            digest, image = row["sha256"], Path(row["image_path"])
            if not image.is_absolute() or not image.is_file():
                raise ValueError(f"Unsafe/missing manifest image: {image}")
            if digest in identities or str(image) in paths:
                raise ValueError(f"Duplicate/cross-split image identity: {digest}")
            if split != "test" and digest in protected:
                raise ValueError("Legacy Test SHA entered Train/Dev")
            identities.add(digest); paths.add(str(image))
            group = row.get("group_id", "")
            if group and group_roles.setdefault(group, split) != split:
                raise ValueError(f"Source group crosses splits: {group}")


def plan(project: Path, seed: int):
    project = project.resolve()
    image_root = (project.parent / "datasets/mosaic_v2").resolve()
    metadata_path = image_root / "metadata.csv"
    legacy_path = project / "e2e/manifests/mosaic_test.csv"
    metadata_sha, legacy_sha = sha256(metadata_path), sha256(legacy_path)
    metadata = read_csv(metadata_path)
    if len(metadata) != EXPECTED_POOL_SIZE:
        raise ValueError(f"Expected {EXPECTED_POOL_SIZE} candidate records, got {len(metadata)}")
    seen, rows = set(), []
    for index, raw in enumerate(metadata, start=2):
        domain, label = normalize_domain(raw["domain"]), normalize_label(raw["label"])
        digest = raw["sha256"].strip().lower()
        if not HEX64.fullmatch(digest) or digest in seen:
            raise ValueError(f"Invalid/duplicate SHA256 at metadata row {index}: {digest}")
        if raw.get("split", "").strip():
            raise ValueError("Candidate metadata already has a split; do not silently resample")
        image = (image_root / raw["file_name"]).resolve()
        if not image.is_relative_to(image_root) or not image.is_file():
            raise ValueError(f"Unsafe/missing image at metadata row {index}: {image}")
        if sha256(image) != digest:
            raise ValueError(f"Image SHA256 mismatch at metadata row {index}: {image}")
        seen.add(digest)
        row = {k: raw.get(k, "").strip() for k in (*SOURCE_FIELDS, "group_id", "candidate_id")}
        row.update(image_path=str(image), domain=domain, label=str(label),
                   label_name="Real" if label == 0 else "Fake", sha256=digest)
        rows.append(row)
    protected, legacy_report = legacy_protection(legacy_path)
    splits, leftovers, group_report = partition(rows, protected, seed)
    if sha256(metadata_path) != metadata_sha or sha256(legacy_path) != legacy_sha:
        raise ValueError("An input manifest changed during planning")
    report = {
        "protocol": "mosaic_v2_4200_600_1200_v1", "seed": seed,
        "metadata_records": len(metadata), "verified_image_sha256": len(rows),
        "metadata_sha256": metadata_sha, "legacy_test_manifest_sha256": legacy_sha,
        "preparation_code_sha256": sha256(Path(__file__).resolve()),
        "source_config_sha256": sha256(project / "configs/visual_expert_mosaic_v2_full.yaml"),
        **legacy_report, "group_checks": group_report,
        "legacy_test_pool_sha_overlap": len(seen & protected),
        "new_test_legacy_sha_overlap": sum(r["sha256"] in protected for r in splits["test"]),
        "train_dev_legacy_sha_overlap": 0,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "per_cell": {k: {f"{d}:{y}": sum(r["domain"] == d and int(r["label"]) == y for r in v) for d, y in CELLS} for k, v in splits.items()},
        "remaining_records": len(leftovers),
        "remaining_per_cell": {f"{d}:{y}": sum(r["domain"] == d and int(r["label"]) == y for r in leftovers) for d, y in CELLS},
        "sha256_cross_split_overlap": 0, "available_group_cross_split_overlap": 0,
        "label_authority": "mosaic_v2_metadata_and_verified_image_files",
    }
    return splits, report, leftovers


def _csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write(project: Path, splits, report, leftovers, *, verify_only=False, pretrained_path=None):
    manifest_dir = project / "e2e/manifests/mosaic_v2_frozen"
    config_dst = project / "configs/visual_expert_mosaic_v2_frozen.yaml"
    config = yaml.safe_load((project / "configs/visual_expert_mosaic_v2_full.yaml").read_text(encoding="utf-8"))
    config["seed"] = report["seed"]
    config["bce_sampler_seed"] = report["seed"] + 1
    config["repair_sampler_seed"] = report["seed"] + 2
    config["manifests"] = {k: f"e2e/manifests/mosaic_v2_frozen/mosaic_{k}.csv" for k in ("train", "dev")}
    run_name = f"outputs/mosaic_v2_frozen_seed{report['seed']}"
    config["paths"] = {k: str(v).replace("outputs/mosaic_v2_full", run_name) for k, v in config["paths"].items()}
    if pretrained_path is not None:
        pretrained_path = Path(pretrained_path).expanduser().resolve()
        initial_sha = sha256(pretrained_path)
        expected_sha = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
        if config["model_name"] != "ViT-L-14-quickgelu" or initial_sha != expected_sha:
            raise ValueError("Local initialization is not the pinned official OpenAI ViT-L/14 file")
        config["pretrained"] = str(pretrained_path)
        config["initialization_sha256"] = initial_sha
        config["initialization_source"] = "OpenAI CLIP official ViT-L/14"
    config["checkpoint_estimate_gib"] = 2.0
    config["minimum_free_disk_gib"] = 1.0
    blobs = {f"mosaic_{k}.csv": _csv_bytes(v) for k, v in splits.items()}
    blobs["mosaic_remaining.csv"] = _csv_bytes(leftovers)
    report = {**report, "split_csv_sha256": {k: hashlib.sha256(blobs[f"mosaic_{k}.csv"]).hexdigest() for k in splits},
              "remaining_csv_sha256": hashlib.sha256(blobs["mosaic_remaining.csv"]).hexdigest()}
    report_bytes = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    blobs["split_report.json"] = report_bytes
    config["data_freeze"] = {"report": "e2e/manifests/mosaic_v2_frozen/split_report.json",
                             "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                             "manifest_sha256": {k: report["split_csv_sha256"][k] for k in ("train", "dev")}}
    config_bytes = yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8")
    digests = {name: hashlib.sha256(value).hexdigest() for name, value in blobs.items()}
    digests["visual_expert_mosaic_v2_frozen.yaml"] = hashlib.sha256(config_bytes).hexdigest()
    blobs["freeze.sha256"] = "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items())).encode("ascii")
    if manifest_dir.exists() or config_dst.exists():
        if not manifest_dir.is_dir() or not config_dst.is_file():
            raise FileExistsError("Partial/existing freeze found; refusing to overwrite")
        if config_dst.read_bytes() != config_bytes:
            raise ValueError("Existing frozen config differs; refusing to overwrite")
        for name, value in blobs.items():
            path = manifest_dir / name
            if not path.is_file() or path.read_bytes() != value:
                raise ValueError(f"Existing freeze differs: {name}; refusing to overwrite")
        print("Existing freeze verified and reused without writing.", flush=True)
        return manifest_dir, config_dst
    if verify_only:
        raise FileNotFoundError("No complete frozen split exists")
    if shutil.disk_usage(project).free < 32 * 1024 * 1024:
        raise OSError("Insufficient disk space to write manifests safely")
    manifest_dir.mkdir(parents=True, exist_ok=False)
    for name, value in blobs.items():
        with (manifest_dir / name).open("xb") as f:
            f.write(value)
    with config_dst.open("xb") as f:
        f.write(config_bytes)
    return manifest_dir, config_dst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pretrained-path", type=Path,
                        help="Optional existing official OpenAI ViT-L/14 file; SHA is pinned")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    splits, report, leftovers = plan(project, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.write or args.verify:
        directory, config = write(project, splits, report, leftovers, verify_only=args.verify,
                                  pretrained_path=args.pretrained_path)
        print(f"Frozen manifests: {directory}\nFrozen config: {config}", flush=True)
    else:
        print("READ-ONLY CHECK OK. --write creates a new freeze; --verify checks an existing freeze.", flush=True)


if __name__ == "__main__":
    main()
