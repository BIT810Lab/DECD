"""Load published image paths, labels, domains and predefined split metadata.

Validation checks paths, labels, sample IDs and optional groups. Unknown CSV
columns are ignored; image contents are decoded only by the dataset transform.
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path, PureWindowsPath

DOMAIN_TO_ID = {"Cartoon": 0, "Chart": 1, "Photo_and_poster": 2}
DOMAIN_ALIASES = {
    "cartoon": "Cartoon", "chart": "Chart", "photo": "Photo_and_poster",
    "photo_and_poster": "Photo_and_poster", "photo & poster": "Photo_and_poster",
    "photoandposter": "Photo_and_poster",
}
PATH_FIELDS = ("image_path", "file_name", "relative_path", "standardized_path",
               "local_path", "path", "filepath", "file_path", "original_image_path")
DOMAIN_FIELDS = ("domain", "category", "source_domain", "dataset", "source")


def canonical_domain(value):
    key = str(value).strip().lower()
    if key not in DOMAIN_ALIASES:
        raise ValueError(f"unsupported domain: {value!r}")
    return DOMAIN_ALIASES[key]


def parse_label(row):
    raw = str(row.get("label", "")).strip().lower()
    aliases = {"0_real": 0, "1_fake": 1, "real": 0, "fake": 1,
               "human": 0, "ai": 1, "generated": 1, "synthetic": 1}
    name = str(row.get("label_name", "")).strip().lower()
    if not raw:
        if name not in aliases:
            raise ValueError("missing/invalid label; expected Real=0 or Fake=1")
        return aliases[name]
    if raw in aliases:
        label = aliases[raw]
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"invalid label: {raw!r}") from exc
        if not math.isfinite(value) or value not in (0., 1.):
            raise ValueError(f"invalid label: {raw!r}")
        label = int(value)
    if name and (name not in aliases or aliases[name] != label):
        raise ValueError("conflicting label / label_name")
    return label


def resolve_image(text, base, *, confined=False):
    text = str(text).strip()
    if not text:
        raise ValueError("empty image path")
    if os.name != "nt" and PureWindowsPath(text).is_absolute():
        raise ValueError("foreign Windows absolute path; provide portable relative paths")
    path = Path(text.replace("\\", "/")).expanduser()
    path = (Path(base) / path).resolve() if not path.is_absolute() else path.resolve()
    if confined and not path.is_relative_to(Path(base).resolve()):
        raise ValueError("image path escapes image_root (including symlinks)")
    return path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("missing or duplicate CSV header")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty manifest: {Path(path).name}")
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("malformed CSV row")
    return fields, rows


def canonical_split(value):
    value = str(value).strip().lower()
    return {"validation": "dev", "val": "dev", "benchmark_test": "test"}.get(value, value)


def assert_role(rows, role):
    for row in rows:
        split = canonical_split(row.get("split", ""))
        if row.get("benchmark_id") and role in ("train", "dev"):
            raise ValueError("Benchmark is test-only: forbidden for training, risk or development")
        if split and split != role:
            raise ValueError(f"manifest split {split!r} cannot be used as {role!r}")


def read_manifest(path, *, image_root=None, require_domain=True, path_field=None,
                  domain_field=None, check_files=True, allow_duplicates=False,
                  role=None):
    path = Path(path).expanduser().resolve()
    fields, raw = read_csv(path)
    pf = path_field or next((k for k in PATH_FIELDS if k in fields), "")
    df = domain_field or next((k for k in DOMAIN_FIELDS if k in fields), "")
    if pf not in fields or (require_domain and df not in fields):
        raise ValueError("missing image path or domain column")
    if domain_field and domain_field not in fields:
        raise ValueError(f"explicit domain field is absent: {domain_field}")
    if role:
        assert_role(raw, role)
    rows = []
    seen_paths, seen_ids = set(), set()
    for i, row in enumerate(raw):
        p = resolve_image(row[pf], image_root or path.parent, confined=image_root is not None)
        if check_files and not p.is_file():
            raise FileNotFoundError(f"manifest row {i}: image file missing")
        label = parse_label(row)
        raw_domain = row[df].strip() if df else ''
        domain = (canonical_domain(raw_domain) if require_domain and raw_domain
                  else DOMAIN_ALIASES.get(raw_domain.lower(), raw_domain or 'all'))
        if require_domain and domain == "all":
            raise ValueError("missing domain")
        sid = str(row.get("sample_id", "")).strip() or f"sample_{i:07d}"
        if not allow_duplicates and (p in seen_paths or sid in seen_ids):
            raise ValueError(f"duplicate image path/sample ID at manifest row {i}")
        seen_paths.add(p)
        seen_ids.add(sid)
        rows.append(dict(image_path=p, domain=domain, domain_id=DOMAIN_TO_ID.get(domain, -1),
                         label=label, label_name="Real" if label == 0 else "Fake",
                         sample_id=sid, manifest_index=i,
                         group_id=str(row.get("group_id", "")).strip(),
                         split=canonical_split(row.get("split", "")),
                         benchmark_id=str(row.get("benchmark_id", "")).strip()))
    return rows


def assert_disjoint(left, right):
    """Check resolved paths and supplied groups across predefined splits."""
    paths_l = {Path(r['image_path']).resolve() for r in left}
    paths_r = {Path(r['image_path']).resolve() for r in right}
    groups_l = {r['group_id'] for r in left if r.get('group_id')}
    groups_r = {r['group_id'] for r in right if r.get('group_id')}
    if paths_l & paths_r or groups_l & groups_r:
        raise ValueError("Train/Dev image path/group overlap")
