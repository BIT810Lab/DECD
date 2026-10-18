#!/usr/bin/env python3
"""Convert an explicitly supplied Train/Dev selection; never invent a split."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from decd.visual_expert.manifests import (
    PATH_FIELDS, read_csv, read_manifest, assert_role, resolve_image,
    file_sha256, public_rows, write_csv_exclusive, parse_label, sha_text,
)


def convert(metadata, image_root, selection_manifest, output, split):
    if split not in ('train', 'dev'):
        raise ValueError('only an explicit train or dev selection can be converted')
    output = Path(output).resolve()
    if output.exists() or output.with_suffix('.provenance.json').exists():
        raise FileExistsError('refusing to overwrite an existing manifest/provenance')
    pool = read_manifest(metadata, image_root=image_root, require_sha=True)
    if any(r.get('benchmark_id') or r.get('split', '').lower() == 'test' for r in pool):
        raise ValueError('a test/benchmark pool cannot become Train/Dev')
    fields, selection = read_csv(selection_manifest)
    assert_role(selection, split)
    pf = next((k for k in PATH_FIELDS if k in fields), '')
    if not pf: raise ValueError('selection must explicitly list file_name or image_path')
    by_path = {r['image_path']: r for r in pool}
    rows, seen, seen_ids = [], set(), set()
    for selected in selection:
        path = resolve_image(selected[pf], image_root, confined=True)
        if path not in by_path or path in seen:
            raise ValueError('selection contains an unknown or duplicate image')
        seen.add(path); row = dict(by_path[path])
        assert_role([row], split)
        if (selected.get('label') or selected.get('label_name')) and parse_label(selected) != row['label']:
            raise ValueError('selection label conflicts with pool metadata')
        for field in ('sha256', 'original_sha256', 'standardized_sha256', 'image_sha256', 'file_sha256'):
            declared = sha_text(selected.get(field))
            if declared and declared not in (row['sha256'], row['original_sha256']):
                raise ValueError('selection SHA conflicts with pool metadata')
        group = str(selected.get('group_id', '')).strip()
        if group:
            if row.get('group_id') and group != row['group_id']:
                raise ValueError('selection group conflicts with pool metadata')
            row['group_id'] = group
        sid = str(selected.get('sample_id', '')).strip() or row['sample_id']
        if sid in seen_ids:
            raise ValueError('selection has duplicate sample IDs')
        seen_ids.add(sid); row['sample_id'] = sid
        rows.append(row)
    write_csv_exclusive(output, public_rows(rows, output, split=split))
    provenance = {'role': split, 'rows': len(rows), 'automatic_split': False,
                  'input_sha256': file_sha256(metadata), 'selection_sha256': file_sha256(selection_manifest),
                  'output_sha256': file_sha256(output), 'near_duplicate_and_source_group_isolation': 'NOT_VERIFIED'}
    with output.with_suffix('.provenance.json').open('x', encoding='utf-8') as f:
        json.dump(provenance, f, indent=2)
    return provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--selection-manifest', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'dev'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(convert(a.metadata, a.image_root, a.selection_manifest, a.output, a.split), indent=2))


if __name__ == '__main__': main()
