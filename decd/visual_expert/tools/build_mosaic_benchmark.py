#!/usr/bin/env python3
"""Freeze a NEW test-only benchmark; never a training split or paper result."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from decd.visual_expert.manifests import (
    DOMAIN_TO_ID, CSV_FIELDS, file_sha256, read_manifest, exclusion_identities,
    public_rows, write_csv_exclusive, identities, read_csv,
)

BENCHMARK_ID = 'mosaic_benchmark_6000_v1'
CELLS = tuple((d, y) for d in DOMAIN_TO_ID for y in (0, 1))


def order_key(row):
    return (DOMAIN_TO_ID[row['domain']], row['label'], row['original_sha256'], row['sample_id'])


def select_stratified(rows, *, seed=2026, exclusion_hashes=frozenset(), exclusion_groups=frozenset()):
    """Fixed six x1000 sampling. Ordering does not depend on metadata row order."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    eligible, rejected, seen = [], [], {}
    for r in sorted(rows, key=order_key):
        cell = (r['domain'], r['label'])
        if cell not in CELLS:
            raise ValueError('unexpected domain-label cell')
        hs = identities([r])
        if hs & exclusion_hashes or (r.get('group_id') and r['group_id'] in exclusion_groups):
            rejected.append((r, 'specified_exclusion_identity_or_group'))
            continue
        prior = [seen[h] for h in hs if h in seen]
        if prior:
            if any((p['domain'], p['label']) != cell for p in prior):
                raise ValueError('conflicting domain/label for duplicate source identity')
            rejected.append((r, 'duplicate_identity'))
            continue
        for h in hs: seen[h] = r
        eligible.append(r)
    counts = Counter((r['domain'], r['label']) for r in eligible)
    insufficient = {f'{d}/{y}': counts[d, y] for d, y in CELLS if counts[d, y] < 1000}
    if insufficient:
        raise ValueError(f'fewer than 1000 eligible identities in cells: {insufficient}')
    rng = random.Random(seed)
    selected = []
    for cell in CELLS:
        pool = [r for r in eligible if (r['domain'], r['label']) == cell]
        selected.extend(rng.sample(pool, 1000))
    selected.sort(key=order_key)
    chosen = {r['original_sha256'] for r in selected}
    remaining = [r for r in eligible if r['original_sha256'] not in chosen]
    assert len(selected) == 6000 and len(identities(selected)) >= 6000
    return selected, remaining, rejected, counts


def freeze_benchmark(metadata, image_root, output_dir, *, seed=2026, exclusions=()):
    out = Path(output_dir).expanduser().resolve()
    if out.exists():
        raise FileExistsError('output directory exists; use --verify, never silently resample')
    metadata = Path(metadata).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    before_sha = file_sha256(metadata)
    rows = read_manifest(metadata, image_root=image_root, require_sha=True, allow_duplicates=True)
    # Pool records must not already designate historical test membership.
    if any(r.get('benchmark_id') or r.get('split', '').lower() in ('test', 'benchmark_test') for r in rows):
        raise ValueError('input pool is already marked as a test/benchmark; do not repurpose it')
    hashes, groups, provenance = exclusion_identities(exclusions)
    selected, remaining, rejected, counts = select_stratified(
        rows, seed=seed, exclusion_hashes=hashes, exclusion_groups=groups)
    if file_sha256(metadata) != before_sha:
        raise ValueError('metadata changed during validation')
    out.mkdir(parents=True, exist_ok=False)
    benchmark = out / 'mosaic_benchmark_6000.csv'
    leftover = out / 'mosaic_remaining.csv'
    write_csv_exclusive(benchmark, public_rows(selected, benchmark, split='test', benchmark_id=BENCHMARK_ID))
    write_csv_exclusive(leftover, public_rows(remaining, leftover, split='unassigned'))
    rejected_path = out / 'mosaic_rejected.csv'
    rejected_rows = []
    for r, reason in rejected:
        item = public_rows([r], rejected_path, split='excluded')[0]
        item['reason'] = reason; rejected_rows.append(item)
    write_csv_exclusive(rejected_path, rejected_rows, CSV_FIELDS + ('reason',))
    report = {
        'protocol': BENCHMARK_ID, 'seed': seed,
        'status': 'FROZEN_CANDIDATE_NOT_HISTORICAL_PAPER_BENCHMARK',
        'intended_use': 'test-only; no training/risk/calibration/hyperparameter selection',
        'input_manifest': metadata.name, 'input_manifest_sha256': before_sha,
        'input_rows': len(rows), 'selected_rows': 6000, 'remaining_rows': len(remaining),
        'rejected_rows': len(rejected), 'exclusions': provenance,
        'sampling_rule': 'sort cells/domain order Cartoon,Chart,Photo_and_poster; label0,1; sort identities; Python Random(seed).sample(cell,1000) without replacement',
        'python_version': sys.version.split()[0],
        'eligible_cell_counts': {f'{d}/{y}': counts[d, y] for d, y in CELLS},
        'selected_cell_counts': {f'{d}/{y}': 1000 for d, y in CELLS},
        'selected_identities': [{'sample_id': r['sample_id'], 'original_sha256': r['original_sha256'],
                                 'sha256': r['sha256']} for r in selected],
        'identity_sha_definition': 'original source-file SHA; sha256 is current input-file bytes',
        'isolation': {
            'specified_exclusion_exact_identity': 'CHECKED' if provenance else 'NO_EXCLUSION_MANIFESTS_SUPPLIED',
            'near_duplicate': 'NOT_VERIFIED', 'source_group_isolation': 'NOT_VERIFIED',
            'trusted_group_id_rows': sum(bool(r.get('group_id')) for r in rows),
            'group_id_claim_is_not_a_verified_group_id': True,
        },
        'rights_and_quality': 'NOT_VERIFIED; metadata claims are not publication approval',
        'outputs': {p.name: file_sha256(p) for p in [benchmark, leftover, rejected_path]},
    }
    freeze = out / 'benchmark_freeze.json'
    with freeze.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    # Separate digest permits detecting accidental edits of the freeze record too.
    with (out / 'benchmark_freeze.sha256').open('x', encoding='ascii') as f:
        f.write(file_sha256(freeze) + '\n')
    return report


def verify_frozen(output_dir, *, metadata=None, image_root=None, exclusions=None):
    """Read-only reuse. Never samples and never writes, even on failure."""
    out = Path(output_dir).expanduser().resolve()
    freeze = out / 'benchmark_freeze.json'
    if file_sha256(freeze) != (out / 'benchmark_freeze.sha256').read_text().strip():
        raise ValueError('freeze record SHA mismatch')
    report = json.loads(freeze.read_text())
    if report.get('protocol') != BENCHMARK_ID or report.get('selected_rows') != 6000:
        raise ValueError('not a valid Benchmark6000 freeze record')
    if set(report['outputs']) != {'mosaic_benchmark_6000.csv', 'mosaic_remaining.csv', 'mosaic_rejected.csv'}:
        raise ValueError('unexpected frozen output names')
    for name, digest in report['outputs'].items():
        if file_sha256(out / name) != digest:
            raise ValueError(f'frozen output SHA mismatch: {name}')
    selected = read_manifest(out / 'mosaic_benchmark_6000.csv', require_sha=True, role='test')
    counts = Counter((r['domain'], r['label']) for r in selected)
    if len(selected) != 6000 or any(counts[c] != 1000 for c in CELLS):
        raise ValueError('frozen cell counts invalid')
    if any(r.get('benchmark_id') != BENCHMARK_ID for r in selected):
        raise ValueError('missing test-only marker')
    actual = [{'sample_id': r['sample_id'], 'original_sha256': r['original_sha256'],
               'sha256': r['sha256']} for r in selected]
    if actual != report['selected_identities']:
        raise ValueError('frozen sample identity/order mismatch')
    if metadata is not None:
        if image_root is None or file_sha256(metadata) != report['input_manifest_sha256']:
            raise ValueError('input manifest changed or missing image_root')
        pool = read_manifest(metadata, image_root=image_root, require_sha=True, allow_duplicates=True)
        pool_keys = {(r['sha256'], r['original_sha256'], r['domain'], r['label']) for r in pool}
        if any((r['sha256'], r['original_sha256'], r['domain'], r['label']) not in pool_keys for r in selected):
            raise ValueError('frozen sample is not in the validated pool')
    if exclusions is not None:
        hashes, groups, provenance = exclusion_identities(exclusions)
        if provenance != report['exclusions']:
            raise ValueError('exclusion manifests changed')
        if identities(selected) & hashes or any(r.get('group_id') in groups for r in selected):
            raise ValueError('frozen benchmark overlaps exclusions')
    elif report['exclusions']:
        raise ValueError('reuse requires the same --exclude-manifest arguments for SHA verification')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--exclude-manifest', action='append', type=Path, default=None)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.verify:
        report = verify_frozen(args.output_dir, metadata=args.metadata,
                               image_root=args.image_root, exclusions=args.exclude_manifest)
    else:
        if args.metadata is None or args.image_root is None:
            p.error('--metadata and --image-root are required when creating a freeze')
        report = freeze_benchmark(args.metadata, args.image_root, args.output_dir,
                                  seed=args.seed, exclusions=args.exclude_manifest or [])
    print(json.dumps({k: report[k] for k in ['status', 'seed', 'selected_rows', 'remaining_rows', 'isolation']}, indent=2))


if __name__ == '__main__':
    main()
