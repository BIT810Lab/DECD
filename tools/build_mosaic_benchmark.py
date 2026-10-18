#!/usr/bin/env python3
"""Select an unassigned MOSAIC6000 pool; never invent official experiment splits.

The filename is retained for CLI compatibility. Historical test-only freeze
records can be verified read-only, but are not the final MOSAIC6000 dataset.
"""
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
SELECTION_ID = 'mosaic6000_selected_pool_v2'
CELLS = tuple((d, y) for d in DOMAIN_TO_ID for y in (0, 1))


def order_key(row):
    return (DOMAIN_TO_ID[row['domain']], row['label'], row['original_sha256'],
            row['sample_id'], row['sha256'], str(row['image_path']))


def select_stratified(rows, *, seed=2026, exclusion_hashes=frozenset(), exclusion_groups=frozenset()):
    """Fixed six x1000 sampling. Ordering does not depend on metadata row order."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    # Close only explicitly supplied identity relationships. A converted file
    # links its current SHA to its source SHA; supplied group IDs propagate
    # exclusions, but do not collapse distinct images into one sample.
    parents = {}
    def find(token):
        parents.setdefault(token, token)
        root = token
        while parents[root] != root:
            root = parents[root]
        while parents[token] != token:
            previous = parents[token]
            parents[token] = root
            token = previous
        return root
    def join(tokens):
        tokens = sorted(tokens)
        first = find(tokens[0])
        for token in tokens[1:]:
            parents[find(token)] = first

    sample_ids = {}
    for row in rows:
        hs = identities([row])
        if not hs:
            raise ValueError('candidate lacks a comparable SHA identity')
        key = (row['sha256'], row['original_sha256'])
        if row['sample_id'] in sample_ids and sample_ids[row['sample_id']] != key:
            raise ValueError('same sample_id identifies different images')
        sample_ids[row['sample_id']] = key
        join(hs)
    components = {id(r): find(sorted(identities([r]))[0]) for r in rows}
    for row in rows:
        if row.get('group_id'):
            join(identities([row]) | {'group:' + row['group_id']})
    blocked = {find(h) for h in exclusion_hashes}
    blocked.update(find('group:' + g) for g in exclusion_groups)
    eligible, rejected, seen = [], [], {}
    for r in sorted(rows, key=order_key):
        cell = (r['domain'], r['label'])
        if cell not in CELLS:
            raise ValueError('unexpected domain-label cell')
        hs = identities([r])
        if find(sorted(hs)[0]) in blocked:
            rejected.append((r, 'specified_exclusion_identity_or_group'))
            continue
        component = components[id(r)]
        if component in seen:
            prior = seen[component]
            if (prior['domain'], prior['label']) != cell:
                raise ValueError('conflicting domain/label for duplicate source identity')
            rejected.append((r, 'duplicate_identity'))
            continue
        seen[component] = r
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


def freeze_selection(metadata, image_root, output_dir, *, seed=2026, exclusions=()):
    """Freeze six x1000 selected candidates, with no Train/Dev/Test assignment."""
    out = Path(output_dir).expanduser().resolve()
    if out.exists():
        raise FileExistsError('output directory exists; use --verify, never silently resample')
    metadata = Path(metadata).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    before_sha = file_sha256(metadata)
    rows = read_manifest(metadata, image_root=image_root, require_sha=True, allow_duplicates=True)
    # A candidate pool is not an already assigned Train/Dev/Test partition.
    if any(r.get('benchmark_id') or r.get('split', '').strip().lower()
           not in ('', 'pool', 'candidate', 'unassigned') for r in rows):
        raise ValueError('input pool has assigned split/benchmark roles; do not repurpose it')
    hashes, groups, provenance = exclusion_identities(exclusions)
    selected, remaining, rejected, counts = select_stratified(
        rows, seed=seed, exclusion_hashes=hashes, exclusion_groups=groups)
    if file_sha256(metadata) != before_sha:
        raise ValueError('metadata changed during validation')
    out.mkdir(parents=True, exist_ok=False)
    benchmark = out / 'mosaic_selected_6000.csv'
    leftover = out / 'mosaic_remaining.csv'
    write_csv_exclusive(benchmark, public_rows(selected, benchmark, split='unassigned'))
    write_csv_exclusive(leftover, public_rows(remaining, leftover, split='unassigned'))
    rejected_path = out / 'mosaic_rejected.csv'
    rejected_rows = []
    for r, reason in rejected:
        item = public_rows([r], rejected_path, split='excluded')[0]
        item['reason'] = reason; rejected_rows.append(item)
    write_csv_exclusive(rejected_path, rejected_rows, CSV_FIELDS + ('reason',))
    report = {
        'protocol': SELECTION_ID, 'seed': seed,
        'status': 'SELECTED_POOL_FROZEN_FORMAL_SPLITS_NOT_CREATED',
        'intended_use': 'unassigned selected dataset pool; not a 6000-image Test set',
        'formal_split_status': 'NOT_CREATED_REUSE_OR_AUTHOR_APPROVAL_REQUIRED',
        'required_partition_counts': {'train': 4200, 'dev': 600, 'test': 1200},
        'required_per_domain_label': {'train': 700, 'dev': 100, 'test': 200},
        'input_manifest': metadata.name, 'input_manifest_sha256': before_sha,
        'input_rows': len(rows), 'selected_rows': 6000, 'remaining_rows': len(remaining),
        'rejected_rows': len(rejected), 'exclusions': provenance,
        'remaining_intended_use': 'unassigned; never automatically a training split',
        'identity_closure': 'transitive supplied source/current SHA and group links propagate exclusions; SHA components deduplicate',
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
    freeze = out / 'selection_freeze.json'
    with freeze.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    # Separate digest permits detecting accidental edits of the freeze record too.
    with (out / 'selection_freeze.sha256').open('x', encoding='ascii') as f:
        f.write(file_sha256(freeze) + '\n')
    return report


# Preserve Python callers, not the former all-Test meaning. New output is always
# unassigned; existing legacy freeze directories are never rewritten.
freeze_benchmark = freeze_selection


def verify_frozen(output_dir, *, metadata=None, image_root=None, exclusions=None):
    """Read-only reuse. Never samples and never writes, even on failure."""
    out = Path(output_dir).expanduser().resolve()
    selected_pool = (out / 'selection_freeze.json').is_file()
    if selected_pool and (out / 'benchmark_freeze.json').exists():
        raise ValueError('ambiguous new-pool and legacy-Test freeze records')
    freeze = out / ('selection_freeze.json' if selected_pool else 'benchmark_freeze.json')
    checksum = out / ('selection_freeze.sha256' if selected_pool else 'benchmark_freeze.sha256')
    if file_sha256(freeze) != checksum.read_text().strip():
        raise ValueError('freeze record SHA mismatch')
    report = json.loads(freeze.read_text(encoding='utf-8'))
    expected_protocol = SELECTION_ID if selected_pool else BENCHMARK_ID
    selected_name = 'mosaic_selected_6000.csv' if selected_pool else 'mosaic_benchmark_6000.csv'
    if report.get('protocol') != expected_protocol or report.get('selected_rows') != 6000:
        raise ValueError('not a valid six-cell 6000-selection freeze record')
    if set(report['outputs']) != {selected_name, 'mosaic_remaining.csv', 'mosaic_rejected.csv'}:
        raise ValueError('unexpected frozen output names')
    for name, digest in report['outputs'].items():
        if file_sha256(out / name) != digest:
            raise ValueError(f'frozen output SHA mismatch: {name}')
    selected = read_manifest(out / selected_name, require_sha=True,
                             role='unassigned' if selected_pool else 'test')
    counts = Counter((r['domain'], r['label']) for r in selected)
    if len(selected) != 6000 or any(counts[c] != 1000 for c in CELLS):
        raise ValueError('frozen cell counts invalid')
    if selected_pool:
        if any(r.get('split') != 'unassigned' or r.get('benchmark_id') for r in selected):
            raise ValueError('selected pool must not carry an assigned role or legacy benchmark marker')
        if (report.get('required_partition_counts') != {'train':4200,'dev':600,'test':1200}
                or report.get('required_per_domain_label') != {'train':700,'dev':100,'test':200}):
            raise ValueError('selected pool records the wrong final dataset protocol')
    elif any(r.get('benchmark_id') != BENCHMARK_ID for r in selected):
        raise ValueError('missing legacy test-only marker')
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
    if not selected_pool:
        # This is a read-only interpretation of an old artifact, not migration
        # or approval to reuse it as the final 4200/600/1200 experiment.
        return dict(report, legacy_test_only_artifact=True,
                    final_mosaic6000_split_status='NOT_VERIFIED_NOT_A_FORMAL_DATASET_SPLIT')
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
        report = freeze_selection(args.metadata, args.image_root, args.output_dir,
                                  seed=args.seed, exclusions=args.exclude_manifest or [])
    fields = ['status','seed','selected_rows','remaining_rows','isolation',
              'formal_split_status','legacy_test_only_artifact','final_mosaic6000_split_status']
    print(json.dumps({k:report[k] for k in fields if k in report}, indent=2))


if __name__ == '__main__':
    main()
