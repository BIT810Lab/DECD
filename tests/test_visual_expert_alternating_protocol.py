from collections import Counter

import pytest

from decd.visual_expert.protocol import (
    validate_balanced_split,
    validate_exact_exposures,
)
from decd.visual_expert.sampler import (
    EXPECTED_CELLS,
    DomainLabelBalancedBatchSampler,
)


def make_rows(per_cell: int):
    rows = []
    for domain, label in EXPECTED_CELLS:
        for i in range(per_cell):
            rows.append({
                "sample_id": f"{domain}-{label}-{i}",
                "domain": domain,
                "label": label,
            })
    return rows


def count_exposures(rows, batches, seed):
    sampler = DomainLabelBalancedBatchSampler(
        rows, batch_size=6, num_batches=batches, seed=seed
    )
    counts = Counter()
    for batch in sampler:
        counts.update(str(rows[index]["sample_id"]) for index in batch)
    return counts


def test_full_train_split_requires_all_4200_samples():
    rows = make_rows(700)
    result = validate_balanced_split(
        rows, expected_per_cell=700, split_name="Train"
    )
    assert len(rows) == 4200
    assert set(result.values()) == {700}


def test_incomplete_split_is_rejected():
    rows = make_rows(700)[:-1]
    with pytest.raises(ValueError, match="does not match the required balanced split"):
        validate_balanced_split(rows, expected_per_cell=700, split_name="Train")


def test_reference_sampler_visits_each_sample_once():
    rows = make_rows(700)
    counts = count_exposures(rows, batches=700, seed=2026)
    summary = validate_exact_exposures(
        rows, counts, expected_repeats=1, phase_name="reference"
    )
    assert summary["unique_samples"] == 4200
    assert summary["total_exposures"] == 4200


def test_continuous_bce_sampler_visits_each_sample_twice_across_cycles():
    rows = make_rows(700)
    # One sampler iterator is consumed continuously: 14 blocks of 100 batches.
    sampler = DomainLabelBalancedBatchSampler(
        rows, batch_size=6, num_batches=1400, seed=2027
    )
    iterator = iter(sampler)
    counts = Counter()
    for _cycle in range(14):
        for _ in range(100):
            batch = next(iterator)
            counts.update(str(rows[index]["sample_id"]) for index in batch)
    summary = validate_exact_exposures(
        rows, counts, expected_repeats=2, phase_name="BCE continuation"
    )
    assert summary["total_exposures"] == 8400


def test_joint_sampler_visits_each_sample_once():
    rows = make_rows(700)
    sampler = DomainLabelBalancedBatchSampler(
        rows, batch_size=6, num_batches=700, seed=2028
    )
    counts = Counter()
    iterator = iter(sampler)
    for _cycle in range(14):
        for _ in range(50):
            batch = next(iterator)
            counts.update(str(rows[index]["sample_id"]) for index in batch)
    summary = validate_exact_exposures(
        rows, counts, expected_repeats=1, phase_name="joint repair"
    )
    assert summary["total_exposures"] == 4200


def test_sampler_start_batch_reconstructs_the_same_suffix():
    rows = make_rows(4)
    full = list(DomainLabelBalancedBatchSampler(rows, batch_size=6, num_batches=8, seed=91))
    resumed = list(
        DomainLabelBalancedBatchSampler(
            rows, batch_size=6, num_batches=8, seed=91, start_batch=3
        )
    )
    assert resumed == full[3:]
    assert len(resumed) == 5
