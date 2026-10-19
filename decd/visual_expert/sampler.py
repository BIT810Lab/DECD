"""Domain-label balanced batch sampling."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence

from torch.utils.data import Sampler

EXPECTED_CELLS: tuple[tuple[str, int], ...] = (
    ("Cartoon", 0),
    ("Cartoon", 1),
    ("Chart", 0),
    ("Chart", 1),
    ("Photo_and_poster", 0),
    ("Photo_and_poster", 1),
)


class DomainLabelBalancedBatchSampler(Sampler[list[int]]):
    """Yield batches balanced across the six expected domain-label cells.

    Each cell is shuffled independently. A cell is reshuffled and cycled when
    exhausted, matching the reported fixed-batch training protocol.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        batch_size: int,
        num_batches: int,
        seed: int,
        start_batch: int = 0,
    ) -> None:
        if batch_size <= 0 or batch_size % len(EXPECTED_CELLS) != 0:
            raise ValueError("batch_size must be a positive multiple of 6")
        if num_batches <= 0:
            raise ValueError("num_batches must be positive")
        if start_batch < 0 or start_batch > num_batches:
            raise ValueError("start_batch must be between 0 and num_batches")

        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.start_batch = int(start_batch)
        self.seed = int(seed)
        self.samples_per_cell = self.batch_size // len(EXPECTED_CELLS)
        self.cell_indices = {cell: [] for cell in EXPECTED_CELLS}

        for index, row in enumerate(rows):
            cell = (str(row["domain"]), int(row["label"]))
            if cell not in self.cell_indices:
                raise ValueError(f"unexpected domain-label cell: {cell}")
            self.cell_indices[cell].append(index)

        empty = [cell for cell, indices in self.cell_indices.items() if not indices]
        if empty:
            raise ValueError(f"empty domain-label cells: {empty}")

    def __len__(self) -> int:
        return self.num_batches - self.start_batch

    def _draw(
        self,
        cell: tuple[str, int],
        count: int,
        rng: random.Random,
        queues: dict[tuple[str, int], list[int]],
        cursors: dict[tuple[str, int], int],
    ) -> list[int]:
        selected: list[int] = []
        while len(selected) < count:
            queue = queues[cell]
            cursor = cursors[cell]
            if cursor >= len(queue):
                queue = list(self.cell_indices[cell])
                rng.shuffle(queue)
                queues[cell] = queue
                cursor = 0

            take = min(count - len(selected), len(queue) - cursor)
            selected.extend(queue[cursor : cursor + take])
            cursors[cell] = cursor + take
        return selected

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        queues: dict[tuple[str, int], list[int]] = {}
        cursors: dict[tuple[str, int], int] = {}

        for cell, indices in self.cell_indices.items():
            queue = list(indices)
            rng.shuffle(queue)
            queues[cell] = queue
            cursors[cell] = 0

        for batch_index in range(self.num_batches):
            batch: list[int] = []
            for cell in EXPECTED_CELLS:
                batch.extend(
                    self._draw(
                        cell,
                        self.samples_per_cell,
                        rng,
                        queues,
                        cursors,
                    )
                )
            rng.shuffle(batch)
            # Reconstruct the exact queue/cursor/RNG state without loading
            # skipped images when resuming at a completed cycle boundary.
            if batch_index >= self.start_batch:
                yield batch
