"""Batch-stream ingestion helpers."""

from __future__ import annotations

from typing import Iterable, Iterator, TypeVar


T = TypeVar("T")


def batched(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Yield small batches from an iterable to demonstrate batch-stream ingestion."""

    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

