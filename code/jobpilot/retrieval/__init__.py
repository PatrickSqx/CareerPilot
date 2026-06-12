"""Retrieval interfaces for JobPilot Phase 2."""

from jobpilot.retrieval.ann_index import ANNRetriever
from jobpilot.retrieval.baseline import TfidfBaselineRetriever
from jobpilot.retrieval.embeddings import EmbeddingStore, build_or_load_job_embeddings, load_job_rows

__all__ = [
    "ANNRetriever",
    "EmbeddingStore",
    "TfidfBaselineRetriever",
    "build_or_load_job_embeddings",
    "load_job_rows",
]
