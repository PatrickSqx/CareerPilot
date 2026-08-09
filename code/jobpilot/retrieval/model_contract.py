"""Immutable production contract for the bundled semantic retrieval model."""

from __future__ import annotations


SENTENCE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SENTENCE_EMBEDDING_DIMENSION = 384
SENTENCE_EMBEDDINGS_NORMALIZED = True
MODEL_MANIFEST_SCHEMA_VERSION = 1
MODEL_MANIFEST_FILENAME = "jobpilot_model_manifest.json"

MODEL_HUB_FILES = (
    "1_Pooling/config.json",
    "README.md",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
MODEL_REQUIRED_FILES = (*MODEL_HUB_FILES, "LICENSE")
MODEL_EXPECTED_SHA256 = {
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
}
