"""Small production-test fixtures; none of them performs network I/O."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from research_rag.settings import Settings


@pytest.fixture
def settings_factory(tmp_path: Path):
    """Create isolated settings whose mutable state lives under ``tmp_path``."""

    def make(**overrides) -> Settings:
        values = {
            "openai_api_key": SecretStr("offline-test-key"),
            "data_dir": tmp_path / "data",
            "corpus_path": tmp_path / "papers.yaml",
            "golden_questions_path": tmp_path / "golden_questions.yaml",
            "download_retries": 0,
        }
        values.update(overrides)
        return Settings(**values)

    return make
