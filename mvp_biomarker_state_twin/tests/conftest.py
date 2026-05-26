from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path() -> Path:
    base_dir = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = base_dir / uuid.uuid4().hex
    temporary_path.mkdir(parents=True, exist_ok=True)
    try:
        yield temporary_path
    finally:
        shutil.rmtree(temporary_path, ignore_errors=True)
