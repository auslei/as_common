import pytest
from pathlib import Path

from src.common.file_utils import ensure_dir


class TestEnsureDir:

    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "new_dir"
        assert not new_dir.exists()
        ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_returns_path_object(self, tmp_path):
        result = ensure_dir(tmp_path / "test")
        assert isinstance(result, Path)

    def test_returned_path_matches_input(self, tmp_path):
        target = tmp_path / "specific_dir"
        result = ensure_dir(target)
        assert result == target

    def test_accepts_string_path(self, tmp_path):
        target = str(tmp_path / "from_string")
        result = ensure_dir(target)
        assert Path(target).exists()
        assert isinstance(result, Path)

    def test_existing_directory_no_error(self, tmp_path):
        existing = tmp_path / "existing"
        existing.mkdir()
        ensure_dir(existing)  # idempotent — must not raise
        assert existing.exists()

    def test_creates_nested_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        ensure_dir(nested)
        assert nested.exists()

    def test_deeply_nested(self, tmp_path):
        deep = tmp_path / "x" / "y" / "z" / "w"
        ensure_dir(deep)
        assert deep.is_dir()
