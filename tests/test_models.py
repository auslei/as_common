import json
import pytest

from src.common.models import ProjectMetadata


class TestProjectMetadata:

    def test_project_name_and_sector_required(self):
        m = ProjectMetadata(project_name="TestProject", sector="Technology")
        assert m.project_name == "TestProject"
        assert m.sector == "Technology"

    def test_default_language_is_en(self):
        m = ProjectMetadata(project_name="P", sector="Finance")
        assert m.language == "en"

    def test_default_description_is_empty(self):
        m = ProjectMetadata(project_name="P", sector="Finance")
        assert m.description == ""

    def test_explicit_language(self):
        m = ProjectMetadata(project_name="P", sector="Finance", language="zh")
        assert m.language == "zh"

    def test_save_creates_project_json(self, tmp_path):
        m = ProjectMetadata(project_name="SaveTest", sector="Energy")
        m.save_to_folder(str(tmp_path))
        assert (tmp_path / "project.json").exists()

    def test_save_load_roundtrip(self, tmp_path):
        original = ProjectMetadata(
            project_name="RoundTrip", sector="Healthcare",
            language="en", description="A detailed description"
        )
        original.save_to_folder(str(tmp_path))
        loaded = ProjectMetadata.load_from_folder(str(tmp_path))
        assert loaded.project_name == original.project_name
        assert loaded.sector == original.sector
        assert loaded.language == original.language
        assert loaded.description == original.description

    def test_load_missing_file_returns_default(self, tmp_path):
        loaded = ProjectMetadata.load_from_folder(str(tmp_path))
        assert loaded.project_name == tmp_path.name
        assert loaded.sector == "General"

    def test_load_uses_folder_name_as_project_name(self, tmp_path):
        folder = tmp_path / "my_project"
        folder.mkdir()
        loaded = ProjectMetadata.load_from_folder(str(folder))
        assert loaded.project_name == "my_project"

    def test_load_strips_trailing_separator(self, tmp_path):
        folder = tmp_path / "proj_a"
        folder.mkdir()
        loaded = ProjectMetadata.load_from_folder(str(folder) + "/")
        assert loaded.project_name == "proj_a"

    def test_saved_json_is_utf8_compatible(self, tmp_path):
        m = ProjectMetadata(project_name="中文项目", sector="科技")
        m.save_to_folder(str(tmp_path))
        raw = (tmp_path / "project.json").read_text(encoding="utf-8")
        assert "中文项目" in raw
        assert "科技" in raw

    def test_saved_json_is_valid_json(self, tmp_path):
        m = ProjectMetadata(project_name="JsonTest", sector="IT")
        m.save_to_folder(str(tmp_path))
        data = json.loads((tmp_path / "project.json").read_text())
        assert data["project_name"] == "JsonTest"
