import pytest
from pathlib import Path
from unittest.mock import patch

from src.common.docker_utils import is_docker_runtime, resolve_project_base, resolve_project_dir


class TestIsDockerRuntime:

    def test_true_when_dockerenv_file_exists(self):
        def path_exists(self):
            return str(self) == "/.dockerenv"

        with patch.object(Path, "exists", path_exists):
            assert is_docker_runtime() is True

    def test_true_when_app_projects_exists(self):
        def path_exists(self):
            return str(self) == "/app/projects"

        with patch.object(Path, "exists", path_exists):
            assert is_docker_runtime() is True

    def test_false_when_neither_path_exists(self):
        with patch.object(Path, "exists", return_value=False):
            assert is_docker_runtime() is False

    def test_true_when_both_paths_exist(self):
        with patch.object(Path, "exists", return_value=True):
            assert is_docker_runtime() is True


class TestResolveProjectBase:

    def test_returns_docker_path_in_container(self):
        with patch("src.common.docker_utils.is_docker_runtime", return_value=True):
            assert resolve_project_base() == Path("/app/projects")

    def test_returns_relative_path_on_host(self):
        with patch("src.common.docker_utils.is_docker_runtime", return_value=False):
            assert resolve_project_base() == Path("projects")

    def test_returns_path_object(self):
        with patch("src.common.docker_utils.is_docker_runtime", return_value=False):
            assert isinstance(resolve_project_base(), Path)


class TestResolveProjectDir:

    def test_docker_path_includes_project_name(self):
        with patch("src.common.docker_utils.is_docker_runtime", return_value=True):
            result = resolve_project_dir("my_project")
            assert result == Path("/app/projects/my_project")

    def test_host_path_includes_project_name(self):
        with patch("src.common.docker_utils.is_docker_runtime", return_value=False):
            result = resolve_project_dir("my_project")
            assert result == Path("projects/my_project")

    def test_project_name_is_last_component(self):
        with patch("src.common.docker_utils.is_docker_runtime", return_value=False):
            result = resolve_project_dir("test-proj")
            assert result.name == "test-proj"

    def test_returns_path_object(self):
        with patch("src.common.docker_utils.is_docker_runtime", return_value=False):
            assert isinstance(resolve_project_dir("p"), Path)

    def test_different_project_names(self):
        with patch("src.common.docker_utils.is_docker_runtime", return_value=False):
            r1 = resolve_project_dir("proj_a")
            r2 = resolve_project_dir("proj_b")
            assert r1 != r2
            assert r1.parent == r2.parent
