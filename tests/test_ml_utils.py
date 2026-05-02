import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open

from src.common.ml_utils import (
    is_running_in_container,
    get_device,
    get_gpu_usage,
    millisec_to_time,
)


class TestIsRunningInContainer:

    def test_true_when_dockerenv_exists(self):
        def os_path_exists(path):
            return path == "/.dockerenv"

        with patch("os.path.exists", side_effect=os_path_exists):
            assert is_running_in_container() is True

    def test_true_when_containerenv_exists(self):
        def os_path_exists(path):
            return path == "/run/.containerenv"

        with patch("os.path.exists", side_effect=os_path_exists):
            assert is_running_in_container() is True

    def test_true_when_cgroup_contains_docker(self):
        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", mock_open(read_data="12:cpu:/docker/abc123\n")):
                assert is_running_in_container() is True

    def test_true_when_cgroup_contains_kubepods(self):
        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", mock_open(read_data="11:cpu:/kubepods/pod123\n")):
                assert is_running_in_container() is True

    def test_true_when_cgroup_contains_containerd(self):
        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", mock_open(read_data="1:name=systemd:/containerd/task\n")):
                assert is_running_in_container() is True

    def test_false_in_normal_environment(self):
        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", side_effect=FileNotFoundError):
                assert is_running_in_container() is False

    def test_false_on_cgroup_permission_error(self):
        with patch("os.path.exists", return_value=False):
            with patch("builtins.open", side_effect=PermissionError):
                assert is_running_in_container() is False


class TestGetDevice:

    def test_force_device_env_overrides_all(self, monkeypatch):
        monkeypatch.setenv("FORCE_DEVICE", "cuda")
        assert get_device() == "cuda"

    def test_force_device_mps(self, monkeypatch):
        monkeypatch.setenv("FORCE_DEVICE", "mps")
        assert get_device() == "mps"

    def test_cuda_when_cuda_provider_available(self, monkeypatch):
        monkeypatch.delenv("FORCE_DEVICE", raising=False)
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ]
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            assert get_device() == "cuda"

    def test_mps_when_coreml_provider_available(self, monkeypatch):
        monkeypatch.delenv("FORCE_DEVICE", raising=False)
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider", "CPUExecutionProvider"
        ]
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            assert get_device() == "mps"

    def test_mps_when_apple_provider_available(self, monkeypatch):
        monkeypatch.delenv("FORCE_DEVICE", raising=False)
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["AppleExecutionProvider"]
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            assert get_device() == "mps"

    def test_cpu_when_only_cpu_provider(self, monkeypatch):
        monkeypatch.delenv("FORCE_DEVICE", raising=False)
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            assert get_device() == "cpu"

    def test_cpu_fallback_when_onnxruntime_unavailable(self, monkeypatch):
        monkeypatch.delenv("FORCE_DEVICE", raising=False)
        with patch.dict(sys.modules, {"onnxruntime": None}):
            assert get_device() == "cpu"

    def test_cpu_fallback_on_provider_exception(self, monkeypatch):
        monkeypatch.delenv("FORCE_DEVICE", raising=False)
        mock_ort = MagicMock()
        mock_ort.get_available_providers.side_effect = RuntimeError("detection failed")
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            assert get_device() == "cpu"


class TestGetGpuUsage:

    def test_returns_correct_percentage(self):
        with patch("subprocess.check_output", return_value="1000, 4000\n"):
            assert get_gpu_usage() == 25.0

    def test_full_gpu_usage(self):
        with patch("subprocess.check_output", return_value="8000, 8000\n"):
            assert get_gpu_usage() == 100.0

    def test_returns_zero_on_subprocess_error(self):
        with patch("subprocess.check_output", side_effect=Exception("nvidia-smi not found")):
            assert get_gpu_usage() == 0.0

    def test_returns_zero_on_parsing_error(self):
        with patch("subprocess.check_output", return_value="not valid output"):
            assert get_gpu_usage() == 0.0


class TestMillisecToTime:

    @pytest.mark.parametrize("ms,expected", [
        (0, "00:00:00"),
        (1000, "00:00:01"),
        (59000, "00:00:59"),
        (60000, "00:01:00"),
        (3599000, "00:59:59"),
        (3600000, "01:00:00"),
        (3661000, "01:01:01"),
        (86400000, "24:00:00"),
        (90061000, "25:01:01"),
        (500, "00:00:00"),    # sub-second truncates to 0
        (1999, "00:00:01"),   # truncates, does not round
    ])
    def test_conversion(self, ms, expected):
        assert millisec_to_time(ms) == expected
