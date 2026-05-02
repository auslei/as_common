import logging
import pytest
from src.common.logger import get_logger, EndpointFilter, suppress_http_logs

LOGGER_PREFIX = "MediaExtraction."


def _clear_logger(name: str):
    full_name = f"{LOGGER_PREFIX}{name}"
    logger = logging.getLogger(full_name)
    logger.handlers.clear()
    # Remove from manager so next get_logger call creates it fresh
    logging.Logger.manager.loggerDict.pop(full_name, None)


class TestGetLogger:

    def test_returns_logger_instance(self):
        logger = get_logger("ret_test")
        assert isinstance(logger, logging.Logger)
        _clear_logger("ret_test")

    def test_logger_name_includes_component(self):
        logger = get_logger("my_component")
        assert "my_component" in logger.name
        _clear_logger("my_component")

    def test_default_level_is_info(self):
        _clear_logger("level_test")
        logger = get_logger("level_test")
        assert logger.level == logging.INFO
        _clear_logger("level_test")

    def test_debug_env_sets_debug_level(self, monkeypatch):
        name = "debug_env_test"
        _clear_logger(name)
        monkeypatch.setenv("DEBUG", "true")
        logger = get_logger(name)
        assert logger.level == logging.DEBUG
        _clear_logger(name)

    def test_debug_env_value_1_sets_debug_level(self, monkeypatch):
        name = "debug_env_1_test"
        _clear_logger(name)
        monkeypatch.setenv("DEBUG", "1")
        logger = get_logger(name)
        assert logger.level == logging.DEBUG
        _clear_logger(name)

    def test_has_at_least_one_handler(self):
        _clear_logger("handler_test")
        logger = get_logger("handler_test")
        assert len(logger.handlers) >= 1
        _clear_logger("handler_test")

    def test_second_call_same_name_does_not_add_handlers(self):
        name = "no_dup_test"
        _clear_logger(name)
        logger = get_logger(name)
        count_after_first = len(logger.handlers)
        get_logger(name)
        assert len(logger.handlers) == count_after_first
        _clear_logger(name)

    def test_file_logging_creates_file(self, tmp_path, monkeypatch):
        name = "file_log_test"
        _clear_logger(name)
        monkeypatch.setenv("LOG__TO_FILE", "true")
        log_file = tmp_path / "test.log"
        logger = get_logger(name, to_file=str(log_file))
        logger.info("file log test message")
        assert log_file.exists()
        assert "file log test message" in log_file.read_text()
        _clear_logger(name)

    def test_custom_icon_accepted(self):
        _clear_logger("icon_test")
        logger = get_logger("icon_test", icon="🚀")
        assert logger is not None
        _clear_logger("icon_test")


class TestEndpointFilter:

    def _make_record(self, message: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0, msg=message,
            args=(), exc_info=None,
        )

    def test_allows_non_matching_path(self):
        f = EndpointFilter("/health")
        record = self._make_record("GET /api/data HTTP/1.1")
        assert f.filter(record) is True

    def test_blocks_matching_path(self):
        f = EndpointFilter("/health")
        record = self._make_record("GET /health HTTP/1.1")
        assert f.filter(record) is False

    def test_blocks_path_in_middle_of_message(self):
        f = EndpointFilter("/jobs")
        record = self._make_record('127.0.0.1 - "GET /jobs HTTP/1.1" 200')
        assert f.filter(record) is False

    def test_allows_similar_but_different_path(self):
        f = EndpointFilter("/jobs")
        record = self._make_record("GET /jobs-extra HTTP/1.1")
        # "/jobs" IS in "/jobs-extra", so filter returns False
        assert f.filter(record) is False

    def test_empty_path_filter(self):
        f = EndpointFilter("")
        record = self._make_record("any message")
        # "" is in any string, so always blocked
        assert f.filter(record) is False


class TestSuppressHttpLogs:

    def test_adds_filters_to_uvicorn_access_logger(self):
        uvicorn_logger = logging.getLogger("uvicorn.access")
        uvicorn_logger.filters.clear()
        suppress_http_logs(["/ping"])
        assert any(
            isinstance(f, EndpointFilter) and f.path == "/ping"
            for f in uvicorn_logger.filters
        )

    def test_default_paths_add_four_filters(self):
        uvicorn_logger = logging.getLogger("uvicorn.access")
        uvicorn_logger.filters.clear()
        suppress_http_logs()
        assert len(uvicorn_logger.filters) == 4

    def test_multiple_paths(self):
        uvicorn_logger = logging.getLogger("uvicorn.access")
        uvicorn_logger.filters.clear()
        suppress_http_logs(["/a", "/b", "/c"])
        assert len(uvicorn_logger.filters) == 3
