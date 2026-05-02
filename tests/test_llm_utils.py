import re
import sys
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.common.llm_utils import (
    get_message_content,
    get_token_count,
    get_current_date,
    clear_gpu_memory,
    unload_ollama_model,
    get_llm,
    get_llm_legacy,
    extract_token_usage,
    get_tokens_for_model,
    detect_output_language,
    STOP_TOKENS,
)
from src.common.LLMProvider import LLMSettings


class TestGetMessageContent:

    def test_plain_string(self):
        msg = AIMessage(content="Hello world")
        assert get_message_content(msg) == "Hello world"

    def test_strips_think_block(self):
        msg = AIMessage(content="<think>internal reasoning</think>Final answer")
        assert get_message_content(msg) == "Final answer"

    def test_strips_tool_call_block(self):
        msg = AIMessage(content="<tool_call>call data</tool_call>Result")
        assert get_message_content(msg) == "Result"

    def test_strips_orphaned_closing_think_tag(self):
        msg = AIMessage(content="text</think>more")
        assert "</think>" not in get_message_content(msg)

    def test_strips_orphaned_closing_tool_call_tag(self):
        msg = AIMessage(content="text</tool_call>more")
        assert "</tool_call>" not in get_message_content(msg)

    def test_list_content_joined(self):
        msg = AIMessage(content=["part1", "part2"])
        result = get_message_content(msg)
        assert "part1" in result
        assert "part2" in result

    def test_multiline_think_block(self):
        msg = AIMessage(content="<think>\nstep1\nstep2\n</think>conclusion")
        assert get_message_content(msg) == "conclusion"

    def test_empty_string(self):
        msg = AIMessage(content="")
        assert get_message_content(msg) == ""

    def test_human_message(self):
        msg = HumanMessage(content="<think>q</think>Hello")
        assert get_message_content(msg) == "Hello"


class TestGetTokenCount:

    def test_single_message(self):
        msgs = [HumanMessage(content="hello")]
        assert get_token_count(msgs) == 5

    def test_multiple_messages(self):
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        assert get_token_count(msgs) == 7

    def test_empty_list(self):
        assert get_token_count([]) == 0

    def test_think_content_stripped_before_counting(self):
        # <think>...</think> is stripped, only "ans" (3 chars) counted
        msgs = [AIMessage(content="<think>long reasoning here</think>ans")]
        assert get_token_count(msgs) == 3

    def test_cjk_counted_per_character(self):
        msgs = [HumanMessage(content="你好")]
        assert get_token_count(msgs) == 2


class TestGetCurrentDate:

    def test_matches_iso_date_format(self):
        date = get_current_date()
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", date)

    def test_matches_todays_date(self):
        date = get_current_date()
        assert date == datetime.now().strftime("%Y-%m-%d")


class TestClearGpuMemory:

    def test_calls_gc_collect(self):
        import gc
        with patch.object(gc, "collect") as mock_gc:
            clear_gpu_memory()
            mock_gc.assert_called_once()


class TestUnloadOllamaModel:

    def test_posts_to_api_generate(self):
        with patch("src.common.llm_utils.requests.post") as mock_post:
            mock_post.return_value = MagicMock()
            unload_ollama_model("qwen3:14b")
            mock_post.assert_called_once()
            url, kwargs = mock_post.call_args.args[0], mock_post.call_args.kwargs
            assert "/api/generate" in url
            assert kwargs["json"]["keep_alive"] == 0
            assert kwargs["json"]["model"] == "qwen3:14b"

    def test_uses_ollama_base_url_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-host:9999")
        with patch("src.common.llm_utils.requests.post") as mock_post:
            mock_post.return_value = MagicMock()
            unload_ollama_model("model")
            url = mock_post.call_args.args[0]
            assert "custom-host:9999" in url

    def test_uses_ollama_api_url_fallback(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.setenv("OLLAMA_API_URL", "http://fallback:8888")
        with patch("src.common.llm_utils.requests.post") as mock_post:
            mock_post.return_value = MagicMock()
            unload_ollama_model("model")
            url = mock_post.call_args.args[0]
            assert "fallback:8888" in url

    def test_silences_network_errors(self):
        with patch("src.common.llm_utils.requests.post", side_effect=ConnectionError("down")):
            unload_ollama_model("model")  # must not raise


class TestGetLlm:
    # get_llm() uses `from common.unified_llm import UnifiedChatModel` internally,
    # so we patch that path (not src.common.*) and check by class name to avoid
    # the double-import problem when the package is installed.

    def test_returns_unified_chat_model(self):
        settings = LLMSettings(provider="ollama", model_name="llama3", temperature=0.0)
        with patch("common.unified_llm.UnifiedChatModel._init_underlying_model") as mock_init:
            mock_init.return_value = MagicMock()
            result = get_llm(settings)
        assert type(result).__name__ == "UnifiedChatModel"

    def test_no_thinking_flag_inverts_to_thinking_false(self):
        settings = LLMSettings(provider="ollama", model_name="llama3",
                               temperature=0.0, no_thinking=True)
        with patch("common.unified_llm.UnifiedChatModel._init_underlying_model") as mock_init:
            mock_init.return_value = MagicMock()
            result = get_llm(settings)
        assert result.thinking is False

    def test_thinking_enabled_by_default(self):
        settings = LLMSettings(provider="ollama", model_name="llama3", temperature=0.0)
        with patch("common.unified_llm.UnifiedChatModel._init_underlying_model") as mock_init:
            mock_init.return_value = MagicMock()
            result = get_llm(settings)
        assert result.thinking is True


class TestGetLlmLegacy:

    def test_unsupported_provider_raises(self):
        settings = LLMSettings(provider="unknown_xyz", model_name="m", temperature=0.0)
        with pytest.raises(ValueError, match="Unsupported"):
            get_llm_legacy(settings)

    def test_deepseek_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        settings = LLMSettings(provider="deepseek", model_name="deepseek-chat", temperature=0.0)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            get_llm_legacy(settings)

    def test_openai_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = LLMSettings(provider="openai", model_name="gpt-4", temperature=0.0)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_llm_legacy(settings)

    def test_llama_cpp_missing_path_raises(self):
        settings = LLMSettings(provider="llama_cpp", model_name="nonexistent-model-xyz",
                               temperature=0.0)
        with pytest.raises(ValueError, match="Model path not defined"):
            get_llm_legacy(settings)

    def test_deepseek_reasoner_swapped_when_no_thinking(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        settings = LLMSettings(provider="deepseek", model_name="deepseek-reasoner",
                               temperature=0.0, no_thinking=True)
        with patch("src.common.llm_utils.ChatDeepSeek") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm_legacy(settings)
            assert mock_cls.call_args.kwargs["model"] == "deepseek-chat"

    def test_deepseek_reasoner_kept_when_thinking_enabled(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        settings = LLMSettings(provider="deepseek", model_name="deepseek-reasoner",
                               temperature=0.0, no_thinking=False)
        with patch("src.common.llm_utils.ChatDeepSeek") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm_legacy(settings)
            assert mock_cls.call_args.kwargs["model"] == "deepseek-reasoner"

    def test_ollama_uses_default_context_window(self):
        settings = LLMSettings(provider="ollama", model_name="llama3", temperature=0.5)
        with patch("src.common.llm_utils.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm_legacy(settings)
            assert mock_cls.call_args.kwargs["num_ctx"] == 16384

    def test_ollama_uses_provided_context_window(self):
        settings = LLMSettings(provider="ollama", model_name="llama3",
                               temperature=0.0, context_window=8192)
        with patch("src.common.llm_utils.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm_legacy(settings)
            assert mock_cls.call_args.kwargs["num_ctx"] == 8192

    def test_temperature_none_coerced_to_zero(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        settings = LLMSettings(provider="openai", model_name="gpt-4-temp-none-test",
                               temperature=None)
        with patch("src.common.llm_utils.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_llm_legacy(settings)
            assert mock_cls.call_args.kwargs["temperature"] == 0.0

    def test_cache_reuse_same_settings(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        unique_model = "gpt-4-cache-reuse-unique-xyz"
        settings = LLMSettings(provider="openai", model_name=unique_model, temperature=0.0)

        import src.common.llm_utils as utils_mod
        cache_key = ("openai", unique_model, 0.0, None, 1, False)
        utils_mod._LLM_CACHE.pop(cache_key, None)

        with patch("src.common.llm_utils.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            r1 = get_llm_legacy(settings)
            r2 = get_llm_legacy(settings)

        assert r1 is r2
        mock_cls.assert_called_once()


class TestExtractTokenUsage:

    def test_from_usage_metadata(self):
        class FakeResponse:
            usage_metadata = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

        result = extract_token_usage(FakeResponse())
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 20
        assert result["total_tokens"] == 30

    def test_from_response_metadata(self):
        class FakeResponse:
            response_metadata = {"token_usage": {"input_tokens": 5, "output_tokens": 15}}

        result = extract_token_usage(FakeResponse())
        assert result["input_tokens"] == 5
        assert result["output_tokens"] == 15

    def test_returns_zeros_when_no_usage_info(self):
        class FakeResponse:
            pass

        result = extract_token_usage(FakeResponse())
        assert result == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def test_usage_metadata_takes_priority(self):
        class FakeResponse:
            usage_metadata = {"input_tokens": 99, "output_tokens": 1, "total_tokens": 100}
            response_metadata = {"token_usage": {"input_tokens": 0}}

        result = extract_token_usage(FakeResponse())
        assert result["input_tokens"] == 99


class TestGetTokensForModel:

    @pytest.mark.parametrize("model_name,expected_key", [
        ("qwen3:14b", "qwen"),
        ("Qwen2.5-72B-Instruct", "qwen"),
        ("deepseek-r1-distill-qwen-14b", "qwen"),
        ("llama-3-8b-instruct", "llama3"),
        ("llama3.1:8b", "llama3"),
        ("deepseek-r1-distill-llama-70b", "llama3"),
        ("mistral-7b-instruct", "mistral"),
        ("mixtral-8x7b-instruct", "mistral"),
        ("gemma2:9b", "gemma"),
        ("phi-3-mini-128k", "phi3"),
        ("phi3:14b", "phi3"),
        ("deepseek-v3", "deepseek_base"),
        ("deepseek-chat", "deepseek_base"),
    ])
    def test_model_family_stop_tokens(self, model_name, expected_key):
        tokens = get_tokens_for_model(model_name)
        assert tokens == STOP_TOKENS[expected_key]

    def test_unknown_model_returns_empty_list(self):
        assert get_tokens_for_model("totally-unknown-model-xyz") == []

    def test_case_insensitive(self):
        assert get_tokens_for_model("QWEN3:14B") == STOP_TOKENS["qwen"]


class TestDetectOutputLanguage:

    def test_chinese(self):
        assert detect_output_language("你好世界") == "Chinese"

    def test_japanese_hiragana(self):
        assert detect_output_language("こんにちは") == "Japanese"

    def test_japanese_katakana(self):
        assert detect_output_language("コンニチハ") == "Japanese"

    def test_korean(self):
        assert detect_output_language("안녕하세요") == "Korean"

    def test_russian(self):
        assert detect_output_language("Привет мир") == "Russian/Cyrillic"

    def test_arabic(self):
        assert detect_output_language("مرحبا") == "Arabic"

    def test_english(self):
        assert detect_output_language("Hello world") == "English"

    def test_empty_string_returns_same_as_request(self):
        assert detect_output_language("") == "same as user's request"

    def test_numbers_only_returns_english(self):
        assert detect_output_language("12345") == "English"
