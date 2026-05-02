import pytest
from src.common.LLMProvider import LLMProvider, LLMSettings, LLAMA_CPP_MODEL_PATHS


class TestLLMProvider:

    def test_field_access(self):
        p = LLMProvider(provider="ollama", model_name="qwen3:14b")
        assert p.provider == "ollama"
        assert p.model_name == "qwen3:14b"

    def test_getitem(self):
        p = LLMProvider(provider="openai", model_name="gpt-4")
        assert p["provider"] == "openai"
        assert p["model_name"] == "gpt-4"

    def test_equality(self):
        p1 = LLMProvider(provider="ollama", model_name="llama3")
        p2 = LLMProvider(provider="ollama", model_name="llama3")
        assert p1 == p2

    def test_inequality_different_model(self):
        p1 = LLMProvider(provider="ollama", model_name="llama3")
        p2 = LLMProvider(provider="ollama", model_name="qwen3")
        assert p1 != p2

    def test_inequality_different_provider(self):
        p1 = LLMProvider(provider="ollama", model_name="llama3")
        p2 = LLMProvider(provider="openai", model_name="llama3")
        assert p1 != p2


class TestLLMSettings:

    def test_required_fields(self):
        s = LLMSettings(provider="ollama", model_name="llama3", temperature=0.7)
        assert s.provider == "ollama"
        assert s.model_name == "llama3"
        assert s.temperature == 0.7

    def test_default_context_window_is_none(self):
        s = LLMSettings(provider="ollama", model_name="llama3", temperature=0.0)
        assert s.context_window is None

    def test_default_top_p(self):
        s = LLMSettings(provider="ollama", model_name="llama3", temperature=0.0)
        assert s.top_p == 1

    def test_default_no_thinking_is_false(self):
        s = LLMSettings(provider="ollama", model_name="llama3", temperature=0.0)
        assert s.no_thinking is False

    def test_getitem_all_fields(self):
        s = LLMSettings(
            provider="deepseek", model_name="deepseek-chat",
            temperature=0.5, context_window=8192, top_p=0.9, no_thinking=True
        )
        assert s["provider"] == "deepseek"
        assert s["model_name"] == "deepseek-chat"
        assert s["temperature"] == 0.5
        assert s["context_window"] == 8192
        assert s["top_p"] == 0.9
        assert s["no_thinking"] is True

    def test_inherits_from_llm_provider(self):
        s = LLMSettings(provider="openai", model_name="gpt-4", temperature=0.0)
        assert isinstance(s, LLMProvider)

    def test_equality(self):
        s1 = LLMSettings(provider="ollama", model_name="llama3", temperature=0.0)
        s2 = LLMSettings(provider="ollama", model_name="llama3", temperature=0.0)
        assert s1 == s2

    def test_inequality_different_temperature(self):
        s1 = LLMSettings(provider="ollama", model_name="llama3", temperature=0.0)
        s2 = LLMSettings(provider="ollama", model_name="llama3", temperature=0.5)
        assert s1 != s2


class TestLlamaCppModelPaths:

    def test_is_dict(self):
        assert isinstance(LLAMA_CPP_MODEL_PATHS, dict)

    def test_has_entries(self):
        assert len(LLAMA_CPP_MODEL_PATHS) > 0

    def test_known_models_present(self):
        assert "qwen3:14b" in LLAMA_CPP_MODEL_PATHS
        assert "qwen3:8b" in LLAMA_CPP_MODEL_PATHS

    def test_values_are_strings(self):
        for key, value in LLAMA_CPP_MODEL_PATHS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_paths_contain_sha256(self):
        for value in LLAMA_CPP_MODEL_PATHS.values():
            assert "sha256" in value
