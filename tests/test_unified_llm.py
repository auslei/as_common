import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration

from src.common.unified_llm import (
    UnifiedChatModel, clean_reasoning_content, _RAW_LLM_CACHE,
    get_tokens_for_model, STOP_TOKENS,
)


class SampleSchema(BaseModel):
    """Sample schema for testing structured output."""
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")


class TestCleanReasoningContent:

    def test_strips_think_block(self):
        content = "<think>internal reasoning</think>Final answer"
        assert clean_reasoning_content(content) == "Final answer"

    def test_strips_tool_call_block(self):
        content = "<tool_call>call_data</tool_call>Result"
        assert clean_reasoning_content(content) == "Result"

    def test_strips_orphaned_closing_tags(self):
        content = "Some text</think>more</tool_call>"
        result = clean_reasoning_content(content)
        assert "</think>" not in result
        assert "</tool_call>" not in result

    def test_returns_non_string_unchanged(self):
        assert clean_reasoning_content(42) == 42
        assert clean_reasoning_content(None) is None

    def test_adjacent_blocks(self):
        content = "<think>reasoning</think><tool_call>call</tool_call>answer"
        assert clean_reasoning_content(content) == "answer"

    def test_multiline_think_block(self):
        content = "<think>\nstep 1\nstep 2\n</think>conclusion"
        assert clean_reasoning_content(content) == "conclusion"


class TestUnifiedChatModel:

    @pytest.fixture
    def mock_llm(self):
        """Creates a mock LangChain LLM instance."""
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="Mocked response")
        llm.bind_tools.return_value = MagicMock()
        llm.bind.return_value = MagicMock()
        llm.stream.return_value = iter([AIMessageChunk(content="M"), AIMessageChunk(content="o")])
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="Async mocked response"))
        return llm

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_initialization(self, mock_init, mock_llm):
        """Test that initialization correctly sets parameters and calls _init_underlying_model."""
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(
            provider="ollama",
            model_name="qwen3:14b",
            temperature=0.7,
            context_window=4096,
            thinking=False
        )

        assert model.provider == "ollama"
        assert model.model_name == "qwen3:14b"
        assert model.temperature == 0.7
        assert model.context_window == 4096
        assert model.thinking is False

        # Verify _init_underlying_model was called with correct parameters
        mock_init.assert_called_once_with(
            provider="ollama",
            model_name="qwen3:14b",
            temperature=0.7,
            context_window=4096,
            top_p=1.0,
            thinking=False
        )

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_basic_invoke(self, mock_init, mock_llm):
        """Test simple string invocation."""
        mock_init.return_value = mock_llm
        model = UnifiedChatModel(provider="openai", model_name="gpt-4")

        response = model.invoke("Hello")

        assert response.content == "Mocked response"
        mock_llm.invoke.assert_called_once_with("Hello", config=None)

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.build_structured_prompt")
    @patch("src.common.unified_llm.parse_structured_output")
    def test_structured_output_invoke_string(self, mock_parse, mock_build, mock_init, mock_llm):
        """Test invoke with a Pydantic schema using a string input."""
        mock_init.return_value = mock_llm
        mock_build.return_value = "Prompt with schema"
        mock_parse.return_value = SampleSchema(name="Alice", age=30)

        model = UnifiedChatModel(provider="deepseek", model_name="deepseek-chat")

        result = model.invoke("Who is Alice?", schema=SampleSchema)

        assert isinstance(result, SampleSchema)
        assert result.name == "Alice"
        assert result.age == 30

        mock_build.assert_called_once_with("Who is Alice?", SampleSchema)
        mock_llm.invoke.assert_called_once_with("Prompt with schema", config=None)
        mock_parse.assert_called_once_with("Mocked response", SampleSchema, llm=mock_llm)

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.build_structured_prompt")
    @patch("src.common.unified_llm.parse_structured_output")
    def test_structured_output_invoke_list(self, mock_parse, mock_build, mock_init, mock_llm):
        """Test invoke with a Pydantic schema using a list of messages."""
        mock_init.return_value = mock_llm
        mock_build.return_value = "Modified message"
        mock_parse.return_value = SampleSchema(name="Bob", age=40)

        model = UnifiedChatModel(provider="ollama", model_name="llama3")
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me about Bob.")
        ]

        result = model.invoke(messages, schema=SampleSchema)

        assert isinstance(result, SampleSchema)
        assert result.name == "Bob"

        # Verify that only the last human message was modified
        args, kwargs = mock_llm.invoke.call_args
        sent_messages = args[0]
        assert len(sent_messages) == 2
        assert sent_messages[0].content == "You are a helpful assistant."
        assert sent_messages[1].content == "Modified message"
        mock_build.assert_called_once_with("Tell me about Bob.", SampleSchema)
        mock_parse.assert_called_once_with("Mocked response", SampleSchema, llm=mock_llm)

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_bind_tools(self, mock_init, mock_llm):
        """Test that bind_tools returns a new UnifiedChatModel wrapping the bound model."""
        mock_init.return_value = mock_llm
        bound_mock = MagicMock()
        mock_llm.bind_tools.return_value = bound_mock

        model = UnifiedChatModel(provider="openai", model_name="gpt-4", temperature=0.5)
        tools = [lambda x: x]

        new_model = model.bind_tools(tools)

        assert isinstance(new_model, UnifiedChatModel)
        assert new_model.provider == "openai"
        assert new_model.temperature == 0.5
        assert new_model.model == bound_mock
        mock_llm.bind_tools.assert_called_once_with(tools)

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_bind_parameters(self, mock_init, mock_llm):
        """Test that bind returns a new UnifiedChatModel wrapping the bound model."""
        mock_init.return_value = mock_llm
        bound_mock = MagicMock()
        mock_llm.bind.return_value = bound_mock

        model = UnifiedChatModel(provider="ollama", model_name="qwen2.5")

        new_model = model.bind(stop=["\n"])

        assert isinstance(new_model, UnifiedChatModel)
        assert new_model.model == bound_mock
        mock_llm.bind.assert_called_once_with(stop=["\n"])

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_stream_proxy(self, mock_init, mock_llm):
        """Test that stream is correctly proxied."""
        mock_init.return_value = mock_llm
        model = UnifiedChatModel(provider="openai", model_name="gpt-4")

        list(model.stream("Hello"))  # consume iterator

        mock_llm.stream.assert_called_once_with("Hello", config=None)

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_invoke_thinking_override_creates_variant(self, mock_init, mock_llm):
        """Test that passing thinking= at call time creates a variant model."""
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(provider="ollama", model_name="qwen3:14b", thinking=True)

        # Call with thinking=False — should create a variant via a second init call
        model.invoke("Hello", thinking=False)

        # _init_underlying_model called once for base model + once for variant
        assert mock_init.call_count == 2
        # Second call should have thinking=False
        _, variant_kwargs = mock_init.call_args
        assert variant_kwargs["thinking"] is False

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_invoke_no_thinking_cleans_content(self, mock_init, mock_llm):
        """Test that thinking=False strips <think> tags from the response."""
        mock_llm.invoke.return_value = AIMessage(content="<think>reasoning</think>Clean answer")
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(provider="ollama", model_name="qwen3:14b", thinking=False)
        response = model.invoke("Hello")

        assert response.content == "Clean answer"

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    async def test_ainvoke_basic(self, mock_init, mock_llm):
        """Test basic async invocation."""
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")
        response = await model.ainvoke("Hello")

        assert response.content == "Async mocked response"
        mock_llm.ainvoke.assert_called_once_with("Hello", config=None)

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.build_structured_prompt")
    @patch("src.common.unified_llm.parse_structured_output")
    async def test_ainvoke_structured_output(self, mock_parse, mock_build, mock_init, mock_llm):
        """Test async invoke with Pydantic schema."""
        mock_init.return_value = mock_llm
        mock_build.return_value = "Prompt with schema"
        mock_parse.return_value = SampleSchema(name="Carol", age=25)

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")
        result = await model.ainvoke("Who is Carol?", schema=SampleSchema)

        assert isinstance(result, SampleSchema)
        assert result.name == "Carol"
        assert result.age == 25
        mock_build.assert_called_once_with("Who is Carol?", SampleSchema)

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    async def test_ainvoke_no_thinking_cleans_content(self, mock_init, mock_llm):
        """Test that async invoke with thinking=False strips <think> tags."""
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="<think>step</think>Answer"))
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(provider="ollama", model_name="qwen3:14b", thinking=False)
        response = await model.ainvoke("Hello")

        assert response.content == "Answer"

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    async def test_astream_proxy(self, mock_init, mock_llm):
        """Test that astream is correctly proxied."""
        async def mock_astream(*args, **kwargs):
            yield AIMessageChunk(content="A")
            yield AIMessageChunk(content="B")

        mock_llm.astream = mock_astream
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")
        chunks = [chunk async for chunk in model.astream("Hello")]

        assert len(chunks) == 2
        assert chunks[0].content == "A"
        assert chunks[1].content == "B"

    @patch("src.common.unified_llm._RAW_LLM_CACHE", {})
    @patch("src.common.unified_llm.ChatOllama")
    def test_cache_reuse(self, mock_ollama_cls):
        """Test that identical init params return the cached model instance."""
        mock_instance = MagicMock()
        mock_ollama_cls.return_value = mock_instance

        result1 = UnifiedChatModel._init_underlying_model(
            provider="ollama", model_name="llama3", temperature=0.0,
            context_window=4096, top_p=1.0, thinking=True
        )
        result2 = UnifiedChatModel._init_underlying_model(
            provider="ollama", model_name="llama3", temperature=0.0,
            context_window=4096, top_p=1.0, thinking=True
        )

        assert result1 is result2
        mock_ollama_cls.assert_called_once()  # Only constructed once

    def test_invalid_provider_raises(self):
        """Test that an unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            UnifiedChatModel._init_underlying_model(
                provider="unknown_provider", model_name="model",
                temperature=0.0, context_window=None, top_p=1.0, thinking=True
            )

    def test_missing_openai_api_key_raises(self, monkeypatch):
        """Test that missing OPENAI_API_KEY raises ValueError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            UnifiedChatModel._init_underlying_model(
                provider="openai", model_name="gpt-4",
                temperature=0.0, context_window=None, top_p=1.0, thinking=True
            )

    def test_missing_deepseek_api_key_raises(self, monkeypatch):
        """Test that missing DEEPSEEK_API_KEY raises ValueError."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            UnifiedChatModel._init_underlying_model(
                provider="deepseek", model_name="deepseek-chat",
                temperature=0.0, context_window=None, top_p=1.0, thinking=True
            )

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.parse_structured_output")
    def test_structured_output_parse_failure_propagates(self, mock_parse, mock_init, mock_llm):
        """Test that a parse failure in invoke raises the exception."""
        mock_init.return_value = mock_llm
        mock_parse.side_effect = ValueError("Parse failed")

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")

        with pytest.raises(ValueError, match="Parse failed"):
            model.invoke("Who is Alice?", schema=SampleSchema)


# ── get_tokens_for_model (defined in unified_llm.py) ──────────────────────────

class TestGetTokensForModel:

    @pytest.mark.parametrize("model_name,expected_key", [
        ("qwen3:14b", "qwen"),
        ("Qwen2.5-72B", "qwen"),
        ("deepseek-r1-distill-qwen-14b", "qwen"),
        ("llama-3-8b-instruct", "llama3"),
        ("llama3.2:3b", "llama3"),
        ("deepseek-r1-distill-llama-70b", "llama3"),
        ("mistral-7b-instruct", "mistral"),
        ("mixtral-8x7b", "mistral"),
        ("gemma2:9b", "gemma"),
        ("phi-3-mini", "phi3"),
        ("phi3:14b", "phi3"),
        ("deepseek-v3", "deepseek_base"),
        ("deepseek-chat", "deepseek_base"),
    ])
    def test_model_family_detection(self, model_name, expected_key):
        assert get_tokens_for_model(model_name) == STOP_TOKENS[expected_key]

    def test_unknown_model_returns_empty(self):
        assert get_tokens_for_model("totally-unknown-model-xyz") == []

    def test_case_insensitive(self):
        assert get_tokens_for_model("QWEN3:14B") == STOP_TOKENS["qwen"]


# ── Provider-specific _init_underlying_model paths ────────────────────────────

class TestProviderInitialization:

    def test_deepseek_swaps_reasoner_to_chat_when_not_thinking(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
        with patch("src.common.unified_llm.ChatDeepSeek") as mock_cls, \
             patch.dict("src.common.unified_llm._RAW_LLM_CACHE", clear=True):
            mock_cls.return_value = MagicMock()
            UnifiedChatModel._init_underlying_model(
                provider="deepseek", model_name="deepseek-reasoner",
                temperature=0.0, context_window=None, top_p=1.0, thinking=False
            )
            assert mock_cls.call_args.kwargs["model"] == "deepseek-chat"

    def test_deepseek_keeps_reasoner_when_thinking_enabled(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
        with patch("src.common.unified_llm.ChatDeepSeek") as mock_cls, \
             patch.dict("src.common.unified_llm._RAW_LLM_CACHE", clear=True):
            mock_cls.return_value = MagicMock()
            UnifiedChatModel._init_underlying_model(
                provider="deepseek", model_name="deepseek-reasoner",
                temperature=0.0, context_window=None, top_p=1.0, thinking=True
            )
            assert mock_cls.call_args.kwargs["model"] == "deepseek-reasoner"

    def test_openai_not_thinking_adds_reasoning_effort_low(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        with patch("src.common.unified_llm.ChatOpenAI") as mock_cls, \
             patch.dict("src.common.unified_llm._RAW_LLM_CACHE", clear=True):
            mock_cls.return_value = MagicMock()
            UnifiedChatModel._init_underlying_model(
                provider="openai", model_name="gpt-4o",
                temperature=0.0, context_window=None, top_p=1.0, thinking=False
            )
            assert mock_cls.call_args.kwargs.get("reasoning_effort") == "low"

    def test_openai_thinking_does_not_add_reasoning_effort(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        with patch("src.common.unified_llm.ChatOpenAI") as mock_cls, \
             patch.dict("src.common.unified_llm._RAW_LLM_CACHE", clear=True):
            mock_cls.return_value = MagicMock()
            UnifiedChatModel._init_underlying_model(
                provider="openai", model_name="gpt-4o",
                temperature=0.0, context_window=None, top_p=1.0, thinking=True
            )
            assert "reasoning_effort" not in mock_cls.call_args.kwargs

    def test_lms_uses_chatopenai_with_hardcoded_key(self):
        with patch("src.common.unified_llm.ChatOpenAI") as mock_cls, \
             patch.dict("src.common.unified_llm._RAW_LLM_CACHE", clear=True):
            mock_cls.return_value = MagicMock()
            UnifiedChatModel._init_underlying_model(
                provider="lms", model_name="local-model",
                temperature=0.5, context_window=None, top_p=1.0, thinking=True
            )
            assert mock_cls.called
            # SecretStr masks str(); use get_secret_value() to read the actual key
            assert mock_cls.call_args.kwargs["api_key"].get_secret_value() == "lm-studio"

    def test_lms_uses_lmstudio_base_url_env(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://custom:5678/v1")
        with patch("src.common.unified_llm.ChatOpenAI") as mock_cls, \
             patch.dict("src.common.unified_llm._RAW_LLM_CACHE", clear=True):
            mock_cls.return_value = MagicMock()
            UnifiedChatModel._init_underlying_model(
                provider="lms", model_name="local-model",
                temperature=0.0, context_window=None, top_p=1.0, thinking=True
            )
            assert mock_cls.call_args.kwargs["base_url"] == "http://custom:5678/v1"

    def test_ollama_not_thinking_adds_reasoning_false(self):
        with patch("src.common.unified_llm.ChatOllama") as mock_cls, \
             patch.dict("src.common.unified_llm._RAW_LLM_CACHE", clear=True):
            mock_cls.return_value = MagicMock()
            UnifiedChatModel._init_underlying_model(
                provider="ollama", model_name="ollama-no-think-unique-xyz",
                temperature=0.0, context_window=None, top_p=1.0, thinking=False
            )
            assert mock_cls.call_args.kwargs.get("reasoning") is False

    def test_llama_cpp_missing_model_path_raises(self):
        with pytest.raises(ValueError, match="Model path not defined"):
            UnifiedChatModel._init_underlying_model(
                provider="llama_cpp", model_name="unknown-model-xyz",
                temperature=0.0, context_window=None, top_p=1.0, thinking=True
            )

    def test_llama_cpp_known_model_initializes(self):
        with patch("src.common.unified_llm.ChatLlamaCpp") as mock_cls, \
             patch.dict("src.common.unified_llm._RAW_LLM_CACHE", clear=True):
            mock_cls.return_value = MagicMock()
            # "qwen3:14b" is a known key in LLAMA_CPP_MODEL_PATHS
            UnifiedChatModel._init_underlying_model(
                provider="llama_cpp", model_name="qwen3:14b",
                temperature=0.0, context_window=None, top_p=1.0, thinking=True
            )
            assert mock_cls.called


# ── _generate and _agenerate ──────────────────────────────────────────────────

class TestGenerateMethods:

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_generate_delegates_to_underlying_generate(self, mock_init):
        mock_underlying = MagicMock()
        chat_result = ChatResult(generations=[ChatGeneration(message=AIMessage(content="gen result"))])
        mock_underlying._generate.return_value = chat_result
        mock_init.return_value = mock_underlying

        model = UnifiedChatModel(provider="ollama", model_name="llama3")
        result = model._generate([HumanMessage(content="test")])

        assert result.generations[0].message.content == "gen result"
        mock_underlying._generate.assert_called_once()

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_generate_falls_back_to_invoke_when_no_generate_method(self, mock_init):
        mock_underlying = MagicMock(spec=["invoke", "ainvoke", "stream", "astream"])
        mock_underlying.invoke.return_value = AIMessage(content="invoke fallback")
        mock_init.return_value = mock_underlying

        model = UnifiedChatModel(provider="ollama", model_name="llama3")
        result = model._generate([HumanMessage(content="test")])

        assert result.generations[0].message.content == "invoke fallback"
        mock_underlying.invoke.assert_called_once()

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_generate_cleans_content_when_not_thinking(self, mock_init):
        mock_underlying = MagicMock()
        msg = AIMessage(content="<think>internal step</think>clean output")
        mock_underlying._generate.return_value = ChatResult(
            generations=[ChatGeneration(message=msg)]
        )
        mock_init.return_value = mock_underlying

        model = UnifiedChatModel(provider="ollama", model_name="llama3", thinking=False)
        result = model._generate([HumanMessage(content="test")])

        assert result.generations[0].message.content == "clean output"

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    async def test_agenerate_delegates_to_underlying(self, mock_init):
        mock_underlying = MagicMock()
        chat_result = ChatResult(generations=[ChatGeneration(message=AIMessage(content="async gen"))])
        mock_underlying._agenerate = AsyncMock(return_value=chat_result)
        mock_init.return_value = mock_underlying

        model = UnifiedChatModel(provider="ollama", model_name="llama3")
        result = await model._agenerate([HumanMessage(content="test")])

        assert result.generations[0].message.content == "async gen"

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    async def test_agenerate_falls_back_to_ainvoke(self, mock_init):
        mock_underlying = MagicMock(spec=["invoke", "ainvoke", "stream", "astream"])
        mock_underlying.ainvoke = AsyncMock(return_value=AIMessage(content="ainvoke fallback"))
        mock_init.return_value = mock_underlying

        model = UnifiedChatModel(provider="ollama", model_name="llama3")
        result = await model._agenerate([HumanMessage(content="test")])

        assert result.generations[0].message.content == "ainvoke fallback"

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    async def test_agenerate_cleans_content_when_not_thinking(self, mock_init):
        mock_underlying = MagicMock()
        msg = AIMessage(content="<think>step</think>clean async output")
        mock_underlying._agenerate = AsyncMock(
            return_value=ChatResult(generations=[ChatGeneration(message=msg)])
        )
        mock_init.return_value = mock_underlying

        model = UnifiedChatModel(provider="ollama", model_name="llama3", thinking=False)
        result = await model._agenerate([HumanMessage(content="test")])

        assert result.generations[0].message.content == "clean async output"


# ── Extended invoke / ainvoke / stream / astream coverage ─────────────────────

class TestUnifiedChatModelExtended:

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="Mocked response")
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="Async mocked response"))
        llm.stream.return_value = iter([AIMessageChunk(content="M")])
        return llm

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_llm_type_property(self, mock_init, mock_llm):
        mock_init.return_value = mock_llm
        model = UnifiedChatModel(provider="ollama", model_name="llama3")
        assert model._llm_type == "unified_ollama"

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.build_structured_prompt")
    @patch("src.common.unified_llm.parse_structured_output")
    def test_invoke_list_content_response_joined_for_schema(
        self, mock_parse, mock_build, mock_init, mock_llm
    ):
        """Response content as list-of-dicts (Claude style) is joined before parsing."""
        mock_init.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(
            content=[{"type": "text", "text": '{"name": "Eve", "age": 20}'}]
        )
        mock_build.return_value = "schema prompt"
        mock_parse.return_value = SampleSchema(name="Eve", age=20)

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")
        result = model.invoke("Who?", schema=SampleSchema)

        assert isinstance(result, SampleSchema)
        passed_content = mock_parse.call_args.args[0]
        assert '{"name": "Eve"' in passed_content

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    async def test_ainvoke_thinking_override_creates_variant(self, mock_init, mock_llm):
        mock_init.return_value = mock_llm
        model = UnifiedChatModel(provider="ollama", model_name="qwen3:14b", thinking=True)
        await model.ainvoke("Hello", thinking=False)
        assert mock_init.call_count == 2
        _, variant_kwargs = mock_init.call_args
        assert variant_kwargs["thinking"] is False

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.build_structured_prompt")
    @patch("src.common.unified_llm.parse_structured_output")
    async def test_ainvoke_structured_output_with_list_messages(
        self, mock_parse, mock_build, mock_init, mock_llm
    ):
        mock_init.return_value = mock_llm
        mock_build.return_value = "Built prompt"
        mock_parse.return_value = SampleSchema(name="Frank", age=45)

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")
        messages = [SystemMessage(content="Be helpful."), HumanMessage(content="Who is Frank?")]
        result = await model.ainvoke(messages, schema=SampleSchema)

        assert isinstance(result, SampleSchema)
        mock_build.assert_called_once_with("Who is Frank?", SampleSchema)

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.build_structured_prompt")
    @patch("src.common.unified_llm.parse_structured_output")
    async def test_ainvoke_list_content_response_joined_for_schema(
        self, mock_parse, mock_build, mock_init, mock_llm
    ):
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(
            content=[{"type": "text", "text": '{"name": "Grace", "age": 33}'}]
        ))
        mock_init.return_value = mock_llm
        mock_build.return_value = "schema prompt"
        mock_parse.return_value = SampleSchema(name="Grace", age=33)

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")
        result = await model.ainvoke("Who?", schema=SampleSchema)

        assert isinstance(result, SampleSchema)
        passed_content = mock_parse.call_args.args[0]
        assert '{"name": "Grace"' in passed_content

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.parse_structured_output")
    async def test_ainvoke_parse_failure_propagates(self, mock_parse, mock_init, mock_llm):
        mock_init.return_value = mock_llm
        mock_parse.side_effect = ValueError("Async parse failed")

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")
        with pytest.raises(ValueError, match="Async parse failed"):
            await model.ainvoke("Who is Alice?", schema=SampleSchema)

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_stream_thinking_override_creates_variant(self, mock_init, mock_llm):
        mock_init.return_value = mock_llm
        model = UnifiedChatModel(provider="ollama", model_name="qwen3:14b", thinking=True)
        list(model.stream("Hello", thinking=False))
        assert mock_init.call_count == 2
        _, variant_kwargs = mock_init.call_args
        assert variant_kwargs["thinking"] is False

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    async def test_astream_thinking_override_creates_variant(self, mock_init, mock_llm):
        async def mock_astream_fn(*args, **kwargs):
            yield AIMessageChunk(content="chunk")

        mock_llm.astream = mock_astream_fn
        mock_init.return_value = mock_llm
        model = UnifiedChatModel(provider="ollama", model_name="qwen3:14b", thinking=True)
        [chunk async for chunk in model.astream("Hello", thinking=False)]
        assert mock_init.call_count == 2
        _, variant_kwargs = mock_init.call_args
        assert variant_kwargs["thinking"] is False
