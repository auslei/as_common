import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_llm():
    """Reusable mock LangChain chat model for unit tests."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="mocked response")
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="mocked async response"))
    return llm
