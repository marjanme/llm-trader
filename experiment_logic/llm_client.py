import json
from typing import Protocol

from experiment_logic.types import ConversationMessage


class LLMClient(Protocol):
    def generate_response(self, complete_conversation: list[ConversationMessage]) -> str:
        ...


class MockLLMClient:
    def generate_response(self, complete_conversation: list[ConversationMessage]) -> str:
        response_object = {
            "reasoning": "Deterministic mock response for pipeline testing.",
            "probability_yes": 0.5,
        }
        return json.dumps(response_object)


def build_llm_client(provider: str) -> LLMClient:
    if provider == "mock":
        return MockLLMClient()

    raise ValueError(f"Unsupported provider: {provider!r}.")
