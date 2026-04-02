import json

from experiment_logic.types import ParsedLLMResponse


def parse_llm_response(raw_response: str) -> ParsedLLMResponse:
    try:
        response_object = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

    if not isinstance(response_object, dict):
        raise ValueError(f"LLM response must be a JSON object, got {type(response_object).__name__}.")

    reasoning = _parse_reasoning(response_object)
    probability_yes = _parse_probability_yes(response_object)

    return ParsedLLMResponse(
        reasoning=reasoning,
        probability_yes=probability_yes,
    )


def _parse_reasoning(response_object: dict) -> str:
    if "reasoning" not in response_object:
        raise ValueError("LLM response is missing required field 'reasoning'.")

    reasoning = response_object["reasoning"]
    if not isinstance(reasoning, str):
        raise ValueError(f"LLM field 'reasoning' must be a string, got {type(reasoning).__name__}.")

    reasoning = reasoning.strip()
    if not reasoning:
        raise ValueError("LLM field 'reasoning' must be a non-empty string.")

    return reasoning


def _parse_probability_yes(response_object: dict) -> float:
    if "probability_yes" not in response_object:
        raise ValueError("LLM response is missing required field 'probability_yes'.")

    probability_yes = response_object["probability_yes"]
    if not isinstance(probability_yes, (int, float)) or isinstance(probability_yes, bool):
        raise ValueError(
            f"LLM field 'probability_yes' must be a number, got {type(probability_yes).__name__}."
        )

    probability_yes = float(probability_yes)
    if probability_yes < 0.0 or probability_yes > 1.0:
        raise ValueError(f"LLM field 'probability_yes' must be between 0 and 1, got {probability_yes}.")

    return probability_yes
