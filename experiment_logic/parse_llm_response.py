import json

from experiment_logic.types import ParsedLLMResponse


def parse_llm_response(raw_response: str) -> ParsedLLMResponse:
    response_object = json.loads(raw_response)

    return ParsedLLMResponse(
        reasoning=response_object["reasoning"].strip(),
        probability_yes=float(response_object["probability_yes"]),
    )
