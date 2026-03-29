import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from experiment_logic.build_prompt import build_prompt
from experiment_logic.llm_client import LLMClient
from experiment_logic.parse_llm_response import parse_llm_response
from experiment_logic.select_news import select_visible_news
from experiment_logic.types import (
    Configuration,
    ConversationMessage,
    DailyDataRow,
    Instruction,
    LoadedInputData,
    Market,
    NewsItem,
    Prediction,
)


def run_experiment(
    loaded_input_data: LoadedInputData,
    market_id: str,
    configuration_id: str,
    llm_client: LLMClient,
    results_dir: Path,
) -> Path:
    market = _get_market(loaded_input_data, market_id)
    configuration = _get_configuration(loaded_input_data, configuration_id)
    instruction = _get_instruction(loaded_input_data, configuration.instruction_id)
    market_daily_data = _get_sorted_daily_data_for_market(loaded_input_data, market.market_id)
    market_news = _get_news_for_market(loaded_input_data, market.market_id)

    predictions: list[Prediction] = []
    for daily_data_row in market_daily_data:
        predictions.append(
            _build_prediction_for_day(
                market=market,
                daily_data_row=daily_data_row,
                market_news=market_news,
                configuration=configuration,
                instruction=instruction,
                llm_client=llm_client,
            )
        )

    run_dir = _create_run_directory(results_dir, market.market_id, configuration.configuration_id)
    _write_predictions(run_dir / "predictions.json", predictions)
    return run_dir


def _build_prediction_for_day(
    market: Market,
    daily_data_row: DailyDataRow,
    market_news: list[NewsItem],
    configuration: Configuration,
    instruction: Instruction,
    llm_client: LLMClient,
) -> Prediction:
    visible_news = select_visible_news(
        news_items=market_news,
        current_date=daily_data_row.date,
        news_mode=configuration.news_mode,
    )

    rendered_prompt = build_prompt(
        market=market,
        daily_data_row=daily_data_row,
        visible_news=visible_news,
        configuration=configuration,
        instruction=instruction,
    )

    user_message = ConversationMessage(role="user", content=rendered_prompt)
    raw_response = llm_client.generate_response([user_message])
    parsed_response = parse_llm_response(raw_response)
    assistant_message = ConversationMessage(role="assistant", content=raw_response)

    return Prediction(
        market_id=market.market_id,
        configuration_id=configuration.configuration_id,
        date=daily_data_row.date,
        reasoning=parsed_response.reasoning,
        p_yes=round(parsed_response.probability_yes, 6),
        p_no=round(1.0 - parsed_response.probability_yes, 6),
        rendered_prompt=rendered_prompt,
        complete_conversation=[user_message, assistant_message],
    )


def _get_market(loaded_input_data: LoadedInputData, market_id: str) -> Market:
    for market in loaded_input_data.markets:
        if market.market_id == market_id:
            return market
    raise ValueError(f"Unknown market_id: {market_id!r}.")


def _get_configuration(loaded_input_data: LoadedInputData, configuration_id: str) -> Configuration:
    for configuration in loaded_input_data.configurations:
        if configuration.configuration_id == configuration_id:
            return configuration
    raise ValueError(f"Unknown configuration_id: {configuration_id!r}.")


def _get_instruction(loaded_input_data: LoadedInputData, instruction_id: str) -> Instruction:
    for instruction in loaded_input_data.instructions:
        if instruction.instruction_id == instruction_id:
            return instruction
    raise ValueError(f"Unknown instruction_id: {instruction_id!r}.")


def _get_sorted_daily_data_for_market(
    loaded_input_data: LoadedInputData,
    market_id: str,
) -> list[DailyDataRow]:
    market_daily_data = [row for row in loaded_input_data.daily_data if row.market_id == market_id]
    return sorted(market_daily_data, key=lambda row: row.date)


def _get_news_for_market(loaded_input_data: LoadedInputData, market_id: str) -> list[NewsItem]:
    return [news_item for news_item in loaded_input_data.news if news_item.market_id == market_id]


def _create_run_directory(results_dir: Path, market_id: str, configuration_id: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"{market_id}_{configuration_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_predictions(path: Path, predictions: list[Prediction]) -> None:
    serializable_predictions = []
    for prediction in predictions:
        prediction_dict = asdict(prediction)
        prediction_dict["date"] = prediction.date.isoformat()
        serializable_predictions.append(prediction_dict)

    path.write_text(
        json.dumps(serializable_predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
