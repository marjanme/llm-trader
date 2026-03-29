import json
from datetime import date
from pathlib import Path

from experiment_logic.types import (
    Configuration,
    DailyDataRow,
    Instruction,
    LoadedInputData,
    Market,
    NewsItem,
)


def load_input_data(input_data_dir: Path) -> LoadedInputData:
    markets = [
        Market(
            market_id=row["market_id"],
            question=row["question"],
            resolution_date=date.fromisoformat(row["resolution_date"]),
        )
        for row in _load_json(input_data_dir / "markets.json")
    ]

    daily_data = [
        DailyDataRow(
            market_id=row["market_id"],
            date=date.fromisoformat(row["date"]),
            probability_yes=float(row["probability_yes"]),
            probability_no=float(row["probability_no"]),
        )
        for row in _load_json(input_data_dir / "daily_data.json")
    ]

    news = [
        NewsItem(
            news_id=row["news_id"],
            market_id=row["market_id"],
            date=date.fromisoformat(row["date"]),
            title=row["title"],
            content=row["content"],
        )
        for row in _load_json(input_data_dir / "news.json")
    ]

    instructions = [
        Instruction(
            instruction_id=row["instruction_id"],
            text=row["text"],
        )
        for row in _load_json(input_data_dir / "instructions.json")
    ]

    configurations = [
        Configuration(
            configuration_id=row["configuration_id"],
            instruction_id=row["instruction_id"],
            show_market_probabilities=row["show_market_probabilities"],
            other_traders=row["other_traders"],
            news_mode=row["news_mode"],
        )
        for row in _load_json(input_data_dir / "configurations.json")
    ]

    return LoadedInputData(
        markets=markets,
        daily_data=daily_data,
        news=news,
        instructions=instructions,
        configurations=configurations,
    )


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
