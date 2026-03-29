from datetime import date

from experiment_logic.types import NewsItem, NewsMode


def select_visible_news(
    news_items: list[NewsItem],
    current_date: date,
    news_mode: NewsMode,
) -> list[NewsItem]:
    if news_mode == "none":
        return []

    if news_mode == "all_until_day":
        return [item for item in news_items if item.date <= current_date]

    raise ValueError(f"Unsupported news_mode: {news_mode!r}.")
