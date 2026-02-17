"""Обработчик стандартизации числовых признаков."""

from dataclasses import dataclass

import pandas as pd

from ..context import PipelineContext
from .base import Handler


@dataclass(frozen=True)
class StandardizeConfig:
    """Конфигурация стандартизации числовых признаков.

    Аргументы:
        columns: Список колонок, которые нужно стандартизировать.
        eps: Малое число для защиты от деления на ноль.
    """

    columns: tuple[str, ...] = (
        "age",
        "education_year",
        "total_experience_months",
        "resume_days_since_update",
        "age_sq",
        "experience_sq",
        "description_len_log",
    )
    eps: float = 1e-8


class StandardizeNumericHandler(Handler):
    """Стандартизировать непрерывные числовые признаки"""

    def __init__(self, config: StandardizeConfig | None = None) -> None:
        """Создать обработчик стандартизации числовых признаков.

        Аргументы:
            config: Конфигурация стандартизации. Если не задана, используются
                значения по умолчанию из `StandardizeConfig`.
        Возвращает:
            None.
        """
        super().__init__()
        self._config = config or StandardizeConfig()

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Применить стандартизацию.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Обновлённый контекст пайплайна.
        """
        if ctx.df is None:
            raise ValueError(
                "В контексте отсутствует df. Проверьте порядок обработчиков."
            )

        df = ctx.df
        stats: dict[str, dict[str, float]] = {}

        for col in self._config.columns:
            if col not in df.columns:
                continue

            df[col] = pd.to_numeric(df[col], errors="coerce")

            s = df[col].astype("float64")
            mean = float(s.mean(skipna=True))
            std = float(s.std(skipna=True, ddof=0))

            denom = std if std > self._config.eps else 1.0
            df[col] = (s - mean) / denom

            stats[col] = {"mean": mean, "std": std}

        ctx.df = df
        ctx.diag.setdefault("standardize_numeric", {})
        ctx.diag["standardize_numeric"] = {
            "columns": list(stats.keys()),
            "stats": stats,
        }
        return ctx
