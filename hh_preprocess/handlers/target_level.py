import logging

import numpy as np

from ..context import PipelineContext
from ..utils.text import safe_lower
from .base import Handler

logger = logging.getLogger(__name__)


class TargetLevelHandler(Handler):
    """Сформировать целевую переменную 'level' (Junior/Middle/Senior)."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Разметить уровень специалиста на основе должности и опыта.

        Логика:
        1. Ищем явные маркеры в названии должности (Senior, Middle, Junior).
        2. Если маркер не найден, используем опыт работы (months).
        3. Кодируем классы: 0 (Junior), 1 (Middle), 2 (Senior).
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")

        df = ctx.df.copy()

        if "total_experience_months" not in df.columns:
            raise ValueError(
                "Требуется total_experience_months для определения уровня."
            )

        # Столбец с желаемой должностью (или текущей) для поиска ключевых слов
        # Обычно это та же колонка, что используется в JobCategoryHandler
        # Для простоты ищем исходную колонку "Ищет работу на должность:"
        desired_col = next(
            (c for c in df.columns if "ищет работу на должность" in c.lower()), None
        )

        conditions = [
            (df["total_experience_months"] < 18),  # < 1.5 года -> Junior
            (df["total_experience_months"] < 60),  # 1.5 - 5 лет -> Middle
            (df["total_experience_months"] >= 60),  # > 5 лет -> Senior
        ]
        choices = ["junior", "middle", "senior"]

        # Сначала размечаем чисто по опыту как fallback
        df["level_raw"] = np.select(conditions, choices, default="middle")

        # Функция уточнения по названию должности
        def refine_by_title(row):
            title = safe_lower(row.get(desired_col, ""))

            if (
                "senior" in title
                or "сеньор" in title
                or "ведущий" in title
                or "sr." in title
                or "lead" in title
            ):
                return "senior"
            if "middle" in title or "mid" in title or "мидл" in title:
                return "middle"
            if (
                "junior" in title
                or "jr" in title
                or "intern" in title
                or "стажер" in title
                or "младший" in title
                or "джун" in title
            ):
                return "junior"

            return row["level_raw"]

        if desired_col:
            df["level_label"] = df.apply(refine_by_title, axis=1)
        else:
            df["level_label"] = df["level_raw"]

        # Маппинг в числа для модели
        label_map = {"junior": 0, "middle": 1, "senior": 2}
        df["target_level"] = df["level_label"].map(label_map)

        # Статистика классов
        counts = df["level_label"].value_counts().to_dict()
        ctx.diag["class_balance"] = counts
        logger.info(f"Баланс классов: {counts}")

        # Убираем вспомогательные колонки, оставляем target_level
        df = df.drop(columns=["level_raw", "level_label"], errors="ignore")

        ctx.df = df

        # Важно: обновляем y в контексте
        ctx.y = df["target_level"].to_numpy(dtype=np.int8)

        return ctx
