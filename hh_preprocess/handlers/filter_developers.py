"""Обработчик фильтрации резюме: исключение нерелевантных профессий (HR, продажи) и отбор IT-специальностей."""

import logging
from typing import Any

from ..context import PipelineContext
from ..utils.text import safe_lower
from .base import Handler

logger = logging.getLogger(__name__)


class FilterDevelopersHandler(Handler):
    """
    Фильтрует датасет, оставляя только резюме IT-специалистов.

    Использует списки ключевых слов для включения (KEEP_KEYWORDS)
    и исключения (EXCLUDE_KEYWORDS) на основе названия должности.
    """

    # Список ключевых слов для поиска целевых специалистов (разработчики, QA, DevOps, и т.д.)
    KEEP_KEYWORDS = [
        "developer",
        "разработчик",
        "programmer",
        "программист",
        "software",
        "engineer",
        "инженер",
        "architect",
        "архитектор",
        "data scien",
        "ds",
        "ml",
        "machine learning",
        "аналитик данных",
        "qa",
        "test",
        "тест",
        "quality",
        "devops",
        "sre",
        "sysadmin",
        "администратор",
        "team lead",
        "tech lead",
        "cto",
        "frontend",
        "backend",
        "fullstack",
        "android",
        "ios",
        "mobile",
        "game",
        "unity",
        "unreal",
        "java",
        "python",
        "php",
        "golang",
        "c++",
        "c#",
        ".net",
        "js",
        "node",
        "1c",
        "1с",
    ]

    # Слова-исключения (например, чтобы не брать HR-менеджеров, ищущих разработчиков)
    EXCLUDE_KEYWORDS = [
        "hr",
        "recruiter",
        "рекрутер",
        "подбор",
        "talent",
        "sales",
        "продаж",
        "account",
        "менеджер по работе",
    ]

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """
        Применяет фильтрацию к DataFrame в контексте.

        Аргументы:
            ctx: Контекст пайплайна с загруженным DataFrame.

        Возвращает:
            Обновленный контекст, где ctx.df содержит только целевые строки.
            Статистика фильтрации сохраняется в ctx.diag["filtered_developers"].

        Исключения:
            ValueError: Если DataFrame не загружен в контекст.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")

        df = ctx.df

        desired_col = next(
            (c for c in df.columns if "ищет работу на должность" in c.lower()), None
        )

        if not desired_col:
            logger.warning("Не найдена колонка с должностью, фильтрация пропущена.")
            return ctx

        def is_target_role(title_obj: Any) -> bool:
            """Проверяет, подходит ли должность под критерии IT-специалиста."""
            title = safe_lower(title_obj)

            if any(bad_word in title for bad_word in self.EXCLUDE_KEYWORDS):
                return False

            return any(keyword in title for keyword in self.KEEP_KEYWORDS)

        mask = df[desired_col].map(is_target_role)

        before_count = len(df)
        df = df[mask].copy()
        after_count = len(df)

        ctx.diag["filtered_developers"] = {
            "before": before_count,
            "after": after_count,
            "removed": before_count - after_count,
        }

        logger.info(f"Фильтрация (Keywords): осталось {after_count} из {before_count}")

        ctx.df = df
        return ctx
