"""Обработчик удаления дубликатов строк в датасете."""

from dataclasses import dataclass

from ..context import PipelineContext
from .base import Handler


@dataclass(frozen=True)
class DeduplicateConfig:
    """Настройки удаления дубликатов.

    Аргументы:
        subset: Подмножество колонок, по которым искать дубликаты.
            Если None, используются все колонки.
        keep: Какую запись оставлять среди дубликатов.
            Допустимые значения: "first" | "last".
    """

    subset: list[str] | None = None
    keep: str = "first"


class DeduplicateHandler(Handler):
    """Удалить дубликаты строк из таблицы.

    Аргументы:
        config: Настройки удаления дубликатов.
    """

    def __init__(self, config: DeduplicateConfig | None = None) -> None:
        super().__init__()
        self._config = config or DeduplicateConfig()

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Удалить дубликаты и записать статистику в диагностику.

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
        before_rows = len(df)
        df = df.drop_duplicates(
            subset=self._config.subset, keep=self._config.keep
        ).copy()
        after_rows = len(df)

        ctx.df = df
        ctx.diag.setdefault("deduplicate", {})
        ctx.diag["deduplicate"].update(
            {
                "rows_before": before_rows,
                "rows_after": after_rows,
                "removed": before_rows - after_rows,
                "subset": self._config.subset,
                "keep": self._config.keep,
            }
        )
        return ctx
