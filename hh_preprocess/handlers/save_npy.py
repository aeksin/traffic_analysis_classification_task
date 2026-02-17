"""Обработчик сохранения подготовленных матриц признаков и таргета на диск."""

import logging

import numpy as np

from ..context import PipelineContext
from .base import Handler

logger = logging.getLogger(__name__)


class SaveNpyHandler(Handler):
    """Сохранить матрицу признаков и таргет в файлы `x_data.npy` и `y_data.npy`."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Сохранить `ctx.X` и `ctx.y` в формате NumPy `.npy`.

        Файлы сохраняются в директорию `ctx.output_dir` с именами:
        - `x_data.npy` — матрица признаков;
        - `y_data.npy` — таргет.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Контекст пайплайна (без изменения данных).

        Исключения:
            ValueError: Если `ctx.X` или `ctx.y` не подготовлены.
        """
        if ctx.X is None or ctx.y is None:
            raise ValueError("X/y not prepared")

        out_x = ctx.output_dir / "x_data.npy"
        out_y = ctx.output_dir / "y_data.npy"
        np.save(out_x, ctx.X)
        np.save(out_y, ctx.y)

        return ctx
