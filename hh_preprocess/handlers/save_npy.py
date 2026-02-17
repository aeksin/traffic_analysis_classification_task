"""Обработчик сохранения подготовленных матриц признаков и таргета на диск."""

import logging

import numpy as np

from ..context import PipelineContext
from .base import Handler

logger = logging.getLogger(__name__)


class SaveNpyHandler(Handler):
    """Сохранить матрицу признаков, таргет и список колонок."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """
        Сохраняет:
        - x_data.npy (матрица)
        - y_data.npy (таргет)
        - columns.txt (список имен колонок в порядке, соответствующем x_data)
        """
        if ctx.X is None or ctx.y is None:
            raise ValueError("X/y not prepared")

        out_x = ctx.output_dir / "x_data.npy"
        out_y = ctx.output_dir / "y_data.npy"
        np.save(out_x, ctx.X)
        np.save(out_y, ctx.y)

        if ctx.feature_names:
            out_cols = ctx.output_dir / "columns.txt"
            try:
                with open(out_cols, "w", encoding="utf-8") as f:
                    for col in ctx.feature_names:
                        f.write(f"{col}\n")
                logger.info(f"Список колонок сохранен: {out_cols}")
            except Exception as e:
                logger.warning(f"Не удалось сохранить список колонок: {e}")

        return ctx
