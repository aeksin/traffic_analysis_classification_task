"""Обработчик определения наличия автомобиля по текстовому описанию."""

import logging

from ..context import PipelineContext
from ..utils.text import safe_lower
from .base import Handler

logger = logging.getLogger(__name__)


_AUTO_COL = "Авто"


class ParseAutoHandler(Handler):
    """Извлечь признак наличия автомобиля в колонку `has_car`."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Добавить бинарный признак `has_car` на основе колонки «Авто».

        Если колонка «Авто» отсутствует, признак `has_car` создаётся со значением `False`
        для всех строк.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Контекст пайплайна с обновлённым `ctx.df`, содержащим колонку `has_car`.

        Исключения:
            ValueError: Если DataFrame отсутствует в контексте.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")
        df = ctx.df.copy()

        if _AUTO_COL not in df.columns:
            df["has_car"] = False
            ctx.df = df
            return ctx

        def parse_has_car(val: object) -> bool:
            """Определить наличие автомобиля по текстовому значению.

            Значения нормализуются функцией `safe_lower`. Далее применяется
            набор правил:
            - пустое значение и «не указано» трактуются как отсутствие автомобиля;
            - ключевые слова «есть», «имеется», а также англоязычные маркеры
              `car` и `own` трактуются как наличие автомобиля.

            Аргументы:
                val: Значение из колонки «Авто».

            Возвращает:
                `True`, если по правилу значение соответствует наличию автомобиля,
                иначе `False`.
            """
            t = safe_lower(val)
            if not t:
                return False
            if "не указано" in t or "no" == t:
                return False
            if "есть" in t or "имеется" in t or "car" in t or "own" in t:
                return True
            return False

        df["has_car"] = df[_AUTO_COL].map(parse_has_car).astype(bool)

        ctx.df = df
        return ctx
