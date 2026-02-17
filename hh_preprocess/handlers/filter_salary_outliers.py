"""Обработчик фильтрации выбросов зарплаты по порогам и IQR."""

from dataclasses import dataclass

import pandas as pd

from ..context import PipelineContext
from .base import Handler


@dataclass(frozen=True)
class SalaryOutlierConfig:
    """Конфигурация удаления выбросов по зарплате.

    Аргументы:
        min_salary_rub: Минимально реалистичная зарплата в рублях.
        use_iqr_filter: Включать ли фильтрацию по IQR (правило 1.5 * IQR).
        iqr_k: Множитель для IQR.
    """

    min_salary_rub: float = 1000.0
    use_iqr_filter: bool = True
    iqr_k: float = 1.5


class FilterSalaryOutliersHandler(Handler):
    """Удалить выбросы и очевидные ошибки в целевой переменной (зарплате).

    Обработчик должен выполняться ПОСЛЕ вычисления `target_salary_rub`
    и ДО финализации в numpy-массивы.

    Логика:
    1) удаляем строки с нереалистично маленькой зарплатой;
    2) опционально удаляем выбросы по правилу IQR.

    В `ctx.diag` записываются счётчики, чтобы было видно, сколько строк удалили.
    """

    def __init__(self, config: SalaryOutlierConfig | None = None) -> None:
        super().__init__()
        self._config = config or SalaryOutlierConfig()

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Применить фильтрацию по зарплате.

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
        if "target_salary_rub" not in df.columns:
            raise ValueError(
                "Колонка target_salary_rub не найдена. "
                "Фильтрацию выбросов нужно ставить после ParseSalaryHandler."
            )

        before_rows = len(df)

        df["target_salary_rub"] = pd.to_numeric(
            df["target_salary_rub"], errors="coerce"
        )

        min_salary = float(self._config.min_salary_rub)
        bad_low_mask = df["target_salary_rub"].notna() & (
            df["target_salary_rub"] < min_salary
        )
        low_removed = int(bad_low_mask.sum())
        df = df.loc[~bad_low_mask].copy()

        iqr_removed = 0
        iqr_bounds: tuple[float, float] | None = None
        if self._config.use_iqr_filter:
            s = df["target_salary_rub"].dropna()
            if len(s) >= 10:
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                iqr = q3 - q1
                k = float(self._config.iqr_k)
                lo = q1 - k * iqr
                hi = q3 + k * iqr
                iqr_bounds = (lo, hi)
                iqr_mask = df["target_salary_rub"].notna() & (
                    (df["target_salary_rub"] < lo) | (df["target_salary_rub"] > hi)
                )
                iqr_removed = int(iqr_mask.sum())
                df = df.loc[~iqr_mask].copy()

        after_rows = len(df)

        ctx.df = df
        ctx.diag.setdefault("salary_outliers", {})
        ctx.diag["salary_outliers"].update(
            {
                "rows_before": before_rows,
                "rows_after": after_rows,
                "removed_low_salary": low_removed,
                "min_salary_rub": min_salary,
                "use_iqr_filter": self._config.use_iqr_filter,
                "iqr_k": float(self._config.iqr_k),
                "removed_iqr": iqr_removed,
                "iqr_bounds": iqr_bounds,
            }
        )
        return ctx
