"""Обработчик группировки городов по категориям (MSK, SPB, BIG, SMALL)."""

from dataclasses import dataclass

from ..context import PipelineContext
from ..utils.text import normalize_city_name
from .base import Handler


@dataclass(frozen=True)
class CityGroupingConfig:
    """Настройки группировки городов.

    Аргументы:
        city_col: Название колонки с городом.
        out_col: Название колонки, куда записывать группу города.
        keep_original: Сохранять ли исходную колонку с городом.
    """

    city_col: str = "city"
    out_col: str = "city_group"
    keep_original: bool = False


class CityGroupingHandler(Handler):
    """Сгруппировать города в категории MSK/SPB/BIG/SMALL."""

    def __init__(self, config: CityGroupingConfig | None = None) -> None:
        super().__init__()
        self._config = config or CityGroupingConfig()

        self._big_cities = {
            "екатеринбург",
            "новосибирск",
            "казань",
            "нижний новгород",
            "челябинск",
            "самара",
            "омск",
            "ростов-на-дону",
            "уфа",
            "красноярск",
            "пермь",
            "воронеж",
            "волгоград",
            "краснодар",
            "саратов",
            "тюмень",
            "тольятти",
            "ижевск",
            "барнаул",
            "ульяновск",
            "иркутск",
            "хабаровск",
            "ярославль",
            "владивосток",
            "махачкала",
            "томск",
            "оренбург",
            "кемерово",
            "новокузнецк",
            "рязань",
            "астрахань",
            "пенза",
            "липецк",
            "киров",
            "чебоксары",
            "калининград",
            "брянск",
            "курск",
            "иваново",
            "тула",
            "сочи",
        }

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Преобразовать колонку города в 4 категории.

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
        col = self._config.city_col
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' не найдена для группировки городов.")

        # Нормализация названия города.
        city_norm = df[col].astype("string").fillna("").map(normalize_city_name)

        def to_group(v: str) -> str:
            if not v:
                return "SMALL"

            if v in {"москва", "moscow"}:
                return "MSK"

            if v in {
                "санкт-петербург",
                "санкт петербург",
                "спб",
                "spb",
                "питер",
                "saint petersburg",
                "st petersburg",
            }:
                return "SPB"

            if v in self._big_cities:
                return "BIG"

            return "SMALL"

        df[self._config.out_col] = city_norm.map(to_group).astype("string")

        if not self._config.keep_original and col != self._config.out_col:
            df = df.drop(columns=[col])

        ctx.df = df
        ctx.diag.setdefault("city_grouping", {})
        ctx.diag["city_grouping"].update(
            {
                "city_col": col,
                "out_col": self._config.out_col,
                "keep_original": self._config.keep_original,
                "value_counts": df[self._config.out_col]
                .value_counts(dropna=False)
                .to_dict(),
            }
        )
        return ctx
