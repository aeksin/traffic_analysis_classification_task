"""Обработчик категоризации названия должности (mapping на укрупненные группы)."""

import logging
from typing import Dict, List

from ..context import PipelineContext
from ..utils.text import safe_lower
from .base import Handler

logger = logging.getLogger(__name__)

_UNKNOWN = "Не указано"

_KEYWORDS: Dict[str, List[str]] = {
    "Системный администратор": [
        "системный администратор",
        "system administrator",
        "sysadmin",
    ],
    "DevOps/SRE": ["devops", "sre", "site reliability"],
    "Администратор баз данных": [
        "dba",
        "администратор баз данных",
        "database administrator",
    ],
    "Data Scientist/ML": ["data scientist", "ds ", "ml engineer", "machine learning"],
    "Аналитик": ["аналитик данных", "data analyst", "bi analyst", "business analyst"],
    "Тестировщик": ["тестировщик", "qa", "quality assurance"],
    "Программист/Разработчик": [
        "разработчик",
        "программист",
        "developer",
        "software engineer",
        "backend",
        "frontend",
        "fullstack",
        "ios",
        "android",
        "java",
        "python",
        "c++",
        "golang",
        "php",
        "javascript",
        "node.js",
        "react",
        "vue",
        "1c",
        "1с",
        "unity",
    ],
    "IT-специалист": ["it", "айти"],
    "Менеджер проектов/Продукта": [
        "product manager",
        "product owner",
        "продакт",
        "product",
        "проектный менеджер",
        "project manager",
        "pm ",
    ],
    "Маркетинг/PR/Контент": [
        "маркетолог",
        "marketing",
        "smm",
        "таргет",
        "seo",
        "контент",
        "pr",
        "copywriter",
        "копирайтер",
    ],
    "Продажи/Клиенты": [
        "продаж",
        "sales",
        "account manager",
        "менеджер по работе с клиентами",
        "клиентами",
        "торговый представитель",
        "кассир",
    ],
    "Финансы/Бухгалтерия": [
        "бухгалтер",
        "accountant",
        "финанс",
        "экономист",
        "аудитор",
        "финансовый",
    ],
    "HR/Рекрутер": ["hr", "рекрутер", "подбор персонала", "recruiter", "talent"],
    "Юрист": ["юрист", "lawyer", "legal"],
    "Логистика/Склад/Транспорт": [
        "логист",
        "logistics",
        "склад",
        "warehouse",
        "курьер",
        "доставка",
        "водитель",
        "driver",
    ],
    "Дизайн/Креатив": [
        "дизайнер",
        "designer",
        "ux",
        "ui",
        "graphic",
        "графический",
        "иллюстратор",
        "illustrator",
        "3d",
        "2d",
    ],
    "Инженерия/Производство/Строительство": [
        "инженер",
        "engineer",
        "технолог",
        "электрик",
        "mechanic",
        "механик",
        "строител",
        "construction",
    ],
    "Административный персонал": [
        "секретарь",
        "assistant",
        "ассистент",
        "офис-менеджер",
        "администратор",
        "reception",
    ],
    "Оператор": ["оператор", "operator"],
    "Специалист (общий)": ["специалист", "specialist"],
}


def categorize_job_title(title: object) -> str:
    """Сгруппировать название должности в укрупнённую категорию .

    Функция получает строку из таблицы набору ключевых слов
    возвращает одну из заранее заданных категорий.

    Аргументы:
        title: Значение из столбца с должностью (может быть строкой или NaN).

    Возвращает:
        Название укрупнённой категории. Если значение отсутствует или не распознано,
        возвращается «Не указано» или «Прочее».
    """
    t = safe_lower(title)
    if not t:
        return _UNKNOWN

    for category, keywords in _KEYWORDS.items():
        if any(k in t for k in keywords):
            return category

    return "Прочее"


class JobCategoryHandler(Handler):
    """Сформировать категориальные признаки профессии по желаемой и текущей должности."""

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")
        df = ctx.df.copy()

        desired_col = next(
            (c for c in df.columns if "ищет работу на должность" in c.lower()), None
        )
        current_title_col = next(
            (c for c in df.columns if "нынешняя должност" in c.lower()), None
        )

        cols_map = {
            desired_col: "job_category",
            current_title_col: "current_job_category",
        }

        for col_name, target_name in cols_map.items():
            if col_name is None:
                logger.warning(
                    f"Column for {target_name} not found; using '{_UNKNOWN}'"
                )
                df[target_name] = _UNKNOWN
            else:
                df[target_name] = df[col_name].map(categorize_job_title)

        drop_cols = [c for c in [desired_col, current_title_col] if c is not None]
        df = df.drop(columns=drop_cols, errors="ignore")

        employer_col = next(
            (c for c in df.columns if "место работы" in c.lower()), None
        )
        if employer_col is not None:
            df = df.drop(columns=[employer_col], errors="ignore")

        ctx.df = df
        return ctx
