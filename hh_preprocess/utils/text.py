"""Утилиты для нормализации строк и обработки текстовых полей."""

import re
from typing import Iterable

_SPACE_RE = re.compile(r"\s+")
_SPB_ALIAS = "санкт-петербург"
_MSK_ALIAS = "москва"


def normalize_spaces(s: str) -> str:
    """Удалить лишние пробелы и неразрывные пробелы."""
    return _SPACE_RE.sub(" ", s.replace("\xa0", " ")).strip()


def safe_lower(s: object) -> str:
    """Безопасное приведение к нижнему регистру."""
    if not isinstance(s, str):
        return ""
    return normalize_spaces(s).lower()


def split_multi_categories(text: object) -> list[str]:
    """Разделить строку с несколькими категориями на список."""
    if not isinstance(text, str):
        return []
    s = normalize_spaces(text)
    if not s:
        return []
    parts = re.split(r"[,;/]", s)
    out: list[str] = []
    for p in parts:
        p = normalize_spaces(p).lower()
        if p:
            out.append(p)
    return out


def contains_any(haystack: str, needles: Iterable[str]) -> bool:
    """Проверить, содержится ли любая из подстрок в строке."""
    return any(n in haystack for n in needles)


def extract_first_int(text: str) -> int | None:
    """Извлечь первое целое число из строки."""
    m = re.search(r"(\d+)", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def normalize_city_name(value: object) -> str:
    """Нормализовать название города для дальнейшей группировки.

    Аргументы:
        value: Исходное значение поля города.

    Возвращает:
        Нормализованная строка.
    """

    s = safe_lower(value)
    if not s:
        return ""

    s = s.replace("\ufeff", "")
    s = re.sub(r"^(г\.|город)\s+", "", s).strip()
    s = s.replace("—", "-").replace("–", "-")

    aliases = {
        "msk": _MSK_ALIAS,
        "moscow": _MSK_ALIAS,
        "spb": _SPB_ALIAS,
        "saint petersburg": _SPB_ALIAS,
        "st petersburg": _SPB_ALIAS,
        "st. petersburg": _SPB_ALIAS,
        "petersburg": _SPB_ALIAS,
        "saint-petersburg": _SPB_ALIAS,
        "санкт петербург": _SPB_ALIAS,
        "питер": _SPB_ALIAS,
    }

    s = aliases.get(s, s)
    return s
