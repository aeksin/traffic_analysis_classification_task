"""Утилиты для работы с курсами валют ЦБ РФ и конвертации в рубли."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.request import urlopen
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


CBR_URL = "https://www.cbr.ru/scripts/XML_daily.asp"


@dataclass(frozen=True)
class FxRates:
    """Курсы валют относительно рубля.

    Поле `rates` содержит отображение `код_валюты -> RUB за 1 единицу валюты`.
    Поле `source` описывает, откуда были получены данные (сеть/кэш/fallback).
    """

    rates: Mapping[str, float]
    source: str


def _parse_cbr_xml(xml_bytes: bytes) -> dict[str, float]:
    """Распарсить XML ЦБ РФ и получить курсы валют в рублях.

    Формирует словарь `код_валюты -> RUB за 1 единицу`, нормализуя `Nominal`
    (например, если в XML курс задан за 10/100 единиц). Для рубля всегда
    добавляется курс `RUB = 1.0`.

    Аргументы:
        xml_bytes: Содержимое XML в байтах.

    Возвращает:
        Словарь курсов валют в формате `currency_code -> RUB per 1 unit`.
    """
    root = ET.fromstring(xml_bytes)
    out: dict[str, float] = {"RUB": 1.0}
    for valute in root.findall("Valute"):
        char_code = (valute.findtext("CharCode") or "").strip().upper()
        nominal = float((valute.findtext("Nominal") or "1").replace(",", "."))
        value = float((valute.findtext("Value") or "").replace(",", "."))
        if char_code and nominal > 0:
            out[char_code] = value / nominal
    return out


def load_fx_rates(cache_dir: Path) -> FxRates:
    """Загрузить курсы валют ЦБ РФ с использованием кэша.

    Алгоритм:
        1) Пытаемся загрузить XML с сайта ЦБ РФ и распарсить курсы.
           При успехе сохраняем словарь в кэш-файл `.fx_rates_cache.json`.
        2) Если сеть недоступна или парсинг не удался — читаем кэш.
        3) Если кэш отсутствует/битый — возвращаем fallback только для `RUB`.

    Аргументы:
        cache_dir: Директория для кэш-файла `.fx_rates_cache.json`.

    Возвращает:
        Объект `FxRates` с курсами и строкой-источником (`source`).

    Примечания:
        - В итоговом наборе курсов всегда присутствует `RUB = 1.0`.
        - Ошибки сети/чтения кэша логируются через `logger.warning`.
    """
    cache_path = cache_dir / ".fx_rates_cache.json"

    try:
        with urlopen(CBR_URL, timeout=10) as resp:  # nosec - intended network fetch
            xml_bytes = resp.read()
        rates = _parse_cbr_xml(xml_bytes)
        cache_path.write_text(json.dumps(rates, ensure_ascii=False), encoding="utf-8")
        return FxRates(rates=rates, source="CBR(XML_daily.asp)")
    except Exception as e:
        logger.warning("Could not fetch CBR rates: %s", e)

    if cache_path.exists():
        try:
            rates = json.loads(cache_path.read_text(encoding="utf-8"))
            rates["RUB"] = 1.0
            return FxRates(rates=rates, source="cache(.fx_rates_cache.json)")
        except Exception as e:
            logger.warning("Could not read FX cache: %s", e)

    return FxRates(rates={"RUB": 1.0}, source="fallback(RUB only)")
