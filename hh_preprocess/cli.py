"""Интерфейс командной строки для запуска процесса предобработки."""

import argparse
import logging
from pathlib import Path

from .context import PipelineContext
from .pipeline import build_pipeline

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки.

    Описывает интерфейс CLI для предобработки датасета HH: путь к входному CSV,
    опциональные параметры кодировки/разделителя и уровень логирования.

    Возвращает:
        Настроенный экземпляр `argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(
        prog="app",
        description=(
            "Предобработка датасета HH. Рядом с входным CSV сохраняет файлы "
            "x_data.npy и y_data.npy."
        ),
    )
    p.add_argument("csv_path", type=Path, help="Путь к файлу hh.csv")
    p.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="Необязательное переопределение кодировки CSV (например, utf-8, cp1251).",
    )
    p.add_argument(
        "--sep",
        type=str,
        default=None,
        help=(
            "Необязательное переопределение разделителя CSV (например, ',' или ';'). "
            "Если не задано, предпринимается попытка автоопределения."
        ),
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень подробности логов.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI: запустить пайплайн предобработки и сохранить результаты.

    Читает входной CSV, собирает и запускает пайплайн обработки, после чего
    сохраняет `x_data.npy` и `y_data.npy` в директорию рядом с входным файлом.
    В конце выводит краткую диагностическую сводку из `ctx.diag` (если она есть).

    Аргументы:
        argv: Список аргументов командной строки без имени программы. Если не задан,
            используются аргументы из `sys.argv`.

    Возвращает:
        Код завершения процесса:
        - `0`, если обработка прошла успешно;
        - `2`, если входной файл не найден.
    """
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    if not args.csv_path.exists():
        logger.error("Input file not found: %s", args.csv_path)
        return 2

    out_dir = args.csv_path.parent
    ctx = PipelineContext(
        input_path=args.csv_path,
        output_dir=out_dir,
        encoding=args.encoding,
        sep=args.sep,
    )

    pipeline = build_pipeline()
    pipeline.handle(ctx)

    logger.info("Saved: %s", (out_dir / "x_data.npy"))
    logger.info("Saved: %s", (out_dir / "y_data.npy"))
    if ctx.feature_names is not None:
        logger.info("Features: %d columns", len(ctx.feature_names))

    if ctx.diag:
        dropped = ctx.diag.get("dropped_rows_total", 0)
        logger.info("Dropped rows total: %s", dropped)
        if ctx.diag.get("drop_reasons"):
            logger.info("Drop reasons: %s", ctx.diag["drop_reasons"])
        if ctx.diag.get("nan_report_top"):
            logger.info("NaN report (top): %s", ctx.diag["nan_report_top"])

    return 0
