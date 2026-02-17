"""Контекст пайплайна для хранения состояния обработки и передачи данных между шагами."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PipelineContext:
    """Контекст выполнения пайплайна предобработки.

    Хранит входные параметры запуска (пути, кодировка, разделитель), промежуточные
    данные (DataFrame) и итоговые артефакты (матрица признаков, таргет, имена
    признаков), а также диагностическую информацию.

    Атрибуты:
        input_path: Путь к входному CSV-файлу.
        output_dir: Директория для сохранения выходных артефактов и кэша.
        encoding: Кодировка CSV (если задана пользователем).
        sep: Разделитель CSV (если задан пользователем).

        df: Текущий DataFrame в пайплайне.
        X: Матрица признаков (NumPy), подготовленная на финальных шагах.
        y: Вектор/массив таргета (NumPy), подготовленный на финальных шагах.
        feature_names: Имена признаков, соответствующие колонкам `X`.

        diag: Словарь с диагностикой и метаданными пайплайна (счётчики, примеры,
            источники данных и т.п.).
    """

    input_path: Path
    output_dir: Path
    encoding: str | None = None
    sep: str | None = None

    df: pd.DataFrame | None = None
    X: np.ndarray | None = None
    y: np.ndarray | None = None
    feature_names: list[str] | None = None

    diag: dict[str, Any] = field(default_factory=dict)
