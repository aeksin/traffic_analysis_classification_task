#!/usr/bin/env python3
"""Скрипт для инференса модели (предсказания зарплаты/уровня) на новых данных."""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np

try:
    from models import (ElasticNetRegressor, LassoRegressor, LinearRegressor,
                        RandomForestWrapper, RidgeRegressor, XGBoostWrapper)
except ImportError as e:
    print(f"[CRITICAL] Не удалось импортировать классы моделей: {e}", file=sys.stderr)
    print(
        "Убедитесь, что папка 'models' существует и вы запускаете скрипт из корня проекта.",
        file=sys.stderr,
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("inference")


class InferenceService:
    """Сервис для загрузки модели и выполнения предсказаний."""

    def __init__(self, resources_dir: Path = None):
        """
        Инициализация сервиса.

        Аргументы:
            resources_dir: Путь к папке с моделью. По умолчанию ищет в ./resources
        """
        if resources_dir is None:
            self.resources_dir = Path(__file__).parent / "resources"
        else:
            self.resources_dir = Path(resources_dir)

        self.model_path = self.resources_dir / "best_model.pkl"
        self._model = self._load_model()

    def _load_model(self):
        """Загружает сериализованную модель (pickle)."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {self.model_path}\n"
                "Сначала запустите обучение: python train.py ."
            )

        logger.info(f"Загрузка модели из {self.model_path}...")

        try:
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            model_name = model.__class__.__name__
            logger.info(f"Модель успешно загружена: {model_name}")
            return model

        except Exception as e:
            raise RuntimeError(f"Ошибка при десериализации модели: {e}")

    def predict(self, x_path: Path) -> List[float]:
        """
        Выполняет предсказание на основе файла с признаками.

        Аргументы:
            x_path: Путь к .npy файлу с матрицей признаков.

        Возвращает:
            Список предсказанных зарплат.
        """
        if not x_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {x_path}")

        logger.info(f"Чтение входных данных: {x_path}")
        try:
            X = np.load(x_path)
        except Exception as e:
            raise ValueError(f"Не удалось прочитать .npy файл: {e}")

        logger.info(f"Запуск инференса для {X.shape[0]} объектов...")

        try:
            predictions = self._model.predict(X)
        except Exception as e:
            raise RuntimeError(f"Ошибка при выполнении predict: {e}")

        return predictions.tolist()


def main():
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(
        description="Инференс модели зарплат (выводит список float в stdout)."
    )
    parser.add_argument(
        "x_path", type=Path, help="Путь к файлу x_data.npy (матрица признаков)."
    )
    args = parser.parse_args()

    try:
        service = InferenceService()
        predictions = service.predict(args.x_path)

        print(predictions)

    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
