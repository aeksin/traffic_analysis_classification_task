"""Базовый абстрактный класс для всех моделей машинного обучения в проекте."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Базовый абстрактный класс для моделей регрессии.

    Определяет единый интерфейс для обучения, предсказания и
    сериализации (сохранения/загрузки) моделей.
    """

    def __init__(self, model_impl: Any) -> None:
        """Инициализировать обертку над моделью.

        Аргументы:
            model_impl: Реализация модели (например, из sklearn).
        """
        self._model = model_impl

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Обучить модель на переданных данных.

        Аргументы:
            X: Матрица признаков.
            y: Вектор целевой переменной.

        Возвращает:
            self (текущий экземпляр модели).
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Сделать предсказание на основе матрицы признаков.

        Аргументы:
            X: Матрица признаков.

        Возвращает:
            Вектор предсказаний.
        """
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        """Сохранить веса (объект модели) в файл.

        Использует модуль pickle для сериализации.

        Аргументы:
            path: Путь к файлу для сохранения (.pkl).
        """
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: Path) -> None:
        """Загрузить веса из файла в текущий объект.

        Аргументы:
            path: Путь к файлу с сериализованной моделью (.pkl).
        """
        with open(path, "rb") as f:
            self._model = pickle.load(f)

    @property
    def coefs(self) -> dict[str, Any]:
        """Получить коэффициенты модели (веса) для инспекции.

        Возвращает:
            Словарь с коэффициентами и отступом.
        """
        return {
            "coef": self._model.coef_.tolist(),
            "intercept": float(self._model.intercept_),
        }
