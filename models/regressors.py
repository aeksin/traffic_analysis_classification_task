"""Реализация регрессионных моделей (линейные, ансамбли) для предсказания зарплаты."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from xgboost import XGBRegressor

from .base import BaseModel


class LinearRegressor(BaseModel):
    """Обертка для классической линейной регрессии (OLS)."""

    def __init__(self) -> None:
        super().__init__(LinearRegression())

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        """Обучить линейную регрессию.

        Аргументы:
            X: Матрица признаков.
            y: Вектор целевой переменной.

        Возвращает:
            self.
        """
        self._model.fit(X, y)
        return self


class RidgeRegressor(BaseModel):
    """Обертка для Ridge регрессии (L2 регуляризация)."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Инициализировать Ridge модель.

        Аргументы:
            alpha: Коэффициент регуляризации.
        """
        super().__init__(Ridge(alpha=alpha))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        """Обучить Ridge регрессию.

        Аргументы:
            X: Матрица признаков.
            y: Вектор целевой переменной.

        Возвращает:
            self.
        """
        self._model.fit(X, y)
        return self


class LassoRegressor(BaseModel):
    """Обертка для Lasso регрессии (L1 регуляризация)."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Инициализировать Lasso модель.

        Аргументы:
            alpha: Коэффициент регуляризации.
        """
        super().__init__(Lasso(alpha=alpha))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegressor":
        """Обучить Lasso регрессию.

        Аргументы:
            X: Матрица признаков.
            y: Вектор целевой переменной.

        Возвращает:
            self.
        """
        self._model.fit(X, y)
        return self


class ElasticNetRegressor(BaseModel):
    """Обертка для ElasticNet (комбинация L1 и L2)."""

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5) -> None:
        """Инициализировать ElasticNet.

        Аргументы:
            alpha: Общий коэффициент регуляризации.
            l1_ratio: Баланс между L1 и L2 (0 = Ridge, 1 = Lasso).
        """
        super().__init__(ElasticNet(alpha=alpha, l1_ratio=l1_ratio))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetRegressor":
        """Обучить ElasticNet регрессию.

        Аргументы:
            X: Матрица признаков.
            y: Вектор целевой переменной.

        Возвращает:
            self.
        """
        self._model.fit(X, y)
        return self


class RandomForestWrapper(BaseModel):
    """Обертка для Random Forest."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: float | int = 1.0,
        n_jobs: int = -1,
    ) -> None:
        """Инициализировать модель Случайного Леса.

        Аргументы:
            n_estimators: Количество деревьев в лесу.
            max_depth: Максимальная глубина дерева.
            min_samples_leaf: Минимальное количество образцов в листе.
            max_features: Количество признаков для поиска лучшего разделения.
            n_jobs: Количество ядер процессора (-1 = все).
        """
        super().__init__(
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=n_jobs,
                random_state=42,
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestWrapper":
        """Обучить модель Случайного Леса.

        Аргументы:
            X: Матрица признаков.
            y: Вектор целевой переменной.

        Возвращает:
            self.
        """
        self._model.fit(X, y)
        return self


class XGBoostWrapper(BaseModel):
    """Обертка для XGBoost (Gradient Boosting)."""

    def __init__(
        self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1
    ) -> None:
        """Инициализировать модель Градиентного Бустинга (XGBoost).

        Аргументы:
            n_estimators: Количество деревьев.
            max_depth: Максимальная глубина дерева.
            learning_rate: Скорость обучения.
        """
        super().__init__(
            XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostWrapper":
        """Обучить модель XGBoost.

        Аргументы:
            X: Матрица признаков.
            y: Вектор целевой переменной.

        Возвращает:
            self.
        """
        self._model.fit(X, y)
        return self
