import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .base import BaseModel

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class LogisticRegressionWrapper(BaseModel):
    """Обертка для Логистической Регрессии."""

    def __init__(self, C: float = 1.0) -> None:
        """Инициализировать модель Логистической Регрессии.

        Аргументы:
            C: Коэффициент обратной регуляризации (меньше -> сильнее регуляризация).
        """
        super().__init__(
            LogisticRegression(
                C=C,
                max_iter=5000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionWrapper":
        """Обучить классификатор.

        Аргументы:
            X: Матрица признаков.
            y: Вектор целевой переменной.

        Возвращает:
            self.
        """
        self._model.fit(X, y)
        return self


class RandomForestClassifierWrapper(BaseModel):
    """Обертка для Random Forest Classifier."""

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None) -> None:
        """Инициализировать модель Случайного Леса.

        Аргументы:
            n_estimators: Количество деревьев в лесу.
            max_depth: Максимальная глубина дерева (для контроля переобучения).
        """
        super().__init__(
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifierWrapper":
        """Обучить классификатор.

        Аргументы:
            X: Матрица признаков.
            y: Вектор целевой переменной.

        Возвращает:
            self.
        """
        self._model.fit(X, y)
        return self


class XGBClassifierWrapper(BaseModel):
    """Обертка для XGBoost Classifier с поддержкой GPU."""

    def __init__(
        self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1
    ) -> None:
        """Инициализировать модель Градиентного Бустинга (XGBoost).

        Аргументы:
            n_estimators: Количество деревьев (итераций бустинга).
            max_depth: Максимальная глубина дерева.
            learning_rate: Скорость обучения.
        """
        super().__init__(
            XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
                objective="multi:softprob",
                num_class=3,
                tree_method="hist",
                device="cuda",
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBClassifierWrapper":
        """Обучить классификатор (с автоматической конвертацией в CuPy).

        Если модель настроена на использование GPU (device='cuda') и доступна
        библиотека CuPy, входные данные автоматически конвертируются в cupy-массивы
        для ускорения передачи данных в видеопамять.

        Аргументы:
            X: Матрица признаков.
            y: Вектор целевой переменной.

        Возвращает:
            self.
        """
        params = self._model.get_params()
        is_gpu = (
            params.get("device") == "cuda" or params.get("tree_method") == "gpu_hist"
        )

        if is_gpu and HAS_CUPY:
            if not isinstance(X, cp.ndarray):
                X = cp.array(X)

            if not isinstance(y, cp.ndarray):
                y = cp.array(y)

        self._model.fit(X, y)
        return self
