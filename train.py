"""Скрипт для обучения моделей, подбора гиперпараметров и сохранения артефактов."""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

from models import (ElasticNetRegressor, LassoRegressor, LinearRegressor,
                    RandomForestWrapper, RidgeRegressor, XGBoostWrapper)
from models.base import BaseModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_data(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Загрузить данные из .npy файлов.

    Аргументы:
        x_path: Путь к файлу с признаками.
        y_path: Путь к файлу с целевой переменной.

    Возвращает:
        Кортеж из массивов numpy (X, y).
    """
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Файлы данных не найдены. Сначала запустите пайплайн обработки."
        )

    logger.info("Загрузка данных: X=%s, y=%s", x_path, y_path)
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y


def tune_and_fit(
    name: str,
    model_wrapper: BaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict[str, list[Any]] | None = None,
) -> Any:
    """Подобрать гиперпараметры (GridSearch) и обучить модель.

    Аргументы:
        name: Имя модели (для логов).
        model_wrapper: Обертка модели.
        X_train: Обучающая выборка.
        y_train: Целевая переменная.
        param_grid: Сетка параметров для перебора.

    Возвращает:
        Лучший обученный эстиматор (sklearn/xgboost).
    """
    sklearn_model = model_wrapper._model

    if not param_grid:
        logger.info("Обучение %s без подбора параметров...", name)
        sklearn_model.fit(X_train, y_train)
        return sklearn_model

    logger.info(
        "Подбор гиперпараметров для %s (grid size: %d)...",
        name,
        _count_grid_size(param_grid),
    )

    grid = GridSearchCV(
        estimator=sklearn_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=4,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    logger.info("Лучшие параметры для %s: %s", name, grid.best_params_)
    return grid.best_estimator_


def _count_grid_size(grid: dict) -> int:
    """Вспомогательная функция для подсчета количества комбинаций в сетке."""
    count = 1
    for v in grid.values():
        count *= len(v)
    return count


def evaluate_model(
    name: str,
    model_wrapper: BaseModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    """Оценить качество модели на тестовой выборке и сохранить результаты.

    Аргументы:
        name: Имя модели.
        model_wrapper: Обученная модель.
        X_test: Тестовые признаки.
        y_test: Тестовые метки.
        output_dir: Папка для сохранения модели.

    Возвращает:
        Словарь с метриками (R2, MAE, MSE, RMSE).
    """
    y_pred = model_wrapper.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    logger.info(
        "RESULT: %-10s | R2: %.4f | MAE: %.0f | RMSE: %.0f", name, r2, mae, rmse
    )

    model_path = output_dir / f"{name.lower()}_model.pkl"
    model_wrapper.save(model_path)

    return {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse}


def main() -> None:
    """Основная функция запуска эксперимента обучения.

    1. Загружает данные.
    2. Делит их на train/test.
    3. Перебирает список моделей (Linear, Ridge, Lasso, ElasticNet, RF, XGBoost).
    4. Запускает подбор гиперпараметров для каждой.
    5. Сохраняет метрики и лучшую модель в папку resources.
    """
    parser = argparse.ArgumentParser(description="Обучение моделей.")
    parser.add_argument("data_dir", type=Path, help="Папка с x_data.npy и y_data.npy")
    args = parser.parse_args()

    x_path = args.data_dir / "x_data.npy"
    y_path = args.data_dir / "y_data.npy"

    resources_dir = Path(__file__).parent / "resources"
    resources_dir.mkdir(exist_ok=True)

    try:
        X, y = load_data(x_path, y_path)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    alphas_grid = np.logspace(-3, 3, 10).tolist()

    models_config = [
        ("Linear", LinearRegressor(), None),
        (
            "Ridge",
            RidgeRegressor(),
            {"alpha": alphas_grid, "solver": ["auto", "svd", "cholesky", "lsqr"]},
        ),
        (
            "Lasso",
            LassoRegressor(),
            {"alpha": alphas_grid, "selection": ["cyclic", "random"]},
        ),
        (
            "ElasticNet",
            ElasticNetRegressor(),
            {"alpha": alphas_grid, "l1_ratio": [0.1, 0.5, 0.9]},
        ),
        (
            "RandomForest",
            RandomForestWrapper(),
            {"n_estimators": [100], "max_depth": [10, 20, None]},
        ),
        (
            "XGBoost",
            XGBoostWrapper(),
            {
                "n_estimators": [1000, 3000],
                "max_depth": [6, 9],
                "learning_rate": [0.1, 0.2],
            },
        ),
    ]

    metrics_report = {}
    best_name = None
    best_rmse = float("inf")
    best_model_wrapper = None

    logger.info("Начало эксперимента с расширенным поиском...")

    for name, wrapper, params in models_config:
        if name in ["Lasso", "ElasticNet"]:
            wrapper._model.max_iter = 1000
            wrapper._model.tol = 1e-3

        try:
            best_est = tune_and_fit(name, wrapper, X_train, y_train, params)
            wrapper._model = best_est
        except Exception as e:
            logger.error(f"Ошибка при обучении {name}: {e}")
            continue

        metrics = evaluate_model(name, wrapper, X_test, y_test, resources_dir)
        metrics_report[name] = metrics

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_name = name
            best_model_wrapper = wrapper

    with open(resources_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4)

    if best_model_wrapper:
        logger.info(
            f"Лучшая модель: {best_name} (RMSE: {best_rmse:.0f}) -> best_model.pkl"
        )
        best_model_wrapper.save(resources_dir / "best_model.pkl")


if __name__ == "__main__":
    main()
