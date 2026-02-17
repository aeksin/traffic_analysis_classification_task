"""Скрипт для обучения моделей, подбора гиперпараметров и сохранения артефактов."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

from models.base import BaseModel
from models.classifiers import (LogisticRegressionWrapper,
                                RandomForestClassifierWrapper,
                                XGBClassifierWrapper)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["Junior", "Middle", "Senior"]


def load_data(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Загрузить матрицы признаков и таргета из .npy файлов.

    Аргументы:
        x_path: Путь к файлу с матрицей признаков (X).
        y_path: Путь к файлу с вектором целевой переменной (y).

    Возвращает:
        Кортеж (X, y), где X — массив float32, y — массив int.

    Исключения:
        FileNotFoundError: Если один из файлов не найден.
    """
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Файлы данных не найдены.")

    X = np.load(x_path)
    y = np.load(y_path)

    y = y.astype(int)

    logger.info(f"Данные загружены: X shape={X.shape}, y shape={y.shape}")
    return X, y


def plot_class_balance(y: np.ndarray, output_dir: Path) -> None:
    """Построить и сохранить график распределения классов.

    Создает файл `class_balance.png` в указанной директории.

    Аргументы:
        y: Вектор целевой переменной.
        output_dir: Директория для сохранения графика.
    """
    unique, counts = np.unique(y, return_counts=True)

    x_labels = [CLASS_NAMES[i] for i in unique]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=x_labels, y=counts, hue=x_labels, legend=False, palette="viridis")
    plt.title("Баланс классов (Junior / Middle / Senior)")
    plt.xlabel("Уровень")
    plt.ylabel("Количество резюме")

    out_path = output_dir / "class_balance.png"
    plt.savefig(out_path)
    plt.close()

    logger.info(f"График баланса классов сохранен: {out_path}")


def tune_and_fit(
    name: str,
    model_wrapper: BaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict[str, list[Any]] | None = None,
) -> Any:
    """Обучить модель с подбором гиперпараметров (GridSearch).

    Если `param_grid` передан, выполняется кросс-валидация (CV=3)
    с оптимизацией метрики `f1_weighted`.

    Аргументы:
        name: Название модели (для логов).
        model_wrapper: Экземпляр обертки модели (наследник BaseModel).
        X_train: Обучающая выборка признаков.
        y_train: Обучающая выборка таргета.
        param_grid: Словарь гиперпараметров для перебора.

    Возвращает:
        Лучший оценщик (Best Estimator) из библиотеки sklearn/xgboost.
    """

    sklearn_model = model_wrapper._model

    if not param_grid:
        logger.info(f"Обучение {name} (без тюнинга)...")
        sklearn_model.fit(X_train, y_train)
        return sklearn_model

    logger.info(f"Запуск GridSearch для {name}...")
    grid = GridSearchCV(
        estimator=sklearn_model,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    logger.info(f"Лучшие параметры для {name}: {grid.best_params_}")
    return grid.best_estimator_


def evaluate_model(
    name: str,
    model_wrapper: BaseModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    """Оценить качество модели на тестовой выборке и сохранить результаты.

    Сохраняет:
    1. Сериализованную модель (.pkl).
    2. Текстовый отчет Classification Report (.txt).

    Аргументы:
        name: Название модели.
        model_wrapper: Обученная обертка модели.
        X_test: Тестовая выборка признаков.
        y_test: Тестовая выборка таргета.
        output_dir: Директория для сохранения артефактов.

    Возвращает:
        Словарь с метриками (f1_weighted и детализированный отчет).
    """
    y_pred = model_wrapper.predict(X_test)

    report_dict = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    report_str = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    logger.info(f"\n--- Report for {name} ---\n{report_str}")

    f1 = f1_score(y_test, y_pred, average="weighted")

    model_path = output_dir / f"{name.lower()}_model.pkl"
    model_wrapper.save(model_path)

    report_path = output_dir / f"{name.lower()}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    return {"f1_weighted": f1, "detail": report_dict}


def main() -> None:
    """Точка входа скрипта обучения.

    1. Загружает подготовленные данные.
    2. Строит график баланса классов.
    3. Делит данные на train/test.
    4. Запускает цикл обучения для списка моделей.
    5. Выбирает лучшую модель по F1-weighted и сохраняет её как `best_model.pkl`.
    """
    parser = argparse.ArgumentParser(
        description="Скрипт обучения моделей классификации."
    )
    parser.add_argument(
        "data_dir", type=Path, help="Папка с файлами x_data.npy и y_data.npy"
    )
    args = parser.parse_args()

    x_path = args.data_dir / "x_data.npy"
    y_path = args.data_dir / "y_data.npy"

    resources_dir = Path(__file__).parent / "resources"
    resources_dir.mkdir(exist_ok=True)

    try:
        X, y = load_data(x_path, y_path)
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        return

    plot_class_balance(y, resources_dir)

    logger.info("Разделение данных на train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models_config = [
        ("LogisticRegression", LogisticRegressionWrapper(), {"C": [0.1, 1.0, 10.0]}),
        (
            "RandomForest",
            RandomForestClassifierWrapper(),
            {
                "n_estimators": [100],
                "max_depth": [20],
                "min_samples_leaf": [4],
                "bootstrap": [True],
            },
        ),
        (
            "XGBoost",
            XGBClassifierWrapper(),
            {
                "n_estimators": [1500],
                "learning_rate": [0.1],
                "max_depth": [6],
                "colsample_bytree": [0.6],
                "subsample": [0.8],
                "reg_alpha": [0],
                "min_child_weight": [3],
            },
        ),
    ]

    best_name = None
    best_f1 = 0.0
    best_wrapper = None
    metrics_all = {}

    for name, wrapper, params in models_config:
        try:
            best_est = tune_and_fit(name, wrapper, X_train, y_train, params)

            wrapper._model = best_est

            metrics = evaluate_model(name, wrapper, X_test, y_test, resources_dir)
            metrics_all[name] = metrics

            if metrics["f1_weighted"] > best_f1:
                best_f1 = metrics["f1_weighted"]
                best_name = name
                best_wrapper = wrapper

        except Exception as e:
            logger.error(f"Ошибка при обучении модели {name}: {e}")

    summary_path = resources_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=4, ensure_ascii=False)

    logger.info(f"Сводка метрик сохранена в {summary_path}")

    if best_wrapper:
        logger.info(f"ЛУЧШАЯ МОДЕЛЬ: {best_name} (F1 Weighted: {best_f1:.4f})")
        best_wrapper.save(resources_dir / "best_model.pkl")
    else:
        logger.warning("Не удалось обучить ни одну модель.")


if __name__ == "__main__":
    main()
