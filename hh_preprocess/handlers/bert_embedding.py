import logging
import os
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from ..context import PipelineContext
from .base import Handler

logger = logging.getLogger(__name__)


class BertEmbeddingHandler(Handler):
    """Преобразование текста опыта в векторные признаки (Embeddings) с помощью LLM."""

    def __init__(
        self, model_name: str = "cointegrated/rubert-tiny2", vector_size: int = 50
    ) -> None:
        """Инициализировать обработчик эмбеддингов.

        Аргументы:
            model_name: Имя модели на HuggingFace (по умолчанию rubert-tiny2).
            vector_size: Количество компонент PCA (размерность выходного вектора).
        """
        super().__init__()
        self.model_name = model_name
        self.vector_size = vector_size

    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Сгенерировать эмбеддинги для колонок с опытом работы.

        Использует SentenceTransformer для векторизации текста и PCA
        для понижения размерности, чтобы не перегружать линейные модели.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Обновленный контекст с колонками emb_0...emb_N.
        """
        if ctx.df is None:
            raise ValueError("DataFrame is not loaded")

        df = ctx.df.copy()

        exp_col = next((c for c in df.columns if "опыт" in str(c).lower()), None)
        if not exp_col:
            return ctx

        logger.info(f"Генерация LLM эмбеддингов (модель: {self.model_name})...")

        model = SentenceTransformer(self.model_name)
        model.max_seq_length = 256

        raw_texts = df[exp_col].fillna("").astype(str).tolist()

        clean_texts = [re.sub(r"\d+", "", t) for t in raw_texts]

        embeddings = model.encode(
            clean_texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,
        )

        if self.vector_size and embeddings.shape[1] > self.vector_size:
            pca = PCA(n_components=self.vector_size, random_state=42)
            embeddings = pca.fit_transform(embeddings)
            logger.info(f"Эмбеддинги сжаты PCA до {self.vector_size} компонент")

        for i in range(embeddings.shape[1]):
            df[f"emb_{i}"] = embeddings[:, i]

        ctx.df = df
        return ctx
