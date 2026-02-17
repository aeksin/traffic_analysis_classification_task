"""Базовый класс обработчика, реализующий паттерн Chain of Responsibility."""

from abc import ABC, abstractmethod
from typing import Optional

from ..context import PipelineContext


class Handler(ABC):
    """Базовый обработчик для паттерна «Цепочка ответственности».

    Каждый обработчик выполняет одно преобразование над контекстом пайплайна
    и передаёт управление следующему обработчику, если он задан.

    Назначение обработчика:
    - инкапсулировать одну логически завершённую операцию;
    - не изменять глобальное состояние;
    - работать только с переданным контекстом.

    Все наследники должны реализовывать метод `_handle`.
    """

    def __init__(self) -> None:
        """Инициализировать обработчик без следующего звена цепочки."""
        self._next: Optional[Handler] = None

    def set_next(self, handler: "Handler") -> "Handler":
        """Установить следующий обработчик в цепочке.

        Аргументы:
            handler: Следующий обработчик.

        Возвращает:
            Обработчик `handler`, чтобы поддерживать цепочное построение.
        """
        self._next = handler
        return handler

    def handle(self, ctx: PipelineContext) -> PipelineContext:
        """Выполнить обработку и передать контекст дальше по цепочке.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Контекст после выполнения всех обработчиков цепочки.
        """
        ctx = self._handle(ctx)
        if self._next is not None:
            return self._next.handle(ctx)
        return ctx

    @abstractmethod
    def _handle(self, ctx: PipelineContext) -> PipelineContext:
        """Реализовать основную логику обработчика.

        Аргументы:
            ctx: Контекст пайплайна.

        Возвращает:
            Обновлённый контекст пайплайна.
        """
        raise NotImplementedError
