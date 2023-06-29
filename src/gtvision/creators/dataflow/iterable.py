from __future__ import annotations

__all__ = ["IterableDataFlowCreator"]

from collections.abc import Iterable
from typing import TypeVar

from gravitorch.engines.base import BaseEngine
from gravitorch.experimental.dataflow import IterableDataFlow

from gtvision.creators.dataflow.base import BaseDataFlowCreator

T = TypeVar("T")


class IterableDataFlowCreator(BaseDataFlowCreator):
    r"""Implements a simple ``IterableDataFlow`` creator.

    Args:
    ----
        iterable (``Iterable``): Specifies the iterable.
        **kwargs: See ``IterableDataFlow`` documentation.
    """

    def __init__(self, iterable: Iterable[T], **kwargs) -> None:
        self._iterable = iterable
        self._kwargs = kwargs

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def create(self, engine: BaseEngine | None = None) -> IterableDataFlow[T]:
        return IterableDataFlow(self._iterable, **self._kwargs)
