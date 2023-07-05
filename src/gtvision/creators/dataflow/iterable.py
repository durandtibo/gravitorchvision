from __future__ import annotations

__all__ = ["IterableDataFlowCreator"]

from collections.abc import Iterable
from typing import TypeVar

from gravitorch.engines.base import BaseEngine
from gravitorch.experimental.dataflow import IterableDataFlow
from gravitorch.utils import setup_object

from gtvision.creators.dataflow.base import BaseDataFlowCreator

T = TypeVar("T")


class IterableDataFlowCreator(BaseDataFlowCreator[T]):
    r"""Implements a simple ``IterableDataFlow`` creator.

    Args:
    ----
        iterable (``Iterable`` or dict): Specifies an iterable or its
            configuration.
        cache (bool, optional): If ``True``, the iterable is created
            only the first time, and then a copy of the iterable is
            returned for each call to the ``create`` method.
            Default: ``False``
        **kwargs: See ``IterableDataFlow`` documentation.

    Example usage:

    .. code-block:: pycon

        >>> from gtvision.creators.dataflow import IterableDataFlowCreator
        >>> creator = IterableDataFlowCreator((1, 2, 3, 4, 5))
        >>> creator.create()
    """

    def __init__(self, iterable: Iterable[T], cache: bool = False, **kwargs) -> None:
        self._iterable = iterable
        self._cache = bool(cache)
        self._kwargs = kwargs

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def create(self, engine: BaseEngine | None = None) -> IterableDataFlow[T]:
        iterable = setup_object(self._iterable)
        if self._cache:
            self._iterable = iterable
        return IterableDataFlow(iterable, **self._kwargs)
