from __future__ import annotations

__all__ = ["BaseDataFlowCreator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from gravitorch.experimental.dataflow.base import BaseDataFlow

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class BaseDataFlowCreator(Generic[T], ABC):
    r"""Define the base class to implement a dataflow creator.

    Example usage:

    .. code-block:: pycon

        >>> from gtvision.creators.dataflow import IterableDataFlowCreator
        >>> creator = IterableDataFlowCreator((1, 2, 3, 4, 5))
        >>> creator.create()
    """

    @abstractmethod
    def create(self, engine: BaseEngine | None = None) -> BaseDataFlow[T]:
        r"""Create a dataflow.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``

        Returns:
        -------
            ``BaseDataFlow``: The created dataflow.
        """
