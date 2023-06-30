from __future__ import annotations

__all__ = ["BaseDataPipeCreator"]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

from torch.utils.data.graph import DataPipe

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class BaseDataPipeCreator(Generic[T], ABC):
    r"""Define the base class to implement a ``DataPipe`` creator.

    Example usage:

    .. code-block:: pycon

        >>> TODO
    """

    @abstractmethod
    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> DataPipe[T]:
        r"""Create a ``DataPipe``.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``
            source_inputs (sequence or ``None``, optional): Specifies
                the first positional arguments of the source
                ``DataPipe``. This argument can be used to create a
                new ``DataPipe`` object, that takes existing
                ``DataPipe`` objects as input. See examples below to
                see how to use it. If ``None``, ``source_inputs`` is
                set to an empty tuple. Default: ``None``

        Returns:
        -------
            ``DataPipe``: The created ``DataPipe``.
        """
