from __future__ import annotations

__all__ = ["BaseDataLoader2Creator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from gravitorch.utils.imports import is_torchdata_available
from objectory import AbstractFactory

if is_torchdata_available():
    from torchdata.dataloader2 import DataLoader2
else:  # pragma: no cover
    DataLoader2 = "DataLoader2"

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class BaseDataLoader2Creator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a dataloader creator.

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gtvision.creators.dataloader2 import DataLoader2Creator
        >>> creator = DataLoader2Creator(IterableWrapper([1, 2, 3, 4, 5]))
        >>> creator.create()
        <torchdata.dataloader2.DataLoader2 object at 0x0123456789>
    """

    @abstractmethod
    def create(self, engine: BaseEngine | None = None) -> DataLoader2[T]:
        r"""Create a dataloader.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``

        Returns:
        -------
            ``torchdata.dataloader2.DataLoader2``: The created dataloader.
        """
