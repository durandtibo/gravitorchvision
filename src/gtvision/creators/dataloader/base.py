from __future__ import annotations

__all__ = ["BaseDataLoaderCreator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class BaseDataLoaderCreator(Generic[T], ABC):
    r"""Define the base class to implement a data loader creator.

    Example usage:

    .. code-block:: pycon

        >>> from gtvision.creators.dataloader import DataLoaderCreator
        >>> creator = DataLoaderCreator(
        ...     {
        ...         "_target_": "gravitorch.data.datasets.DummyMultiClassDataLoader",
        ...         "num_examples": 10,
        ...         "num_classes": 2,
        ...         "feature_size": 4,
        ...     }
        ... )
        >>> creator.create()
        DummyMultiClassDataLoader(num_examples=10, num_classes=2, feature_size=4, noise_std=0.2)
    """

    @abstractmethod
    def create(self, engine: BaseEngine | None = None) -> DataLoader[T]:
        r"""Create a data loader.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``

        Returns:
        -------
            ``torch.utils.data.DataLoader``: The created data loader.
        """
