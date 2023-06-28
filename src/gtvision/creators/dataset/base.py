from __future__ import annotations

__all__ = ["BaseDatasetCreator"]

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from gravitorch.engines.base import BaseEngine
from torch.utils.data import Dataset

T = TypeVar("T")


class BaseDatasetCreator(Generic[T], ABC):
    r"""Define the base class to implement a dataset creator.

    Example usage:

    .. code-block:: pycon

        >>> from gtvision.creators.dataset import DatasetCreator
        >>> creator = DatasetCreator(
        ...     {
        ...         "_target_": "gravitorch.data.datasets.DummyMultiClassDataset",
        ...         "num_examples": 10,
        ...         "num_classes": 2,
        ...         "feature_size": 4,
        ...     }
        ... )
        >>> creator.create()
        DummyMultiClassDataset(num_examples=10, num_classes=2, feature_size=4, noise_std=0.2)
    """

    @abstractmethod
    def create(self, engine: BaseEngine | None = None) -> Dataset[T]:
        r"""Create a dataset.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``

        Returns:
        -------
            ``torch.utils.data.Dataset``: The created dataset.
        """
