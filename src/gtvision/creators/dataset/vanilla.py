from __future__ import annotations

__all__ = ["DatasetCreator"]

from typing import TypeVar

from gravitorch.data.datasets import setup_dataset
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent
from torch.utils.data import Dataset

from gtvision.creators.dataset.base import BaseDatasetCreator

T = TypeVar("T")


class DatasetCreator(BaseDatasetCreator[T]):
    r"""Implements a simple dataset creator.

    Args:
    ----
        dataset (``torch.utils.data.Dataset`` or dict): Specifies
            the dataset or its configuration.
        cache (bool, optional): If ``True``, the dataset is created
            only the first time, and then the same data is returned
            for each call to the ``create`` method.
            Default: ``False``
    """

    def __init__(self, dataset: Dataset[T] | dict, cache: bool = True) -> None:
        self._dataset = dataset
        self._cache = bool(cache)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  dataset={str_indent(self._dataset)}\n"
            f"  cache={self._cache},"
            ")"
        )

    def create(self, engine: BaseEngine | None = None) -> Dataset[T]:
        dataset = setup_dataset(self._dataset)
        if self._cache and not isinstance(self._dataset, Dataset):
            self._dataset = dataset
        return dataset
