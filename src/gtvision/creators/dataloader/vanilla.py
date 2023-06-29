from __future__ import annotations

__all__ = ["DataLoaderCreator"]

from typing import TypeVar

from gravitorch.data.dataloaders import create_dataloader
from gravitorch.engines import BaseEngine
from gravitorch.utils import setup_object
from gravitorch.utils.format import str_indent, str_pretty_dict
from gravitorch.utils.seed import get_torch_generator
from torch.utils.data import DataLoader, Dataset

from gtvision.creators.dataloader.base import BaseDataLoaderCreator
from gtvision.creators.dataset.base import BaseDatasetCreator

T = TypeVar("T")


class DataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Implements a simple dataloader creator.

    Args:
    ----
        dataset (``torch.utils.data.Dataset``): Specifies a
            dataset (or its configuration) or a dataset creator
            (or its configuration).
        seed (int, optional): Specifies the random seed used to
            reproduce the shuffling of the samples. Default: ``0``
        **kwargs: See ``torch.utils.data.DataLoader`` documentation.
    """

    def __init__(
        self, dataset: Dataset | BaseDatasetCreator | dict, seed: int = 0, **kwargs
    ) -> None:
        self._dataset: Dataset | BaseDatasetCreator = setup_object(dataset)
        self._seed = int(seed)
        self._kwargs = kwargs

    def __str__(self) -> str:
        config = {"dataset": self._dataset, "seed": self._seed} | self._kwargs
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_pretty_dict(config, sorted_keys=True))}\n)"
        )

    def create(self, engine: BaseEngine | None = None) -> DataLoader[T]:
        dataset = self._dataset
        if isinstance(dataset, BaseDatasetCreator):
            dataset = dataset.create(engine)
        epoch = 0 if engine is None else engine.epoch
        return create_dataloader(
            dataset, generator=get_torch_generator(self._seed + epoch), **self._kwargs
        )
