from __future__ import annotations

__all__ = ["DataLoaderCreator", "VanillaDataLoaderCreator"]

from typing import TypeVar

from gravitorch.data.dataloaders import create_dataloader
from gravitorch.data.datasets import is_dataset_config
from gravitorch.engines import BaseEngine
from gravitorch.utils import setup_object
from gravitorch.utils.format import str_indent, str_pretty_dict
from gravitorch.utils.seed import get_torch_generator
from torch.utils.data import DataLoader, Dataset

from gtvision.creators.dataloader.base import BaseDataLoaderCreator
from gtvision.creators.dataset.base import BaseDatasetCreator, setup_dataset_creator
from gtvision.creators.dataset.vanilla import DatasetCreator

T = TypeVar("T")


class DataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Implements a simple dataloader creator.

    Args:
    ----
        dataset (``torch.utils.data.Dataset`` or ``BaseDatasetCreator``
            or ``dict``): Specifies a dataset (or its configuration)
            or a dataset creator (or its configuration).
        seed (int, optional): Specifies the random seed used to
            reproduce the shuffling of the samples. Default: ``0``
        **kwargs: See ``torch.utils.data.DataLoader`` documentation.
    """

    def __init__(
        self, dataset: Dataset | BaseDatasetCreator | dict, seed: int = 0, **kwargs
    ) -> None:
        if isinstance(dataset, Dataset) or (
            isinstance(dataset, dict) and is_dataset_config(dataset)
        ):
            dataset = DatasetCreator(dataset)
        self._dataset = setup_dataset_creator(dataset)
        self._seed = int(seed)
        self._kwargs = kwargs

    def __str__(self) -> str:
        config = {"dataset": self._dataset, "seed": self._seed} | self._kwargs
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_pretty_dict(config, sorted_keys=True))}\n)"
        )

    def create(self, engine: BaseEngine | None = None) -> DataLoader[T]:
        epoch = 0 if engine is None else engine.epoch
        return create_dataloader(
            self._dataset.create(engine),
            generator=get_torch_generator(self._seed + epoch),
            **self._kwargs,
        )


class VanillaDataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Implements a simple dataloader creator.

    Args:
    ----
        dataloader (``torch.utils.data.DataLoader`` or dict): Specifies
            the dataloader or its configuration.
        cache (bool, optional): If ``True``, the dataloader is created
            only the first time, and then the same data is returned
            for each call to the ``create`` method.
            Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gtvision.creators.dataloader import VanillaDataLoaderCreator
        >>> from gravitorch.data.datasets import ExampleDataset
        >>> dataset = ExampleDataset((1, 2, 3, 4))
        >>> creator = VanillaDataLoaderCreator(
        ...     {"_target_": "torch.utils.data.DataLoader", "dataset": dataset},
        ... )
        >>> creator.create()
    )
    """

    def __init__(self, dataloader: DataLoader | dict, cache: bool = True) -> None:
        self._dataloader = dataloader
        self._cache = bool(cache)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  dataloader={str_indent(self._dataloader)}\n"
            f"  cache={self._cache},"
            ")"
        )

    def create(self, engine: BaseEngine | None = None) -> DataLoader[T]:
        dataloader = setup_object(self._dataloader)  # TODO: setup_dataloader
        if self._cache and not isinstance(self._dataloader, DataLoader):
            self._dataloader = dataloader
        return dataloader
