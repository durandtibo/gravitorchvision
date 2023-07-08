from __future__ import annotations

__all__ = ["AutoDataLoaderCreator"]

from typing import TypeVar

from gravitorch.distributed import comm as dist
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent
from torch.utils.data import DataLoader, Dataset

from gtvision.creators.dataloader.base import BaseDataLoaderCreator
from gtvision.creators.dataloader.distributed import DistributedDataLoaderCreator
from gtvision.creators.dataloader.vanilla import VanillaDataLoaderCreator
from gtvision.creators.dataset.base import BaseDatasetCreator

T = TypeVar("T")


class AutoDataLoaderCreator(BaseDataLoaderCreator[T]):
    r"""Defines a PyTorch dataloader creator that automatically chooses
    the dataloader creator based on the context.

    If the distributed package is activated, it uses the
    ``DistributedDataLoaderCreator``, otherwise it uses
    ``DataLoaderCreator``.

    Note the behavior of this class may change based on the new data
    loader creators.

    Args:
    ----
        dataset (``torch.utils.data.Dataset``): Specifies a
            dataset (or its configuration) or a dataset creator
            (or its configuration).
        **kwargs: See ``DataLoaderCreator`` or
            ``DistributedDataLoaderCreator`` documentation.

    Example usage:

    .. code-block:: pycon

        >>> from gtvision.creators.dataloader import AutoDataLoaderCreator
        >>> creator = AutoDataLoaderCreator(
        ...     {
        ...         "_target_": "gravitorch.data.datasets.DummyMultiClassDataset",
        ...         "num_examples": 10,
        ...         "num_classes": 2,
        ...         "feature_size": 4,
        ...     }
        ... )
        >>> creator.create()  # doctest: +ELLIPSIS
        <torch.utils.data.dataloader.DataLoader object at 0x...>
    """

    def __init__(self, dataset: Dataset | BaseDatasetCreator | dict, **kwargs) -> None:
        if dist.is_distributed():
            self._dataloader = DistributedDataLoaderCreator(dataset, **kwargs)
        else:
            self._dataloader = VanillaDataLoaderCreator(dataset, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  dataloader={str_indent(self._dataloader)}\n)"

    def create(self, engine: BaseEngine | None = None) -> DataLoader[T]:
        return self._dataloader.create(engine=engine)
