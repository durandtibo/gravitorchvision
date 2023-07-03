from __future__ import annotations

__all__ = ["VanillaDataLoaderCreator"]

from typing import TypeVar

from gravitorch.data.dataloaders import setup_dataloader
from gravitorch.engines import BaseEngine
from gravitorch.utils.format import str_indent
from torch.utils.data import DataLoader

from gtvision.creators.dataloader.base import BaseDataLoaderCreator

T = TypeVar("T")


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
        dataloader = setup_dataloader(self._dataloader)
        if self._cache and not isinstance(self._dataloader, DataLoader):
            self._dataloader = dataloader
        return dataloader
