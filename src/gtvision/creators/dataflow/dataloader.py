from __future__ import annotations

__all__ = ["DataLoaderDataFlowCreator"]

from typing import TypeVar

from gravitorch.engines.base import BaseEngine
from gravitorch.experimental.dataflow.dataloader import DataLoaderDataFlow
from gravitorch.utils import setup_object
from torch.utils.data import DataLoader

from gtvision.creators.dataflow.base import BaseDataFlowCreator
from gtvision.creators.dataloader.base import BaseDataLoaderCreator

T = TypeVar("T")


class DataLoaderDataFlowCreator(BaseDataFlowCreator):
    r"""Implements a simple ``DataLoaderDataFlow`` creator.

    Args:
    ----
        dataset (``torch.utils.data.DataLoader``): Specifies a
            data loader (or its configuration) or a data loader
            creator (or its configuration).
    """

    def __init__(self, dataloader: DataLoader | BaseDataLoaderCreator | dict) -> None:
        self._dataloader: DataLoader | BaseDataLoaderCreator = setup_object(dataloader)

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def create(self, engine: BaseEngine | None = None) -> DataLoaderDataFlow[T]:
        dataloader = self._dataloader
        if isinstance(dataloader, BaseDataLoaderCreator):
            dataloader = dataloader.create(engine)
        return DataLoaderDataFlow(dataloader)
