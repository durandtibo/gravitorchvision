from __future__ import annotations

__all__ = ["DataLoaderDataFlowCreator"]

from typing import TypeVar

from gravitorch.data.dataloaders import is_dataloader_config
from gravitorch.engines.base import BaseEngine
from gravitorch.experimental.dataflow.dataloader import DataLoaderDataFlow
from gravitorch.utils.format import str_indent, str_pretty_dict
from torch.utils.data import DataLoader

from gtvision.creators.dataflow.base import BaseDataFlowCreator
from gtvision.creators.dataloader import DataLoaderCreator, setup_dataloader_creator
from gtvision.creators.dataloader.base import BaseDataLoaderCreator

T = TypeVar("T")


class DataLoaderDataFlowCreator(BaseDataFlowCreator[T]):
    r"""Implements a simple ``DataLoaderDataFlow`` creator.

    Args:
    ----
        dataloader (``torch.utils.data.DataLoader`` or
            ``BaseDataLoaderCreator``): Specifies a dataloader (or its
            configuration) or a dataloader creator (or its
            configuration).
    """

    def __init__(self, dataloader: DataLoader | BaseDataLoaderCreator | dict) -> None:
        if isinstance(dataloader, DataLoader) or (
            isinstance(dataloader, dict) and is_dataloader_config(dataloader)
        ):
            dataloader = DataLoaderCreator(dataloader)
        self._dataloader = setup_dataloader_creator(dataloader)

    def __str__(self) -> str:
        config = {"dataloader": self._dataloader}
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_pretty_dict(config, sorted_keys=True))}\n)"
        )

    def create(self, engine: BaseEngine | None = None) -> DataLoaderDataFlow[T]:
        return DataLoaderDataFlow(self._dataloader.create(engine))
