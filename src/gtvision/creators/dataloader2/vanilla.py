from __future__ import annotations

__all__ = ["DataLoader2Creator"]

from collections.abc import Iterable
from typing import TypeVar

from gravitorch.data.dataloaders import create_dataloader2
from gravitorch.engines import BaseEngine
from gravitorch.utils import setup_object
from gravitorch.utils.format import str_indent, str_pretty_dict
from torch.utils.data.graph import DataPipe
from torchdata.dataloader2 import DataLoader2, ReadingServiceInterface
from torchdata.dataloader2.adapter import Adapter

from gtvision.creators.dataloader2.base import BaseDataLoader2Creator
from gtvision.creators.datapipe.base import BaseDataPipeCreator

T = TypeVar("T")


class DataLoader2Creator(BaseDataLoader2Creator[T]):
    r"""Implements a simple dataloader creator.

    Args:
    ----
        datapipe (``torch.utils.data.graph.DataPipe``): Specifies a
            datapipe (or its configuration) or a datapipe creator
            (or its configuration).
        datapipe_adapter_fn: Specifies the ``Adapter`` function(s)
            that will be applied to the DataPipe. Default: ``None``
        reading_service: Defines how ``DataLoader2`` should execute
            operations over the ``DataPipe``, e.g.
            multiprocessing/distributed. Default: ``None``
    """

    def __init__(
        self,
        datapipe: DataPipe | BaseDataPipeCreator | dict,
        datapipe_adapter_fn: Iterable[Adapter | dict] | Adapter | dict | None = None,
        reading_service: ReadingServiceInterface | dict | None = None,
    ) -> None:
        self._datapipe = datapipe
        self._datapipe_adapter_fn = datapipe_adapter_fn
        self._reading_service = reading_service

    def __str__(self) -> str:
        config = {
            "datapipe": self._datapipe,
            "datapipe_adapter_fn": self._datapipe_adapter_fn,
            "reading_service": self._reading_service,
        }
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_pretty_dict(config, sorted_keys=True))}\n)"
        )

    def create(self, engine: BaseEngine | None = None) -> DataLoader2[T]:
        datapipe = setup_object(self._datapipe)
        if isinstance(datapipe, BaseDataPipeCreator):
            datapipe = datapipe.create(engine)
        return create_dataloader2(
            datapipe=datapipe,
            datapipe_adapter_fn=self._datapipe_adapter_fn,
            reading_service=self._reading_service,
        )
