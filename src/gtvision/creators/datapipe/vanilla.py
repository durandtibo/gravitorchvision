from __future__ import annotations

__all__ = ["DataPipeCreator"]

from collections.abc import Sequence
from typing import TypeVar

from gravitorch.datapipes import clone_datapipe, setup_datapipe
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent
from torch.utils.data.graph import DataPipe

from gtvision.creators.datapipe.base import BaseDataPipeCreator

T = TypeVar("T")


class DataPipeCreator(BaseDataPipeCreator[T]):
    r"""Implements a simple ``DataPipe`` creator.

    Args:
    ----
        datapipe (``torch.utils.data.graph.DataPipe`` or dict):
            Specifies the ``DataPipe`` or its configuration.
        cache (bool, optional): If ``True``, the ``DataPipe`` is
            created only the first time, and then the same
            ``DataPipe`` is returned for each call to the
            ``create`` method. Default: ``False``
        deepcopy (bool, optional): If ``True``, the ``DataPipe``
            object is deep-copied before to iterate over the data.
            It allows a deterministic behavior when in-place
            operations are performed on the data.
            Default: ``False``
    """

    def __init__(
        self, datapipe: DataPipe[T] | dict, cache: bool = True, deepcopy: bool = False
    ) -> None:
        self._datapipe = datapipe
        self._cache = bool(cache)
        self._deepcopy = bool(deepcopy)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  datapipe={str_indent(self._datapipe)}\n"
            f"  cache={self._cache},"
            f"  deepcopy={self._deepcopy},"
            ")"
        )

    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> DataPipe[T]:
        datapipe = setup_datapipe(self._datapipe)
        if self._cache and not isinstance(self._datapipe, DataPipe):
            self._datapipe = datapipe
        if self._deepcopy:
            datapipe = clone_datapipe(datapipe, raise_error=False)
        return datapipe
