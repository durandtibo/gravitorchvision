from __future__ import annotations

__all__ = ["DataPipeCreator"]

from collections.abc import Sequence
from typing import TypeVar

from gravitorch.datapipes import clone_datapipe, setup_datapipe
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent, str_pretty_dict
from torch.utils.data import IterDataPipe, MapDataPipe

from gtvision.creators.datapipe.base import BaseDataPipeCreator

T = TypeVar("T")


class DataPipeCreator(BaseDataPipeCreator[T]):
    r"""Implements a simple ``DataPipe`` creator.

    Args:
    ----
        datapipe (``IterDataPipe`` or ``MapDataPipe`` or dict):
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
        self,
        datapipe: IterDataPipe[T] | MapDataPipe[T] | dict,
        cache: bool = True,
        deepcopy: bool = False,
    ) -> None:
        self._datapipe = datapipe
        self._cache = bool(cache)
        self._deepcopy = bool(deepcopy)

    def __str__(self) -> str:
        config = {"datapipe": self._datapipe, "cache": self._cache, "deepcopy": self._deepcopy}
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_pretty_dict(config, sorted_keys=True))}\n)"
        )

    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe[T] | MapDataPipe[T]:
        datapipe = setup_datapipe(self._datapipe)
        if self._cache:
            self._datapipe = datapipe
        if self._deepcopy:
            datapipe = clone_datapipe(datapipe, raise_error=False)
        return datapipe
