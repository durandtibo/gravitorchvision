from __future__ import annotations

__all__ = ["ChainedDataPipeCreator"]

from collections.abc import Sequence
from typing import TypeVar

from gravitorch.datapipes import create_chained_datapipe
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent, str_torch_sequence
from torch.utils.data import IterDataPipe, MapDataPipe

from gtvision.creators.datapipe.base import BaseDataPipeCreator

T = TypeVar("T")


class ChainedDataPipeCreator(BaseDataPipeCreator[T]):
    r"""Implements an ``DataPipe`` creator to create a chain of
    ``DataPipe``s from their configuration.

    Args:
    ----
        config (dict or sequence of dict): Specifies the configuration
            of the ``DataPipe`` object to create. See description
            of the ``create_chained_datapipe`` function to
            learn more about the expected values.

    Raises:
    ------
        ValueError if the configuration sequence is empty.

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gtvision.creators.datapipe import ChainedDataPipeCreator
        >>> # Create an DataPipe object using a single DataPipe object and no source input
        >>> creator = ChainedDataPipeCreator(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": [1, 2, 3, 4],
        ...     }
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # Equivalent to
        >>> creator = ChainedDataPipeCreator(
        ...     [
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4],
        ...         },
        ...     ]
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # It is possible to use the source_inputs to create the same DataPipe object.
        >>> # The data is given by the source_inputs
        >>> creator = ChainedDataPipeCreator(
        ...     config={"_target_": "torch.utils.data.datapipes.iter.IterableWrapper"},
        ... )
        >>> datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # Create an DataPipe object using two DataPipe objects and no source input
        >>> creator = ChainedDataPipeCreator(
        ...     [
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4],
        ...         },
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...             "batch_size": 2,
        ...         },
        ...     ]
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        >>> # It is possible to use the source_inputs to create the same DataPipe object.
        >>> # A source DataPipe object is specified by using source_inputs
        >>> creator = ChainedDataPipeCreator(
        ...     config=[
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...             "batch_size": 2,
        ...         },
        ...     ],
        ... )
        >>> datapipe = creator.create(source_inputs=[IterableWrapper([1, 2, 3, 4])])
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        >>> # It is possible to create a chained ``DataPipe`` object that takes several
        >>> # DataPipe objects as input.
        >>> creator = ChainedDataPipeCreator(
        ...     config=[
        ...         {"_target_": "torch.utils.data.datapipes.iter.Multiplexer"},
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...             "batch_size": 2,
        ...         },
        ...     ],
        ... )
        >>> datapipe = creator.create(
        ...     source_inputs=[
        ...         IterableWrapper([1, 2, 3, 4]),
        ...         IterableWrapper([11, 12, 13, 14]),
        ...     ],
        ... )
        >>> tuple(datapipe)
        ([1, 11], [2, 12], [3, 13], [4, 14])
    """

    def __init__(self, config: dict | Sequence[dict]) -> None:
        if not config:
            raise ValueError("It is not possible to create a DataPipe because the config is empty")
        self._config = config

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n  {str_indent(str_torch_sequence(self._config))}\n)"
        )

    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe[T] | MapDataPipe[T]:
        return create_chained_datapipe(config=self._config, source_inputs=source_inputs)
