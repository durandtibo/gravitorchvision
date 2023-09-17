from __future__ import annotations

__all__ = ["SequentialDataPipeCreator"]

from collections.abc import Sequence
from typing import TypeVar

from coola.utils.format import str_indent, str_sequence
from gravitorch.engines.base import BaseEngine
from torch.utils.data import IterDataPipe, MapDataPipe

from gtvision.creators.datapipe.base import BaseDataPipeCreator, setup_datapipe_creator

T = TypeVar("T")


class SequentialDataPipeCreator(BaseDataPipeCreator):
    r"""Implements an ``DataPipe`` creator to create an ``DataPipe``
    object by using a sequence ``DataPipe`` creators.

    Args:
    ----
        creators: Specifies the sequence of ``DataPipe`` creators
            or their configurations. The sequence of creators follows
            the order of the ``DataPipe``s. The first creator is
            used to create the first ``DataPipe`` (a.k.a. source),
            and the last creator is used to create the last
            ``DataPipe`` (a.k.a. sink). This creator assumes all
            the DataPipes have a single source DataPipe as their first
            argument, excepts for the source ``DataPipe``.

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data import IterDataPipe, MapDataPipe
        >>> from torch.utils.data.datapipes.iter import Batcher, IterableWrapper
        >>> from gtvision.creators.datapipe import (
        ...     SequentialDataPipeCreator,
        ...     ChainedDataPipeCreator,
        ... )
        >>> # Create an DataPipe object using a single DataPipe creator and no source input
        >>> creator = SequentialDataPipeCreator(
        ...     [
        ...         ChainedDataPipeCreator(
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                 "iterable": [1, 2, 3, 4],
        ...             },
        ...         ),
        ...     ]
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # It is possible to use the source_inputs to create the same DataPipe object.
        >>> # The data is given by the source_inputs
        >>> creator = SequentialDataPipeCreator(
        ...     [
        ...         ChainedDataPipeCreator(
        ...             {"_target_": "torch.utils.data.datapipes.iter.IterableWrapper"},
        ...         ),
        ...     ]
        ... )
        >>> datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # Create an DataPipe object using two DataPipe creators and no source input
        >>> creator = SequentialDataPipeCreator(
        ...     [
        ...         ChainedDataPipeCreator(
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                 "iterable": [1, 2, 3, 4],
        ...             },
        ...         ),
        ...         ChainedDataPipeCreator(
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...                 "batch_size": 2,
        ...             },
        ...         ),
        ...     ]
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        >>> # It is possible to use the source_inputs to create the same DataPipe object.
        >>> # A source DataPipe object is specified by using source_inputs
        >>> creator = SequentialDataPipeCreator(
        ...     creators=[
        ...         ChainedDataPipeCreator(
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...                 "batch_size": 2,
        ...             },
        ...         ),
        ...     ]
        ... )
        >>> datapipe = creator.create(source_inputs=[IterableWrapper([1, 2, 3, 4])])
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        >>> # It is possible to create a sequential ``DataPipe`` object that takes several
        >>> # DataPipe objects as input.
        >>> creator = SequentialDataPipeCreator(
        ...     [
        ...         ChainedDataPipeCreator(
        ...             [
        ...                 {"_target_": "torch.utils.data.datapipes.iter.Multiplexer"},
        ...                 {
        ...                     "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...                     "batch_size": 2,
        ...                 },
        ...             ],
        ...         ),
        ...     ]
        ... )
        >>> datapipe = creator.create(
        ...     source_inputs=(IterableWrapper([1, 2, 3, 4]), IterableWrapper([11, 12, 13, 14])),
        ... )
        >>> tuple(datapipe)
        ([1, 11], [2, 12], [3, 13], [4, 14])
    """

    def __init__(self, creators: Sequence[BaseDataPipeCreator | dict]) -> None:
        if not creators:
            raise ValueError("It is not possible to create a DataPipe because creators is empty")
        self._creators = [setup_datapipe_creator(creator) for creator in creators]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n" f"  {str_indent(str_sequence(self._creators))},\n)"
        )

    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe[T] | MapDataPipe[T]:
        datapipe = self._creators[0].create(engine=engine, source_inputs=source_inputs)
        for creator in self._creators[1:]:
            datapipe = creator.create(engine=engine, source_inputs=(datapipe,))
        return datapipe
