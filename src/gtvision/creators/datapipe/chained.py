from __future__ import annotations

__all__ = [
    "ChainedDataPipeCreator",
    "create_chained_datapipe",
]

from collections.abc import Sequence
from typing import TypeVar

from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent, str_torch_sequence
from objectory import OBJECT_TARGET, factory
from torch.utils.data import IterDataPipe, MapDataPipe

from gtvision.creators.datapipe.base import BaseDataPipeCreator

T = TypeVar("T")


class ChainedDataPipeCreator(BaseDataPipeCreator):
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

        >>> from gtvision.creators.datapipe import ChainedDataPipeCreator
        # Create an DataPipe object using a single DataPipe object and no source input
        >>> creator = ChainedDataPipeCreator(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": [1, 2, 3, 4],
        ...     }
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        # Equivalent to
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
        # It is possible to use the source_inputs to create the same DataPipe object.
        # The data is given by the source_inputs
        >>> creator = ChainedDataPipeCreator(
        ...     config={"_target_": "torch.utils.data.datapipes.iter.IterableWrapper"},
        ... )
        >>> datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        # Create an DataPipe object using two DataPipe objects and no source input
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
        # It is possible to use the source_inputs to create the same DataPipe object.
        # A source DataPipe object is specified by using source_inputs
        >>> creator = ChainedDataPipeCreator(
        ...     config=[
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...             "batch_size": 2,
        ...         },
        ...     ],
        ... )
        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> datapipe = creator.create(source_inputs=[IterableWrapper(data=[1, 2, 3, 4])])
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        # It is possible to create a chained ``DataPipe`` object that takes several
        # DataPipe objects as input.
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
        ...         IterableWrapper(data=[1, 2, 3, 4]),
        ...         IterableWrapper(data=[11, 12, 13, 14]),
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
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_torch_sequence(self._config))},\n)"
        )

    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe[T] | MapDataPipe[T]:
        return create_chained_datapipe(config=self._config, source_inputs=source_inputs)


def create_chained_datapipe(
    config: dict | Sequence[dict],
    source_inputs: Sequence | None = None,
) -> IterDataPipe[T] | MapDataPipe[T]:
    r"""Creates a chained ``DataPipe`` object.

    A chained ``DataPipe`` object has a single source (which
    can takes multiple ``DataPipe`` objects) and a single sink.
    The structure should look like:

        SourceDatapipe -> DataPipe1 -> DataPipe2 -> SinkDataPipe

    The structure of the ``config`` input depends on the chained
    ``DataPipe`` object that is created:

        - If ``config`` is a ``dict`` object, it creates a chained
            ``DataPipe`` object with a single ``DataPipe``
            object. The dictionary should contain the parameters used
            to initialize the ``DataPipe`` object. It should
            follow the ``object_factory`` syntax. Using a dict allows
            to initialize a single ``DataPipe`` object. If you
            want to create a ``DataPipe`` object recursively, you
            need to give a sequence of dict.
        - If ``config`` is a sequence of ``dict`` objects, this
            function creates an ``DataPipe`` object with a
            chained structure. The sequence of configurations
            follows the order of the ``DataPipe``s. The first
            config is used to create the first ``DataPipe``
            (a.k.a. source), and the last config is used to create the
            last ``DataPipe`` (a.k.a. sink). This function assumes
            all the DataPipes have a single source DataPipe as their
            first argument, excepts for the source ``DataPipe``.

    Note: it is possible to create chained ``DataPipe`` objects
    without using this function.

    Args:
    ----
        config (dict or sequence of dict): Specifies the configuration
            of the ``DataPipe`` object to create. See description
            above to know when to use a dict or a sequence of dicts.
        source_inputs (sequence or ``None``): Specifies the first
            positional arguments of the source ``DataPipe``. This
            argument can be used to create a new ``DataPipe``
            object, that takes existing ``DataPipe`` objects as
            input. See examples below to see how to use it.
            If ``None``, ``source_inputs`` is set to an empty tuple.
            Default: ``None``

    Returns:
    -------
        ``DataPipe``: The last (a.k.a. sink) ``DataPipe`` of
            the sequence.

    Raises:
    ------
        RuntimeError if the configuration is empty (empty dict or
            sequence).

    Example usage:

    .. code-block:: pycon

        >>> from gtvision.creators.datapipe.chained import create_chained_datapipe
        # Create an DataPipe object using a single DataPipe object and no source input
        >>> datapipe = create_chained_datapipe(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": [1, 2, 3, 4],
        ...     }
        ... )
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        # Equivalent to
        >>> datapipe = create_chained_datapipe(
        ...     [
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4],
        ...         },
        ...     ]
        ... )
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        # It is possible to use the source_inputs to create the same DataPipe object.
        # The data is given by the source_inputs
        >>> datapipe = create_chained_datapipe(
        ...     config={"_target_": "torch.utils.data.datapipes.iter.IterableWrapper"},
        ...     source_inputs=([1, 2, 3, 4],),
        ... )
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        # Create an DataPipe object using two DataPipe objects and no source input
        >>> datapipe = create_chained_datapipe(
        ...     [
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4],
        ...         },
        ...         {"_target_": "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ...     ]
        ... )
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        # It is possible to use the source_inputs to create the same DataPipe object.
        # A source DataPipe object is specified by using source_inputs
        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> datapipe = create_chained_datapipe(
        ...     config=[
        ...         {"_target_": "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ...     ],
        ...     source_inputs=[IterableWrapper(data=[1, 2, 3, 4])],
        ... )
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        # It is possible to create a chained ``DataPipe`` object that takes several
        # DataPipe objects as input.
        >>> datapipe = create_chained_datapipe(
        ...     config=[
        ...         {"_target_": "torch.utils.data.datapipes.iter.Multiplexer"},
        ...         {"_target_": "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ...     ],
        ...     source_inputs=[
        ...         IterableWrapper(data=[1, 2, 3, 4]),
        ...         IterableWrapper(data=[11, 12, 13, 14]),
        ...     ],
        ... )
        >>> tuple(datapipe)
        ([1, 11], [2, 12], [3, 13], [4, 14])
    """
    if not config:
        raise RuntimeError("It is not possible to create a DataPipe because the config is empty")
    source_inputs = source_inputs or ()
    if isinstance(config, dict):
        config = config.copy()  # Make a copy because the dict is modified below.
        target = config.pop(OBJECT_TARGET)
        return factory(target, *source_inputs, **config)
    datapipe = create_chained_datapipe(config[0], source_inputs)
    for cfg in config[1:]:
        datapipe = create_chained_datapipe(cfg, source_inputs=(datapipe,))
    return datapipe
