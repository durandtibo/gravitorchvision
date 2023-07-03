from __future__ import annotations

__all__ = ["BaseDataPipeCreator", "setup_datapipe_creator"]

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

from gravitorch.utils.format import str_target_object
from objectory import AbstractFactory
from torch.utils.data import IterDataPipe, MapDataPipe

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataPipeCreator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a ``DataPipe`` creator.

    Example usage:

    .. code-block:: pycon

        >>> from gtvision.creators.datapipe import ChainedDataPipeCreator
        >>> creator = ChainedDataPipeCreator(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": [1, 2, 3, 4],
        ...     }
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
    """

    @abstractmethod
    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe[T] | MapDataPipe[T]:
        r"""Create a ``DataPipe``.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``
            source_inputs (sequence or ``None``, optional): Specifies
                the first positional arguments of the source
                ``DataPipe``. This argument can be used to create a
                new ``DataPipe`` object, that takes existing
                ``DataPipe`` objects as input. See examples below to
                see how to use it. If ``None``, ``source_inputs`` is
                set to an empty tuple. Default: ``None``

        Returns:
        -------
            ``IterDataPipe`` or ``MapDataPipe``: The created
                ``DataPipe``.
        """


def setup_datapipe_creator(creator: BaseDataPipeCreator | dict) -> BaseDataPipeCreator:
    r"""Sets up the datapipe creator.

    The datapipe creator is instantiated from its configuration by
    using the ``BaseDataPipeCreator`` factory function.

    Args:
    ----
        creator (``BaseDataPipeCreator`` or dict): Specifies the
            datapipe creator or its configuration.

    Returns:
    -------
        ``BaseDataPipeCreator``: The instantiated datapipe creator.
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing the datapipe creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataPipeCreator.factory(**creator)
    return creator
