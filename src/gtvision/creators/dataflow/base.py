from __future__ import annotations

__all__ = ["BaseDataFlowCreator", "setup_dataflow_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from gravitorch.experimental.dataflow.base import BaseDataFlow
from gravitorch.utils.format import str_target_object
from objectory import AbstractFactory

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataFlowCreator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a dataflow creator.

    Example usage:

    .. code-block:: pycon

        >>> from gtvision.creators.dataflow import IterableDataFlowCreator
        >>> creator = IterableDataFlowCreator((1, 2, 3, 4, 5))
        >>> creator.create()
    """

    @abstractmethod
    def create(self, engine: BaseEngine | None = None) -> BaseDataFlow[T]:
        r"""Create a dataflow.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``

        Returns:
        -------
            ``BaseDataFlow``: The created dataflow.
        """


def setup_dataflow_creator(creator: BaseDataFlowCreator[T] | dict) -> BaseDataFlowCreator[T]:
    r"""Sets up the dataflow creator.

    The dataflow creator is instantiated from its configuration by
    using the ``BaseDataFlowCreator`` factory function.

    Args:
    ----
        creator (``BaseDataFlowCreator`` or dict): Specifies the
            dataflow creator or its configuration.

    Returns:
    -------
        ``BaseDataFlowCreator``: The instantiated dataflow creator.
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing the dataflow creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataFlowCreator.factory(**creator)
    return creator
