r"""This module defines some utility functions for the dataloader
creators."""

from __future__ import annotations

__all__ = ["setup_dataloader_creator"]

import logging

from gravitorch.utils.format import str_target_object

from gtvision.creators.dataloader.base import BaseDataLoaderCreator

logger = logging.getLogger(__name__)


def setup_dataloader_creator(
    creator: BaseDataLoaderCreator | dict | None,
) -> BaseDataLoaderCreator:
    r"""Sets up a dataloader creator.

    Args:
    ----
        creator (``BaseDataLoaderCreator`` or dict or None):
            Specifies the dataloader creator or its configuration.
            If ``None``, a dataloader creator will be created
            automatically.

    Returns:
    -------
        ``BaseDataLoaderCreator``: The dataloader creator.
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing a dataloader creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataLoaderCreator.factory(**creator)
    return creator
