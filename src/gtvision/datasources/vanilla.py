from __future__ import annotations

__all__ = ["VanillaDataSource"]

import logging
from collections.abc import Mapping
from typing import Any

from coola.utils import str_indent, str_mapping
from gravitorch.datasources import BaseDataSource, LoaderNotFoundError
from gravitorch.engines import BaseEngine
from gravitorch.experimental.dataflow import BaseDataFlow
from gravitorch.utils.asset import AssetManager

from gtvision.creators.dataflow.base import BaseDataFlowCreator, setup_dataflow_creator

logger = logging.getLogger(__name__)


class VanillaDataSource(BaseDataSource):
    r"""Implement a simple data source using dataflow creators.

    Args:
    ----
        dataflow_creators (``Mapping``): Specifies the dataflow
            creators or their configuration.

    Example usage:

    .. code-block:: pycon

        >>> from gtvision.datasources import VanillaDataSource
        >>> from gtvision.creators.dataflow import IterableDataFlowCreator
        >>> datasource = VanillaDataSource(
        ...     {
        ...         "train": {
        ...             "_target_": "gtvision.creators.dataflow.IterableDataFlowCreator",
        ...             "iterable": [1, 2, 3, 4],
        ...         },
        ...         "eval": IterableDataFlowCreator(["a", "b", "c"]),
        ...     }
        ... )
        >>> datasource
        VanillaDataSource(
          (train): IterableDataFlowCreator(cache=False, length=4)
          (eval): IterableDataFlowCreator(cache=False, length=3)
        )
    """

    def __init__(self, dataflow_creators: Mapping[str, BaseDataFlowCreator | dict]) -> None:
        self._asset_manager = AssetManager()
        self._dataflow_creators: Mapping[str, BaseDataFlowCreator] = {
            key: setup_dataflow_creator(value) for key, value in dataflow_creators.items()
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_mapping(self._dataflow_creators))}\n)"
        )

    def attach(self, engine: BaseEngine) -> None:
        logger.info("Attach the data source to an engine")

    def get_asset(self, asset_id: str) -> Any:
        return self._asset_manager.get_asset(asset_id)

    def has_asset(self, asset_id: str) -> bool:
        return self._asset_manager.has_asset(asset_id)

    def get_dataloader(self, loader_id: str, engine: BaseEngine | None = None) -> BaseDataFlow:
        if not self.has_dataloader(loader_id):
            raise LoaderNotFoundError(f"{loader_id} does not exist")
        return self._dataflow_creators[loader_id].create(engine=engine)

    def has_dataloader(self, loader_id: str) -> bool:
        return loader_id in self._dataflow_creators
