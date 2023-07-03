from __future__ import annotations

from unittest.mock import Mock

from gravitorch.testing import torchdata_available
from gravitorch.utils.imports import is_torchdata_available
from objectory import OBJECT_TARGET
from pytest import fixture, mark
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

from gtvision.creators.dataloader2 import DataLoader2Creator
from gtvision.creators.datapipe import ChainedDataPipeCreator

if is_torchdata_available():
    from torchdata.dataloader2 import (
        DataLoader2,
        MultiProcessingReadingService,
        ReadingServiceInterface,
    )
    from torchdata.dataloader2.adapter import Adapter, Shuffle
else:  # pragma: no cover
    Adapter = "Adapter"
    MultiProcessingReadingService, Shuffle = Mock(), Mock()


@fixture
def datapipe() -> IterDataPipe:
    return IterableWrapper([1, 2, 3, 4, 5])


########################################
#     Tests for DataLoader2Creator     #
########################################


@torchdata_available
def test_dataloader2_creator_str(datapipe: IterDataPipe) -> None:
    assert str(DataLoader2Creator(datapipe)).startswith("DataLoader2Creator(")


@torchdata_available
def test_dataloader2_creator_datapipe(datapipe: IterDataPipe) -> None:
    dataloader = DataLoader2Creator(datapipe).create()
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, IterableWrapper)
    assert dataloader.datapipe_adapter_fns is None
    assert dataloader.reading_service is None
    assert tuple(dataloader) == (1, 2, 3, 4, 5)


@torchdata_available
def test_dataloader2_creator_datapipe_creator(datapipe: IterDataPipe) -> None:
    dataloader = DataLoader2Creator(
        ChainedDataPipeCreator(
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4, 5],
            }
        )
    ).create()
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, IterableWrapper)
    assert dataloader.datapipe_adapter_fns is None
    assert dataloader.reading_service is None
    assert tuple(dataloader) == (1, 2, 3, 4, 5)


@torchdata_available
@mark.parametrize(
    "datapipe_adapter_fn",
    (
        Shuffle(),
        {OBJECT_TARGET: "torchdata.dataloader2.adapter.Shuffle"},
    ),
)
def test_dataloader2_creator_datapipe_adapter_fn(
    datapipe: IterDataPipe, datapipe_adapter_fn: Adapter | dict
) -> None:
    dataloader = DataLoader2Creator(datapipe, datapipe_adapter_fn=datapipe_adapter_fn).create()
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, ShufflerIterDataPipe)
    assert len(dataloader.datapipe_adapter_fns) == 1
    assert isinstance(dataloader.datapipe_adapter_fns[0], Shuffle)
    assert dataloader.reading_service is None
    assert set(dataloader) == {1, 2, 3, 4, 5}


@torchdata_available
@mark.parametrize(
    "reading_service",
    (
        MultiProcessingReadingService(),
        {OBJECT_TARGET: "torchdata.dataloader2.MultiProcessingReadingService"},
    ),
)
def test_dataloader2_creator_reading_service(
    datapipe: IterDataPipe,
    reading_service: ReadingServiceInterface | dict,
) -> None:
    dataloader = DataLoader2Creator(datapipe, reading_service=reading_service).create()
    assert isinstance(dataloader, DataLoader2)
    assert isinstance(dataloader.datapipe, IterableWrapper)
    assert dataloader.datapipe_adapter_fns is None
    assert isinstance(dataloader.reading_service, MultiProcessingReadingService)
    assert tuple(dataloader) == (1, 2, 3, 4, 5)
