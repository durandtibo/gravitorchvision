from unittest.mock import Mock

import torch
from coola import objects_are_equal
from gravitorch.engines import BaseEngine
from objectory import OBJECT_TARGET
from pytest import fixture
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from gtvision.creators.dataloader import VanillaDataLoaderCreator


class FakeDataset(Dataset):
    def __len__(self) -> int:
        return 20

    def __getitem__(self, item: int) -> Tensor:
        return torch.ones(5).mul(item)


@fixture
def dataset() -> Dataset:
    return FakeDataset()


##############################################
#     Tests for VanillaDataLoaderCreator     #
##############################################


def test_vanilla_dataloader_creator_str() -> None:
    assert str(VanillaDataLoaderCreator(Mock(spec=DataLoader))).startswith(
        "VanillaDataLoaderCreator("
    )


def test_vanilla_dataloader_creator_create_dataset_object(dataset: Dataset) -> None:
    dl = DataLoader(dataset, batch_size=8)
    dataloader = VanillaDataLoaderCreator(dl).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader is dl
    batch = next(iter(dataloader))
    assert batch.equal(
        torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7],
            ],
            dtype=torch.float,
        )
    )


def test_vanilla_dataloader_creator_create_dataset_config(dataset: Dataset) -> None:
    dataloader = VanillaDataLoaderCreator(
        {OBJECT_TARGET: "torch.utils.data.DataLoader", "dataset": dataset, "batch_size": 8},
    ).create()
    assert isinstance(dataloader, DataLoader)
    batch = next(iter(dataloader))
    assert batch.equal(
        torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7],
            ],
            dtype=torch.float,
        )
    )


def test_vanilla_dataloader_creator_create_dataset_cache_true(dataset: Dataset) -> None:
    creator = VanillaDataLoaderCreator(
        {OBJECT_TARGET: "torch.utils.data.DataLoader", "dataset": dataset, "batch_size": 8},
    )
    dataloader1 = creator.create()
    dataloader2 = creator.create()
    assert isinstance(dataloader1, DataLoader)
    assert isinstance(dataloader2, DataLoader)
    assert dataloader1 is dataloader2
    assert objects_are_equal(tuple(dataloader1), tuple(dataloader2))


def test_vanilla_dataloader_creator_create_dataloader_cache_false(dataset: Dataset) -> None:
    creator = VanillaDataLoaderCreator(
        {OBJECT_TARGET: "torch.utils.data.DataLoader", "dataset": dataset, "batch_size": 8},
        cache=False,
    )
    dataloader1 = creator.create()
    dataloader2 = creator.create()
    assert isinstance(dataloader1, DataLoader)
    assert isinstance(dataloader2, DataLoader)
    assert dataloader1 is not dataloader2
    assert objects_are_equal(tuple(dataloader1), tuple(dataloader2))


def test_vanilla_dataloader_creator_create_with_engine(dataset: Dataset) -> None:
    dataloader = VanillaDataLoaderCreator(DataLoader(dataset, batch_size=8)).create(
        engine=Mock(spec=BaseEngine)
    )
    assert isinstance(dataloader, DataLoader)
    batch = next(iter(dataloader))
    assert batch.equal(
        torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7],
            ],
            dtype=torch.float,
        )
    )
