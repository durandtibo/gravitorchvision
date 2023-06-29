from unittest.mock import Mock

import torch
from coola import objects_are_equal
from gravitorch.data.dataloaders.collators import PaddedSequenceCollator
from gravitorch.engines import BaseEngine
from objectory import OBJECT_TARGET
from pytest import fixture, mark
from torch import Tensor
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader, default_collate

from gtvision.creators.dataloader import DataLoaderCreator
from gtvision.creators.dataset import DatasetCreator


class FakeDataset(Dataset):
    def __len__(self) -> int:
        return 20

    def __getitem__(self, item: int) -> Tensor:
        return torch.ones(5).mul(item)


@fixture
def dataset() -> Dataset:
    return FakeDataset()


#######################################
#     Tests for DataLoaderCreator     #
#######################################


def test_dataloader_creator_str(dataset: Dataset) -> None:
    assert str(DataLoaderCreator(dataset)).startswith("DataLoaderCreator(")


def test_dataloader_creator_dataset(dataset: Dataset) -> None:
    dataloader = DataLoaderCreator(dataset, batch_size=8).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.dataset is dataset
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


def test_dataloader_creator_dataset_creator(dataset: Dataset) -> None:
    dataloader = DataLoaderCreator(DatasetCreator(dataset), batch_size=8).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.dataset is dataset
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


@mark.parametrize("batch_size", (1, 2, 4))
def test_dataloader_creator_batch_size(dataset: Dataset, batch_size: int) -> None:
    dataloader = DataLoaderCreator(dataset, batch_size=batch_size).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.batch_size == batch_size
    batch = next(iter(dataloader))
    assert torch.is_tensor(batch)
    assert batch.shape == (batch_size, 5)


def test_dataloader_creator_shuffle_false(dataset: Dataset) -> None:
    dataloader = DataLoaderCreator(dataset, batch_size=8, shuffle=False).create()
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.sampler, SequentialSampler)
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


def test_dataloader_creator_shuffle_true(dataset: Dataset) -> None:
    dataloader = DataLoaderCreator(dataset, batch_size=8, shuffle=True).create()
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.sampler, RandomSampler)
    batch = next(iter(dataloader))
    assert not batch.equal(
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


@mark.parametrize("num_workers", (0, 1, 2))
def test_dataloader_creator_num_workers(dataset: Dataset, num_workers: int) -> None:
    dataloader = DataLoaderCreator(dataset, num_workers=num_workers).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.num_workers == num_workers


def test_dataloader_creator_pin_memory_false(dataset: Dataset) -> None:
    dataloader = DataLoaderCreator(dataset, pin_memory=False).create()
    assert isinstance(dataloader, DataLoader)
    assert not dataloader.pin_memory


def test_dataloader_creator_pin_memory_true(dataset: Dataset) -> None:
    dataloader = DataLoaderCreator(dataset, pin_memory=True).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.pin_memory


def test_dataloader_creator_drop_last_false(dataset: Dataset) -> None:
    dataloader = DataLoaderCreator(dataset, drop_last=False).create()
    assert isinstance(dataloader, DataLoader)
    assert not dataloader.drop_last


def test_dataloader_creator_drop_last_true(dataset: Dataset) -> None:
    dataloader = DataLoaderCreator(dataset, drop_last=True).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.drop_last


def test_dataloader_creator_same_random_seed(dataset: Dataset) -> None:
    assert objects_are_equal(
        tuple(DataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create()),
        tuple(DataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create()),
    )


def test_dataloader_creator_different_random_seeds(dataset: Dataset) -> None:
    assert not objects_are_equal(
        tuple(DataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create()),
        tuple(DataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=2).create()),
    )


def test_dataloader_creator_same_random_seed_same_epoch(dataset: Dataset) -> None:
    assert objects_are_equal(
        tuple(
            DataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create(
                engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
        tuple(
            DataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create(
                engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
    )


def test_dataloader_creator_same_random_seed_different_epochs(dataset: Dataset) -> None:
    assert not objects_are_equal(
        tuple(
            DataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create(
                engine=Mock(spec=BaseEngine, epoch=0)
            )
        ),
        tuple(
            DataLoaderCreator(dataset, batch_size=8, shuffle=True, seed=1).create(
                engine=Mock(spec=BaseEngine, epoch=1)
            )
        ),
    )


def test_dataloader_creator_collate_fn(dataset: Dataset) -> None:
    dataloader = DataLoaderCreator(dataset, collate_fn=default_collate).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.collate_fn == default_collate


def test_dataloader_creator_collate_fn_none(dataset: Dataset) -> None:
    dataloader = DataLoaderCreator(dataset, collate_fn=None).create()
    assert isinstance(dataloader, DataLoader)
    assert dataloader.collate_fn == default_collate


def test_dataloader_creator_collate_fn_from_config(dataset: Dataset) -> None:
    dataloader = DataLoaderCreator(
        dataset,
        collate_fn={OBJECT_TARGET: "gravitorch.data.dataloaders.collators.PaddedSequenceCollator"},
    ).create()
    assert isinstance(dataloader, DataLoader)
    assert isinstance(dataloader.collate_fn, PaddedSequenceCollator)
