from gravitorch.data.datasets import ExampleDataset
from objectory import OBJECT_TARGET
from pytest import fixture
from torch.utils.data import Dataset

from gtvision.creators.dataloader import (
    VanillaDataLoaderCreator,
    setup_dataloader_creator,
)


@fixture
def dataset() -> Dataset:
    return ExampleDataset((1, 2, 3, 4, 5))


###############################################
#     Tests for setup_dataloader_creator     #
###############################################


def test_setup_dataloader_creator_object(dataset: Dataset) -> None:
    creator = VanillaDataLoaderCreator(dataset)
    assert setup_dataloader_creator(creator) is creator


def test_setup_dataloader_creator_config(dataset: Dataset) -> None:
    assert isinstance(
        setup_dataloader_creator(
            {
                OBJECT_TARGET: "gtvision.creators.dataloader.VanillaDataLoaderCreator",
                "dataset": dataset,
            }
        ),
        VanillaDataLoaderCreator,
    )
