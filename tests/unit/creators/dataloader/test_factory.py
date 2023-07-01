from gravitorch.data.datasets import ExampleDataset
from objectory import OBJECT_TARGET
from pytest import fixture
from torch.utils.data import Dataset

from gtvision.creators.dataloader import DataLoaderCreator, setup_data_loader_creator


@fixture
def dataset() -> Dataset:
    return ExampleDataset((1, 2, 3, 4, 5))


###############################################
#     Tests for setup_data_loader_creator     #
###############################################


def test_setup_data_loader_creator_object(dataset: Dataset) -> None:
    data_loader_creator = DataLoaderCreator(dataset)
    assert setup_data_loader_creator(data_loader_creator) is data_loader_creator


def test_setup_data_loader_creator_config(dataset: Dataset) -> None:
    data_loader_creator = setup_data_loader_creator(
        {OBJECT_TARGET: "gtvision.creators.dataloader.DataLoaderCreator", "dataset": dataset}
    )
    assert isinstance(data_loader_creator, DataLoaderCreator)
