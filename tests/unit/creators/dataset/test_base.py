from gravitorch.data.datasets import ExampleDataset
from objectory import OBJECT_TARGET

from gtvision.creators.dataset import DatasetCreator, setup_dataset_creator

###########################################
#     Tests for setup_dataset_creator     #
###########################################


def test_setup_dataset_creator_object() -> None:
    creator = DatasetCreator(ExampleDataset((1, 2, 3, 4, 5)))
    assert setup_dataset_creator(creator) is creator


def test_setup_dataset_creator_dict() -> None:
    assert isinstance(
        setup_dataset_creator(
            {
                OBJECT_TARGET: "gtvision.creators.dataset.DatasetCreator",
                "dataset": {
                    OBJECT_TARGET: "gravitorch.data.datasets.ExampleDataset",
                    "examples": (1, 2, 3, 4, 5),
                },
            },
        ),
        DatasetCreator,
    )
