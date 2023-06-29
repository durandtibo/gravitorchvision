from gravitorch.data.datasets import ExampleDataset
from gravitorch.experimental.dataflow import DataLoaderDataFlow
from torch.utils.data import DataLoader

from gtvision.creators.dataflow import DataLoaderDataFlowCreator
from gtvision.creators.dataloader import DataLoaderCreator

###############################################
#     Tests for DataLoaderDataFlowCreator     #
###############################################


def test_dataloader_dataflow_creator_str() -> None:
    assert str(DataLoaderDataFlowCreator(DataLoader(ExampleDataset((1, 2, 3, 4, 5))))).startswith(
        "DataLoaderDataFlowCreator("
    )


def test_dataloader_dataflow_creator_create_dataloader() -> None:
    dataflow = DataLoaderDataFlowCreator(DataLoader(ExampleDataset((1, 2, 3, 4, 5)))).create()
    assert isinstance(dataflow, DataLoaderDataFlow)
    assert list(dataflow) == [1, 2, 3, 4, 5]


def test_dataloader_dataflow_creator_create_dataloader_creator() -> None:
    dataflow = DataLoaderDataFlowCreator(
        DataLoaderCreator(ExampleDataset((1, 2, 3, 4, 5)))
    ).create()
    assert isinstance(dataflow, DataLoaderDataFlow)
    assert list(dataflow) == [1, 2, 3, 4, 5]
