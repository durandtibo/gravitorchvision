import logging
from unittest.mock import Mock

from gravitorch.datasources import LoaderNotFoundError
from gravitorch.engines import BaseEngine
from gravitorch.experimental.dataflow import IterableDataFlow
from gravitorch.utils.asset import AssetNotFoundError
from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, fixture, raises

from gtvision.creators.dataflow import BaseDataFlowCreator, IterableDataFlowCreator
from gtvision.datasources import DataFlowDataSource

########################################
#     Tests for DataFlowDataSource     #
########################################


@fixture
def datasource() -> DataFlowDataSource:
    return DataFlowDataSource(
        {
            "train": {
                OBJECT_TARGET: "gtvision.creators.dataflow.IterableDataFlowCreator",
                "iterable": [1, 2, 3, 4],
            },
            "eval": IterableDataFlowCreator(["a", "b", "c"]),
        }
    )


def test_dataflow_datasource_str(datasource: DataFlowDataSource) -> None:
    assert str(datasource).startswith("DataFlowDataSource(")


def test_dataflow_datasource_attach(
    caplog: LogCaptureFixture, datasource: DataFlowDataSource
) -> None:
    with caplog.at_level(logging.INFO):
        datasource.attach(engine=Mock(spec=BaseEngine))
        assert len(caplog.messages) >= 1


def test_dataflow_datasource_get_asset_exists(datasource: DataFlowDataSource) -> None:
    datasource._asset_manager.add_asset("something", 2)
    assert datasource.get_asset("something") == 2


def test_dataflow_datasource_get_asset_does_not_exist(datasource: DataFlowDataSource) -> None:
    with raises(AssetNotFoundError, match="The asset 'something' does not exist"):
        datasource.get_asset("something")


def test_dataflow_datasource_has_asset_true(datasource: DataFlowDataSource) -> None:
    datasource._asset_manager.add_asset("something", 1)
    assert datasource.has_asset("something")


def test_dataflow_datasource_has_asset_false(datasource: DataFlowDataSource) -> None:
    assert not datasource.has_asset("something")


def test_dataflow_datasource_get_data_loader_train(datasource: DataFlowDataSource) -> None:
    dataflow = datasource.get_data_loader("train")
    assert isinstance(dataflow, IterableDataFlow)
    with dataflow as flow:
        assert tuple(flow) == (1, 2, 3, 4)


def test_dataflow_datasource_get_data_loader_eval(datasource: DataFlowDataSource) -> None:
    dataflow = datasource.get_data_loader("eval")
    assert isinstance(dataflow, IterableDataFlow)
    with dataflow as flow:
        assert tuple(flow) == ("a", "b", "c")


def test_dataflow_datasource_get_data_loader_missing(datasource: DataFlowDataSource) -> None:
    with raises(LoaderNotFoundError):
        datasource.get_data_loader("missing")


def test_dataflow_datasource_get_data_loader_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    dataflow_creator = Mock(spec=BaseDataFlowCreator, create=Mock(return_value=["a", "b", "c"]))
    datasource = DataFlowDataSource({"train": dataflow_creator})
    datasource.get_data_loader("train", engine=engine)
    dataflow_creator.create.assert_called_once_with(engine=engine)


def test_dataflow_datasource_get_data_loader_without_engine() -> None:
    dataflow_creator = Mock(spec=BaseDataFlowCreator, create=Mock(return_value=["a", "b", "c"]))
    datasource = DataFlowDataSource({"train": dataflow_creator})
    datasource.get_data_loader("train")
    dataflow_creator.create.assert_called_once_with(engine=None)


def test_dataflow_datasource_has_data_loader_true(datasource: DataFlowDataSource) -> None:
    assert datasource.has_data_loader("train")


def test_dataflow_datasource_has_data_loader_false(datasource: DataFlowDataSource) -> None:
    assert not datasource.has_data_loader("missing")


def test_dataflow_datasource_load_state_dict(datasource: DataFlowDataSource) -> None:
    datasource.load_state_dict({})


def test_dataflow_datasource_state_dict(datasource: DataFlowDataSource) -> None:
    assert datasource.state_dict() == {}
