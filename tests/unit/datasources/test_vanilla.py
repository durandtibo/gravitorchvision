import logging
from unittest.mock import Mock

from gravitorch.datasources import LoaderNotFoundError
from gravitorch.engines import BaseEngine
from gravitorch.experimental.dataflow import IterableDataFlow
from gravitorch.utils.asset import AssetNotFoundError
from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, fixture, raises

from gtvision.creators.dataflow import BaseDataFlowCreator, IterableDataFlowCreator
from gtvision.datasources import VanillaDataSource

#######################################
#     Tests for VanillaDataSource     #
#######################################


@fixture
def datasource() -> VanillaDataSource:
    return VanillaDataSource(
        {
            "train": {
                OBJECT_TARGET: "gtvision.creators.dataflow.IterableDataFlowCreator",
                "iterable": [1, 2, 3, 4],
            },
            "eval": IterableDataFlowCreator(["a", "b", "c"]),
        }
    )


def test_vanilla_data_source_str(datasource: VanillaDataSource) -> None:
    assert str(datasource).startswith("VanillaDataSource(")


def test_vanilla_data_source_attach(
    caplog: LogCaptureFixture, datasource: VanillaDataSource
) -> None:
    with caplog.at_level(logging.INFO):
        datasource.attach(engine=Mock(spec=BaseEngine))
        assert len(caplog.messages) >= 1


def test_vanilla_data_source_get_asset_exists(datasource: VanillaDataSource) -> None:
    datasource._asset_manager.add_asset("something", 2)
    assert datasource.get_asset("something") == 2


def test_vanilla_data_source_get_asset_does_not_exist(datasource: VanillaDataSource) -> None:
    with raises(AssetNotFoundError, match="The asset 'something' does not exist"):
        datasource.get_asset("something")


def test_vanilla_data_source_has_asset_true(datasource: VanillaDataSource) -> None:
    datasource._asset_manager.add_asset("something", 1)
    assert datasource.has_asset("something")


def test_vanilla_data_source_has_asset_false(datasource: VanillaDataSource) -> None:
    assert not datasource.has_asset("something")


def test_vanilla_data_source_get_data_loader_train(datasource: VanillaDataSource) -> None:
    dataflow = datasource.get_data_loader("train")
    assert isinstance(dataflow, IterableDataFlow)
    with dataflow as flow:
        assert tuple(flow) == (1, 2, 3, 4)


def test_vanilla_data_source_get_data_loader_eval(datasource: VanillaDataSource) -> None:
    dataflow = datasource.get_data_loader("eval")
    assert isinstance(dataflow, IterableDataFlow)
    with dataflow as flow:
        assert tuple(flow) == ("a", "b", "c")


def test_vanilla_data_source_get_data_loader_missing(datasource: VanillaDataSource) -> None:
    with raises(LoaderNotFoundError):
        datasource.get_data_loader("missing")


def test_vanilla_data_source_get_data_loader_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    dataflow_creator = Mock(spec=BaseDataFlowCreator, create=Mock(return_value=["a", "b", "c"]))
    datasource = VanillaDataSource({"train": dataflow_creator})
    datasource.get_data_loader("train", engine=engine)
    dataflow_creator.create.assert_called_once_with(engine=engine)


def test_vanilla_data_source_get_data_loader_without_engine() -> None:
    dataflow_creator = Mock(spec=BaseDataFlowCreator, create=Mock(return_value=["a", "b", "c"]))
    datasource = VanillaDataSource({"train": dataflow_creator})
    datasource.get_data_loader("train")
    dataflow_creator.create.assert_called_once_with(engine=None)


def test_vanilla_data_source_has_data_loader_true(datasource: VanillaDataSource) -> None:
    assert datasource.has_data_loader("train")


def test_vanilla_data_source_has_data_loader_false(datasource: VanillaDataSource) -> None:
    assert not datasource.has_data_loader("missing")


def test_vanilla_data_source_load_state_dict(datasource: VanillaDataSource) -> None:
    datasource.load_state_dict({})


def test_vanilla_data_source_state_dict(datasource: VanillaDataSource) -> None:
    assert datasource.state_dict() == {}
