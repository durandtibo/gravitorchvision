from __future__ import annotations

from gravitorch.experimental.dataflow import IterableDataFlow
from objectory import OBJECT_TARGET

from gtvision.creators.dataflow import IterableDataFlowCreator


def create_list() -> list:
    return [1, 2, 3, 4, 5]


#############################################
#     Tests for IterableDataFlowCreator     #
#############################################


def test_iterable_dataflow_creator_str() -> None:
    assert str(IterableDataFlowCreator((1, 2, 3, 4, 5))).startswith("IterableDataFlowCreator(")


def test_iterable_dataflow_creator_create() -> None:
    dataflow = IterableDataFlowCreator((1, 2, 3, 4, 5)).create()
    assert isinstance(dataflow, IterableDataFlow)
    assert list(dataflow) == [1, 2, 3, 4, 5]


def test_iterable_dataflow_creator_create_cache() -> None:
    creator = IterableDataFlowCreator(
        {OBJECT_TARGET: "unit.creators.dataflow.test_iterable.create_list"}, cache=True
    )
    assert isinstance(creator._iterable, dict)
    dataflow = creator.create()
    assert creator._iterable == [1, 2, 3, 4, 5]
    assert isinstance(dataflow, IterableDataFlow)
    assert list(dataflow) == [1, 2, 3, 4, 5]


def test_iterable_dataflow_creator_create_deepcopy() -> None:
    dataflow = IterableDataFlowCreator((1, 2, 3, 4, 5), deepcopy=True).create()
    assert isinstance(dataflow, IterableDataFlow)
    assert dataflow._deepcopy
    assert list(dataflow) == [1, 2, 3, 4, 5]
