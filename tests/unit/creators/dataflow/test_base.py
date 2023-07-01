from objectory import OBJECT_TARGET

from gtvision.creators.dataflow import IterableDataFlowCreator, setup_dataflow_creator

############################################
#     Tests for setup_dataflow_creator     #
############################################


def test_setup_dataflow_creator_object() -> None:
    creator = IterableDataFlowCreator((1, 2, 3, 4, 5))
    assert setup_dataflow_creator(creator) is creator


def test_setup_dataflow_creator_dict() -> None:
    assert isinstance(
        setup_dataflow_creator(
            {
                OBJECT_TARGET: "gtvision.creators.dataflow.IterableDataFlowCreator",
                "iterable": (1, 2, 3, 4, 5),
            }
        ),
        IterableDataFlowCreator,
    )
