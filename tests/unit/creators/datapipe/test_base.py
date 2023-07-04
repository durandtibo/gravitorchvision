from __future__ import annotations

from objectory import OBJECT_TARGET

from gtvision.creators.datapipe import ChainedDataPipeCreator, setup_datapipe_creator

############################################
#     Tests for setup_datapipe_creator     #
############################################


def test_setup_datapipe_creator_object() -> None:
    creator = ChainedDataPipeCreator(
        [
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4],
            }
        ]
    )
    assert setup_datapipe_creator(creator) is creator


def test_setup_datapipe_creator_dict() -> None:
    assert isinstance(
        setup_datapipe_creator(
            {
                OBJECT_TARGET: "gtvision.creators.datapipe.ChainedDataPipeCreator",
                "config": {
                    OBJECT_TARGET: "gtvision.creators.datapipe.IterableDataFlowCreator",
                    "iterable": (1, 2, 3, 4, 5),
                },
            },
        ),
        ChainedDataPipeCreator,
    )
