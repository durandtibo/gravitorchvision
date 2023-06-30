from __future__ import annotations

from gravitorch.datapipes.iter import SourceWrapper
from objectory import OBJECT_TARGET
from pytest import mark, raises
from torch.utils.data.datapipes.iter import Batcher, Multiplexer

from gtvision.creators.datapipe import (
    SequentialDataPipeCreator,
    create_sequential_datapipe,
)

###############################################
#     Tests for SequentialDataPipeCreator     #
###############################################


def test_sequential_datapipe_creator_str() -> None:
    creator = SequentialDataPipeCreator(
        [
            {
                OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            }
        ]
    )
    assert str(creator).startswith("SequentialDataPipeCreator(")


def test_sequential_datapipe_creator_empty() -> None:
    with raises(
        ValueError, match="It is not possible to create a DataPipe because config is empty"
    ):
        SequentialDataPipeCreator([])


def test_sequential_datapipe_creator_dict() -> None:
    creator = SequentialDataPipeCreator(
        {
            OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
            "source": [1, 2, 3, 4],
        }
    )
    datapipe = creator.create()
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_sequential_datapipe_creator_dict_source_inputs() -> None:
    creator = SequentialDataPipeCreator(
        config={OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper"},
    )
    datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_sequential_datapipe_creator_dict_one_input_datapipe() -> None:
    creator = SequentialDataPipeCreator(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
    )
    datapipe = creator.create(source_inputs=[SourceWrapper([1, 2, 3, 4])])
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_sequential_datapipe_creator_dict_two_input_datapipes() -> None:
    creator = SequentialDataPipeCreator(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
    )
    datapipe = creator.create(
        source_inputs=[
            SourceWrapper([1, 2, 3, 4]),
            SourceWrapper([11, 12, 13, 14]),
        ]
    )
    assert isinstance(datapipe, Multiplexer)
    assert tuple(datapipe) == (1, 11, 2, 12, 3, 13, 4, 14)


def test_sequential_datapipe_creator_sequence_1() -> None:
    creator = SequentialDataPipeCreator(
        [
            {
                OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            }
        ]
    )
    datapipe = creator.create()
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_sequential_datapipe_creator_sequence_2() -> None:
    creator = SequentialDataPipeCreator(
        [
            {
                OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            },
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ]
    )
    datapipe = creator.create()
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_sequential_datapipe_creator_sequence_source_inputs() -> None:
    creator = SequentialDataPipeCreator(
        config=[
            {OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
    )
    datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_sequential_datapipe_creator_sequence_source_inputs_datapipe() -> None:
    creator = SequentialDataPipeCreator(
        config=[{OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2}],
    )
    datapipe = creator.create(source_inputs=[SourceWrapper([1, 2, 3, 4])])
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_sequential_datapipe_creator_sequence_multiple_input_datapipes() -> None:
    creator = SequentialDataPipeCreator(
        config=[
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
    )
    datapipe = creator.create(
        source_inputs=[
            SourceWrapper([1, 2, 3, 4]),
            SourceWrapper([11, 12, 13, 14]),
        ]
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 11], [2, 12], [3, 13], [4, 14])


################################################
#     Tests for create_sequential_datapipe     #
################################################


@mark.parametrize("config", (list(), tuple(), dict()))
def test_create_sequential_datapipe_empty(config: list | tuple | dict) -> None:
    with raises(
        RuntimeError, match="It is not possible to create a DataPipe because config is empty"
    ):
        create_sequential_datapipe(config)


def test_create_sequential_datapipe_dict() -> None:
    datapipe = create_sequential_datapipe(
        {
            OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
            "source": [1, 2, 3, 4],
        }
    )
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_sequential_datapipe_dict_source_inputs() -> None:
    datapipe = create_sequential_datapipe(
        config={OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper"},
        source_inputs=([1, 2, 3, 4],),
    )
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_sequential_datapipe_dict_one_input_datapipe() -> None:
    datapipe = create_sequential_datapipe(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        source_inputs=[SourceWrapper([1, 2, 3, 4])],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_sequential_datapipe_dict_two_input_datapipes() -> None:
    datapipe = create_sequential_datapipe(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
        source_inputs=[
            SourceWrapper([1, 2, 3, 4]),
            SourceWrapper([11, 12, 13, 14]),
        ],
    )
    assert isinstance(datapipe, Multiplexer)
    assert tuple(datapipe) == (1, 11, 2, 12, 3, 13, 4, 14)


def test_create_sequential_datapipe_sequence_1() -> None:
    datapipe = create_sequential_datapipe(
        [
            {
                OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            }
        ]
    )
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_sequential_datapipe_sequence_2() -> None:
    datapipe = create_sequential_datapipe(
        [
            {
                OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            },
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ]
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_sequential_datapipe_sequence_source_inputs() -> None:
    datapipe = create_sequential_datapipe(
        config=[
            {OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
        source_inputs=([1, 2, 3, 4],),
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_sequential_datapipe_sequence_source_inputs_datapipe() -> None:
    datapipe = create_sequential_datapipe(
        config=[{OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2}],
        source_inputs=[SourceWrapper([1, 2, 3, 4])],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_sequential_datapipe_sequence_multiple_input_datapipes() -> None:
    datapipe = create_sequential_datapipe(
        config=[
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
        source_inputs=[
            SourceWrapper([1, 2, 3, 4]),
            SourceWrapper([11, 12, 13, 14]),
        ],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 11], [2, 12], [3, 13], [4, 14])
