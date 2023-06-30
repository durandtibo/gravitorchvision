from __future__ import annotations

from objectory import OBJECT_TARGET
from pytest import mark, raises
from torch.utils.data.datapipes.iter import Batcher, IterableWrapper, Multiplexer

from gtvision.creators.datapipe import ChainedDataPipeCreator, create_chained_datapipe

############################################
#     Tests for ChainedDataPipeCreator     #
############################################


def test_chained_datapipe_creator_str() -> None:
    creator = ChainedDataPipeCreator(
        [
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4],
            }
        ]
    )
    assert str(creator).startswith("ChainedDataPipeCreator(")


def test_chained_datapipe_creator_empty() -> None:
    with raises(
        ValueError, match="It is not possible to create a DataPipe because the config is empty"
    ):
        ChainedDataPipeCreator([])


def test_chained_datapipe_creator_dict() -> None:
    creator = ChainedDataPipeCreator(
        {
            OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
            "iterable": [1, 2, 3, 4],
        }
    )
    datapipe = creator.create()
    assert isinstance(datapipe, IterableWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_chained_datapipe_creator_dict_source_inputs() -> None:
    creator = ChainedDataPipeCreator(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper"},
    )
    datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
    assert isinstance(datapipe, IterableWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_chained_datapipe_creator_dict_one_input_datapipe() -> None:
    creator = ChainedDataPipeCreator(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
    )
    datapipe = creator.create(source_inputs=[IterableWrapper([1, 2, 3, 4])])
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_chained_datapipe_creator_dict_two_input_datapipes() -> None:
    creator = ChainedDataPipeCreator(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
    )
    datapipe = creator.create(
        source_inputs=[
            IterableWrapper([1, 2, 3, 4]),
            IterableWrapper([11, 12, 13, 14]),
        ]
    )
    assert isinstance(datapipe, Multiplexer)
    assert tuple(datapipe) == (1, 11, 2, 12, 3, 13, 4, 14)


def test_chained_datapipe_creator_sequence_1() -> None:
    creator = ChainedDataPipeCreator(
        [
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4],
            }
        ]
    )
    datapipe = creator.create()
    assert isinstance(datapipe, IterableWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_chained_datapipe_creator_sequence_2() -> None:
    creator = ChainedDataPipeCreator(
        [
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4],
            },
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ]
    )
    datapipe = creator.create()
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_chained_datapipe_creator_sequence_source_inputs() -> None:
    creator = ChainedDataPipeCreator(
        config=[
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
    )
    datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_chained_datapipe_creator_sequence_source_inputs_datapipe() -> None:
    creator = ChainedDataPipeCreator(
        config=[{OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2}],
    )
    datapipe = creator.create(source_inputs=[IterableWrapper([1, 2, 3, 4])])
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_chained_datapipe_creator_sequence_multiple_input_datapipes() -> None:
    creator = ChainedDataPipeCreator(
        config=[
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
    )
    datapipe = creator.create(
        source_inputs=[
            IterableWrapper([1, 2, 3, 4]),
            IterableWrapper([11, 12, 13, 14]),
        ]
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 11], [2, 12], [3, 13], [4, 14])


#############################################
#     Tests for create_chained_datapipe     #
#############################################


@mark.parametrize("config", (list(), tuple(), dict()))
def test_create_chained_datapipe_empty(config: list | tuple | dict) -> None:
    with raises(
        RuntimeError, match="It is not possible to create a DataPipe because the config is empty"
    ):
        create_chained_datapipe(config)


def test_create_chained_datapipe_dict() -> None:
    datapipe = create_chained_datapipe(
        {
            OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
            "iterable": [1, 2, 3, 4],
        }
    )
    assert isinstance(datapipe, IterableWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_chained_datapipe_dict_source_inputs() -> None:
    datapipe = create_chained_datapipe(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper"},
        source_inputs=([1, 2, 3, 4],),
    )
    assert isinstance(datapipe, IterableWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_chained_datapipe_dict_one_input_datapipe() -> None:
    datapipe = create_chained_datapipe(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        source_inputs=[IterableWrapper([1, 2, 3, 4])],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_chained_datapipe_dict_two_input_datapipes() -> None:
    datapipe = create_chained_datapipe(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
        source_inputs=[
            IterableWrapper([1, 2, 3, 4]),
            IterableWrapper([11, 12, 13, 14]),
        ],
    )
    assert isinstance(datapipe, Multiplexer)
    assert tuple(datapipe) == (1, 11, 2, 12, 3, 13, 4, 14)


def test_create_chained_datapipe_sequence_1() -> None:
    datapipe = create_chained_datapipe(
        [
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4],
            }
        ]
    )
    assert isinstance(datapipe, IterableWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_chained_datapipe_sequence_2() -> None:
    datapipe = create_chained_datapipe(
        [
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4],
            },
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ]
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_chained_datapipe_sequence_source_inputs() -> None:
    datapipe = create_chained_datapipe(
        config=[
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
        source_inputs=([1, 2, 3, 4],),
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_chained_datapipe_sequence_source_inputs_datapipe() -> None:
    datapipe = create_chained_datapipe(
        config=[{OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2}],
        source_inputs=[IterableWrapper([1, 2, 3, 4])],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_chained_datapipe_sequence_multiple_input_datapipes() -> None:
    datapipe = create_chained_datapipe(
        config=[
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
        source_inputs=[
            IterableWrapper([1, 2, 3, 4]),
            IterableWrapper([11, 12, 13, 14]),
        ],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 11], [2, 12], [3, 13], [4, 14])
