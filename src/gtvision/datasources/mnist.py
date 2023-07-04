r"""This module implements a data source for MNIST dataset."""

from __future__ import annotations

__all__ = ["MNISTDataSource"]

from pathlib import Path

from gravitorch import constants as ct
from gravitorch.creators.dataloader import BaseDataLoaderCreator
from gravitorch.datasources import DatasetDataSource

from gtvision.datasets import MNIST


class MNISTDataSource(DatasetDataSource):
    r"""Implements a data source for the MNIST dataset.

    Args:
    ----
        path (str, optional): Specifies the path where to save/load
            the MNIST data.
        dataloader_creators (dict): Specifies the dataloader
            creators to initialize. Each key indicates a dataloader
            creator name. The value can be a ``BaseDataLoaderCreator``
            object, or its configuration, or ``None``. ``None`` means
            a default dataloader will be created. Each dataloader
            creator takes a ``Dataset`` object as input, so you need
            to specify a dataset with the same name. The expected
            keys are ``'train'`` and ``'eval'``.
        download (bool, optional): If ``True``, downloads the
            dataset from the internet and puts it in root
            directory. If dataset is already downloaded, it is
            not downloaded again. Default: ``False``
    """

    def __init__(
        self,
        path: Path | str,
        dataloader_creators: dict[str, BaseDataLoaderCreator | dict | None],
        download: bool = False,
    ) -> None:
        super().__init__(
            datasets={
                ct.TRAIN: MNIST.create_with_default_transforms(path, train=True, download=download),
                ct.EVAL: MNIST.create_with_default_transforms(path, train=False, download=download),
            },
            dataloader_creators=dataloader_creators,
        )
