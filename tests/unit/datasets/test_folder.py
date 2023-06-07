from pathlib import Path

import numpy as np
from gravitorch import constants as ct
from PIL import Image
from pytest import TempPathFactory, fixture
from torchvision.transforms import ToTensor

from gtvision.datasets import ImageFolder

#################################
#     Tests for ImageFolder     #
#################################


def create_image_folder(path: Path) -> None:
    r"""Creates an image folder dataset with 2 classes: cat vs dog.

    Args:
    ----
        path (str): Specifies the path where to write the images of the dataset.
    """
    cat_path = path.joinpath("cat")
    cat_path.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng()
    for n in range(3):
        im_out = Image.fromarray(rng.uniform(0, 256, (16, 16, 3)).astype("uint8")).convert("RGB")
        im_out.save(cat_path.joinpath(f"out{n}.jpg"))
    dog_path = path.joinpath("dog")
    dog_path.mkdir(exist_ok=True, parents=True)
    for n in range(2):
        im_out = Image.fromarray(rng.uniform(0, 256, (16, 16, 3)).astype("uint8")).convert("RGB")
        im_out.save(dog_path.joinpath(f"out{n}.jpg"))


@fixture(scope="module")
def dataset_path(tmp_path_factory: TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    create_image_folder(path)
    return path


def test_image_folder(dataset_path: Path) -> None:
    dataset = ImageFolder(dataset_path.as_posix())
    assert len(dataset) == 5
    for i in range(3):
        assert len(dataset[i]) == 2
        assert ct.INPUT in dataset[i]
        assert dataset[i][ct.TARGET] == 0
    for i in range(3, 5):
        assert len(dataset[i]) == 2
        assert ct.INPUT in dataset[i]
        assert dataset[i][ct.TARGET] == 1


def test_image_folder_transform(dataset_path: Path) -> None:
    assert isinstance(
        ImageFolder(dataset_path.as_posix(), transform=ToTensor()).transform, ToTensor
    )


def test_image_folder_target_transform(dataset_path: Path) -> None:
    assert isinstance(
        ImageFolder(dataset_path.as_posix(), target_transform=ToTensor()).target_transform, ToTensor
    )
