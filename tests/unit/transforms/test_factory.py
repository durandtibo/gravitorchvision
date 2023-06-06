from objectory import OBJECT_TARGET
from torchvision import transforms

from gtvision.transforms import create_compose

##########################
#     create_compose     #
##########################


def test_create_compose_1() -> None:
    transform = create_compose([{OBJECT_TARGET: "torchvision.transforms.CenterCrop", "size": 10}])
    assert isinstance(transform, transforms.Compose)
    assert len(transform.transforms) == 1
    assert isinstance(transform.transforms[0], transforms.CenterCrop)


def test_create_compose_2() -> None:
    transform = create_compose(
        [
            {OBJECT_TARGET: "torchvision.transforms.CenterCrop", "size": 10},
            transforms.PILToTensor(),
        ]
    )
    assert isinstance(transform, transforms.Compose)
    assert len(transform.transforms) == 2
    assert isinstance(transform.transforms[0], transforms.CenterCrop)
    assert isinstance(transform.transforms[1], transforms.PILToTensor)
