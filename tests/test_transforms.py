import pytest
import torch
import ttach as tta


@pytest.mark.parametrize(
    "transform",
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
        tta.Scale(scales=[1, 2, 4], interpolation="nearest"),
        tta.Resize(sizes=[(4, 5), (8, 10)], original_size=(4, 5), interpolation="nearest")
    ],
)
def test_aug_deaug_mask(transform):
    a = torch.arange(20).reshape(1, 1, 4, 5).float()
    for p in transform.params:
        aug = transform.apply_aug_image(a, **{transform.pname: p})
        deaug = transform.apply_deaug_mask(aug, **{transform.pname: p})
        assert torch.allclose(a, deaug)


@pytest.mark.parametrize(
    "transform",
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
        tta.Scale(scales=[1, 2, 4], interpolation="nearest"),
        tta.Add(values=[-1, 0, 1, 2]),
        tta.Multiply(factors=[-1, 0, 1, 2]),
        tta.FiveCrops(crop_height=3, crop_width=5),
        tta.Resize(sizes=[(4, 5), (8, 10), (2, 2)], interpolation="nearest")
    ],
)
def test_label_is_same(transform):
    a = torch.arange(20).reshape(1, 1, 4, 5).float()
    for p in transform.params:
        aug = transform.apply_aug_image(a, **{transform.pname: p})
        deaug = transform.apply_deaug_label(aug, **{transform.pname: p})
        assert torch.allclose(aug, deaug)


@pytest.mark.parametrize(
    "transform",
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip()
    ],
)
def test_flip_keypoints(transform):
    keypoints = torch.tensor([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9], [0.4, 0.3]])
    for p in transform.params:
        aug = transform.apply_deaug_keypoints(keypoints.detach().clone(), **{transform.pname: p})
        deaug = transform.apply_deaug_keypoints(aug, **{transform.pname: p})
        assert torch.allclose(keypoints, deaug)


@pytest.mark.parametrize(
    "transform",
    [
        tta.Rotate90(angles=[0, 90, 180, 270])
    ],
)
def test_rotate90_keypoints(transform):
    keypoints = torch.tensor([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9], [0.4, 0.3]])
    for p in transform.params:
        aug = transform.apply_deaug_keypoints(keypoints.detach().clone(), **{transform.pname: p})
        deaug = transform.apply_deaug_keypoints(aug, **{transform.pname: -p})
        assert torch.allclose(keypoints, deaug)


def test_add_transform():
    transform = tta.Add(values=[-1, 0, 1])
    a = torch.arange(20).reshape(1, 1, 4, 5).float()
    for p in transform.params:
        aug = transform.apply_aug_image(a, **{transform.pname: p})
        assert torch.allclose(aug, a + p)


def test_multiply_transform():
    transform = tta.Multiply(factors=[-1, 0, 1])
    a = torch.arange(20).reshape(1, 1, 4, 5).float()
    for p in transform.params:
        aug = transform.apply_aug_image(a, **{transform.pname: p})
        assert torch.allclose(aug, a * p)


def test_fivecrop_transform():
    transform = tta.FiveCrops(crop_height=1, crop_width=1)
    a = torch.arange(25).reshape(1, 1, 5, 5).float()
    output = [0, 20, 24, 4, 12]
    for i, p in enumerate(transform.params):
        aug = transform.apply_aug_image(a, **{transform.pname: p})
        assert aug.item() == output[i]

#
# def test_resize_transform():
#     transform = tta.Resize(sizes=[(10, 10), (5, 5)], original_size=(5, 5))
#     a = torch.arange(25).reshape(1, 1, 5, 5).float()
#     for i, p in enumerate(transform.params):
#         aug = transform.apply_aug_image(a, **{transform.pname: p})
#         assert aug.item() == output[i]