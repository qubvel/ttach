import torch
import pytest
import numpy as np
import ttach as tta


@pytest.mark.parametrize(
    "transform",
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
        tta.Scale(scales=[1, 2, 4], interpolation="nearest"),
        tta.Resize(
            sizes=[(4, 5), (8, 10)], original_size=(4, 5), interpolation="nearest"
        ),
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
        tta.Resize(sizes=[(4, 5), (8, 10), (2, 2)], interpolation="nearest"),
    ],
)
def test_label_is_same(transform):
    a = torch.arange(20).reshape(1, 1, 4, 5).float()
    for p in transform.params:
        aug = transform.apply_aug_image(a, **{transform.pname: p})
        deaug = transform.apply_deaug_label(aug, **{transform.pname: p})
        assert torch.allclose(aug, deaug)


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


@pytest.mark.parametrize(
    "transform",
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ],
)
def test_bbox_is_same(transform):
    image = torch.zeros([10, 12]).reshape(1, 1, 10, 12)
    image[:, :, 1:4, 2:7] = 1.0

    def to_bbox(x):
        np_x = x.numpy().squeeze()
        y_min = np.argwhere(np_x)[0][0]
        x_min = np.argwhere(np_x)[0][1]
        y_max = np.argwhere(np_x)[-1][0]
        x_max = np.argwhere(np_x)[-1][1]
        return torch.tensor([x_min, y_min, x_max, y_max]).reshape(1, 1, 4).float()

    gt_bbox = to_bbox(image).reshape(1, 1, 4)

    for i, p in enumerate(transform.params):
        aug_image = transform.apply_aug_image(image, **{transform.pname: p})
        h, w = aug_image.shape[2:4]
        aug_bbox = to_bbox(aug_image)
        deaug_bbox = transform.apply_deaug_bbox(aug_bbox, h, w, **{transform.pname: p})
        assert torch.allclose(deaug_bbox, gt_bbox)
