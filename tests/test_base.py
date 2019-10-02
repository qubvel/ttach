import pytest
import torch
import ttach as tta


def test_compose_1():
    transform = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
            tta.Scale(scales=[1, 2, 4], interpolation="nearest"),
        ]
    )

    assert len(transform) == 2 * 2 * 4 * 3  # all combinations for aug parameters

    dummy_label = torch.ones(2).reshape(2, 1).float()
    dummy_image = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()
    dummy_model = lambda x: {"label": dummy_label, "mask": x}

    for augmenter in transform:
        augmented_image = augmenter.augment_image(dummy_image)
        model_output = dummy_model(augmented_image)
        deaugmented_mask = augmenter.deaugment_mask(model_output["mask"])
        deaugmented_label = augmenter.deaugment_label(model_output["label"])
        assert torch.allclose(deaugmented_mask, dummy_image)
        assert torch.allclose(deaugmented_label, dummy_label)


@pytest.mark.parametrize(
    "case",
    [
        ("mean", 0.5),
        ("gmean", 0.0),
        ("max", 1.0),
        ("min", 0.0),
        ("sum", 1.5),
        ("tsharpen", 0.56903558),
    ],
)
def test_merger(case):
    merge_type, output = case
    input = [1.0, 0.0, 0.5]
    merger = tta.base.Merger(type=merge_type, n=len(input))
    for i in input:
        merger.append(torch.tensor(i))
    assert torch.allclose(merger.result, torch.tensor(output))
