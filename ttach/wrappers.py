import torch
import torch.nn as nn
from typing import Optional, Mapping, Union, Tuple

from .base import Merger, Compose


class SegmentationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (segmentation model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): segmentation model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `mask`
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: Compose,
        merge_mode: str = "mean",
        output_mask_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_mask_key

    def forward(
        self, image: torch.Tensor, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        merger = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_mask(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result


class ClassificationTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (classification model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): classification model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_mask_key (str): if model output is `dict`, specify which key belong to `label`
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: Compose,
        merge_mode: str = "mean",
        output_label_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_label_key

    def forward(
        self, image: torch.Tensor, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        merger = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_label(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result


class KeypointsTTAWrapper(nn.Module):
    """Wrap PyTorch nn.Module (keypoints model) with test time augmentation transforms

    Args:
        model (torch.nn.Module): classification model with single input and single output
            (.forward(x) should return either torch.Tensor or Mapping[str, torch.Tensor])
        transforms (ttach.Compose): composition of test time transforms
        merge_mode (str): method to merge augmented predictions mean/gmean/max/min/sum/tsharpen
        output_keypoints_key (str): if model output is `dict`, specify which key belong to `label`
        scaled (bool): True if model return x, y scaled values in [0, 1], else False
        original_size(tuple[int, int]): optional, need to rescale keypoints to absolute values if scaled=True

    """

    def __init__(
        self,
        model: nn.Module,
        transforms: Compose,
        merge_mode: str = "mean",
        output_keypoints_key: Optional[str] = None,
        scaled: bool = False,
        original_size: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_keypoints_key
        self.scaled = scaled
        self.original_size = original_size

        if scaled and original_size is None:
            raise ValueError("For scaled mode need set original_size.")

    def forward(
        self, image: torch.Tensor, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        merger = Merger(type=self.merge_mode, n=len(self.transforms))
        batch_size = image.size()[0]

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)

            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]

            augmented_output = augmented_output.reshape(batch_size, -1, 2)
            if self.scaled:
                augmented_output[..., 0] *= self.original_size[1]
                augmented_output[..., 1] *= self.original_size[0]

            deaugmented_output = transformer.deaugment_keypoints(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.scaled:
            result[..., 0] /= self.original_size[1]
            result[..., 1] /= self.original_size[0]
        result = result.reshape(batch_size, -1)

        if self.output_key is not None:
            result = {self.output_key: result}

        return result
