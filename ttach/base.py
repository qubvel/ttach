import itertools
from functools import partial
from typing import List, Union


class BaseTransform:
    identity_param = None

    def __init__(self, name: str, params: Union[list, tuple]):
        self.params = params
        self.pname = name

    def apply_aug_image(self, image, *args, **kwargs):
        raise NotImplementedError

    def apply_deaug_mask(self, mask, *args, **kwargs):
        raise NotImplementedError

    def apply_deaug_label(self, label, *args, **kwargs):
        raise NotImplementedError

    def apply_deaug_bbox(self, bbox, *args, **kwargs):
        raise NotImplementedError


class ImageOnlyTransform(BaseTransform):
    def apply_deaug_mask(self, mask, *args, **kwargs):
        return mask

    def apply_deaug_label(self, label, *args, **kwargs):
        return label

    def apply_deaug_bbox(self, bbox, *args, **kwargs):
        return bbox


class DualTransform(BaseTransform):
    pass


class Chain:
    def __init__(self, functions: List[callable]):
        self.functions = functions or []

    def __call__(self, x, *args, **kwargs):
        for f in self.functions:
            x = f(x, *args, **kwargs)
        return x


class Transformer:
    def __init__(
        self,
        image_pipeline: Chain,
        mask_pipeline: Chain,
        label_pipeline: Chain,
        bbox_pipeline: Chain,
    ):
        self.image_pipeline = image_pipeline
        self.mask_pipeline = mask_pipeline
        self.label_pipeline = label_pipeline
        self.bbox_pipeline = bbox_pipeline

    def augment_image(self, image, *args, **kwargs):
        return self.image_pipeline(image, *args, **kwargs)

    def deaugment_mask(self, mask, *args, **kwargs):
        return self.mask_pipeline(mask, *args, **kwargs)

    def deaugment_label(self, label, *args, **kwargs):
        return self.label_pipeline(label, *args, **kwargs)

    def deaugment_bbox(self, bbox, h, w, *args, **kwargs):
        return self.bbox_pipeline(bbox, h, w, *args, **kwargs)


class Compose:
    def __init__(self, transforms: List[BaseTransform]):
        self.aug_transforms = transforms
        self.aug_transform_parameters = list(
            itertools.product(*[t.params for t in self.aug_transforms])
        )
        self.deaug_transforms = transforms[::-1]
        self.deaug_transform_parameters = [
            p[::-1] for p in self.aug_transform_parameters
        ]

    def __iter__(self) -> Transformer:
        for aug_params, deaug_params in zip(
            self.aug_transform_parameters, self.deaug_transform_parameters
        ):
            image_aug_chain = Chain(
                [
                    partial(t.apply_aug_image, **{t.pname: p})
                    for t, p in zip(self.aug_transforms, aug_params)
                ]
            )
            mask_deaug_chain = Chain(
                [
                    partial(t.apply_deaug_mask, **{t.pname: p})
                    for t, p in zip(self.deaug_transforms, deaug_params)
                ]
            )
            label_deaug_chain = Chain(
                [
                    partial(t.apply_deaug_label, **{t.pname: p})
                    for t, p in zip(self.deaug_transforms, deaug_params)
                ]
            )
            bbox_deaug_chain = Chain(
                [
                    partial(t.apply_deaug_bbox, **{t.pname: p})
                    for t, p in zip(self.deaug_transforms, deaug_params)
                ]
            )

            yield Transformer(
                image_pipeline=image_aug_chain,
                mask_pipeline=mask_deaug_chain,
                label_pipeline=label_deaug_chain,
                bbox_pipeline=bbox_deaug_chain,
            )

    def __len__(self) -> int:
        return len(self.aug_transform_parameters)
