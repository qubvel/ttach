import itertools
from functools import partial
from typing import List, Optional, Union

from . import functional as F


class BaseTransform:
    identity_param = None

    def __init__(
            self,
            name: str,
            params: Union[list, tuple],
    ):
        self.params = params
        self.pname = name

    def apply_aug_image(self, image, *args, **params):
        raise NotImplementedError

    def apply_deaug_mask(self, mask, *args, **params):
        raise NotImplementedError

    def apply_deaug_label(self, label, *args, **params):
        raise NotImplementedError

    def apply_deaug_keypoints(self, keypoints, *args, **params):
        raise NotImplementedError


class ImageOnlyTransform(BaseTransform):

    def apply_deaug_mask(self, mask, *args, **params):
        return mask

    def apply_deaug_label(self, label, *args, **params):
        return label

    def apply_deaug_keypoints(self, keypoints, *args, **params):
        return keypoints


class DualTransform(BaseTransform):
    pass


class Chain:

    def __init__(
            self,
            functions: List[callable]
    ):
        self.functions = functions or []

    def __call__(self, x):
        for f in self.functions:
            x = f(x)
        return x


class Transformer:
    def __init__(
            self,
            image_pipeline: Chain,
            mask_pipeline: Chain,
            label_pipeline: Chain,
            keypoints_pipeline: Chain
    ):
        self.image_pipeline = image_pipeline
        self.mask_pipeline = mask_pipeline
        self.label_pipeline = label_pipeline
        self.keypoints_pipeline = keypoints_pipeline

    def augment_image(self, image):
        return self.image_pipeline(image)

    def deaugment_mask(self, mask):
        return self.mask_pipeline(mask)

    def deaugment_label(self, label):
        return self.label_pipeline(label)

    def deaugment_keypoints(self, keypoints):
        return self.keypoints_pipeline(keypoints)


class Compose:

    def __init__(
            self,
            transforms: List[BaseTransform],
    ):
        self.aug_transforms = transforms
        self.aug_transform_parameters = list(itertools.product(*[t.params for t in self.aug_transforms]))
        self.deaug_transforms = transforms[::-1]
        self.deaug_transform_parameters = [p[::-1] for p in self.aug_transform_parameters]

    def __iter__(self) -> Transformer:
        for aug_params, deaug_params in zip(self.aug_transform_parameters, self.deaug_transform_parameters):
            image_aug_chain = Chain([partial(t.apply_aug_image, **{t.pname: p})
                                     for t, p in zip(self.aug_transforms, aug_params)])
            mask_deaug_chain = Chain([partial(t.apply_deaug_mask, **{t.pname: p})
                                      for t, p in zip(self.deaug_transforms, deaug_params)])
            label_deaug_chain = Chain([partial(t.apply_deaug_label, **{t.pname: p})
                                       for t, p in zip(self.deaug_transforms, deaug_params)])
            keypoints_deaug_chain = Chain([partial(t.apply_deaug_keypoints, **{t.pname: p})
                                           for t, p in zip(self.deaug_transforms, deaug_params)])
            yield Transformer(
                image_pipeline=image_aug_chain,
                mask_pipeline=mask_deaug_chain,
                label_pipeline=label_deaug_chain,
                keypoints_pipeline=keypoints_deaug_chain
            )

    def __len__(self) -> int:
        return len(self.aug_transform_parameters)


class Merger:

    def __init__(
            self,
            type: str = 'mean',
            n: int = 1,
    ):

        if type not in ['mean', 'gmean', 'sum', 'max', 'min', 'tsharpen']:
            raise ValueError('Not correct merge type `{}`.'.format(type))

        self.output = None
        self.type = type
        self.n = n

    def append(self, x):

        if self.type == 'tsharpen':
            x = x ** 0.5

        if self.output is None:
            self.output = x
        elif self.type in ['mean', 'sum', 'tsharpen']:
            self.output = self.output + x
        elif self.type == 'gmean':
            self.output = self.output * x
        elif self.type == 'max':
            self.output = F.max(self.output, x)
        elif self.type == 'min':
            self.output = F.min(self.output, x)

    @property
    def result(self):
        if self.type in ['sum', 'max', 'min']:
            result = self.output
        elif self.type in ['mean', 'tsharpen']:
            result = self.output / self.n
        elif self.type in ['gmean']:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError('Not correct merge type `{}`.'.format(self.type))
        return result
