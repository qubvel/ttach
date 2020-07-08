import torch
import torch.nn.functional as F


def rot90(x, k=1):
    """rotate batch of images by 90 degrees k times"""
    return torch.rot90(x, k, (2, 3))


def hflip(x):
    """flip batch of images horizontally"""
    return x.flip(3)


def vflip(x):
    """flip batch of images vertically"""
    return x.flip(2)


def sum(x1, x2):
    """sum of two tensors"""
    return x1 + x2


def add(x, value):
    """add value to tensor"""
    return x + value


def max(x1, x2):
    """compare 2 tensors and take max values"""
    return torch.max(x1, x2)


def min(x1, x2):
    """compare 2 tensors and take min values"""
    return torch.min(x1, x2)


def multiply(x, factor):
    """multiply tensor by factor"""
    return x * factor


def scale(x, scale_factor, interpolation="nearest", align_corners=None):
    """scale batch of images by `scale_factor` with given interpolation mode"""
    h, w = x.shape[2:]
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    return F.interpolate(
        x, size=(new_h, new_w), mode=interpolation, align_corners=align_corners
    )


def resize(x, size, interpolation="nearest", align_corners=None):
    """resize batch of images to given spatial size with given interpolation mode"""
    return F.interpolate(x, size=size, mode=interpolation, align_corners=align_corners)


def crop(x, x_min=None, x_max=None, y_min=None, y_max=None):
    """perform crop on batch of images"""
    return x[:, :, y_min:y_max, x_min:x_max]


def crop_lt(x, crop_h, crop_w):
    """crop left top corner"""
    return x[:, :, 0:crop_h, 0:crop_w]


def crop_lb(x, crop_h, crop_w):
    """crop left bottom corner"""
    return x[:, :, -crop_h:, 0:crop_w]


def crop_rt(x, crop_h, crop_w):
    """crop right top corner"""
    return x[:, :, 0:crop_h, -crop_w:]


def crop_rb(x, crop_h, crop_w):
    """crop right bottom corner"""
    return x[:, :, -crop_h:, -crop_w:]


def center_crop(x, crop_h, crop_w):
    """make center crop"""

    center_h = x.shape[2] // 2
    center_w = x.shape[3] // 2
    half_crop_h = crop_h // 2
    half_crop_w = crop_w // 2

    y_min = center_h - half_crop_h
    y_max = center_h + half_crop_h + crop_h % 2
    x_min = center_w - half_crop_w
    x_max = center_w + half_crop_w + crop_w % 2

    return x[:, :, y_min:y_max, x_min:x_max]


def _disassemble_keypoints(keypoints):
    x = keypoints[..., 0]
    y = keypoints[..., 1]
    return x, y


def _assemble_keypoints(x, y):
    return torch.stack([x, y], dim=-1)


def keypoints_hflip(keypoints):
    x, y = _disassemble_keypoints(keypoints)
    return _assemble_keypoints(1. - x, y)


def keypoints_vflip(keypoints):
    x, y = _disassemble_keypoints(keypoints)
    return _assemble_keypoints(x, 1. - y)


def keypoints_rot90(keypoints, k=1):

    if k not in {0, 1, 2, 3}:
        raise ValueError("Parameter k must be in [0:3]")
    if k == 0:
        return keypoints
    x, y = _disassemble_keypoints(keypoints)

    if k == 1:
        xy = [y, 1. - x]
    elif k == 2:
        xy = [1. - x, 1. - y]
    elif k == 3:
        xy = [1. - y, x]

    return _assemble_keypoints(*xy)
