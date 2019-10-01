import torch
import torch.nn.functional as F


def rot90(x, k=1):
    return torch.rot90(x, k, (2, 3))


def hflip(x):
    return x.flip(3)


def vflip(x):
    return x.flip(2)


def sum(x1, x2):
    return x1 + x2


def add(x, value):
    return x + value


def max(x1, x2):
    return torch.max(x1, x2)


def min(x1, x2):
    return torch.min(x1, x2)


def multiply(x, factor):
    return x * factor


def scale(x, scale_factor, interpolation="nearest", align_corners=None):
    h, w = x.shape[2:]
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    return F.interpolate(
        x, size=(new_h, new_w), mode=interpolation, align_corners=align_corners
    )


def crop(x, x_min=None, x_max=None, y_min=None, y_max=None):
    return x[:, :, y_min:y_max, x_min:x_max]


def crop_lt(x, crop_h, crop_w):
    return x[:, :, 0:crop_h, 0:crop_w]


def crop_lb(x, crop_h, crop_w):
    return x[:, :, -crop_h:, 0:crop_w]


def crop_rt(x, crop_h, crop_w):
    return x[:, :, 0:crop_h, -crop_w:]


def crop_rb(x, crop_h, crop_w):
    return x[:, :, -crop_h:, -crop_w:]


def center_crop(x, crop_h, crop_w):
    # assert crop_h % 2 == 0
    # assert crop_w % 2 == 0

    center_h = x.shape[2] // 2
    center_w = x.shape[3] // 2
    half_crop_h = crop_h // 2
    half_crop_w = crop_w // 2

    y_min = center_h - half_crop_h
    y_max = center_h + half_crop_h + crop_h % 2
    x_min = center_w - half_crop_w
    x_max = center_w + half_crop_w + crop_w % 2

    return x[:, :, y_min:y_max, x_min:x_max]
