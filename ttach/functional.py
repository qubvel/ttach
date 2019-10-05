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


def _disassemble_bbox(bbox):
    x_min = bbox[..., 0]
    y_min = bbox[..., 1]
    x_max = bbox[..., 2]
    y_max = bbox[..., 3]
    return x_min, y_min, x_max, y_max


def _assemble_bbox(x_min, y_min, x_max, y_max):
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def bbox_hflip(bbox, image_height, image_width):
    """Flip a bounding box horizontally around the y-axis.

    Args:
        bbox: 3D (B, N, 4) or 2D (N, 4) tensor with box in format (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = _disassemble_bbox(bbox)
    return _assemble_bbox(
        image_width - x_max - 1, y_min, image_width - x_min - 1, y_max
    )


def bbox_vflip(bbox, image_height, image_width):
    """Flip a bounding box vertically around the x-axis.

    Args:
        bbox: 3D tensor (B, N, 4) with box in format (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = _disassemble_bbox(bbox)
    return _assemble_bbox(
        x_min, image_height - y_max - 1, x_max, image_height - y_min - 1
    )


def bbox_rot90(bbox, image_height, image_width, k=1):
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)
    Args:
        bbox (tuple): 3D tensor (B, N, 4) with box in format (x_min, y_min, x_max, y_max).
        image_height (int): height of image 
        k (int): Number of CCW rotations. Must be in range [0;3] See np.rot90.
    """
    if k == 0:
        return bbox

    if k < 0 or k > 3:
        raise ValueError("Parameter n must be in range [0;3]")

    x_min, y_min, x_max, y_max = _disassemble_bbox(bbox)

    if k == 1:
        bbox = [y_min, image_width - x_max - 1, y_max, image_width - x_min - 1]
    elif k == 2:
        bbox = [
            image_width - x_max - 1,
            image_height - y_max - 1,
            image_width - x_min - 1,
            image_height - y_min - 1,
        ]
    elif k == 3:
        bbox = [image_height - y_max - 1, x_min, image_height - y_min - 1, x_max]

    return _assemble_bbox(*bbox)


def bbox_scale(bbox, image_height, image_width, scale=1):
    """Scale bounding box"""
    x_min, y_min, x_max, y_max = _disassemble_bbox(bbox)
    return _assemble_bbox(
        int(x_min * scale), int(y_min * scale), int(x_max * scale), int(y_max * scale)
    )


def bbox_resize(bbox, image_height, image_width, new_image_size):
    """Resize bbox to new image size"""

    new_image_height = new_image_size[0]
    new_image_width = new_image_size[1]

    x_factor = new_image_width / image_width
    y_factor = new_image_height / image_height

    x_min, y_min, x_max, y_max = _disassemble_bbox(bbox)
    return _assemble_bbox(
        x_min * x_factor, y_min * y_factor, x_max * x_factor, y_max * y_factor
    )
