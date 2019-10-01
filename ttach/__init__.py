from .wrappers import (
    SegmentationTTAWrapper,
    ClassificationTTAWrapper,
)
from .base import Compose

from .transforms import (
    HorizontalFlip, VerticalFlip, Rotate90, Scale, Add, Multiply, FiveCrops
)

from . import aliases
