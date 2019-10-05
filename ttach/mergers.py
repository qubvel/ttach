from . import functional as F
import torchvision

class Merger:
    def __init__(self, type: str = "mean", n: int = 1):

        if type not in ["mean", "gmean", "sum", "max", "min", "tsharpen"]:
            raise ValueError("Not correct merge type `{}`.".format(type))

        self.output = None
        self.type = type
        self.n = n

    def append(self, x):

        if self.type == "tsharpen":
            x = x ** 0.5

        if self.output is None:
            self.output = x
        elif self.type in ["mean", "sum", "tsharpen"]:
            self.output = self.output + x
        elif self.type == "gmean":
            self.output = self.output * x
        elif self.type == "max":
            self.output = F.max(self.output, x)
        elif self.type == "min":
            self.output = F.min(self.output, x)

    @property
    def result(self):
        if self.type in ["sum", "max", "min"]:
            result = self.output
        elif self.type in ["mean", "tsharpen"]:
            result = self.output / self.n
        elif self.type in ["gmean"]:
            result = self.output ** (1 / self.n)
        else:
            raise ValueError("Not correct merge type `{}`.".format(self.type))
        return result

class NMSMerger:

    def __init__(self, tta_iou_threshold=0.9):
        self.tta_iou_threshold = tta_iou_threshold
        self._bboxes = []
        self._scores = []

    def append(self, boxes, scores):
        self._bboxes.append(boxes)
        self._scores.append(scores)

    @property
    def result(self):

        if not self._bboxes:
            return [], []

        # boxes = torch.concatenate()

        return 

