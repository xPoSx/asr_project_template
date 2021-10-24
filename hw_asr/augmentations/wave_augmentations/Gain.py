import torch_audiomentations
from torch import Tensor
import random

from hw_asr.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < 0.3:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        else:
            return data
