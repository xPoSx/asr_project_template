from torchaudio import transforms
from torch import Tensor
import numpy as np

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = transforms.FrequencyMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        p = np.random.binomial(1, 0.5)
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1) if p else data
