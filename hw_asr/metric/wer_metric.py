from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1)
        predictions = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(predictions, kwargs['log_probs_length'])
        ]
        for log_prob_vec, target_text in zip(predictions, text):
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec)
            else:
                pred_text = self.text_encoder.decode(log_prob_vec)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], *args, **kwargs):
        wers = []
        preds = []
        for i in range(log_probs.size(0)):
            tmp_probs = log_probs[i][:int(kwargs['log_probs_length'][i])]
            tmp_probs = tmp_probs.reshape(1, tmp_probs.size(0), tmp_probs.size(1))
            preds.append(tmp_probs)

        for log_prob_vec, target_text in zip(preds, text):
            if hasattr(self.text_encoder, "ctc_beam_search"):
                pred_text = self.text_encoder.ctc_beam_search(log_prob_vec)
            else:
                pred_text = self.text_encoder.decode(log_prob_vec)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
