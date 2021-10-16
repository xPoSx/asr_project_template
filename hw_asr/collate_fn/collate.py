import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {'text_encoded_length': [], 'spectrogram_length': [], 'spectrogram': [], 'text_encoded': [],
                    'text': []}
    for elem in dataset_items:
        # print(elem)
        result_batch['text_encoded_length'].append(elem['text_encoded'].shape[-1])
        result_batch['spectrogram_length'].append(elem['spectrogram'].shape[-1])
        result_batch['text'].append(elem['text'])
    text_res_shape = max(result_batch['text_encoded_length'])
    spec_res_shape = max(result_batch['spectrogram_length'])

    for elem in dataset_items:
        result_batch['spectrogram'].append(
            F.pad(elem['spectrogram'], (0, spec_res_shape - elem['spectrogram'].shape[-1])))
        result_batch['text_encoded'].append(
            F.pad(elem['text_encoded'], (0, text_res_shape - elem['text_encoded'].shape[-1])))

    result_batch['spectrogram'] = torch.cat(result_batch['spectrogram']).permute(0, 2, 1)
    result_batch['text_encoded'] = torch.cat(result_batch['text_encoded'])
    result_batch['text_encoded_length'] = torch.tensor(result_batch['text_encoded_length'])
    result_batch['spectrogram_length'] = torch.tensor(result_batch['spectrogram_length'])
    return result_batch
