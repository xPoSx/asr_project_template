import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {'text_encoded_length': [], 'spectrogram_length': []}
    for elem in dataset_items:
        # print(elem)
        result_batch['text_encoded_length'].append(elem['text_encoded'].shape[-1])
        result_batch['spectrogram_length'].append(elem['spectrogram'].shape[-1])
        for k, v in elem.items():
            if k in ['audio', 'spectrogram', 'text_encoded']:
                # print(v.shape)
                if k not in result_batch.keys():
                    # print('da')
                    result_batch[k] = v
                else:
                    # print('net')
                    diff = result_batch[k].shape[-1] - v.shape[-1]
                    if diff > 0:
                        result_batch[k] = torch.cat((result_batch[k], F.pad(v, (0, diff))))
                    else:
                        result_batch[k] = torch.cat((v, F.pad(result_batch[k], (0, -diff))))
            else:
                if k not in result_batch.keys():
                    result_batch[k] = []
                result_batch[k].append(v)
    result_batch['spectrogram'] = result_batch['spectrogram'].permute(0, 2, 1)
    result_batch['text_encoded_length'] = torch.tensor(result_batch['text_encoded_length'])
    result_batch['spectrogram_length'] = torch.tensor(result_batch['spectrogram_length'])
    return result_batch
