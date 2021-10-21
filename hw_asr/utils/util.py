import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import pandas as pd
import torch
import gzip
import os, shutil, wget

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader

def init_lm():
    lm_gzip_path = '3-gram.pruned.1e-7.arpa.gz'
    if not os.path.exists(lm_gzip_path):
        print('Downloading pruned 3-gram model.')
        lm_url = 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'
        lm_gzip_path = wget.download(lm_url)
        print('Downloaded the 3-gram language model.')
    else:
        print('Pruned .arpa.gz already exists.')
    #
    # lm_vocab_path = 'librispeech-vocab.txt'
    # if not os.path.exists(lm_vocab_path):
    #     print('Downloading librspeech vocabulary')
    #     vocab_url = 'https://www.openslr.org/resources/11/librispeech-vocab.txt'
    #     lm_vocab_path = wget.download(vocab_url)
    #     print('Downloaded librispeech vocabulary')
    # else:
    #     print('Librispeech vocabulary already exists')

    uppercase_lm_path = '3-gram.pruned.1e-7.arpa'
    if not os.path.exists(uppercase_lm_path):
        with gzip.open(lm_gzip_path, 'rb') as f_zipped:
            with open(uppercase_lm_path, 'wb') as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)
        print('Unzipped the 3-gram language model.')
    else:
        print('Unzipped .arpa already exists.')

    lm_path = 'lowercase_3-gram.pruned.1e-7.arpa'
    if not os.path.exists(lm_path):
        with open(uppercase_lm_path, 'r') as f_upper:
            with open(lm_path, 'w') as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())
    print('Converted language model file to lowercase.')

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()
