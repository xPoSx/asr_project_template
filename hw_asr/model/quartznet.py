import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio

from hw_asr.base import BaseModel


class QuartzCell(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch, stride=1, dilation=1, activation=True):
        super(QuartzCell, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size, stride, dilation=dilation, groups=in_ch),
            nn.Conv1d(in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch)
        )
        self.activation = activation
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layers(x)
        if self.activation:
            x = self.relu(x)
        return x


class QuartzBlock(nn.Module):
    def __init__(self, n_cells, kernel_size, in_ch, out_ch):
        super(QuartzBlock, self).__init__()
        self.layers = nn.ModuleList([QuartzCell(kernel_size, in_ch, out_ch)])
        self.layers.extend([QuartzCell(kernel_size, out_ch, out_ch) for _ in range(n_cells - 2)])
        self.layers.append(QuartzCell(kernel_size, out_ch, out_ch, activation=False))
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x_skip = self.bn(self.conv(x))
        for layer in self.layers:
            x = layer(x)
        self.relu(x + x_skip)


class QuartzNet(BaseModel):
    def __init__(self, n_feats, n_class, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = nn.Sequential(
            QuartzCell(33, n_feats, 256, stride=2),
            QuartzBlock(5, 33, 256, 256),
            QuartzBlock(5, 39, 256, 256),
            QuartzBlock(5, 51, 256, 512),
            QuartzBlock(5, 63, 512, 512),
            QuartzBlock(5, 75, 512, 512),
            QuartzCell(87, 512, 512),
            QuartzCell(1, 512, 1024),
            QuartzCell(1, 1024, n_class, dilation=2)
        )

    def forward(self, spectrogram, *args, **kwargs):
        return self.net(spectrogram)

    def transform_input_lengths(self, input_lengths):
        return input_lengths
