import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from hw_asr.base import BaseModel


class RnnCell(nn.Module):
    def __init__(self, n_feats, hidden_size, dropout):
        super(RnnCell, self).__init__()
        self.rnn = nn.RNN(n_feats, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(n_feats)

    def forward(self, inputs):
        total_length = inputs.size(0)
        inputs = F.relu(self.bn(inputs.permute(0, 2, 1)))
        output = inputs.permute(0, 2, 1)
        output, _ = self.rnn(output)

        return output


class DeepSpeech(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size=512, num_layers=5, dropout=0.1, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        self.layers = nn.ModuleList()
        n_feats = int(math.floor(n_feats + 2 * 20 - 41) / 2 + 1)
        n_feats = int(math.floor(n_feats + 2 * 10 - 21) / 2 + 1)
        n_feats <<= 5
        rnn_out = hidden_size << 1
        for i in range(num_layers):
            self.layers.append(RnnCell(
                n_feats=n_feats if i == 0 else rnn_out,
                hidden_size=hidden_size,
                dropout=dropout
            ))
        self.fc = nn.Sequential(
            nn.Linear(rnn_out, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_class, bias=False)
        )

    def forward(self, spectrogram, spectrogram_length, *args, **kwargs):
        inputs = spectrogram.unsqueeze(1).permute(0, 1, 3, 2)
        outputs = self.conv(inputs)
        batch_size, num_channels, hidden_dim, seq_length = outputs.size()
        outputs = outputs.view(batch_size, num_channels * hidden_dim, seq_length).permute(2, 0, 1).contiguous()

        for layer in self.layers:
            outputs = layer(outputs)

        outputs = outputs.transpose(0, 1)
        outputs = self.fc(outputs)
        outputs = F.log_softmax(outputs, dim=-1)

        return outputs

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2

