import torch.nn as nn
import torch
import torch.nn.functional as F
from hw_asr.base import BaseModel


class MaskedConv(nn.Module):
    def __init__(self, sequential: nn.Sequential) -> None:
        super(MaskedConv, self).__init__()
        self.sequential = sequential

    def forward(self, inputs, lengths):
        output = None
        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0).cuda()
            lengths = self.get_lengths(module, lengths)
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(dim=2, start=length, length=mask[i].size(2) - length).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, lengths

    def get_lengths(self, module, lengths):
        if isinstance(module, nn.Conv2d):
            res = (lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) // \
                  module.stride[1] + 1
            return res
        return lengths


class RnnCell(nn.Module):
    def __init__(self, n_feats, hidden_size, dropout):
        super(RnnCell, self).__init__()
        self.rnn = nn.RNN(n_feats, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(n_feats)

    def forward(self, inputs, inputs_length):
        total_length = inputs.size(0)
        inputs = F.relu(self.bn(inputs.transpose(1, 2)))
        output = inputs.transpose(1, 2)
        output = nn.utils.rnn.pack_padded_sequence(output, inputs_length.cpu(), enforce_sorted=False)
        output, _ = self.rnn(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=total_length)

        return output


class DeepSpeech(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size=512, num_layers=5, dropout=0.1, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.conv = MaskedConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        self.layers = nn.ModuleList()
        n_feats = (n_feats - 1) // 2 + 1
        n_feats = ((n_feats - 1) // 2 + 1) * 32
        rnn_out = hidden_size * 2
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
        inputs = spectrogram.unsqueeze(1).transpose(2, 3)
        outputs, output_lengths = self.conv(inputs, spectrogram_length)
        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2).view(batch_size, seq_lengths, channels * dimension)
        outputs = outputs.permute(1, 0, 2).contiguous()
        for layer in self.layers:
            outputs = layer(outputs, output_lengths)
        outputs = self.fc(outputs.transpose(0, 1)).log_softmax(dim=-1)
        return outputs

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2

