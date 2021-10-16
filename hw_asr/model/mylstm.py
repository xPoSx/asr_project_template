import torch.nn as nn
from hw_asr.base import BaseModel


class MyLSTM(BaseModel):
    def __init__(self, n_feats, n_class, num_layers, fc_hidden=128, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.net = nn.LSTM(n_feats, fc_hidden, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fc_hidden, n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        return {"logits": self.fc(self.net(spectrogram)[0])}

    def transform_input_lengths(self, input_lengths):
        return input_lengths
