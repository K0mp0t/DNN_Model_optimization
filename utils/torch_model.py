from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_ksize=(2, 2)):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                   nn.LeakyReLU(0.1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.MaxPool2d(pool_ksize))

    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    def __init__(self, alphabet_len):
        super(CRNN, self).__init__()

        self.feature_extractor = nn.Sequential(ConvBlock(1, 32),
                                               ConvBlock(32, 64, (2, 1)),
                                               ConvBlock(64, 64),
                                               ConvBlock(64, 128),
                                               ConvBlock(128, 256, (2, 1)))
        self.lstm1 = nn.LSTM(258, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 256, batch_first=True)

        self.fc = nn.Sequential(nn.Linear(256, alphabet_len+1),
                                nn.Softmax(dim=2))

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1).squeeze(2)
        f1 = torch.permute(f1, (0, 2, 1))

        x = torch.cat([f1, x2], dim=2)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)

        return x
