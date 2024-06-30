import torch
import torch.nn as nn


class adaIN(nn.Module):

    def __init__(self, eps=1e-5):
        super(adaIN, self).__init__()
        self.eps = eps

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        out = out_in
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out


class ResnetAdaINBlock(nn.Module):

    def __init__(self, dim):
        super(ResnetAdaINBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = adaIN()
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = adaIN()

    def forward(self, x, gamma, beta):
        out = self.conv1(x)
        out = self.norm1(x, gamma, beta)
        out = self.relu1(x)
        out = self.conv2(x)
        out = self.norm2(x, gamma, beta)
        return x+out
