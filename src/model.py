import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


def generate_square_subsequent_mask(sz: int, device):
    return (torch.triu(torch.ones((sz, sz), dtype = torch.bool, device = device)) != 1).transpose(0, 1)

# LSTM model. Applied skip connection
class lstm(nn.Module):
    def __init__(self, input_size: int, num_output: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, input_size, num_layers=num_layers)
        self.linear = nn.Linear(input_size, num_output, bias = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_0, c_0 = torch.zeros((self.num_layers, self.input_size), device = x.device), torch.zeros((self.num_layers, self.input_size), device = x.device)
        x_0 = x
        x, (_, _) = self.rnn(x, (h_0, c_0))
        return self.linear(F.silu(x) + x_0)

class trf(nn.Module):
    def __init__(self, input_size: int, num_output: int, max_len: int = 252, nhead: int = 1, dim_feedforward: int = 256, num_layers: int = 1):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, input_size)
        layer = nn.TransformerEncoderLayer(d_model = input_size, nhead = nhead, dim_feedforward = dim_feedforward, activation = F.silu)
        self.trf = nn.TransformerEncoder(layer, num_layers = num_layers)
        self.linear = nn.Linear(input_size, num_output, bias = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positional_embedding(torch.arange(x.shape[0], device = x.device).flip(0))
        x = F.silu(self.trf(x, mask = generate_square_subsequent_mask(x.shape[0], x.device), is_causal=True)) + x
        return self.linear(x)

# Mamba Model. Applied skip connection
class mamba(nn.Module):
    def __init__(self, input_size: int, num_output: int):
        super().__init__()
        self.mamba = Mamba(d_model=input_size)
        self.linear = nn.Linear(in_features=input_size, out_features=num_output, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.mamba(x.view(1, -1, x.shape[1]))[0]) + x
        x = self.linear(x)
        return x
