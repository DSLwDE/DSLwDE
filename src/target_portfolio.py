from typing import List, Dict
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sophia import SophiaG


class OneHotWeightCollection:
    def __init__(self, yearmons: List[str], num_item: int, device: torch.device):
        super().__init__()
        self.device = device

        self.yearmons = yearmons
        self.num_item = num_item
        self.weight = {yearmon:torch.zeros((self.num_item), dtype = torch.float, device = self.device) for yearmon in yearmons}

    def calculate_one_hot_weights(self, data: pd.DataFrame, missing_mask: Dict[str, torch.Tensor]):
        for yearmon in tqdm.tqdm(self.yearmons):
            argmax = np.argmax(data['log_tr'].loc[yearmon].sum(0))
            self.weight[yearmon][argmax] = 1

    def get_weights(self):
        return {yearmon:self.weight[yearmon].cpu() for yearmon in self.yearmons}

class MaxSortinoWeightCollection(nn.Module):
    def __init__(self, yearmons: List[str], num_item: int, device: torch.device):
        super().__init__()
        self.device = device

        self.yearmons = yearmons
        self.num_item = num_item
        self.log_weight = nn.ParameterDict(
            {yearmon:torch.zeros((self.num_item), dtype = torch.float, device = self.device)
             for yearmon in yearmons}
        ) 

    def objective(self, yearmon:str, mean: torch.Tensor, cov: torch.Tensor, target_data: torch.Tensor, mask: torch.Tensor, prev_sortino: torch.float):
        weight = self.log_weight[yearmon][~mask].softmax(dim = 0)
        target_data = target_data[:, ~mask]
        seq_len = target_data.shape[0]

        if not torch.isfinite(weight).all():
            raise ValueError(f"Invalid weight at index {yearmon}")

        mu = (mean * weight).sum()
        sigma = (weight.reshape(1, -1) @ cov @ weight).sqrt()
        loss = - mu * seq_len + prev_sortino * sigma * np.sqrt(seq_len)
        
        with torch.no_grad():
            backtest = (target_data.exp() * weight).sum(-1).log()
            sortino = (backtest.mean(0) * seq_len) / (backtest.var() * seq_len).sqrt()
            entropy = -(weight * weight.log()).sum(-1)

        return loss, sortino, entropy

    def train_max_sortino_weights(self, data: pd.DataFrame, missing_mask: Dict[str, torch.Tensor], num_epoch: int = 1000):
        self.missing_mask = missing_mask

        for yearmon in (pbar := tqdm.tqdm(self.yearmons)):
            target_data = torch.tensor(data['log_tr'].loc[yearmon].values, dtype = torch.float, device = self.device)

            mean = target_data.mean(0)
            cov = target_data.clip(max=0).mT.cov()

            for lr in [5e-1, 1e-1, 5e-2]:
                try:
                    self.log_weight[yearmon].data.fill_(0)
                    optimizer = SophiaG([self.log_weight[yearmon]], lr = lr)

                    sortino = 0.0
                    for epoch in range(num_epoch):
                        optimizer.zero_grad(set_to_none = True)
                        loss, sortino, entropy = self.objective(yearmon, mean, cov, target_data, missing_mask[yearmon], sortino)
                        loss.backward()
                        optimizer.step()

                    pbar.set_description(f"Sortino = {sortino:.6f}, Entropy = {entropy:.6f}")

                    break
                except Exception as e:
                    print(e)
                    continue

                raise(f'failed portfolio optimization at {date}')

    def get_weights(self):
        return {yearmon:torch.where(self.missing_mask[yearmon], -np.inf, self.log_weight[yearmon].data).softmax(-1).cpu() for yearmon in self.yearmons}
