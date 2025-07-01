import random
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sophia import SophiaG
from src.mSSRM_PGA import mSSRM_PGA


def loss_fn(pred: torch.Tensor, target: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred[:, ~missing_mask], target[:, ~missing_mask])

def max_sharpe_port(returns, device):
    seq_len, dim = returns.shape
    log_weight = nn.Parameter(torch.zeros(dim, device = device))
    optimizer = SophiaG([log_weight], lr=5e-1)

    mean = returns.mean(0)
    cov = returns.T.cov()

    prev_weight = torch.zeros(dim, device = device).fill_(torch.inf)
    prev_sharpe = 0.0
    for _ in range(1000):
        weight = log_weight.softmax(0)
        if torch.allclose(weight, prev_weight):
            return weight.data

        if not torch.isfinite(weight).all():
            return torch.empty(dim, device = device).fill_(1 / dim)

        prev_weight = weight.clone()

        optimizer.zero_grad()
        mu = (mean * weight).sum()
        sigma = (weight.reshape(1, -1) @ cov @ weight).sqrt()
        loss = - mu * seq_len + prev_sharpe * sigma * np.sqrt(seq_len)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            backtest = (returns.exp() * weight).sum(-1).log()
            prev_sharpe = (backtest.mean(0) * seq_len) / (backtest.var() * seq_len).sqrt()

    return log_weight.data.softmax(0)

def max_sortino_port(returns, device):
    seq_len, dim = returns.shape
    log_weight = nn.Parameter(torch.zeros(dim, device = device))
    optimizer = SophiaG([log_weight], lr=5e-1)

    mean = returns.mean(0)
    cov = returns.clip(max=0).T.cov()

    prev_weight = torch.zeros(dim, device = device).fill_(torch.inf)
    prev_sharpe = 0.0
    for _ in range(1000):
        weight = log_weight.softmax(0)
        if torch.allclose(weight, prev_weight):
            return weight.data

        if not torch.isfinite(weight).all():
            return torch.empty(dim, device = device).fill_(1 / dim)

        prev_weight = weight.clone()

        optimizer.zero_grad()
        mu = (mean * weight).sum()
        sigma = (weight.reshape(1, -1) @ cov @ weight).sqrt()
        loss = - mu * seq_len + prev_sharpe * sigma * np.sqrt(seq_len)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            backtest = (returns.exp() * weight).sum(-1).log()
            prev_sharpe = (backtest.mean(0) * seq_len) / (backtest.var() * seq_len).sqrt()

    return log_weight.data.softmax(0)

def train(
    yearmons,
    backtest_start,
    backtest_end,
    date2ind,
    yearmon2indices,
    data,
    torch_data,
    torch_target_data,
    missing_mask,
    portfolio,
    model_cls,
    optimizer_cls,
    #loss_fn,
    device: torch.device,
    window_size = 0,
    num_repeat = 100,
    num_epoch = 100,
    lr = 1e-2,
    stop_count = 10,
    trading_cost = 0.004,
    lagging = 2):
    
    input_size = torch_data.shape[1]
    num_item = len(data['log_tr'].columns)
    num_features = input_size // num_item
    
    first_date = data.index.min()
    last_date = data.index.max()
    
    all_backtest = pd.DataFrame(index=data[backtest_start:backtest_end].index)
    all_weights = []
    all_logits = []

    for rep in range(num_repeat):
        random.seed(rep)
        np.random.seed(rep)
        torch.manual_seed(rep)
    
        model = model_cls(input_size = input_size, num_output = num_item).to(device)
    
        optimizer = optimizer_cls(model.parameters(), lr = lr)
    
        backtest = pd.DataFrame(0, index=data[backtest_start:backtest_end].index, columns=['backtest'])
        weights = pd.DataFrame(0, index=yearmons, columns=data['log_tr'].columns).loc[backtest_start:backtest_end]

        train_historical_missing_mask = missing_mask[yearmons[1]].clone()
        valid_historical_missing_mask = missing_mask[yearmons[2]].clone()
        test_historical_missing_mask = missing_mask[yearmons[3]].clone()
    
        for ind in (pbar := tqdm.tqdm(range(4, len(yearmons)))):
            train_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-3]].index]]
            train_target_data = torch_target_data[[date2ind[x] for x in data.loc[yearmons[ind-2]:].index]]
            valid_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-2]].index]]
            valid_target_data = torch_target_data[[date2ind[x] for x in data.loc[yearmons[ind-1]:].index]]
            test_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-1]].index]]
            test_target_data = torch_target_data[[date2ind[x] for x in data.loc[yearmons[ind]:].index]]

            train_sortino = 0.0
            valid_sortino = 0.0
            best_valid_loss = torch.inf
            count = 0
            for epoch in range(num_epoch):
                model.train()
                
                yearmon = yearmons[ind-2]
                optimizer.zero_grad(set_to_none=True)
                train_historical_missing_mask = train_historical_missing_mask & missing_mask[yearmon]
                x = train_data[-window_size:].clone()
                x[:, train_historical_missing_mask.repeat(num_features)] = 0
                output = model(x)
                train_loss = loss_fn(
                    output,
                    train_target_data[:window_size],
                    missing_mask[yearmon])
                train_loss.backward()
                optimizer.step()
        
                model.eval()
                with torch.no_grad():
                    yearmon = yearmons[ind-1]
                    valid_historical_missing_mask = valid_historical_missing_mask & missing_mask[yearmon]
                    x = valid_data[-window_size:].clone()
                    x[:, valid_historical_missing_mask.repeat(num_features)] = 0
                    output = model(x)
                    valid_loss = loss_fn(
                        output,
                        valid_target_data[:window_size],
                        missing_mask[yearmon])
        
                    if best_valid_loss > valid_loss:
                        best_valid_loss = valid_loss
                        best_state_dict = model.state_dict()
                        count = 0
                    else:
                        count += 1

                    if count == stop_count:
                        break
        
            model.load_state_dict(best_state_dict)
            model.eval()
    
            yearmon = yearmons[ind]
            with torch.no_grad():
                test_historical_missing_mask = test_historical_missing_mask & missing_mask[yearmon]
                x = test_data[-window_size:].clone()
                x[:, test_historical_missing_mask.repeat(num_features)] = 0
                output = model(x)

            predict_start, predict_end = yearmon2indices[yearmon]

            if yearmon < backtest_start:
                test_loss = loss_fn(
                    output,
                    test_target_data[:window_size],
                    missing_mask[yearmon])
                pbar.set_description(f"Test loss: {test_loss:.6f}")

            else:
                mask = missing_mask[yearmon]
                cur_returns = torch.where(
                    mask.expand(predict_end - predict_start + 1, -1),
                    0,
                    torch_target_data[predict_start + lagging:predict_end + 1 + lagging]).exp()

                weight = torch.zeros(mask.shape, device = device)
                weight[~mask] = max_sortino_port(output[:, ~mask], device) if portfolio == 'maxsortino' else max_sharpe_port(output[:, ~mask], device)
                weights.loc[yearmon] = weight.cpu().numpy()

                if yearmon == backtest_start:
                    cur_returns[0] -= torch.where(
                        missing_mask[yearmon],
                        0,
                        weight * trading_cost)
                else:
                    cur_returns[0] -= torch.where(
                        missing_mask[yearmon],
                        0,
                        (weight - prev_weight).abs() * trading_cost)

                backtest_path = weight.reshape(1, -1).expand(predict_end - predict_start + 1, -1) * cur_returns
                backtest_path = backtest_path.sum(-1).log()
                backtest.loc[yearmon, 'backtest'] = backtest_path.cpu().numpy()

                prev_weight = weight * cur_returns.prod(0)
                prev_weight /= prev_weight.sum(-1)

                test_return = backtest.loc[:yearmon].apply(np.exp).prod().iloc[0]
                test_sharpe = (backtest.loc[:yearmon].mean() * np.sqrt(252) / backtest.loc[:yearmon].std()).iloc[0]
                pbar.set_description(f"Test Return: {test_return:.6f}, Test Sharpe: {test_sharpe:.6f}")
    
        backtest = backtest.shift(lagging).fillna(0)
        
        all_backtest = pd.concat((all_backtest, backtest), axis=1)
        all_weights.append(weights)

    return all_backtest, all_weights