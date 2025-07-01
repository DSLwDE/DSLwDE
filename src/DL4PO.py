from typing import List, Tuple, Dict

import tqdm
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sophia import SophiaG
from .utils import get_yearmons

def preprocess_data(file_name: str, start_date: str, device: torch.device) -> Tuple[List[str], pd.DataFrame, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    data = pd.read_csv(file_name)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['ticker', 'date'])
    data = data.set_index(['ticker', 'date'])
    
    # 데이터 전처리
    data['return'] = np.log(data.groupby('ticker')['div_adj_close'].pct_change() + 1)
    data = data.drop(['open', 'low', 'high', 'close', 'volume'], axis = 1)
    data = data.unstack(level = 0)
    data = data[(~data['return'].isna()).sum(1) > 1]
    data = data[(~(data['return'].isna() | (data['return'] == 0))).sum(1) > 0]
    data = data.loc[data.index.intersection(pd.bdate_range(data.index.min(), data.index.max()))]
    data = data.loc[start_date:]

    torch_data = torch.tensor(data.values, dtype = torch.float, device = device).nan_to_num(0)
    torch_target_data = torch_data[:, -data['return'].columns.__len__():]

    yearmons = get_yearmons(data)

    missing_mask = {yearmon:torch.tensor(data['return'].loc[yearmon].iloc[0].isna().values, dtype = torch.bool, device = device)
                    for yearmon in yearmons}

    return yearmons, data.fillna(0), torch_data, torch_target_data, missing_mask


def preprocess_data_rolling_univ(file_name: str, start_date: str, device: torch.device) -> Tuple[List[str], pd.DataFrame, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    data = pd.read_csv(file_name)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['tradingitemid', 'date'])
    data = data.set_index(['tradingitemid', 'date'])
    
    # 데이터 전처리
    data['return'] = np.log(data.groupby('tradingitemid')['div_adj_close'].pct_change() + 1)
    data = data.drop(['open', 'low', 'high', 'close', 'volume'], axis = 1)
    data = data.unstack(level = 0)
    data = data[(~data['return'].isna()).sum(1) > 1]
    data = data[(~(data['return'].isna() | (data['return'] == 0))).sum(1) > 0]
    data = data.loc[data.index.intersection(pd.bdate_range(data.index.min(), data.index.max()))]
    data = data.loc[start_date:]

    data_yearmons = get_yearmons(data)

    torch_data = torch.tensor(data.values, dtype = torch.float, device = device).nan_to_num(0)
    torch_target_data = torch_data[:, -data['return'].columns.__len__():]

    member_data = pd.read_csv(file_name.replace('.', '_member.'))
    member_data['date'] = pd.to_datetime(member_data['date'])
    member_data = member_data.set_index('date')
    member_data = member_data.loc[pd.bdate_range(member_data.index.min(), member_data.index.max())]

    member_yearmons = get_yearmons(member_data)

    yearmons = member_yearmons if data_yearmons[0] < member_yearmons[0] else data_yearmons

    df = pd.DataFrame(True, index = yearmons, columns = data['return'].columns)
    for yearmon in yearmons:
        df.loc[yearmon, list(set(df.columns).intersection(set(member_data.loc[member_data.loc[yearmon].index.min(), 'tradingitemid'].values)))] = False

    missing_mask = {yearmon:torch.tensor(df.loc[yearmon].values, dtype = torch.bool, device = device)
                    for yearmon in yearmons}

    return yearmons, data.fillna(0), torch_data, torch_target_data, missing_mask

def max_sharpe_loss_fn(pred: torch.Tensor, target_returns: torch.Tensor, missing_mask: torch.Tensor):
    R = (pred[..., ~missing_mask] * target_returns[..., ~missing_mask]).sum(-1)
    R_mean = R.mean()
    R_std = R.std()
    return R_mean / R_std.clamp(min = 1e-8)

def max_sortino_loss_fn(pred: torch.Tensor, target_returns: torch.Tensor, missing_mask: torch.Tensor):
    R = (pred[..., ~missing_mask] * target_returns[..., ~missing_mask]).sum(-1)
    R_mean = R.mean()
    R_semistd = R.clamp(max = 0).std()
    return R_mean / R_semistd.clamp(min = 1e-8)

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
    model_cls,
    optimizer_cls,
    device: torch.device,
    objective = 'sharpe',
    window_size = 21,
    num_repeat = 100,
    num_epoch = 100,
    lr = 3e-4,
    stop_count = 10,
    trading_cost = 0.004,
    lagging = 2):

    if objective == 'maxsortino':
        loss_fn = max_sortino_loss_fn
    else:
        loss_fn = max_sharpe_loss_fn
    
    input_size = torch_data.shape[1]
    num_item = len(data['div_adj_close'].columns)
    num_features = input_size // num_item
    
    first_date = data.index.min()
    last_date = data.index.max()
    
    all_backtest = pd.DataFrame(index=data[backtest_start:backtest_end].index)
    all_weights = []
    
    for rep in range(num_repeat):
        random.seed(rep)
        np.random.seed(rep)
        torch.manual_seed(rep)
    
        model = model_cls(input_size = input_size, num_output = num_item).to(device)
    
        optimizer = optimizer_cls(model.parameters(), lr = lr)
    
        backtest = pd.DataFrame(0, index=data[backtest_start:backtest_end].index, columns=['backtest'])
        weights = pd.DataFrame(0, index=yearmons, columns=data['div_adj_close'].columns).loc[backtest_start:backtest_end]

        train_historical_missing_mask = missing_mask[yearmons[1]].clone()
        valid_historical_missing_mask = missing_mask[yearmons[2]].clone()
        test_historical_missing_mask = missing_mask[yearmons[3]].clone()
    
        for ind in (pbar := tqdm.tqdm(range(5, len(yearmons)))):
            train_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-4]].index]]
            train_target_data = torch_target_data[[date2ind[x] for x in data.loc[:yearmons[ind-4]].index]]
            valid_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-3]].index]]
            valid_target_data = torch_target_data[[date2ind[x] for x in data.loc[:yearmons[ind-3]].index]]
            test_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-2]].index]]
            test_target_data = torch_target_data[[date2ind[x] for x in data.loc[:yearmons[ind-2]].index]]
    
            train_sortino = 0.0
            valid_sortino = 0.0
            best_valid_loss = torch.inf
            count = 0
            for epoch in range(num_epoch):
                yearmon = yearmons[ind-3]
                cur_start, cur_end = yearmon2indices[yearmon]
    
                model.train()
                optimizer.zero_grad(set_to_none=True)
                train_historical_missing_mask = train_historical_missing_mask & missing_mask[yearmon]
                x = train_data[-window_size:].clone()
                x[:, train_historical_missing_mask.repeat(num_features)] = 0
                output = model(x)
                train_loss = loss_fn(
                    output,
                    train_target_data[-window_size:],
                    missing_mask[yearmon])
                train_loss.backward()
                optimizer.step()
        
                model.eval()
                with torch.no_grad():
                    yearmon = yearmons[ind-2]
                    cur_start, cur_end = yearmon2indices[yearmon]

                    valid_historical_missing_mask = valid_historical_missing_mask & missing_mask[yearmon]
                    x = valid_data[-window_size:].clone()
                    x[:, valid_historical_missing_mask.repeat(num_features)] = 0
                    output = model(x)
                    output = output[-1]
                    valid_loss = loss_fn(
                        output,
                        valid_target_data[-window_size:],
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
    
            with torch.no_grad():
                yearmon = yearmons[ind-1]
                cur_start, cur_end = yearmon2indices[yearmon]

                test_historical_missing_mask = test_historical_missing_mask & missing_mask[yearmon]
                x = test_data[-window_size:].clone()
                x[:, test_historical_missing_mask.repeat(num_features)] = 0
                output = model(x)
    
                yearmon = yearmons[ind]
                predict_start, predict_end = yearmon2indices[yearmon]
    
                if yearmon < backtest_start:
                    output = output[-1]
                    test_loss = loss_fn(
                        output,
                        test_target_data[-window_size:],
                        missing_mask[yearmon])
                    pbar.set_description(f"Test loss: {test_loss:.6f}")

                else:
                    output = output[-1]
                    mask = missing_mask[yearmon]
                    cur_returns = torch.where(
                        mask.expand(predict_end - predict_start + 1, -1),
                        0,
                        torch_target_data[predict_start + lagging:predict_end + 1 + lagging]).exp()
    
                    weight = torch.where(
                        missing_mask[yearmon],
                        -torch.inf,
                        output).softmax(dim=-1)
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


class model_cls(nn.Module):
    def __init__(self, input_size: int, num_output: int):
        super().__init__()
        self.neural_layer = nn.LSTM(input_size, input_size, 1)
        self.output_layer = nn.Linear(input_size, num_output, bias = True)

    def forward(self, x):
        x, _ = self.neural_layer(x)
        return self.output_layer(x)