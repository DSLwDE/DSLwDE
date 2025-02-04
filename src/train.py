import tqdm
import random

import numpy as np
import pandas as pd
import torch


def train(
    yearmons,
    backtest_start,
    backtest_end,
    date2ind,
    yearmon2indices,
    data,
    torch_data,
    torch_target_data,
    target_weight,
    missing_mask,
    model_cls,
    optimizer_cls,
    loss_fn,
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
    
    for rep in range(num_repeat):
        random.seed(rep)
        np.random.seed(rep)
        torch.manual_seed(rep)
    
        model = model_cls(input_size = input_size, num_output = num_item).to(device)
    
        optimizer = optimizer_cls(model.parameters(), lr = lr)
    
        backtest = pd.DataFrame(0, index=data[backtest_start:backtest_end].index, columns=['backtest'])
        weights = pd.DataFrame(0, index=yearmons, columns=data['log_tr'].columns).loc[backtest_start:backtest_end]
    
        for ind in (pbar := tqdm.tqdm(range(3, len(yearmons)))):
            train_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-3]].index]]
            valid_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-2]].index]]
            test_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-1]].index]]
    
            train_sortino = 0.0
            valid_sortino = 0.0
            best_valid_loss = torch.inf
            count = 0
            for epoch in range(num_epoch):
                yearmon = yearmons[ind-3]
                cur_start, cur_end = yearmon2indices[yearmon]
    
                model.train()
                optimizer.zero_grad(set_to_none=True)
                output = model(train_data[-window_size:])
                train_loss = loss_fn(
                    output,
                    target_weight[yearmon],
                    missing_mask[yearmon])
                train_loss.backward()
                optimizer.step()
        
                model.eval()
                with torch.no_grad():
                    yearmon = yearmons[ind-2]
                    cur_start, cur_end = yearmon2indices[yearmon]
                    
                    output = model(valid_data[-window_size:])
                    output = output[-1]
                    valid_loss = loss_fn(
                        output,
                        target_weight[yearmon],
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
                
                output = model(test_data[-window_size:])
    
                yearmon = yearmons[ind]
                predict_start, predict_end = yearmon2indices[yearmon]
    
                if yearmon < backtest_start:
                    output = output[-1]
                    test_loss = loss_fn(
                        output,
                        target_weight[yearmon],
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

