import numpy as np
import pandas as pd

def aggregate_weights(yearmons, data, all_weights, backtest_start):
    weights = pd.DataFrame(index=yearmons, columns=data['log_tr'].columns).loc[backtest_start:]
    weights.loc[:, :] = np.stack([x.loc[weights.index] for x in all_weights]).mean(0)
    weights = weights.astype(float)
    return weights

def backtest(yearmons, data, missing_mask, weights, backtest_start, trading_cost = 0.004, lagging = 2):
    result = pd.DataFrame(0, index = data.loc[backtest_start:].index, columns = ['strategy'])
    prev_strategy_weight = None
    prev_ew_weight = None
    for yearmon in yearmons:
        if yearmon < backtest_start:
            continue

        cur_returns = data.shift(-lagging).loc[yearmon, 'log_tr'].fillna(0).apply(np.exp).values
        cur_return_agg = np.prod(cur_returns, axis = 0)
        cur_available = ~missing_mask[yearmon].cpu().numpy()
    
        cur_strategy_returns = cur_returns.copy()
        cur_strategy_weight = weights.loc[yearmon].values
        cur_strategy_weight[~cur_available] = 0
        if np.sum(cur_strategy_weight) == 0:
            cur_strategy_weight[cur_available] = 1 / np.sum(cur_available)
        
        if prev_strategy_weight is None:
            cur_strategy_returns[0] -= trading_cost * cur_strategy_weight
        else:
            cur_strategy_returns[0] -= np.abs(prev_strategy_weight - cur_strategy_weight) * trading_cost

        result.loc[yearmon, 'strategy'] = np.log(np.sum(cur_strategy_weight * cur_strategy_returns, axis = 1))
 
        prev_strategy_weight = cur_strategy_weight * cur_return_agg
        prev_strategy_weight /= np.sum(prev_strategy_weight, axis = -1)

    return result

def backtest_baseline(yearmons, data, missing_mask, backtest_start, trading_cost = 0.004, lagging = 2):
    result = pd.DataFrame(0, index = data.loc[backtest_start:].index, columns = ['equal_weight', 'value_weight'])
    prev_ew_weight = None
    prev_vw_weight = None
    for yearmon in yearmons:
        if yearmon < backtest_start:
            continue
    
        cur_returns = data.shift(-lagging).loc[yearmon, 'log_tr'].fillna(0).apply(np.exp).values
        cur_return_agg = np.prod(cur_returns, axis = 0)
        cur_available = ~missing_mask[yearmon].cpu().numpy()
        cur_market_cap = data.loc[yearmon, 'close'].iloc[0] * data.loc[yearmon, 'volume'].iloc[0]
        cur_market_cap = cur_market_cap.fillna(0).values
    
        cur_ew_returns = cur_returns.copy()
        cur_vw_returns = cur_returns.copy()
        
        cur_ew_weight = cur_available / np.sum(cur_available)
        cur_vw_weight = cur_market_cap / np.sum(cur_market_cap)
        
        if prev_ew_weight is None:
            cur_ew_returns[0] -= trading_cost * cur_ew_weight
            cur_vw_returns[0] -= trading_cost * cur_vw_weight
        else:
            cur_ew_returns[0] -= np.abs(prev_ew_weight - cur_ew_weight) * trading_cost
            cur_ew_returns[0] -= np.abs(prev_vw_weight - cur_vw_weight) * trading_cost
        
        result.loc[yearmon, 'equal_weight'] = np.log(np.sum(cur_ew_weight.reshape(1, -1) * cur_ew_returns, axis = 1))
        result.loc[yearmon, 'value_weight'] = np.log(np.sum(cur_vw_weight.reshape(1, -1) * cur_vw_returns, axis = 1))
    
        prev_ew_weight = cur_ew_weight * cur_return_agg
        prev_ew_weight /= np.sum(prev_ew_weight)
        prev_vw_weight = cur_vw_weight * cur_return_agg
        prev_vw_weight /= np.sum(prev_vw_weight)

    return result