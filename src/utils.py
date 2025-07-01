from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def get_yearmons(data: pd.DataFrame) -> List[str]:
    first_date = data.index.min()
    last_date = data.index.max()
    yearmons = []
    for year in range(first_date.year, last_date.year + 1):
        for month in range(12):
            if year == first_date.year and month + 1 < first_date.month:
                continue
    
            yearmon = f'{year}-{month + 1:02d}'
    
            if year == last_date.year and month + 1 == last_date.month:
                break
    
            yearmons.append(yearmon)
    return yearmons

def preprocess_data(file_name: str, start_date: str, device: torch.device) -> Tuple[List[str], pd.DataFrame, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    data = pd.read_csv(file_name)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['ticker', 'date'])
    data = data.set_index(['ticker', 'date'])
    
    # 데이터 전처리
    data['open'] = np.log(data['open'] / data['close'])
    data['high'] = np.log(data['high'] / data['close'])
    data['low'] = np.log(data['low'] / data['close'])
    data['volume'] = np.log(data['volume'])
    data['log_tr'] = np.log(data.groupby('ticker')['div_adj_close'].pct_change() + 1)
    data = data.drop(['close', 'div_adj_close'], axis = 1)
    data = data.unstack(level = 0)
    data = data[(~data['log_tr'].isna()).sum(1) > 1]
    data = data[(~(data['log_tr'].isna() | (data['log_tr'] == 0))).sum(1) > 0]
    data = data.loc[data.index.intersection(pd.bdate_range(data.index.min(), data.index.max()))]
    data = data.loc[start_date:]

    torch_data = torch.tensor(data.values, dtype = torch.float, device = device).nan_to_num(0)
    torch_target_data = torch_data[:, -data['log_tr'].columns.__len__():]

    yearmons = get_yearmons(data)

    missing_mask = {yearmon:torch.tensor(data['log_tr'].loc[yearmon].iloc[0].isna().values, dtype = torch.bool, device = device)
                    for yearmon in yearmons}

    return yearmons, data.fillna(0), torch_data, torch_target_data, missing_mask

def preprocess_data_rolling_univ(file_name: str, start_date: str, device: torch.device) -> Tuple[List[str], pd.DataFrame, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    data = pd.read_csv(file_name)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['tradingitemid', 'date'])
    data = data.set_index(['tradingitemid', 'date'])
    
    # 데이터 전처리
    data['open'] = np.log(data['open'] / data['close'])
    data['high'] = np.log(data['high'] / data['close'])
    data['low'] = np.log(data['low'] / data['close'])
    data['volume'] = np.log(data['volume'])
    data['log_tr'] = np.log(data.groupby('tradingitemid')['div_adj_close'].pct_change() + 1)
    data = data.drop(['close', 'div_adj_close'], axis = 1)
    data = data.unstack(level = 0)
    data = data[(~data['log_tr'].isna()).sum(1) > 1]
    data = data[(~(data['log_tr'].isna() | (data['log_tr'] == 0))).sum(1) > 0]
    data = data.loc[data.index.intersection(pd.bdate_range(data.index.min(), data.index.max()))]
    data = data.loc[start_date:]

    data_yearmons = get_yearmons(data)
    
    torch_data = torch.tensor(data.values, dtype = torch.float, device = device).nan_to_num(0)
    torch_target_data = torch_data[:, -data['log_tr'].columns.__len__():]

    member_data = pd.read_csv(file_name.replace('.', '_member.'))
    member_data['date'] = pd.to_datetime(member_data['date'])
    member_data = member_data.set_index('date')
    member_data = member_data.loc[pd.bdate_range(member_data.index.min(), member_data.index.max())]

    member_yearmons = get_yearmons(member_data)
    
    yearmons = member_yearmons if data_yearmons[0] < member_yearmons[0] else data_yearmons

    df = pd.DataFrame(True, index = yearmons, columns = data['log_tr'].columns)
    for yearmon in yearmons:
        df.loc[yearmon, list(set(df.columns).intersection(set(member_data.loc[member_data.loc[yearmon].index.min(), 'tradingitemid'].values)))] = False

    missing_mask = {yearmon:torch.tensor(df.loc[yearmon].values, dtype = torch.bool, device = device)
                    for yearmon in yearmons}

    return yearmons, data.fillna(0), torch_data, torch_target_data, missing_mask

def preprocess_backtest_data(file_name: str, device: torch.device) -> Tuple[List[str], pd.DataFrame]:
    data = pd.read_csv(file_name)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['ticker', 'date'])
    data = data.set_index(['ticker', 'date'])

    data['log_tr'] = np.log(data.groupby('ticker')['div_adj_close'].pct_change() + 1)
    data = data[['volume', 'close', 'log_tr']]
    data = data.unstack(level = 0)
    data = data[(~data['log_tr'].isna()).sum(1) > 0]
    data = data[(~(data['log_tr'].isna() | (data['log_tr'] == 0))).sum(1) > 0]
    data = data.loc[data.index.intersection(pd.bdate_range(data.index.min(), data.index.max()))]

    yearmons = get_yearmons(data)
    
    missing_mask = {yearmon:torch.tensor(data['log_tr'].loc[yearmon].iloc[0].isna().values, dtype = torch.bool, device = device)
                    for yearmon in yearmons}

    return yearmons, data, missing_mask

def preprocess_backtest_data_rolling_univ(file_name: str, device: torch.device) -> Tuple[List[str], pd.DataFrame]:
    data = pd.read_csv(file_name)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['tradingitemid', 'date'])
    data = data.set_index(['tradingitemid', 'date'])

    data['log_tr'] = np.log(data.groupby('tradingitemid')['div_adj_close'].pct_change() + 1)
    data = data[['volume', 'close', 'log_tr']]
    data = data.unstack(level = 0)
    data = data[(~data['log_tr'].isna()).sum(1) > 0]
    data = data[(~(data['log_tr'].isna() | (data['log_tr'] == 0))).sum(1) > 0]
    data = data.loc[data.index.intersection(pd.bdate_range(data.index.min(), data.index.max()))]

    data_yearmons = get_yearmons(data)

    member_data = pd.read_csv(file_name.replace('.', '_member.'))
    member_data['date'] = pd.to_datetime(member_data['date'])
    member_data = member_data.set_index('date')
    member_data = member_data.loc[pd.bdate_range(member_data.index.min(), member_data.index.max())]

    member_yearmons = get_yearmons(member_data)

    yearmons = member_yearmons if data_yearmons[0] < member_yearmons[0] else data_yearmons
    
    df = pd.DataFrame(True, index = yearmons, columns = data['log_tr'].columns)
    for yearmon in yearmons:
        df.loc[yearmon, list(set(df.columns).intersection(set(member_data.loc[member_data.loc[yearmon].index.min(), 'tradingitemid'].values)))] = False

    missing_mask = {yearmon:torch.tensor(df.loc[yearmon].values, dtype = torch.bool, device = device)
                    for yearmon in yearmons}

    return yearmons, data, missing_mask