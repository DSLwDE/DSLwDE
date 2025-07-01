import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import preprocess_data, preprocess_data_rolling_univ
from src.model import lstm, trf, mamba
from src.loss import loss_fn
from src.train import train
from sophia import SophiaG


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda') # Set GPU by CUDA_VISIBLE_DEVICES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--universe', default = 'universe1', type = str)
    parser.add_argument('-i', '--rolling_universe', action = 'store_true')
    parser.add_argument('-p', '--portfolio', default = 'maxsortino', choices = ['maxsharpe', 'maxsortino', 'onehot'])
    parser.add_argument('-o', '--model', default = 'mamba', choices = ['lstm', 'trf', 'mamba'])
    parser.add_argument('-z', '--optimizer', default = 'adam', choices = ['adam', 'sophia'])
    parser.add_argument('-s', '--train_start', default = '2010-01')
    parser.add_argument('-b', '--backtest_start', default = '2019-11')
    parser.add_argument('-l', '--lr', default = 1e-4, type = float)
    parser.add_argument('-w', '--window_size', default = 252, type = int)
    parser.add_argument('-r', '--num_repeat', default = 1000, type = int)
    parser.add_argument('-e', '--num_epoch', default = 50)
    parser.add_argument('-t', '--trading_cost', default = 0.004)
    parser.add_argument('-g', '--lagging', default = 2, type = int)
    args = parser.parse_args()

    with open(f'result/{args.universe}_weights_{args.portfolio}.pkl', 'rb') as f:
        target_weight = pickle.load(f)

    for key in target_weight.keys():
        target_weight[key] = target_weight[key].to(device)

    if args.rolling_universe:
        yearmons, data, torch_data, torch_target_data, missing_mask = preprocess_data_rolling_univ(f'data/{args.universe}.csv', args.train_start, device)
    else:
        yearmons, data, torch_data, torch_target_data, missing_mask = preprocess_data(f'data/{args.universe}.csv', args.train_start, device)

    backtest_start = args.backtest_start
    backtest_end = yearmons[-1]
    date2ind = {x:ind for ind, x in enumerate(data.index)}
    yearmon2indices = {x:(date2ind[data.loc[x].index[0]], date2ind[data.loc[x].index[-1]]) for ind, x in enumerate(yearmons)}

    all_backtest, all_weights, all_logits = train(
        yearmons,
        backtest_start,
        backtest_end,
        date2ind,
        yearmon2indices,
        data,
        torch_data,
        torch_target_data,
        missing_mask,
        target_weight,
        eval(args.model),
        SophiaG if args.optimizer == 'sophia' else torch.optim.Adam,
        loss_fn,
        device,
        window_size = args.window_size,
        num_repeat = args.num_repeat,
        num_epoch = args.num_epoch,
        lr = args.lr,
        trading_cost = args.trading_cost,
        lagging = args.lagging)

    common_path = f'result/{args.universe}_ensemble_{args.portfolio}_{args.model}_{args.optimizer}_{args.window_size}_{args.num_repeat}'
    
    with open(common_path + '_all_weights.pkl', 'wb') as f:
        pickle.dump(all_weights, f)

    with open(common_path + '_all_logits.pkl', 'wb') as f:
        pickle.dump(all_logits, f)