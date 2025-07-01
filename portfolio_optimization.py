import argparse
import pickle

import torch

from src.utils import preprocess_data, preprocess_data_rolling_univ, get_yearmons
from src.target_portfolio import OneHotWeightCollection, MaxSharpeWeightCollection, MaxSortinoWeightCollection


device = torch.device('cuda')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', default = 'universe1', type = str)
    parser.add_argument('-i', '--rolling_universe', action = 'store_true')
    parser.add_argument('-n', '--num_epoch', default = 1000, type = int)
    args = parser.parse_args()

    if args.rolling_universe:
        yearmons, data, torch_data, torch_target_data, missing_mask = preprocess_data_rolling_univ('data/' + args.file_name + '.csv', device)
    else:
        yearmons, data, torch_data, torch_target_data, missing_mask = preprocess_data('data/' + args.file_name + '.csv', device)

    # One Hot
    weight_collection = OneHotWeightCollection(yearmons, data['log_tr'].columns.__len__(), device)
    weight_collection.calculate_one_hot_weights(data, missing_mask)
    weights = weight_collection.get_weights()
    with open(f'result/{args.file_name}_weights_onehot.pkl', 'wb') as f:
        pickle.dump(weights, f)

    # Max Sharpe
    weight_collection = MaxSharpeWeightCollection(yearmons, data['log_tr'].columns.__len__(), device)
    weight_collection.train_max_sharpe_weights(data, missing_mask, num_epoch = args.num_epoch)
    weights = weight_collection.get_weights()
    with open(f'result/{args.file_name}_weights_maxsharpe.pkl', 'wb') as f:
        pickle.dump(weights, f)

    # Max-Sortino
    weight_collection = MaxSortinoWeightCollection(yearmons, data['log_tr'].columns.__len__(), device)
    weight_collection.train_max_sortino_weights(data, missing_mask, num_epoch = args.num_epoch)
    weights = weight_collection.get_weights()
    with open(f'result/{args.file_name}_weights_maxsortino.pkl', 'wb') as f:
        pickle.dump(weights, f)
