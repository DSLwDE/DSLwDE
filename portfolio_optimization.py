import argparse
import pickle

import torch

from src.utils import preprocess_data, get_yearmons
from src.target_portfolio import OneHotWeightCollection, MaxSortinoWeightCollection


device = torch.device('cuda')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', default = 'data/universe1.csv', type = str)
    parser.add_argument('-n', '--num_epoch', default = 100, type = int)
    args = parser.parse_args()
    
    yearmons, data, torch_data, torch_target_data, missing_mask = preprocess_data(file_name, device)

    weight_collection = OneHotWeightCollection(yearmons, data['div_adj_close'].columns.__len__(), device)
    
    weight_collection.calculate_one_hot_weights(data, missing_mask)
    
    weights = weight_collection.get_weights()
    
    with open(f'result/{args.file_name}_{args.mode}_weights_onehot.pkl', 'wb') as f:
        pickle.dump(weights, f)
    
    weight_collection = MaxSortinoWeightCollection(yearmons, data['div_adj_close'].columns.__len__(), device)
    
    weight_collection.train_max_sortino_weights(data, missing_mask, num_epoch = args.num_epoch)
    
    weights = weight_collection.get_weights()
    
    with open(f'result/{args.file_name}_{args.mode}_weights_maxsortino.pkl', 'wb') as f:
        pickle.dump(weights, f)
