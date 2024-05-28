from itertools import count
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from config import *
from homo_import_balanced import import_data, gen_import_dict, balanced_loader, self_loader
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score
import random

# set seeds
torch.manual_seed(12345)
random.seed(12345)
np.random.seed(12345)


# choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device


def test(loader, source, target):
    with torch.no_grad():
        user_ids = []
        pred_cache = []
        true_cache = []
        model.eval()
        total_correct = 0
        train_loader_len = 0
        possible_correct = 0
        total_examples = total_correct = 0
        for batch in tqdm(loader):
            batch = batch.to(device)
            if HOMO:
                batch_size = batch.batch_size
            else:
                batch_size = batch['users'].batch_size
            if HOMO:
                out = model(batch.x, batch.edge_index)[:batch_size]
            else:
                out = model(batch.x_dict, batch.edge_index_dict)['users'][:batch_size]
            pred = out.argmax(dim=-1)
            
            possible_correct += batch_size
            if HOMO:
                total_correct += int((pred == batch.y[:batch_size]).sum())
            else:
                total_correct += int((pred == batch['users'].y[:batch_size]).sum())                
            
            train_loader_len += 1
            
            if HOMO:
                # for a classification report
                user_ids.append(batch.n_id[:batch_size].cpu().tolist())
                pred_cache.append((pred.cpu()).tolist())
                true_cache.append((batch.y[:batch_size].cpu()).tolist())
            else:
                # for a classification report
                user_ids.append(batch['users'].n_id[:batch_size].cpu().tolist())
                pred_cache.append((pred.cpu()).tolist())
                true_cache.append((batch['users'].y[:batch_size].cpu()).tolist())

    
    pred_flat = [item for items in pred_cache for item in items]
    true_flat = [item for items in true_cache for item in items]
    user_ids_flat = [item for items in user_ids for item in items]
    
    df = pd.DataFrame({'user_id': user_ids_flat,
                       'predicted_label': pred_flat,
                       'true_label': true_flat})
            
    acc_score_s = accuracy_score(true_flat, pred_flat)
    f1_score_s = f1_score(true_flat, pred_flat, average = 'macro')
    precision_s = precision_score(true_flat, pred_flat, average = 'macro')
    recall_s = recall_score(true_flat, pred_flat, average = 'macro')

    #valid_acc = total_correct / possible_correct           

    return acc_score_s, f1_score_s, precision_s, recall_s


if __name__ == '__main__':
    ## Run the script
    seed_countries = ['Bolivia', 'Chile', 'Colombia', 'Ecuador']
    training_stats = []
    
    for source_model in seed_countries:
        model = torch.load(f'{MODEL_FOLDER}{source_model}_bal_gnn.pt')
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        target_countries = [source_model]
        for country_name in target_countries:
            print(f'{source_model} is predicting {country_name}')
            
            import_dict = gen_import_dict(country_name)
            att_df, idx_df, mask_and_y_cache, true_y = import_data(import_dict, balanced_data=False)
            train_loader, valid_loader, test_loader = self_loader(att_df, idx_df, mask_and_y_cache, true_y)
            
            acc_score_s, f1_score_s, precision_s, recall_s = test(test_loader, source_model, country_name)
            
            training_stats.append({f"{country_name}": \
                {"test_accuracy":f'{acc_score_s}',\
                'test_f1':f'{f1_score_s}',\
                'test_macro_precision':f'{precision_s}',\
                'test_macro_recall':f'{recall_s}'}})
            
    df = pd.concat([pd.DataFrame(i).T for i in training_stats])
    df = df.reset_index()
    df.to_csv(f'../results/{DATA_OUT}.csv', index = False)