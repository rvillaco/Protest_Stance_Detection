from itertools import count
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from config import *
from sklearn.metrics import  accuracy_score, f1_score
from referendum_import import import_data, balanced_loader
import random
from torch_geometric.loader import NeighborLoader


def gen_referendum_dict(country):
    
    import_dict = {}
    ## Set train/test/val/edge paths
    import_dict['train_path'] = f'../data/referendum/ref_splits/2a-train_dataframe.csv'
    import_dict['val_path'] = f'../data/referendum/ref_splits/2b-validation_dataframe.csv'
    import_dict['test_path'] = f'../data/referendum/ref_splits/2c-test_dataframe.csv'

    # lets try w/ retweets and see how that goes
    import_dict['edge_path'] = f'../data/referendum/ref_edges/labeled_user_response_edges.csv'

    # paths to embeddings calculated by the transformer model
    import_dict['train_embed'] = f'../data/referendum/ref_embeddings/train_embeddings.pt'
    import_dict['train_map'] = f'../data/referendum/ref_embeddings/train_predictions.csv'
    import_dict['val_embed'] = f'../data/referendum/ref_embeddings/val_embeddings.pt'
    import_dict['val_map'] = f'../data/referendum/ref_embeddings/val_predictions.csv'
    import_dict['test_embed'] = f'../data/referendum/ref_embeddings/test_embeddings.pt'
    import_dict['test_map'] = f'../data/referendum/ref_embeddings/test_predictions.csv'
    import_dict['onehop_embed'] = f'../data/referendum/ref_embeddings/first_neighbor_embeddings.pt'
    import_dict['onehop_map'] = f'../data/referendum/ref_embeddings/first_neighbor_predictions.csv'
    import_dict['twohop_embed'] = f'../data/referendum/ref_embeddings/second_neighbor_embeddings.pt'
    import_dict['twohop_map'] = f'../data/referendum/ref_embeddings/second_neighbor_predictions.csv'


    # for heterogenous import
    import_dict['path_to_responses'] = f'../data/referendum/ref_edges/labeled_user_response_edges.csv'
    import_dict['path_to_retweets'] = f'../data/referendum/ref_edges/labeled_user_retweet_edges.csv'

    import_dict['path_to_ul_responses'] =  f'../data/referendum/ref_edges/first_neighbor_response_edges.csv'
    import_dict['path_to_ul_retweets'] = f'../data/referendum/ref_edges/first_neighbor_retweet_edges.csv'

    return import_dict


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

    #valid_acc = total_correct / possible_correct           

    return df, acc_score_s, f1_score_s


if __name__ == '__main__':
    ## Run the script
    MODEL_FOLDERS = ["../models/homo_rt_only/", "../models/homo_rp_only/","../models/homo_full_net/"]
    target_countries = ['Chile']
    for country_name in target_countries:
        
        training_stats = []

        import_dict = gen_referendum_dict(country_name)
        att_df, idx_df, mask_and_y_cache, true_y = import_data(import_dict)
        #data, train_loader, valid_loader, test_loader = balanced_loader(att_df, idx_df, mask_and_y_cache, true_y, 0, inference=True)
        data, train_loader, valid_loader, test_loader = balanced_loader(att_df, idx_df, mask_and_y_cache, true_y, 0)
        
        del train_loader, valid_loader
        
        # explore this..
        everything_loader = NeighborLoader(data, input_nodes = (data.n_id),
                            num_neighbors=[1], batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)
        
        source_model = 'Chile'
        df_cache = []
        acc_cache = []
        for idx, mtype in enumerate(MODEL_FOLDERS):
            model = torch.load(f'{mtype}{source_model}_bal_gnn.pt')
            print(f'{source_model} is predicting {country_name}')

            #tacc, tf1 = test(train_loader, source_model, country_name)
            #vacc, vf1 = test(valid_loader, source_model, country_name)
            df, test_acc, test_f1 = test(test_loader, source_model, country_name)
            
            print(f'Test accuracy: {test_acc}, Test f1: {test_f1}')
            
            acc_cache.append({f'{idx}': 
                {'test_acc': f'{test_acc}', 
                'test_f1':f'{test_f1}'}
            })
            
            df, test_acc, test_f1 = test(everything_loader, source_model, country_name)

            df_cache.append(df)
            del model


dfs = [pd.DataFrame(i).T for i in acc_cache]
df = pd.concat(dfs)
df['run'] = ['homo_retweet_only', 'homo_reply_only', 'homo_full_network']

# rt
#83.2
#16.8

#rp
# 86.3%
# 13.7

#full
#87.6
#12.4

df.to_csv('homogenous_referendum_runs.csv', index = False)