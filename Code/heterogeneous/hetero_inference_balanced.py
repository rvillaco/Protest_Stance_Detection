from itertools import count
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from config import *
from hetero_import_balanced import import_data, balanced_loader
import random

# set seeds
torch.manual_seed(12345)
random.seed(12345)
np.random.seed(12345)


# choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device



def gen_xcountry_dict(source_country, target_country):
    import_dict = {}
    ## Set train/test/val/edge paths
    import_dict['train_path'] = f'../data/labeled/{target_country}/2a-train_{target_country}_dataframe.csv'
    import_dict['val_path'] = f'../data/labeled/{target_country}/2b-validation_{target_country}_dataframe.csv'
    import_dict['test_path'] = f'../data/labeled/{target_country}/2c-test_{target_country}_dataframe.csv'

    # lets try w/ retweets and see how that goes
    import_dict['edge_path'] = f'../data/edges/{target_country}/labeled_user_response_edges.csv'

    # paths to embeddings calculated by the transformer model
    import_dict['train_embed'] = f'../data/cross_country/embeddings/{source_country}/{target_country}_train_embeddings.pt'
    import_dict['train_map'] = f'../data/cross_country/embeddings/{source_country}/{target_country}_train_predictions.csv'
    import_dict['val_embed'] = f'../data/cross_country/embeddings/{source_country}/{target_country}_val_embeddings.pt'
    import_dict['val_map'] = f'../data/cross_country/embeddings/{source_country}/{target_country}_val_predictions.csv'
    import_dict['test_embed'] = f'../data/cross_country/embeddings/{source_country}/{target_country}_test_embeddings.pt'
    import_dict['test_map'] = f'../data/cross_country/embeddings/{source_country}/{target_country}_test_predictions.csv'
    import_dict['onehop_embed'] = f'../data/cross_country/embeddings/{source_country}/{target_country}_first_neighbor_embeddings.pt'
    import_dict['onehop_map'] = f'../data/cross_country/embeddings/{source_country}/{target_country}_first_neighbor_predictions.csv'
    import_dict['twohop_embed'] = f'../data/cross_country/embeddings/{source_country}/{target_country}_second_neighbor_embeddings.pt'
    import_dict['twohop_map'] = f'../data/cross_country/embeddings/{source_country}/{target_country}_second_neighbor_predictions.csv'


    # for heterogenous import
    import_dict['path_to_responses'] = f'../data/edges/{target_country}/labeled_user_response_edges.csv'
    import_dict['path_to_retweets'] = f'../data/edges/{target_country}/labeled_user_retweet_edges.csv'

    import_dict['path_to_ul_responses'] =  f'../data/edges/{target_country}/complete_unlabeled_response_edges.csv'
    import_dict['path_to_ul_retweets'] = f'../data/edges/{target_country}/complete_unlabeled_retweet_edges.csv'

    return import_dict



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
            batch_size = batch['users'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict)['users'][:batch_size]
            pred = out.argmax(dim=-1)
            
            possible_correct += batch_size
            total_correct += int((pred == batch['users'].y[:batch_size]).sum())
            
            train_loader_len += 1
            
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
            
    #acc_score_s = accuracy_score(true_flat, pred_flat)
    #f1_score_s = f1_score(true_flat, pred_flat)

    #valid_acc = total_correct / possible_correct           

    return df


if __name__ == '__main__':
    ## Run the script

    model_list = ['Bolivia', 'Chile', 'Colombia', 'Ecuador']
    for source_model in model_list:
        
        training_stats = []
        model = torch.load(f'{MODEL_FOLDER}{source_model}_bal_gnn.pt')

        target_countries = ['Bolivia', 'Colombia', 'Ecuador','Chile']
        target_countries.remove(source_model)
        for country_name in target_countries:
            import_dict = gen_xcountry_dict(source_model, country_name)
            print(f'{source_model} is predicting {country_name}')

            att_df, idx_df, mask_and_y_cache, true_y = import_data(import_dict)
            data, train_loader, valid_loader, test_loader = balanced_loader(att_df, idx_df, mask_and_y_cache, true_y, 0)
            #train_loader, valid_loader, test_loader = balanced_loader(att_df, idx_df, mask_and_y_cache, true_y, 0)
            # free up some memory
            del att_df, idx_df, mask_and_y_cache, true_y
            tdf = test(train_loader, source_model, country_name)
            vdf = test(valid_loader, source_model, country_name)
            test_df = test(test_loader, source_model, country_name)
            
            tdf.to_csv(f'../results/{source_model}/cross_country/{DATA_OUT}/{country_name}_train_predictions.csv', index = False)
            vdf.to_csv(f'../results/{source_model}/cross_country/{DATA_OUT}/{country_name}_val_predictions.csv', index = False)
            test_df.to_csv(f'../results/{source_model}/cross_country/{DATA_OUT}/{country_name}_test_predictions.csv', index = False)
