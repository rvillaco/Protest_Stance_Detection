import pandas as pd
import torch
import pandas as pd
from torch_geometric.data import Data
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from config import *


def gen_import_dict(country):
    
    import_dict = {}
    ## Set train/test/val/edge paths
    import_dict['train_path'] = f'../data/labeled/{country}/2a-train_{country}_dataframe.csv'
    import_dict['val_path'] = f'../data/labeled/{country}/2b-validation_{country}_dataframe.csv'
    import_dict['test_path'] = f'../data/labeled/{country}/2c-test_{country}_dataframe.csv'

    # lets try w/ retweets and see how that goes
    import_dict['edge_path'] = f'../data/edges/{country}/labeled_user_response_edges.csv'

    # paths to embeddings calculated by the transformer model
    import_dict['train_embed'] = f'../data/embeddings/{country}/train_embeddings.pt'
    import_dict['train_map'] = f'../data/embeddings/{country}/train_predictions.csv'
    import_dict['val_embed'] = f'../data/embeddings/{country}/val_embeddings.pt'
    import_dict['val_map'] = f'../data/embeddings/{country}/val_predictions.csv'
    import_dict['test_embed'] = f'../data/embeddings/{country}/test_embeddings.pt'
    import_dict['test_map'] = f'../data/embeddings/{country}/test_predictions.csv'
    import_dict['onehop_embed'] = f'../data/embeddings/{country}/first_neighbor_embeddings.pt'
    import_dict['onehop_map'] = f'../data/embeddings/{country}/first_neighbor_predictions.csv'
    import_dict['twohop_embed'] = f'../data/embeddings/{country}/second_neighbor_embeddings.pt'
    import_dict['twohop_map'] = f'../data/embeddings/{country}/second_neighbor_predictions.csv'


    # for heterogenous import
    import_dict['path_to_responses'] = f'../data/edges/{country}/labeled_user_response_edges.csv'
    import_dict['path_to_retweets'] = f'../data/edges/{country}/labeled_user_retweet_edges.csv'

    import_dict['path_to_ul_responses'] =  f'../data/edges/{country}/complete_unlabeled_response_edges.csv'
    import_dict['path_to_ul_retweets'] = f'../data/edges/{country}/complete_unlabeled_retweet_edges.csv'

    return import_dict



### Import all embeddings ###
def import_embeddings(train_map, val_map, test_map, onehop_map, twohop_map,
                              train_embed, val_embed, test_embed, onehop_embed, twohop_embed):
    """[grabs the user embedding for each node and makes sure they're aligend with y, and the masks.
    Not all nodes will have attributes. For those that do not, we NaNs with the column-wise mean.]

    Returns:
        [type]: [description]
    """

    embedding_maps = []
    for map in [train_map, val_map, test_map, onehop_map, twohop_map]:
        embedding_maps.append((pd.read_csv(map, dtype = object))['user_id'].astype(str))
    embedding_maps = (pd.concat(embedding_maps))

    embedding_tensors = []
    for tensor_i in [train_embed, val_embed, test_embed, onehop_embed, twohop_embed]:
        embedding_tensors.append(pd.DataFrame(torch.load(tensor_i).cpu().numpy()))
    embedding_tensors = (pd.concat(embedding_tensors))

    embedding_tensors['old_user_id'] = embedding_maps

    embedding_tensors = embedding_tensors.drop_duplicates(subset = ['old_user_id'])
    return embedding_tensors


# inspect this: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py
def import_and_index(import_dict, str_idx = True):
    """Takes the path to an edge list with two columns named 'user_in'
    and 'tw_out'. This function maps each of those user IDs to an index starting
    at 0 and ending at the number of unique accounts in the network. It returns
    the mapping dictionary as well as the edgelist where account names have been
    mapped to indices.

    Args:
        path_to_edgelist ([str]): [path to edgelist (.csv)]

    Returns:
        [idx_df]: [edgelist following index-mapping]
        [idx_mapper]: [a dictionary for going to and from the mapping space]

    """
    
    assert RP_ONLY + RT_ONLY != 2, "Only RT or RP only can be True. Not both."
    
    if str_idx == True:
        if RT_ONLY == True:
            #for homogenous network, lets try rt only
                df_rt = pd.read_csv(import_dict['path_to_retweets'], dtype=object)
                df_rt = df_rt.astype(str)
                df_u_rt = pd.read_csv(import_dict['path_to_ul_retweets'], dtype = object)
                df_u_rt = df_u_rt.astype(str)
                df_rt = pd.concat([df_rt, df_u_rt])
                df = df_rt
        
        elif RP_ONLY == True:
                df_rt = pd.read_csv(import_dict['path_to_responses'], dtype=object)
                df_rt = df_rt.astype(str)
                df_u_rt = pd.read_csv(import_dict['path_to_ul_responses'], dtype = object)
                df_u_rt = df_u_rt.astype(str)
                df_rt = pd.concat([df_rt, df_u_rt])
                df = df_rt
        else:
            # bring in responses
            df_re = pd.read_csv(import_dict['path_to_responses'], dtype=object)
            df_re = df_re.astype(str)
            if ORDER == 'complete':
                df_u_re = pd.read_csv(import_dict['path_to_ul_responses'], dtype = object)
                df_u_re = df_u_re.astype(str)
                df_re = pd.concat([df_re, df_u_re])
            df_re['attribute'] = 1

            # bring in retweets
            df_rt = pd.read_csv(import_dict['path_to_retweets'], dtype=object)
            df_rt = df_rt.astype(str)
            if ORDER == 'complete':
            # bring in unlabeled users
                df_u_rt = pd.read_csv(import_dict['path_to_ul_retweets'], dtype = object)
                df_u_rt = df_u_rt.astype(str)
                df_rt = pd.concat([df_rt, df_u_rt])
            df_rt['attribute'] = 0
            df = pd.concat([df_re, df_rt])
    
    # extract unique ids for both user_in and user_out 
    unique_ids = pd.DataFrame({'unique_keys' : (df['user_in'].unique().tolist() + df['user_out'].unique().tolist())})
    unique_ids = unique_ids['unique_keys'].unique().tolist()
    edge_ids = pd.DataFrame({'user_id':unique_ids}).astype(str)

    # we're only going to use edges where embeddings are present
    embeddings = import_embeddings(import_dict['train_map'], import_dict['val_map'], import_dict['test_map'], import_dict['onehop_map'], import_dict['twohop_map'],
                        import_dict['train_embed'], import_dict['val_embed'], import_dict['test_embed'], import_dict['onehop_embed'], import_dict['twohop_embed'])
    embeddings.old_user_id = embeddings.old_user_id.astype(str)
    embeddings = embeddings.rename(columns = {'old_user_id': 'user_id'})

    embeddings = edge_ids.merge(embeddings, how= 'left', on='user_id')
    embeddings = embeddings.dropna(subset=[0])

    ids_with_embeddings = embeddings.user_id.astype(str).tolist()
    idx_mapper = {name:str(idx) for idx, name in enumerate(ids_with_embeddings)}

    # map to dataframe
    df['source_idx'] = df['user_in'].map(idx_mapper)
    df['target_idx'] = df['user_out'].map(idx_mapper)
    df = df.dropna()

    # use the indices for the edgelist we feed to the GNN. We can mask
    # train/ test/ ec. accordingly
    #idx_df = df[['source_idx', 'target_idx', 'attribute']]
    idx_df = df.drop(columns = ['user_in','user_out'])
    embeddings = embeddings.drop(columns = ['user_id'])
    embeddings = torch.FloatTensor(embeddings.to_numpy())

    return idx_df, idx_mapper, embeddings

def df_to_train_test_val_tensor(df, idx_mapper, str_idx = True):
    """
    Helper function:
    Maps train, test, and validation data to idx_mapper. Returns a list of integers 
    corresponding to the index of these nodes. Pyg requires nodes to be indexed
    from 0 to num_nodes.

    Args:
        df : [pd.Dataframe] df containing the edge list of the data
        idx_mapper : [dict] dictionary that maps user ids to 0 - num_nodes
    """

    # rip out user and stance columns
    df = df[['user_id', 'user_government_stance']].rename(columns = {'user_id':'old_user_id'})
    df = df.drop_duplicates(subset = 'old_user_id')
    if str_idx == True:
        df['old_user_id'] = df['old_user_id'].astype(str)    
    
    embed_df = pd.DataFrame([idx_mapper]).T.reset_index()
    embed_df.columns = ['old_user_id', 'user_id']

    subset_with_labs = embed_df.merge(df, how = 'left', on='old_user_id').dropna()
    subset_with_labs['user_id'] = subset_with_labs['user_id'].values.astype(str)
    subset_with_labs = subset_with_labs.drop(columns = ['old_user_id'])
    
    uStance = subset_with_labs.groupby('user_government_stance').count().reset_index().rename(columns = {'user_id': 'counts'})
    uStance['weights'] = subset_with_labs.shape[0] / uStance['counts']
    newdf = pd.merge(subset_with_labs, uStance[['user_government_stance', 'weights']], on = 'user_government_stance')
    #label_info = {'class_count': len(uStance), 'min_class_count': uStance.counts.min()}

    return newdf


def train_test_val_import(train_path, val_path, test_path, idx_mapper):
    """
    imports train, test, and validation data and uses left joins to create
    train, val, and test masks - each of length len(nodes). Each node will have
    a binary (and mutually exclusive) mask value for each mask. Addditionally,
    we extract the stance predictions, y. Nodes with no labels are given a y of -1

    Args:
        train_path [str] : relative path to training data
        val_path [str] : path to validation data
        test_path [str] : path to test data
        idx_mapper [dict] : output of import and index- maps nodes to (0-num(node)) indices
    """
    
    # just for debugging
    #train_path = import_dict['train_path']
    #val_path = import_dict['val_path']
    #test_path = import_dict['test_path']
    
    df = pd.read_csv(train_path, dtype=object)
    train = df_to_train_test_val_tensor(df, idx_mapper)
    train['train_mask'] = True

    df = pd.read_csv(val_path, dtype=object)
    val = df_to_train_test_val_tensor(df, idx_mapper)
    val['val_mask'] = True

    df = pd.read_csv(test_path, dtype=object)
    test = df_to_train_test_val_tensor(df, idx_mapper)
    test['test_mask'] = True

    # left join labels to a dataframe- we need separate masks for train, test, and validation,
    # we want their labels associated with them and we want all non-labeled nodes to receive
    # -1 as their label
    label_mapping_df = pd.DataFrame({'user_id':range(len(idx_mapper.keys()))})
    label_mapping_df['user_id'] = label_mapping_df['user_id'].astype(str)

    joined_split = pd.concat([train, val, test]).fillna(False)
    
    assert joined_split[['train_mask', 'val_mask', 'test_mask']].sum(axis = 1).sum() == joined_split.shape[0], 'train, test, val not disjoint'
    
    label_mapping_df = (label_mapping_df.merge(joined_split, how='left', left_on = 'user_id', right_on = 'user_id')
                                        .rename(columns = {'user_government_stance':'y'}))

    # users we don't have labels for will all be set as -1
    # this will allow us to use neighborhood values but not update loss based on
    # predictions for these nodes
    label_mapping_df['y'] = label_mapping_df['y'].fillna(-1)
    # train, test, and val masks want lists of True and False
    label_mapping_df = label_mapping_df.fillna(False)

    return label_mapping_df

def import_data(import_dict, balanced_data = False):
    """
    This function calls import_and_index as well as train_test_val import. It
    uses the output of those functions to create a pyG data object as well as two
    dataloader-like objects created via Neighbor Sampling. Train loader only samples
    for nodes in the training set. The subgraph loader does not mask

    TODO: We will need to replace att_df (which is currently randomly generated) with
    embeddings extracted from the transformer model

    edge_path [str]: path to edge list
    train_path [str] : path to training data
    val_path [str] : path to validation data
    test_path [str] : path to test data
    """
    # create 0 to n_node indexed edgelist and import embeddings
    idx_df, idx_mapper, att_df = import_and_index(import_dict)
    # extract train, val, and test masks along with y
    label_mapping_df = train_test_val_import(import_dict['train_path'], import_dict['val_path'], import_dict['test_path'], idx_mapper)

    # randomly generating attributes - will change when we get roberta embeddings 
    #att_df = torch.FloatTensor(len(idx_mapper.keys()), 5).uniform_(1, 5)
  
    if balanced_data:
        mask_and_y_cache ={}
        sample_keys = pd.DataFrame({'user_id':range(len(idx_mapper.keys()))})
        sample_keys['user_id'] = label_mapping_df['user_id'].astype('int64')
        for i in range(EPOCHS):
            # picking a weird random state fromula so i can still set seed for torch numpy and samples will still be different across epochs
            masked_train = label_mapping_df[label_mapping_df['train_mask'] == True].copy()
            balanced_train = masked_train.sample(masked_train.train_mask.sum(), weights = masked_train.weights, replace=True, random_state=(i+13)*128)
            train_users = torch.Tensor(balanced_train.user_id.astype('int64').to_numpy()).long()
            
            masked_val = label_mapping_df[label_mapping_df['val_mask'] == True].copy()
            balanced_val = masked_val.sample(masked_val.val_mask.sum(), weights = masked_val.weights, replace=True, random_state = (i+13)*128)
            val_users = torch.Tensor(balanced_val.user_id.astype('int64').to_numpy()).long()

            masked_test = label_mapping_df[label_mapping_df['test_mask'] == True].copy()
            balanced_test = masked_test.sample(n = masked_test.test_mask.sum(), weights = masked_test.weights, replace=True, random_state = (i+13)*128)
            test_users = torch.Tensor(balanced_test.user_id.astype('int64').to_numpy()).long()
            
            true_y = torch.Tensor(label_mapping_df.y.astype(int).values).long()

            #balanced_data = pd.concat([balanced_train, balanced_val, balanced_test])
            #nb = pd.merge(sample_keys, balanced_data, how='inner', left_on='user_id', right_on='user_id')
            #nb.y = nb.y.fillna(-1)
            #nb = nb.fillna(False)
            mask_and_y_cache[i] = {'train_users':train_users,
                                   'val_users':val_users, 
                                   'test_users':test_users
            }            
    else:
        mask_and_y_cache = {}
        masked_train = label_mapping_df[label_mapping_df['train_mask'] == True].copy()
        train_users = torch.Tensor(masked_train.user_id.astype('int64').to_numpy()).long()
        
        masked_val = label_mapping_df[label_mapping_df['val_mask'] == True].copy()
        val_users = torch.Tensor(masked_val.user_id.astype('int64').to_numpy()).long()

        masked_test = label_mapping_df[label_mapping_df['test_mask'] == True].copy()
        test_users = torch.Tensor(masked_test.user_id.astype('int64').to_numpy()).long()
        
        true_y = torch.Tensor(label_mapping_df.y.astype(int).values).long()

        mask_and_y_cache[0] = {'train_users':train_users,
                                'val_users':val_users, 
                                'test_users':test_users
        }            
    return att_df, idx_df, mask_and_y_cache, true_y


def balanced_loader(att_df, idx_df, mask_and_y_cache, true_y, epoch, inference = False):
    """[generates different loaders (with different samples) for 
    different epochs while keeping the ratio near 50-50]

    Args:
        att_df ([FloatTensor]): [embedding values]
        edge_list ([LongTensor]): [edge list]
        mask_and_y_cache ([dict of long tensors]): [each epoch is balanced, but different majority class samples are taken each epoch]
        epoch ([int]): [epoch #]

    Returns:
        [data]: [pyg heterogeneous data object]
        [train_loader, test_loader, val_loader]: data loaders for the model
    """
    train_users = mask_and_y_cache[epoch]['train_users']
    val_users = mask_and_y_cache[epoch]['val_users']
    test_users = mask_and_y_cache[epoch]['test_users']
    #y = mask_and_y_cache[epoch]['y']
    y = true_y
    
    edge_list = torch.tensor([idx_df.source_idx.astype('int64').values, idx_df.target_idx.astype('int64').values])

    # create pyg data object - for heterogenous objects masks need to be boolean apparently... weird that it's not that way for homogenous networks...
    #data = Data(x=att_df, edge_index=edge_list, y=y, train_mask=(train_mask==1), val_mask=(val_mask==1), test_mask = (test_mask==1))
    data = Data(x = att_df, edge_index = edge_list, y=true_y, train_mask=train_users, val_mask=val_users, test_mask=test_users)
    
    # allows us to get original node indices from neighborloader
    data.n_id = torch.arange(data.num_nodes)
    if HOMO:
        if inference:
            # shuffle train if we're training
            train_loader = NeighborLoader(data, input_nodes=(data.train_mask),
                        num_neighbors=[LAYER_ONE_NEIGHBORS, LAYER_TWO_NEIGHBORS], batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0)
        else:
            train_loader = NeighborLoader(data, input_nodes=(data.train_mask),
                                    num_neighbors=[LAYER_ONE_NEIGHBORS, LAYER_TWO_NEIGHBORS], batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=0)
        valid_loader = NeighborLoader(data, input_nodes = (data.val_mask),
                                num_neighbors=[LAYER_ONE_NEIGHBORS, LAYER_TWO_NEIGHBORS], batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0)
        test_loader = NeighborLoader(data, input_nodes = (data.test_mask),
                                num_neighbors=[LAYER_ONE_NEIGHBORS, LAYER_TWO_NEIGHBORS], batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0)
    else:
        data = data.to_heterogeneous(edge_type = torch.tensor(idx_df.attribute.values), node_type_names = ['users'])
        #data = T.ToUndirected()(data)
        # sample neighbors - we use this for minibatching
        if inference:
            #don't shuffle on inference
            train_loader = NeighborLoader(data, input_nodes=('users', data['users'].train_mask),
                                        num_neighbors=[LAYER_ONE_NEIGHBORS, LAYER_TWO_NEIGHBORS], batch_size=BATCH_SIZE, shuffle=False,
                                        num_workers=0)        
        else:

            train_loader = NeighborLoader(data, input_nodes=('users', data['users'].train_mask),
                                        num_neighbors=[LAYER_ONE_NEIGHBORS, LAYER_TWO_NEIGHBORS], batch_size=BATCH_SIZE, shuffle=True,
                                        num_workers=0)
        valid_loader = NeighborLoader(data, input_nodes = ('users', data['users'].val_mask),
                                num_neighbors=[LAYER_ONE_NEIGHBORS, LAYER_TWO_NEIGHBORS], batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0)
        test_loader = NeighborLoader(data, input_nodes = ('users', data['users'].test_mask),
                            num_neighbors=[LAYER_ONE_NEIGHBORS, LAYER_TWO_NEIGHBORS], batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    return train_loader, valid_loader, test_loader


def self_loader(att_df, idx_df, mask_and_y_cache, true_y):
    """data loader for self-edge-list i.e. a is only connected to a, b is only connected to b, etc,
    just a sanity check to make sure our model is working

    Args:
        att_df ([FloatTensor]): [embedding values]
        edge_list ([LongTensor]): [edge list]
        mask_and_y_cache ([dict of long tensors]): [each epoch is balanced, but different majority class samples are taken each epoch]
        epoch ([int]): [epoch #]

    Returns:
        [data]: [pyg heterogeneous data object]
        [train_loader, test_loader, val_loader]: data loaders for the model
    """
    train_users = mask_and_y_cache[0]['train_users']
    val_users = mask_and_y_cache[0]['val_users']
    test_users = mask_and_y_cache[0]['test_users']
    #y = mask_and_y_cache[epoch]['y']
    y = true_y
    
    # generate an edge list where each user is only connected to herself
    edge_list = torch.tensor([np.arange(len(true_y)), np.arange(len(true_y))])

    data = Data(x = att_df, edge_index = edge_list, y=true_y, train_mask=train_users, val_mask=val_users, test_mask=test_users)
    
    # allows us to get original node indices from neighborloader
    data.n_id = torch.arange(data.num_nodes)
    
    train_loader = NeighborLoader(data, input_nodes=(data.train_mask),
                            num_neighbors=[1], batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0)
    valid_loader = NeighborLoader(data, input_nodes = (data.val_mask),
                            num_neighbors=[1], batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)
    test_loader = NeighborLoader(data, input_nodes = (data.test_mask),
                            num_neighbors=[1], batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)
    
    return train_loader, valid_loader, test_loader