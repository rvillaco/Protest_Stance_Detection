# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 15:53:10 2021

@author: rvillaco
"""

from torch.utils.data import Dataset, WeightedRandomSampler
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizer
import torch
import pandas as pd

################################ Tweet level Classification Dataset ################################
@dataclass
class TweetDatasetConfig:
    """
    Configuration for the preprocessing of the stance dataset.
    """
    user_file_name: str = field(
        metadata={"help": "Path to file containing stance dataset."}
    )
    tokenizer: PreTrainedTokenizer =  field(
                    metadata={"help": "Transformer Tokenizer."}
    )    
    max_seq_length: int =  field(
        default = None, metadata={"help": "Maximum sequence lenght for the model."}
    )
    interaction_categories: list =  field(
        default_factory = ['Original', 'Quote', 'Reply'], metadata={"help": "Possible tweet types, based on their interaction (eg: 'Original', 'Reply', etc.). Must include '<pad>' and <cls>' token."}
    )
    label_column: str =  field(
        default = None, metadata={"help": "Column name in Stance Dataset file that contains the user stance label."}
    )
    tweet_label_column: str =  field(
        default = 'tweet_government_stance', metadata={"help": "Column name in Stance Dataset file that contains the user stance label."}
    )
    keep_all_tweets: bool =  field(
        default = False, metadata={"help": "Whether to maintain all user tweets or only the one that had a weak label."}
    )    
    
    def __post_init__(self):
        if self.max_seq_length is None:    self.max_seq_length = self.tokenizer.model_max_length

            
class TweetDataset(Dataset):
    def __init__(self, config: TweetDatasetConfig):

        self.tokenizer = config.tokenizer 
        self.config = config
        
        # Load Dataframe
        Tweet_DF = pd.read_csv(config.user_file_name, dtype = {'tweet_id': 'Int64', 'user_id': str})
        # Encode Interaction Type
        Tweet_DF['interaction_type'] = pd.Categorical(Tweet_DF.interaction_type, categories = config.interaction_categories)
        Tweet_DF['interaction_type'] = Tweet_DF.interaction_type.cat.codes
        
        
        if config.label_column is not None:        
            Tweet_DF = Tweet_DF.rename(columns = {config.label_column: 'user_stance', config.tweet_label_column: 'tweet_stance'})

            if not config.keep_all_tweets: # Just keep weak labeled tweets
                Tweet_DF = Tweet_DF[Tweet_DF.tweet_stance != -1]

            # Get Tweet weights
            twStance = Tweet_DF.groupby('user_stance').tweet_id.count().reset_index().rename(columns = {'tweet_id': 'counts'})
            twStance['weights'] = len(twStance) / twStance['counts']
            Tweet_DF = pd.merge(Tweet_DF, twStance[['user_stance', 'weights']], on = 'user_stance').reset_index(drop = True)

            # Define Sampling weights
            self.label_info = {'class_count': len(twStance), 'min_class_count': twStance.counts.min()}
            self.weights = torch.DoubleTensor(Tweet_DF.weights.values)
        
        else:
            assert config.keep_all_tweets, 'If no labels are provided, the dataset must keep all tweets'
            Tweet_DF['user_stance'] = None
            self.weights = None
            
        # Define parameters
        self.ids = Tweet_DF.tweet_id.values
        self.labels = Tweet_DF.user_stance.values
        self.text = Tweet_DF.text.values
        self.interaction_type = Tweet_DF.interaction_type.values       
           
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, item):
        return self.ids[item], self.text[item], self.interaction_type[item], self.labels[item]    
    
    def _Tweet_datacollator(self, data_batch): # Could produce problems of memory overflow in multiprocessing (not sure if the self reference will cause duplication of the dataset) 
        # TODO: if it doest work for multiprocessing make it take tokenizer as a parameter
        ''' 
        Collate Batches of observations given by this dataset. 
            data_batch: List of Dictionaries as produced by self.__getitem__
        '''
        # If __getitem__ returned a mapping we could use: {key: default_collate([d[key] for d in batch]) for key in elem} # (but I think is slower)
        # Transpose the Batched Input
        batched_ids, batched_text, interaction_types, labels = zip(*data_batch)

        batched_examples = self.tokenizer.batch_encode_plus(
                                      list(batched_text),
                                      add_special_tokens = True,
                                      max_length = self.config.max_seq_length,
                                      return_token_type_ids = False,
                                      padding = True, # 'max_length', # We can use this parameter if we want to padd to the maximum model size
                                      return_attention_mask=True,
                                      return_tensors='pt',
                                      truncation = 'longest_first'
                                        )
        
        # Resolve Label
        if labels[0] is None: # We could be working with unlabeled data
            labels = None
        else:
            labels = torch.tensor(labels).long()        

        return {
            'batched_ids': batched_ids,
            'input_ids': batched_examples['input_ids'],
            'attention_mask': batched_examples['attention_mask'],
            'interaction_types': torch.tensor(interaction_types).long(),         
            'labels': labels  
        }    
    
    def _Balanced_sampler(self, num_samples: int = None, replacement: bool = False, generator = None):
        '''
        This sampler can be provided to the data_loader to produce balanced labeled samples.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to retrieve. If None, then the sampler will retrieve the minimum class number times the number of classes. The default is None.
        replacement : bool, optional
            Whether to sample with replacement. The default is False.

        Returns
        -------
        torch.utils.data.WeightedRandomSampler.

        '''
        if num_samples is None:
            finalN = self.__len__() if replacement else int(self.label_info['class_count'] * self.label_info['min_class_count'])
        else:
            finalN =  num_samples
        return WeightedRandomSampler(weights = self.weights, num_samples = finalN, replacement = replacement, generator=generator)

    



################################ User level Classification Dataset ################################
@dataclass
class StanceDatasetConfig:
    """
    Configuration for the preprocessing of the stance dataset.
    """
    user_file_name: str = field(
        metadata={"help": "Path to file containing stance dataset."}
    )
    tokenizer: PreTrainedTokenizer =  field(
                    metadata={"help": "Transformer Tokenizer."}
    )    
    max_seq_len: int =  field(
        default = None, metadata={"help": "Maximum sequence lenght for the model."}
    )
    interaction_categories: list =  field(
        default_factory = ['<cls>', '<pad>', 'Original', 'Quote', 'Reply', 'Retweet'], metadata={"help": "Possible tweet types, based on their interaction (eg: 'Original', 'Reply', etc.). Must include '<pad>' and <cls>' token."}
    )
    max_tw_per_user: int =  field(
        default = 15, metadata={"help": "Maximum number of tweets included in the user model."}
    )
    label_column: str =  field(
        default = None, metadata={"help": "Column name in Stance Dataset file that contains the user stance label."}
    )
    
    SEED: int =  field(
        default = None, metadata={"help": "Seed to use for tweet sampling."}
    )    
    def __post_init__(self):
        # Create Tweet Pad and CLS token IDs
        try:
            self.tweet_pad_id = self.interaction_categories.index('<pad>')
            self.tweet_cls_id = self.interaction_categories.index('<cls>')
            self.retwet_id = self.interaction_categories.index('Retweet')
        except ValueError:
            print('"interaction_cats" must include a "<pad>" and "<cls>" token')        
        
        if self.max_seq_len is None:    self.max_seq_len = self.tokenizer.model_max_length

            
class StanceDataset(Dataset):
    def __init__(self, config: StanceDatasetConfig):

        self.tokenizer = config.tokenizer 
        self.config = config
        
        # Load Dataframe
        User_DF = pd.read_csv(config.user_file_name, dtype = {'tweet_id': 'Int64', 'user_id': str})
        # Encode Interaction Type
        User_DF['interaction_type'] = pd.Categorical(User_DF.interaction_type, categories = config.interaction_categories)
        User_DF['interaction_type'] = User_DF.interaction_type.cat.codes
        
        # Preprocess the DataFrame so that we can keep at most max_tw_per_user tweets per each user.
        self.User_DF = User_DF.sort_values(['user_id', 'tweet_id'], ascending=[True, True]).reset_index(drop = True)
        # Manage Label Column
        if config.label_column is not None:
            self.User_DF = self.User_DF.rename(columns = {config.label_column :'user_stance'})
            users = self.User_DF.groupby('user_id').user_stance.first().reset_index()
            # Get User Counts per Label
            uStance = users.groupby('user_stance').count().reset_index().rename(columns = {'user_id': 'counts'})
            uStance['weights'] = len(users) / uStance['counts']
            users = pd.merge(users, uStance[['user_stance', 'weights']], on = 'user_stance')
            
            self.label_info = {'class_count': len(uStance), 'min_class_count': uStance.counts.min()}
            self.users = users.user_id.values
            self.weights = torch.DoubleTensor(users.weights.values)
            del(users)
        else:
            self.User_DF['user_stance'] = None
            # Get user weights for balanced labeled 
            self.users = list(set(User_DF.user_id))
            self.weights = None
           
    def __len__(self):
        return len(self.users)   
    
    def _Balanced_sampler(self, num_samples: int = None, replacement: bool = False, generator = None):
        '''
        This sampler can be provided to the data_loader to produce balanced labeled samples.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to retrieve. If None, then the sampler will retrieve the minimum class number times the number of classes. The default is None.
        replacement : bool, optional
            Whether to sample with replacement. The default is False.

        Returns
        -------
        torch.utils.data.WeightedRandomSampler.

        '''
        if num_samples is None:
            finalN = self.__len__() if replacement else int(self.label_info['class_count'] * self.label_info['min_class_count'])
        else:
            finalN =  num_samples
        return WeightedRandomSampler(weights = self.weights, num_samples = finalN, replacement = replacement, generator=generator)
                
    
    def _Stance_datacollator(self, data_batch):
        ''' 
        Collate Batches of observations given by this dataset. The purpose of this is to reduce the size of the input tensors to the maximum sequence length and 
        tweet number per user seen in the batch instead of the model maximum.
            data_batch: List of Dictionaries as produced by self.__getitem__
        '''      
        # If __getitem__ returned a mapping we could use: {key: default_collate([d[key] for d in batch]) for key in elem} # (but I think is slower)
        # Transpose the Batched Input
        batched_ids, temp_in_ids, temp_att_mask, interaction_types, tweet_masks, labels, max_lengths, tweet_numbers = zip(*data_batch)
        
        # Get the max number of tweets and their max sequence length seen in the batch
        batch_tw_len, batch_tweet_num, batch_size = max(max_lengths), max(tweet_numbers), len(batched_ids)
        # Concatenate the inputs
        input_ids = torch.cat(temp_in_ids) # size() = [batch_size, max_tw_per_user, max_seq_len]
        attention_mask = torch.cat(temp_att_mask) # size() = [batch_size, max_tw_per_user, max_seq_len]
        interaction_types = torch.cat(interaction_types) # size() = [batch_size, max_tw_per_user]
        tweet_masks = torch.cat(tweet_masks) # size() = [batch_size, max_tw_per_user]
        
        # Resolve Label
        if labels[0] is None: # We could be working with unlabeled data
            labels = None
        else:
            labels = torch.tensor(labels).long()
            
        if batch_tw_len < self.config.max_seq_len: # Reduce sequence size to batch maximum if necessary
            # Reduce Input IDs
            # Create container of reduced size
            temp_in_ids = torch.zeros(input_ids.shape[0], batch_tweet_num, batch_tw_len) # size() = [batch_size, batch_tweet_num, batch_tw_len]     
            temp_in_ids[:, :, :].copy_(input_ids[:, :batch_tweet_num, :batch_tw_len])
            input_ids = temp_in_ids.long()           
            # Reduce Masks
            temp_att_mask = torch.zeros(attention_mask.shape[0], batch_tweet_num, batch_tw_len) # size() = [batch_size, batch_tweet_num, batch_tw_len]    
            temp_att_mask[:, :, :].copy_(attention_mask[:, :batch_tweet_num, :batch_tw_len])
            attention_mask = temp_att_mask.long()            
            
        if batch_tweet_num < self.config.max_tw_per_user: # Reduce tweet number dimension to batch maximum if necessary
            # Reduce Interaction Types
            temp_inter_type = torch.zeros(interaction_types.shape[0], batch_tweet_num) # size() = [batch_size, batch_tweet_num]
            temp_inter_type[:, :].copy_(interaction_types[:, :batch_tweet_num])
            interaction_types = temp_inter_type.long()
            # Reduce Tweet Mask
            temp_tw_mask = torch.zeros(tweet_masks.shape[0], batch_tweet_num) # size() = [batch_size, batch_tweet_num]
            temp_tw_mask[:, :].copy_(tweet_masks[:, :batch_tweet_num])
            tweet_masks = temp_tw_mask.long()            
            
        return {
            'batched_ids': batched_ids,
            'input_ids': input_ids.reshape(batch_size * batch_tweet_num, batch_tw_len).contiguous(), # size() = [batch_size * batch_tweet_num, batch_tw_len] 
            'attention_mask': attention_mask.reshape(batch_size * batch_tweet_num, batch_tw_len).contiguous(), # size() = [batch_size * batch_tweet_num, batch_tw_len],
            'interaction_types': interaction_types,
            'tweet_masks': tweet_masks,
            'labels': labels,            
        }        

    def __getitem__(self, item):
        # Items will be User IDs
        user_id = self.users[item]
        uDF = self.User_DF.loc[self.User_DF.user_id == user_id]
        
        # Get Subsample of User DataFrame if its longer than max_tw_per_user
        if len(uDF) > self.config.max_tw_per_user:
            uDF = uDF.sample(n = self.config.max_tw_per_user, random_state = self.config.SEED).sort_index()

        # Prepare Label
        target = uDF.user_stance.head(1).item()
        target = torch.tensor(target, dtype=torch.long) if target is not None else None

        # Get Interaction Types
        interaction_types = torch.tensor(uDF.interaction_type.to_numpy(), dtype=torch.long)

        # Tokenize the text
        examples = self.tokenizer.batch_encode_plus(
                                      uDF.text.to_list(),
                                      max_length = self.config.max_seq_len,
                                      return_token_type_ids = False,
                                      padding = 'max_length', # We can use this parameter if we want to padd to the maximum model size
                                      return_attention_mask=True,
                                      return_tensors='pt',
                                      truncation = 'longest_first'
                                        )
        input_ids, attention_mask = examples['input_ids'], examples['attention_mask']
        max_tweet_length = attention_mask.sum(axis = 1).max().item()
        num_user_tweets = len(uDF)
        # Define Tweet Masks
        tweet_mask = torch.ones((1, self.config.max_tw_per_user), dtype=torch.long) # size() = [1, max_tweet_num]
        len_to_pad = self.config.max_tw_per_user - num_user_tweets

        if len_to_pad > 0: ## Append Padding to user with less than necessary tweets
            # Pad tokens
            inp_temp = torch.ones((len_to_pad, input_ids.shape[1]), dtype=torch.long) * self.tokenizer.pad_token_id
            # Pad inputs
            input_ids = torch.cat([input_ids, inp_temp]) # size() = [1, max_tw_per_user, max_seq_len]
            # Pad attention
            attention_mask = torch.cat([attention_mask, torch.zeros((len_to_pad, attention_mask.shape[1]), dtype=torch.long)]) # size() = [1, max_tw_per_user, max_seq_len]
            # Pad interaction type
            interaction_types = torch.cat([interaction_types, torch.ones((len_to_pad), dtype=torch.long) * self.config.tweet_pad_id]) # size() = [1, max_tw_per_user]
            # Mask padding Tweets
            tweet_mask[:, len(uDF):] = 0
        # Unsqueeze Tensors
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        interaction_types = interaction_types.unsqueeze(0)
        return user_id, input_ids, attention_mask, interaction_types, tweet_mask, target, max_tweet_length, num_user_tweets


class EmbeddingDataset(Dataset):
    '''
    Dataset used when working with pretrained user embeddings resulting from a User_Stance_Classifier
    '''
    def __init__(self, user_labels_file, embedding_file, label_column = 'true_label'):
        self.User_DF = pd.read_csv(user_labels_file, dtype = {'user_id': str})
        userEmbeddings = torch.load(embedding_file, weights_only=False)
        
        # Manage Label Column
        if label_column is not None:
            self.User_DF = self.User_DF.rename(columns = {label_column :'user_stance'})            
        else:
            self.User_DF['user_stance'] = None
           
        # Define parameters
        self.ids = self.User_DF.user_id.values
        self.labels = self.User_DF.user_stance.values
        self.embeddings = userEmbeddings
           
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, item):
        return self.ids[item], self.embeddings[item], self.labels[item]    

    def _embedding_datacollator(self, data_batch): # Could produce problems of memory overflow in multiprocessing (not sure if the self reference will cause duplication of the dataset) 
        # TODO: if it doest work for multiprocessing make it take tokenizer as a parameter
        ''' 
        Collate Batches of observations given by this dataset. 
            data_batch: List of Dictionaries as produced by self.__getitem__
        '''
        batched_ids, batched_embs, labels = zip(*data_batch)

        return {
            'batched_ids': batched_ids,
            'user_embeddings': torch.stack(batched_embs, dim = 0), 
            'labels': torch.tensor(labels).long()  
        }

