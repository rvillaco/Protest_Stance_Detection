# -*- coding: utf-8 -*-
"""
"""

import torch
import torch.nn as nn
from transformers import RobertaModel
from .ModelConfiguration import RoBERTaEncoderConfig, ModelEmbeddingsConfig, UserEncoderConfig


##################### Define User Classification Modules #####################

#### Tweet Text Embedder ####
class FastText_Embedder(nn.Module):
    ''' Gets tweet embeddings by averaging the unit vectors of the words (a la FastText embeddings)'''
    def __init__(self, pad_idx, pretrained_embedding = None, vocab_size = None, embedding_dim = None, 
                  freeze_embedding = False, dropout = 0.1):      
        super().__init__()
        
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.hidden_size = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding,
                                                          padding_idx = pad_idx)
        else:
            assert (vocab_size is not None) and (embedding_dim is not None), 'If a pretrained embedding is not provided, an embedding and vocab dimension must be provided.'
            self.vocab_size, self.hidden_size = vocab_size, embed_dim
            self.embedding = nn.Embedding(num_embeddings = vocab_size,
                                          embedding_dim = embed_dim,
                                          padding_idx = pad_idx,
                                          max_norm = 5.0)
        
    def forward(self, input_ids, attention_mask):

        embedded = self.embedding(input_ids) # embedded = [batch size, tw_per_user, sent len, emb dim]
        
        # Get the Norm of the of the vectors
        normW = torch.linalg.norm(embedded, ord = 2, dim = 3, keepdim = True)
        mask = attention_mask.unsqueeze(-1) > 0
        normW = normW.data.masked_fill_(~mask, float('inf'))
        
        # Make embeddings unit vectors and sum them (for the sentence embedding)
        embedded = (embedded.div(normW)).sum(axis = 2) # embedded = [batch size, tw_per_user, emb dim]
        
        # Get the average sentence embeddings by dividing them by the sequence length
        seq_len = attention_mask.sum(axis = 2)
        seq_len[seq_len == 0] = float('inf')
        embedded = embedded.div(seq_len.unsqueeze(-1))     
        
        return embedded
   
class RoBETO_Embedder(nn.Module):
    def __init__(self, config: RoBERTaEncoderConfig):      
        super().__init__()
        self.RoBETO = RobertaModel.from_pretrained(config.Model_Dir, add_pooling_layer = False)
        
        self.hidden_size = self.RoBETO.config.hidden_size  
        self.FC = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.xavier_uniform_(self.FC.weight)
        
        self.activation = getattr(nn, config.activation)()
        self.dropout = nn.Dropout(config.dropout)

        if config.freeze_bert_embeddings:
            self.freeze_embeddings()
        
    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.RoBETO(
          input_ids = input_ids,
          attention_mask = attention_mask
        )['last_hidden_state'] # (batch_size, input_len, hidden_size)        
        
        first_token_tensor = last_hidden_state[:, 0, :]
        first_token_tensor = self.dropout(first_token_tensor)
        pooled_output = self.FC(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        
        return pooled_output
    
    def freeze_embeddings(self):
        for param in self.RoBETO.parameters():
              param.requires_grad = False    
              
              
#### Tweet Level Embedder ####
class Model_Embeddings_v3(nn.Module):
    ''''''
    def __init__(self, config: ModelEmbeddingsConfig):      
        super().__init__()
        
        self.mask_embeddings = config.mask_embeddings
        # Get Sentence Embeddings
        if config.tweet_encoder_config.model_type == 'FastText':
            self.tweet_embedder = FastText_Embedder(config.tweet_encoder_config)
        elif config.tweet_encoder_config.model_type == 'RoBERTa':
            self.tweet_embedder = RoBETO_Embedder(config.tweet_encoder_config)
        self.embedding_dim = self.tweet_embedder.hidden_size
        
        # Create CLS parameter that will encapsulate a Users embedding based on her tweets
        self.cls_idx = config.cls_idx
        
        self.max_tweet_number = config.max_tweet_number + 1
        
        # Define Tweet Type Embeddings
        self.type_embedding = nn.Embedding(num_embeddings = config.tweet_type_number,
                                           embedding_dim = self.embedding_dim,
                                           padding_idx = config.pad_idx,
                                           max_norm = 1.0)     

        # Define Tweet Type Embeddings
        # TODO: Could be worth persuing time embeddings (based on minutes or hours) extrapolated via sin or cos functions (discuss with Evan)
        self.position_embedding = nn.Embedding(num_embeddings = self.max_tweet_number,
                                               embedding_dim = self.embedding_dim,
                                               max_norm = 1.0)           
        
        self.LayerNorm = nn.LayerNorm(self.embedding_dim, eps = config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.dropout)        
               
    def forward(self, input_ids, attention_mask, interaction_types, tweet_masks):
        """
        Arguments:
        input_ids -- Tokenized batch of user tweets. Dimension: [batch_size * batch_tweet_number, max_batch_seq_length]
        attention_mask -- Batch of Input Padding Masks. Dimension: [batch_size * batch_tweet_number, max_batch_seq_length]
        interaction_types -- Interaction types of the tweets in the batch. Dimension: [batch_size, batch_tweet_number]
        tweet_masks -- Batch of Tweet Padding Masks. Dimension: [batch_size, batch_tweet_number]
        """
        batch_size, batch_num_tweets = interaction_types.size()
        device = input_ids.device
        
        ### Get Sentence Embeddings
        input_embeddings = self.tweet_embedder(input_ids, attention_mask).reshape(batch_size, batch_num_tweets, 
                                                                                  self.embedding_dim)# size() = [batch size, batch_tw_per_user + 1, emb dim]       
          # Add zero tensor to the first position
        zero_batch = torch.zeros(batch_size, 1, self.embedding_dim, device = device)
        input_embeddings = torch.cat((zero_batch, input_embeddings), dim =1)# size() = [batch size, batch_tw_per_user + 1, emb dim]
         
        ### Get embeddings of interaction types
          # Append CLS token to interaction types
        cls_batch = torch.ones(batch_size, 1, device = device, dtype = torch.long) * self.cls_idx
        inter_types = torch.cat((cls_batch, interaction_types), dim =1)
          # Embedd extended types
        type_embeddings = self.type_embedding(inter_types) # size() = [batch size, batch_tw_per_user + 1, emb dim]
        
        # Get possition embeddings
        input_shape = input_embeddings.size()
        position_ids = torch.arange(input_shape[1]).repeat(input_shape[0], 1).to(device)
        pos_embeddings = self.position_embedding(position_ids) # size() = [batch size, batch_tw_per_user + 1, emb dim]
        
        # Add embeddings and normalize
        embeddings = input_embeddings + pos_embeddings + type_embeddings
        
        # Mask Padding Tweets
        if self.mask_embeddings:
            embeddings = embeddings * tweet_masks.unsqueeze(-1)
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings) # size() = [batch size, batch_tw_per_user, emb dim]
        
        return embeddings

#### User Encoder ####
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class User_Encoder(nn.Module):
    def __init__(self, config: UserEncoderConfig):
        super(User_Encoder, self).__init__()
        self.initializer_range = config.initializer_range
        # if config.model_embedder_version == 'v1':
        #     self.tweet_encoder = Model_Embeddings_v1(config.model_embeddings_config)
        # elif config.model_embedder_version == 'v2':
        #     self.tweet_encoder = Model_Embeddings_v2(config.model_embeddings_config)
        if config.model_embedder_version == 'v3':
            self.tweet_encoder = Model_Embeddings_v3(config.model_embeddings_config)            
        else:
            raise Exception('Only v1, v2 and v3 of Model_Embeddings is implemented.')
            
        self.embedding_dim = self.tweet_encoder.embedding_dim
        encoder_layers = TransformerEncoderLayer(self.embedding_dim, config.num_attention_heads, config.intermidiate_size, batch_first = True, 
                                                 activation = config.transformer_activation, dropout = config.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.num_encoder_layers)
        
        self.FC = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_uniform_(self.FC.weight)
        
        self.dropout = nn.Dropout(config.dropout)
        self.activation = getattr(nn, config.user_activation)()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std = self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, interaction_types, tweet_masks):
        """
        Arguments:
        input_ids -- Tokenized batch of user tweets. Dimension: [batch_size * batch_tweet_number, max_batch_seq_length]
        attention_mask -- Batch of Input Padding Masks. Dimension: [batch_size * batch_tweet_number, max_batch_seq_length]
        interaction_types -- Interaction types of the tweets in the batch. Dimension: [batch_size, batch_tweet_number]
        tweet_masks -- Batch of Tweet Padding Masks. Dimension: [batch_size, batch_tweet_number]
        """
        batch_size, batch_num_tweets = interaction_types.size()
        device = input_ids.device

        # Add the cls token to the tweet_masks (this is ugly) 
        # TODO: I should probably change the Stance_Dataset so that the CLS token is included in the interaction_types and the mask is updated there (then I could remove this and the iter_type part in the Model_Embedder_v*). This does make the input dimension of the input_ids to be different from the other two tensors (which is still ugly).
        one_batch = torch.ones(batch_size, 1, device = device, dtype = torch.long)
        tm = torch.cat((one_batch, tweet_masks), dim =1)
        
        # Encode Batch of Tweets per User
        x = self.tweet_encoder(input_ids, attention_mask, interaction_types, tm) # size() = [batch size, batch_tw_per_user + 1, emb dim]
        
        output = self.transformer_encoder(x, src_key_padding_mask = tm) # size() = [batch size, batch_tw_per_user, emb dim]
        first_token_tensor = output[:, 0, :] # size() = [batch size, emb dim] # First Token contains CLS parameter
        first_token_tensor = self.dropout(first_token_tensor)
        
        pooled_output = self.FC(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

#### Classification Head ####
class User_Stance_Classifier(nn.Module):
    def __init__(self, num_classes, user_config: UserEncoderConfig):
        super(User_Stance_Classifier, self).__init__()
        self.user_embedder = User_Encoder(user_config)
        self.out_fc = nn.Linear(self.user_embedder.embedding_dim, num_classes)
        nn.init.xavier_uniform_(self.out_fc.weight)
        
    def forward(self, input_ids, attention_mask, interaction_types, tweet_masks):
        """
        Arguments:
        input_ids -- Tokenized batch of user tweets. Dimension: [batch_size * batch_tweet_number, max_batch_seq_length]
        attention_mask -- Batch of Input Padding Masks. Dimension: [batch_size * batch_tweet_number, max_batch_seq_length]
        interaction_types -- Interaction types of the tweets in the batch. Dimension: [batch_size, batch_tweet_number]
        tweet_masks -- Batch of Tweet Padding Masks. Dimension: [batch_size, batch_tweet_number]
        """               
        user_emb = self.user_embedder(input_ids, attention_mask, interaction_types, tweet_masks)
        output = self.out_fc(user_emb)
        return output, user_emb
