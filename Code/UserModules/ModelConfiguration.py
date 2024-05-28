# -*- coding: utf-8 -*-
"""
"""
from dataclasses import dataclass, field

ENCODER_TYPES = ['RoBERTa', 'FastText', 'CNN']

##################### Define Configuration Modules #####################
class TweetEncoderConfig:
    """
    Configuration for the model used to encode a user's tweets.
    """    
    def __init__(self, model_type: str):
        self.model_type = model_type
        assert self.model_type in ENCODER_TYPES, 'Only the following encoders have been implemented: ' + ", ".join(ENCODER_TYPES)    
             
    
class FastTextEncoderConfig(TweetEncoderConfig):
    # TODO: Implement the config
    pass

class CNN_EncoderConfig(TweetEncoderConfig):
    # TODO: Implement the config
    pass
    
class RoBERTaEncoderConfig(TweetEncoderConfig):
    """
    Configuration for the model used to encode a user's tweets.
    """   
    def __init__(self, Model_Dir: str, dropout: float = 0.1, activation: str = 'Tanh', freeze_bert_embeddings: bool = True):
        self.Model_Dir = Model_Dir # Directory with the Pretrained RoBERTa Model Weights.
        self.dropout = dropout # Dropout rate used on the pooled model embeddings.
        self.activation = activation # Activation function used after pooling (must be a member of torch.nn).
        self.freeze_bert_embeddings = freeze_bert_embeddings # Freeze BERT embeddings.
        super().__init__(model_type = 'RoBERTa')
        
@dataclass
class ModelEmbeddingsConfig:
    """
    Configuration for the model used to encode a user's tweets.
    """
    tweet_encoder_config: TweetEncoderConfig = field(
        metadata={"help": "Configuration for the Tweet Encoder."}
    )
    cls_idx: int =  field(
        default = 0, metadata={"help": "Index of the padding token user at the tweet level (not necessarily the same as the one used at a sequence level)."}
    )    
    pad_idx: int =  field(
        default = 1, metadata={"help": "Index of the padding token user at the tweet level (not necessarily the same as the one used at a sequence level)."}
    )
    max_tweet_number: int =  field(
        default = 15, metadata={"help": "Maximum number of tweets allowed for a given user."}
    )
    tweet_type_number: int =  field(
        default = 6, metadata={"help": "Number of tweet types modeled (includes the pad and cls tokens)."}
    ) 
    dropout: float = field(
        default = 0.1, metadata={"help": "Dropout rate used for the combined embeddings."}
    )    
    layer_norm_eps: float = field(
        default = 1e-12, metadata={"help": "Stability parameter used on normalization layer."}
    )
    mask_embeddings: bool = field(
        default = True, metadata={"help": "Wether to multiply the embeddings with the tweet mask."}
    )


@dataclass
class UserEncoderConfig:
    """
    Configuration for the model used to encode a user's tweets.
    """
    model_embeddings_config: ModelEmbeddingsConfig = field(
        metadata={"help": "Configuration for the Tweet Model Embeddings."}
    )
    num_attention_heads: int =  field(
        default = 6, metadata={"help": "Number of attention heads used in a user Transformer Encoder Layer."}
    )
    intermidiate_size: int =  field(
        default = 2048, metadata={"help": "Dimension of the hidden feedforward layer in a user Transformer Encoder Layer."}
    )
    transformer_activation: str =  field(
        default = 'gelu', metadata={"help": "Activation function used in a user Transformer Encoder Layer."}
    )
    num_encoder_layers: int =  field(
        default = 3, metadata={"help": "Number of layers to use for the User transformers."}
    )    
    dropout: float = field(
        default = 0.1, metadata={"help": "Dropout rate used for the transformer layers."}
    )   
    initializer_range: float = field(
        default = 0.02, metadata={"help": "Maximum range for layer initialization."}
    )
    user_activation: str = field(
        default = 'Tanh', metadata={"help": "Activation used for the last user encoder layer."}
    )
    model_embedder_version: str = field(
        default = 'v1', metadata={"help": "Activation used for the last user encoder layer."}
    )   
