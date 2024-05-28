# select batch size
BATCH_SIZE = 128
LAYER_ONE_NEIGHBORS = 25 #orig 25
LAYER_TWO_NEIGHBORS = 15 # orig 15
ATTN_HEADS = 1
EPOCHS = 10
BALANCED_DATA = False
# if complete, use first-order neighborhood + labeled nodes
# if anything else, only use labeled nodes.
ORDER = 'complete'
if BALANCED_DATA:
    MODEL_FOLDER = "../models/hetero_balanced_final/"
else:
    MODEL_FOLDER = "../models/hetero_imbalanced_final/"
if BALANCED_DATA:
    DATA_OUT = "refhetero_balpreds"
else:
    DATA_OUT = "refbalpreds"