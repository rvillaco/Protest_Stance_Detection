# select batch size
BATCH_SIZE = 128
LAYER_ONE_NEIGHBORS = 25 #orig 25
LAYER_TWO_NEIGHBORS = 15
ATTN_HEADS = 4
EPOCHS = 10
ORDER = 'complete'
HOMO = True
RT_ONLY = False
RP_ONLY = False
BALANCED_DATA = False

if RT_ONLY == True:
    MODEL_FOLDER = "../models/homo_rt_only/"
elif RP_ONLY ==True:
    MODEL_FOLDER = "../models/homo_rp_only/"
else:
    MODEL_FOLDER = "../models/homo_full_net/"

if RT_ONLY == True:
    DATA_OUT = "homo_rt_only"
elif RP_ONLY ==True:
    DATA_OUT = "homo_rp_only"
else:
    DATA_OUT = "homo_full_net"
