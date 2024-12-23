import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 333
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True # whether use cosineLR or not
n_channels = 3
n_labels = 1

epochs = 200
#epochs = 2000
img_size = 256
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 10

pretrain = False
#task_name = 'InstrumentsSeg' # GlaS MoNuSeg
task_name = 'InstrumentsSeg'
#task_name = 'TNK3'
learning_rate = 1e-3
batch_size = 16

model_name = 'CFFANet_2'
#model_name = 'DCFFANet'
#model_name = 'ABCNet'
#model_name = 'DMCSNet'
#model_name = 'MANet'
#model_name = 'UCTransNet'
#model_name = 'UCTransNet_pretrain'

train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Test_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'
session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config

# used in testing phase, copy the session name in training phase
#test_session = "Test_session_04.01_17h40"  # dice_pred 0.9475994576632925 CFFANet_1

#test_session = "Test_session_04.13_22h23"    # CFFR-Net dice_pred 0.9500377699085027

#test_session = "Test_session_04.23_18h44"   # dice_pred 0.9512527143729314 learning rate: 1e-5

test_session = "Test_session_05.21_17h59"  # 0.96790.9679

#test_session = "Test_session_05.28_19h10"

#test_session = "Test_session_06.11_16h22" # CASE 4

#test_session = "Test_session_06.04_11h43"

#test_session = "Test_session_06.12_15h01"   FINAL MODEL ACCURACIESF FROM THIS

#test_session = 'Test_session_06.12_22h13'  # CASE 5