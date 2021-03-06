import torch

DATA_PATH = "C:/Moein/AI/Datasets/Flicker-8k" 
DF_PATH = "C:/Moein/AI/Datasets/Flicker-8k" 
# DATA_PATH = "/kaggle/input/flickr8k"
# DF_PATH = "/kaggle/working/Image-Captioning/input"
BATCH_SIZE = 8
LR = 1e-4
NUM_WORKERS = 0
FREQ_THRESHOLD = 5
SIZE = 224
MODEL_NAME = 'resnet18'
PRETRAINED = True
ATTENTION_DIM = 128 # 256
EMBED_DIM = 128 # 256
DECODER_DIM = 256 # 512
ENCODER_DIM = 512
FEATURES_GRID_SIZE = 7
N_HEAD = 8
N_ENC_LAYERS = 6
N_DEC_LAYERS = 6
DIM_FF = 2048
DROPOUT = 0.3
ENCODER_LR = 1e-4
DECODER_LR = 4e-4
FACTOR = 0.5
PATIENCE = 3
EPOCHS = 5
MAX_GRAD_NORM = 5
MAX_LEN_PRED = 20 # check the dataframe
# VOCAB_SIZE = 2994
MAX_LEN_TRANFORMER = 100
PAD_TOKEN_ID = 0
D_MODEL = 512
LR_SCHEDULER = "ReduceLROnPlateau"
STEP = "epoch"
MODEL_PATH = "."
MODEL_SAVE_NAME = "best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")