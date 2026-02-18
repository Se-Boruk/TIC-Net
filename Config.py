import os


#Basic params and variabels
########################################################
DATABASE_RAW_PATH = os.path.join("Database", "raw")
DATABASE_PATH = os.path.join("Database", "processed")

RANDOM_STATE = 243

TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.15
TEST_SPLIT = 0.1

DEFAULT_IMG_SIZE = 224

SOURCE_MAP = {
              "coco": 0,
              "ade20k": 1,
              "flickr30k": 2
              }

SPLIT_HASHES = {
    "train": "7686595d368b2c9647cc9c77e168d700c0ead13e808de647ac09af2864108f53",
    "val": "fa6f29ac8d62881053f9d57f5c65a94405f96644a28d5b95736f244d9ce2a2f7",
    "test": "dfad87ad4a698f84bcdfaa1bd61c833e9bfbfe7d9f6d5d693e30db1c115a3d9d"
}


#Training hyperparams
########################################################

#0 order parameters
EPOCHS = 100
BATCH_SIZE = 64
N_WORKERS = 5
MAX_QUEUE = 10


#1st order parameters
LR = 1e-4               #Learning rate... no need to explain
BASE_FILTERS_CNN = 64   #Base n of filters for the network CNN
HIDDEN_DIM_LSTM = 512   #Hidden dim of lstm
VOCAB_SIZE = 14859       #Size of the vocabulary
TOKEN_DIM = 512         #Length of the token encoding (single token length after processing)
LATENT_SPACE = 512      #Length of vector models are producing
SEQUENCE_LENGTH = 96   #Numbers of tokens coming into the text encoder

LSTM_DEPTH = 3

TRAIN_SET_FRACTION = 1   #Fraction of the train set which wll be used in the epoch
LOSS_MARGIN = 0.6           #How far the negatives should be pushed
PATIENCE = 5







