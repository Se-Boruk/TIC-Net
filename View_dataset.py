###################################################################
# ( 0 ) Libs and dependencies 
###################################################################
from tqdm import tqdm
import os
import warnings
import textwrap
import csv
import numpy as np

from transformers import logging as transformers_logging

import torch
from torchinfo import summary
import torch.optim as optim

#My libs
from DataBase_functions import Custom_DataSet_Manager
from DataBase_functions import Async_DataLoader
import DataBase_functions as d_func

import Tokenizer_lib as Tok_lib
import Functions as functions
import Architectures as Arch
import Config
import matplotlib.pyplot as plt
###################################################################
# ( 1 ) Hardware setup
###################################################################

print("\nSearching for cuda device...")
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print available GPUs
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")



# Silence the "letters" and diagnostic noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

###################################################################
# ( 2 ) Hyperparameters
###################################################################

TRAIN_SPLIT = Config.TRAIN_SPLIT
VAL_SPLIT = Config.VAL_SPLIT
TEST_SPLIT = Config.TEST_SPLIT

RANDOM_STATE = Config.RANDOM_STATE
DATABASE_PATH = Config.DATABASE_PATH


#0 order parameters
EPOCHS = Config.EPOCHS
BATCH_SIZE = Config.BATCH_SIZE
N_WORKERS = Config.N_WORKERS
MAX_QUEUE = Config.MAX_QUEUE



#1st order parameters
LR = Config.LR     
BASE_FILTERS_CNN = Config.BASE_FILTERS_CNN    
HIDDEN_DIM_LSTM = Config.HIDDEN_DIM_LSTM
VOCAB_SIZE = Config.VOCAB_SIZE      
TOKEN_DIM = Config.TOKEN_DIM         
LATENT_SPACE = Config.LATENT_SPACE     
SEQUENCE_LENGTH = Config.SEQUENCE_LENGTH
LSTM_DEPTH = Config.LSTM_DEPTH


LOSS_MARGIN = Config.LOSS_MARGIN       
TRAIN_SET_FRACTION = Config.TRAIN_SET_FRACTION
PATIENCE = Config.PATIENCE


###################################################################
# ( 3 ) Loading data
###################################################################

#Load manager and execute
manager = Custom_DataSet_Manager(DataSet_path = DATABASE_PATH,
                                 train_split = TRAIN_SPLIT,
                                 val_split = VAL_SPLIT,
                                 test_split = TEST_SPLIT,
                                 random_state = RANDOM_STATE
                                 )


#Load dataset
Train_set, Val_set, Test_set = manager.load_dataset_from_disk()

#Verify stratified splits (different datasets could cause not optimal learning. Theyre stratified by the source)
print("="*60)
print("\nStratification test:")
d_func.verify_splits(Train_set, Val_set, Test_set)


print("="*60)
print("\nHash verification if dataset splits are the same across runs:")
#d_func.verify_dataset_integrity(Train_set, Val_set, Test_set, Config.SPLIT_HASHES)


###################################################################
# ( 4 ) Model creation, dataloader preparation
###################################################################

# Training loader
train_loader = Async_DataLoader(dataset = Train_set,
                                batch_size=BATCH_SIZE,
                                sequence_length = SEQUENCE_LENGTH,
                                num_workers=N_WORKERS,
                                device='cuda',
                                max_queue=MAX_QUEUE,
                                image_augmentation = True,
                                fraction = TRAIN_SET_FRACTION
                                )



train_loader.start_epoch(shuffle=True)

batch = train_loader.get_batch()


Tokenizer = Tok_lib.SimpleTokenizer("vocab.json", "lemma_map.json", SEQUENCE_LENGTH)


def show_image(batch, Tokenizer, i):


    image = batch['image_original'][i]
    image_aug = batch['image_augmented'][i]
    
    caption_p = batch['caption_positive'][i]
    caption_n = batch['caption_negative'][i]
    origin = batch['origin'][i]
    
    
    
    print("="*50)
    print("Caption positive:")
    print(caption_p)
    
    caption_p_trans = Tokenizer.decode(caption_p)
    print("Caption enc+dec:")
    print(caption_p_trans)
    
    
    print("="*50)
    print("Caption positive:")
    print(caption_n)
    
    caption_n_trans = Tokenizer.decode(caption_n)
    print("Caption enc+dec:")
    print(caption_n_trans)
    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 1. Move to CPU and Permute to (H, W, C)
    image_aug_np = image_aug.permute(1, 2, 0).cpu().numpy()
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    # 2. Reverse the Z-score: (Z * std) + mean
    # NumPy broadcasting handles the channel-wise operation automatically
    image_aug_unnorm = (image_aug_np * std) + mean
    image_unnorm = (image_np * std) + mean
    
    # 3. Clip to [0, 1] range to remove floating point artifacts
    image_aug_final = np.clip(image_aug_unnorm, 0, 1)
    image_final = np.clip(image_unnorm, 0, 1)
    
    
    
    
    
    # Set the maximum width for the caption part
    wrapper = textwrap.TextWrapper(width=70) 
    
    def format_title(label, caption):
        wrapped_caption = wrapper.fill(text=caption)
        return f"{label}\n{wrapped_caption}"
    
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Origin: {origin}")
    # Positive subplot
    plt.subplot(1, 2, 1)
    plt.imshow(image_final)
    plt.title(format_title("POSITIVE CASE", caption_p_trans), 
              fontsize=10, fontweight='bold', pad=20, loc='center')
    plt.axis('off')
    
    # Negative subplot
    plt.subplot(1, 2, 2)
    plt.imshow(image_aug_final)
    plt.title(format_title("NEGATIVE CASE", caption_n_trans), 
              fontsize=10, fontweight='bold', pad=20, loc='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()




for i in range(7):
    show_image(batch, Tokenizer, i)









