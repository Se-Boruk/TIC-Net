###################################################################
# ( 0 ) Libs and dependencies 
###################################################################
from tqdm import tqdm
import os
import warnings

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

# Validation loader
val_loader = Async_DataLoader(dataset = Val_set,
                                batch_size=BATCH_SIZE,
                                sequence_length = SEQUENCE_LENGTH,
                                num_workers=N_WORKERS,
                                device='cuda',
                                max_queue=MAX_QUEUE,
                                image_augmentation = True,
                                fraction = 1
                                )


#No augmentation: 3,5 min
#Image augmentation: 3,5min - 4min

###################################################################
# ( 5 ) Getting hyperparams, setting models
###################################################################

text_model = Arch.Text_encoder(vocab_size = VOCAB_SIZE,
                               word_dim = TOKEN_DIM,
                               hidden_dim = HIDDEN_DIM_LSTM,
                               embed_dim = LATENT_SPACE,
                               depth = LSTM_DEPTH
                               )

image_model = Arch.Image_encoder(embed_dim = LATENT_SPACE
                                 )


        
model = Arch.Siamese_model(Image_model = image_model,
                           Text_model = text_model,
                           device = device
                           )


optimizer = optim.AdamW(model.parameters(),
                        lr=LR,
                        betas=(0.9, 0.98),
                        weight_decay=0.01,     
                        eps=1e-8
                    )


criterion = functions.Custom_loss(margin = LOSS_MARGIN,
                                  triplet_weight=1.0, 
                                  contrastive_weight=2.0,
                                  temp=0.07
                                  )


###################################################################
# ( 5.5 ) Showing the models architecture and size
###################################################################


model.move_to_device("cpu")

print("\n" + "="*30 + " IMAGE ENCODER " + "="*30)
# We use print() to force it to show in the console/notebook output
print(summary(model.img_enc, 
              input_size=(BATCH_SIZE, 3, 224, 224), 
              col_names=["input_size", "output_size", "num_params"],
              device=device))

print("\n" + "="*30 + " TEXT ENCODER " + "="*30)
print(summary(model.txt_enc, 
              input_size=(BATCH_SIZE, SEQUENCE_LENGTH),
              dtypes=[torch.long], # Critical for Embedding layer
              col_names=["input_size", "output_size", "num_params"],
              device=device))




###################################################################
# ( 6 ) Training the model
###################################################################





# --- CONFIGURATION ---
CSV_LOG_PATH = "Plots/training_log.csv"
CHECKPOINT_PATH = "Models/Trained/best_model.pth"
CHECKPOINT_DIR = "Models/Trained/checkpoints"
best_val_loss = float('inf')
patience_counter = 0
SAVE_EVERY = 5
# Initialize CSV with expanded headers
if not os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "T_Loss", "T_B_Acc", "T_Recall", "T_Spec", 
            "V_Loss", "V_B_Acc", "V_Recall", "V_Spec", "Thresh"
        ])
        
###############

accumulation_steps = 8
import math
steps_per_epoch = math.ceil(train_loader.get_num_batches() / accumulation_steps)
total_updates = steps_per_epoch * EPOCHS

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    total_steps=total_updates,
    pct_start=0.2,  # 20% warmup
    div_factor=25,
    final_div_factor=1000
)


scaler = torch.amp.GradScaler(device) 

#Model preparation just for sanity
model.move_to_device()
model.train_mode()




print(f"Starting training on {device}!\n\n")

for e in range(EPOCHS):
    # ==========================================
    # 1. TRAINING PHASE
    # ==========================================
    model.train()
    model.train_mode()
    optimizer.zero_grad()
    train_loader.start_epoch(shuffle=True)
    
    train_epoch_loss = 0.0
    train_pos_scores = []
    train_neg_scores = []
    num_train_batches = train_loader.get_num_batches()
    c = 0 
    
    with tqdm(total=num_train_batches, desc=f"Epoch {e+1} [Train]", unit=" batch") as pbar:
        while True:
            batch = train_loader.get_batch()
            if batch is None: break
    
            batch_image = batch['image_original']
            batch_image_aug = batch['image_augmented']
            batch_caption_p = batch['caption_positive']
            batch_caption_n = batch['caption_negative']
    
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                v_main, v_aug, t_pos, t_neg = model(batch_image, batch_image_aug, batch_caption_p, batch_caption_n)
                
                # 1. Collect scores for threshold calibration BEFORE loss scaling
                # We use detach().cpu() to prevent memory leaks on the GPU
                train_pos_scores.append((v_main * t_pos).sum(dim=1).detach().cpu())
                train_neg_scores.append((v_main * t_neg).sum(dim=1).detach().cpu())
    
                # 2. Calculate and normalize loss
                loss = criterion(v_main, v_aug, t_pos, t_neg) / accumulation_steps
    
            # 3. Backward pass (scaled)
            scaler.scale(loss).backward()
            c += 1
            
            is_last_batch = (pbar.n + 1 == num_train_batches)
            
            # 4. Optimizer Step (Every N batches)
            if (c % accumulation_steps == 0) or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # SAFETY: Prevent scheduler from stepping past total_steps
                if scheduler.last_epoch < scheduler.total_steps:
                    scheduler.step()
                    
                c = 0 # Reset counter
    
            # 5. Metrics logging
            # Multiply by accumulation_steps for the "real" loss to show in tqdm
            current_loss = loss.item() * accumulation_steps
            train_epoch_loss += current_loss
            pbar.update(1)
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})


    # Calibrate threshold and calculate Train Metrics
    train_pos_scores = torch.cat(train_pos_scores).numpy()
    train_neg_scores = torch.cat(train_neg_scores).numpy()
    

    avg_train_loss = train_epoch_loss / num_train_batches

    # ==========================================
    # 2. VALIDATION PHASE
    # ==========================================
    model.eval()
    model.eval_mode()
    val_loader.start_epoch(shuffle=False)
    
    val_epoch_loss = 0.0
    val_pos_scores = []
    val_neg_scores = []
    num_val_batches = val_loader.get_num_batches()
    
    with torch.no_grad():
        with tqdm(total=num_val_batches, desc=f"Epoch {e+1} [Val]", unit=" batch") as pbar_val:
            batch_idx = 0
            while True:
                batch = val_loader.get_batch()
                if batch is None: break
                
                img = batch['image_original']
                aug = batch['image_augmented']
                pos = batch['caption_positive']
                neg = batch['caption_negative']
                
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    v_m, v_a, t_p, t_n = model(img, aug, pos, neg)
                    loss = criterion(v_m, v_a, t_p, t_n)
                    
                    val_pos_scores.append((v_m * t_p).sum(dim=1).cpu())
                    val_neg_scores.append((v_m * t_n).sum(dim=1).cpu())

                val_epoch_loss += loss.item()
                batch_idx += 1
                pbar_val.update(1)
                pbar_val.set_postfix({"avg_val_loss": f"{val_epoch_loss / batch_idx:.4f}"})

    val_pos_scores = torch.cat(val_pos_scores).numpy()
    val_neg_scores = torch.cat(val_neg_scores).numpy()
    
    current_threshold = functions.calibrate_threshold(val_pos_scores, val_neg_scores)
    
    t_b_acc, t_recall, t_spec = functions.calculate_metrics(train_pos_scores, train_neg_scores, current_threshold)
    v_b_acc, v_recall, v_spec = functions.calculate_metrics(val_pos_scores, val_neg_scores, current_threshold)
    
    avg_val_loss = val_epoch_loss / num_val_batches

    print(f"Epoch {e+1} Result:")
    print(f"T_Loss: {avg_train_loss:.4f} | T_B_Acc: {t_b_acc:.4f} | T_Rec: {t_recall:.4f} | T_Spec: {t_spec:.4f}")
    print(f"V_Loss: {avg_val_loss:.4f} | V_B_Acc: {v_b_acc:.4f} | V_Rec: {v_recall:.4f} | V_Spec: {v_spec:.4f} | Thresh: {current_threshold:.2f}")

    # ==========================================
    # 3. LOGGING (IRT)
    # ==========================================
    with open(CSV_LOG_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            e + 1, 
            f"{avg_train_loss:.4f}", f"{t_b_acc:.4f}", f"{t_recall:.4f}", f"{t_spec:.4f}",
            f"{avg_val_loss:.4f}", f"{v_b_acc:.4f}", f"{v_recall:.4f}", f"{v_spec:.4f}",
            f"{current_threshold:.2f}"
        ])
        f.flush()
        os.fsync(f.fileno())

    # ==========================================
    # 4. CHECKPOINTING & PERIODIC SAVING
    # ==========================================
    checkpoint_data = {
        'epoch': e + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'threshold': float(current_threshold),      
        'val_loss': float(avg_val_loss),             
        'val_balanced_acc': float(v_b_acc)          
        }

    # Condition A: Improvement (Save as Best)
    if avg_val_loss < best_val_loss:
        print(f"Improvement: Saving best model to {CHECKPOINT_PATH}")
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(checkpoint_data, CHECKPOINT_PATH)
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

    # Condition B: Periodic Save (Save every N epochs)
    if (e + 1) % SAVE_EVERY == 0:
        periodic_name = f"{CHECKPOINT_DIR}checkpoint_epoch_{e+1}.pth"
        print(f"Periodic Save: Saving checkpoint to {periodic_name}")
        torch.save(checkpoint_data, periodic_name)

    if patience_counter >= PATIENCE:
        print(f"Early Stopping. Best Val Loss: {best_val_loss:.4f}")
        break
    print("="*30)




"""
import matplotlib.pyplot as plt
i = 5

Tokenizer = Tok_lib.SimpleTokenizer("vocab.json", TOKEN_LENGTH)

origin = batch["origin"][i]
image_original = batch['image_original'][i]
image_augmented = batch['image_augmented'][i]

caption_positive_1 = Tokenizer.decode(batch['caption_positive_1'][i])
caption_positive_2 = Tokenizer.decode(batch['caption_positive_2'][i])
caption_negative = Tokenizer.decode(batch['caption_negative'][i])

print("="*50)
print("Caption positive 1:\n")
print(caption_positive_1)

print("="*50)
print("Caption positive 2:\n")
print(caption_positive_2)

print("="*50)
print("Caption negative:\n")
print(caption_negative)

image_augmented = image_augmented.permute(1,2,0).cpu().numpy()
image_original = image_original.permute(1,2,0).cpu().numpy()



plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image_original)

plt.subplot(1,2,2)
plt.title("Augmented")
plt.imshow(image_augmented)
"""



