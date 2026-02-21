import torch
import os

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Path to your BEST model (the one you want to submit)
INPUT_MODEL_PATH = os.path.join("Models", "Trained", "best_model.pth")

# 2. Path where the submission file should be saved
# (Name it exactly what the system expects, usually 'weights.pth')
OUTPUT_MODEL_PATH = "weights.pth" 

print(f"Processing: {INPUT_MODEL_PATH}")

# ==========================================
# EXTRACTION SCRIPT
# ==========================================
try:
    # 1. Load the full checkpoint (which contains 'epoch', 'optimizer', etc.)
    # We use map_location='cpu' to avoid needing a GPU for this simple task
    full_checkpoint = torch.load(INPUT_MODEL_PATH, map_location='cpu')
    
    # 2. Extract ONLY the model weights
    # The submission system fails because it sees keys like "epoch" 
    # instead of "txt_enc.embedding.weight" at the top level.
    if 'model_state_dict' in full_checkpoint:
        flat_weights = full_checkpoint['model_state_dict']
        print("Successfully extracted 'model_state_dict' from checkpoint.")
    else:
        # Fallback: Maybe it's already flat?
        print("Warning: 'model_state_dict' key not found. Assuming file is already flat.")
        flat_weights = full_checkpoint

    # 3. Save the FLAT dictionary
    # This creates a file that contains ONLY the weights, no metadata.
    torch.save(flat_weights, OUTPUT_MODEL_PATH)
    
    print(f"\nSuccess! Saved compatible weights to: {OUTPUT_MODEL_PATH}")
    print("You can now submit this file.")

except Exception as e:
    print(f"\n[ERROR] Failed to process file: {e}")