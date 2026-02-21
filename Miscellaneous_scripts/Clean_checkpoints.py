import torch
import os
import numpy as np
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Folder containing your .pth files
INPUT_FOLDER = os.path.join("Models", "Trained") 

# Folder to save the fixed files (creates if not exists)
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "Cleaned")

def sanitize_value(value):
    """
    Recursively converts NumPy scalars to Python native types.
    Leaves PyTorch tensors and other safe types alone.
    """
    # 1. Handle NumPy Floats (float32, float64)
    if isinstance(value, (np.floating, float)):
        return float(value)
    
    # 2. Handle NumPy Integers (int64, int32)
    elif isinstance(value, (np.integer, int)):
        return int(value)
    
    # 3. Handle Lists/Tuples (recurse)
    elif isinstance(value, list):
        return [sanitize_value(v) for v in value]
    elif isinstance(value, tuple):
        return tuple(sanitize_value(v) for v in value)
    
    # 4. Handle Dictionaries (recurse)
    elif isinstance(value, dict):
        return {k: sanitize_value(v) for k, v in value.items()}
        
    # 5. Return everything else as-is (Tensors, Strings, None)
    return value

def clean_checkpoints():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get list of .pth files
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".pth")]
    
    if not files:
        print("No .pth files found in the directory.")
        return

    print(f"Found {len(files)} checkpoints to clean.\n")

    for filename in tqdm(files, desc="Cleaning Checkpoints"):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        
        try:
            # 1. Load unsafe (trusted local file)
            # We use map_location='cpu' to avoid VRAM usage during cleaning
            checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
            
            # 2. Create a new clean dictionary
            clean_checkpoint = {}
            
            # 3. Iterate and Sanitize
            for key, value in checkpoint.items():
                # Skip state_dicts (they contain Tensors which are safe and huge)
                if 'state_dict' in key:
                    clean_checkpoint[key] = value
                else:
                    # Sanitize metrics like threshold, loss, epoch
                    clean_checkpoint[key] = sanitize_value(value)
            
            # 4. Save safely
            torch.save(clean_checkpoint, output_path)
            
        except Exception as e:
            print(f"\n[Error] Failed to process {filename}: {e}")

    print("\n" + "="*50)
    print(f"Done! Cleaned models are saved in:\n{OUTPUT_FOLDER}")
    print("You can now load these files with weights_only=True")
    print("="*50)

if __name__ == "__main__":
    clean_checkpoints()