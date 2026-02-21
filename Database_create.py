import os
import sys
import pandas as pd
import ast
import shutil
import gc
import math
from PIL import Image as PILImage
from datasets import Dataset, Features, Image, Value, Sequence, concatenate_datasets, load_from_disk
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
from Config import DATABASE_RAW_PATH, DATABASE_PATH, DEFAULT_IMG_SIZE
from DataBase_functions import preprocess_image_to_square

CACHE_DIR = os.path.join(DATABASE_RAW_PATH, "temp_shards")
BATCH_SIZE = 2000
NUM_WORKERS = 4

BASE_IMG_PATHS = {
    "coco": os.path.join(DATABASE_RAW_PATH, "coco", "images"),
    "flickr30k": os.path.join(DATABASE_RAW_PATH, "flickr30k", "images"),
    "ade20k": os.path.join(DATABASE_RAW_PATH, "ade20k", "images"),
}

DATASET_DIRS = {
    "coco": "coco",
    "flickr30k": "flickr30k",
    "ade20k": "ade20k" 
}

# --- HELPERS ---

def parse_captions(caption_string):
    try:
        if isinstance(caption_string, list): return caption_string
        return ast.literal_eval(caption_string)
    except (ValueError, SyntaxError):
        return [str(caption_string)]

def worker_task(args):
    """
    Function executed by worker processes.
    args: (df_chunk, images_dir, shard_idx, dataset_name, features)
    """
    df_chunk, images_dir, shard_idx, dataset_name, features = args
    shard_path = os.path.join(CACHE_DIR, f"shard_{dataset_name}_{shard_idx}")
    
    if os.path.exists(shard_path):
        return shard_path

    data = {"image": [], "image_id": [], "captions": [], "origin": []}

    for _, row in df_chunk.iterrows():
        image_name = str(row['image_name']).strip()
        full_path = os.path.join(images_dir, image_name)
        
        # Path fallback for COCO
        if dataset_name == "coco" and not os.path.exists(full_path):
            try:
                base_id = image_name.split('.')[0]
                full_path = os.path.join(images_dir, f"{int(base_id):012d}.jpg")
            except: pass

        if not os.path.exists(full_path): continue

        try:
            with PILImage.open(full_path) as img:
                processed_img = preprocess_image_to_square(img.convert("RGB"), DEFAULT_IMG_SIZE)
                data["image"].append(processed_img)
                data["image_id"].append(image_name)
                data["captions"].append(parse_captions(row['captions']))
                data["origin"].append(dataset_name)
        except Exception: continue

    if not data["image"]: return None

    shard_ds = Dataset.from_dict(data, features=features)
    shard_ds.save_to_disk(shard_path)
    
    del data, shard_ds
    gc.collect()
    return shard_path

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    features = Features({
        "image": Image(),
        "image_id": Value("string"),
        "captions": Sequence(Value("string")),
        "origin": Value("string")
    })

    all_shard_paths = []

    for ds_name, folder_name in DATASET_DIRS.items():
        images_dir = BASE_IMG_PATHS.get(ds_name)
        base_dir = os.path.join(DATABASE_RAW_PATH, folder_name)
        csv_path = os.path.join(base_dir, f"captions_{folder_name}.csv")

        if not os.path.exists(csv_path): continue

        print(f"\n🚀 Dispatching {ds_name} to {NUM_WORKERS} workers...")
        df = pd.read_csv(csv_path)
        
        num_shards = math.ceil(len(df) / BATCH_SIZE)
        
        # Prepare arguments for the worker pool
        tasks = []
        for i in range(num_shards):
            chunk = df.iloc[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            if not chunk.empty:
                tasks.append((chunk, images_dir, i, ds_name, features))

        # Execute in Parallel
        with Pool(processes=NUM_WORKERS) as pool:
            # imap allows us to use tqdm to track worker completion
            for result in tqdm(pool.imap_unordered(worker_task, tasks), total=len(tasks), desc=f"Parallel {ds_name}"):
                if result:
                    all_shard_paths.append(result)

    if not all_shard_paths:
        print("❌ No data processed."); sys.exit(1)

    print(f"\n🔗 Final Phase: Concatenating {len(all_shard_paths)} shards...")
    
    # Concatenation is IO bound, do it sequentially to keep RAM flat
    loaded_shards = [load_from_disk(p) for p in tqdm(all_shard_paths, desc="Loading Shards")]
    final_ds = concatenate_datasets(loaded_shards)

    if os.path.exists(DATABASE_PATH):
        shutil.rmtree(DATABASE_PATH)
    
    print(f"💾 Saving final database to: {DATABASE_PATH}")
    final_ds.save_to_disk(DATABASE_PATH, max_shard_size="2GB")

    print("🧹 Cleaning up...")
    shutil.rmtree(CACHE_DIR)

    print(f"\n✅ DONE. Total records: {len(final_ds)}")