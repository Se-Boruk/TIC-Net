"""

Script to be run once and to create shared initially lightly pre-processed dataset in the arrow file format

It contains only the positive samples. 
Further processing will be made on the run or in other data processing steps

"""

import os
import json
from PIL import Image as PILImage
from datasets import Dataset, Features, Image, Value, concatenate_datasets
from Config import DATABASE_RAW_PATH, DATABASE_PATH, DEFAULT_IMG_SIZE
from tqdm import tqdm
import shutil

from DataBase_functions import preprocess_image_to_square

#####################################################################################
BASE_IMG_PATHS = {
    "coco": os.path.join(DATABASE_RAW_PATH, "coco"),
    "flick30k": os.path.join(DATABASE_RAW_PATH, "flickr30k/images"),
    "ade20k": os.path.join(DATABASE_RAW_PATH, "ade20k", "ADEChallengeData2016", "images", "training"),
}

JSON_PATHS = {
    "coco": os.path.join(DATABASE_RAW_PATH, "coco_train_captions.jsonl"),
    "flick30k": os.path.join(DATABASE_RAW_PATH, "flickr30k_train_captions.jsonl"),
    "ade20k": os.path.join(DATABASE_RAW_PATH, "ade20k_train_captions.jsonl"),
}
N_WORKERS = 1
TMP_CACHE_DIR =  os.path.join(DATABASE_RAW_PATH, "temp_cache")
DATABASES = ["flick30k", "ade20k", "coco"]

#####################################################################################


def get_universal_path(data):
    ds_id = data['dataset_id'].lower()
    img_id = data['image_id']
    
    if "coco" in ds_id:
        #Padding as coco has the leading 0 so it must match (in the localized narratives there is just number)
        padded_id = f"{int(img_id):012d}"
        return os.path.join(BASE_IMG_PATHS["coco"], f"{padded_id}.jpg")
    if "ade20k" in ds_id:
        return os.path.join(BASE_IMG_PATHS["ade20k"], f"{img_id}.jpg")
    if "flick30k" in ds_id:
        return os.path.join(BASE_IMG_PATHS["flick30k"], f"{img_id}.jpg")
    return None


def prepare_worker_data(input_jsonl):
    """
    Returns ONE single list of all valid entries
    """
    
    print(f"Filtering {input_jsonl}...")
    filtered_data = []
    
    with open(input_jsonl, 'r') as f:
        total = sum(1 for _ in open(input_jsonl, 'r'))
        f.seek(0)
        
        
        for line in tqdm(f, total=total, desc="Reading JSONL"):
            item = json.loads(line)
            full_path = get_universal_path(item)
            
            if full_path and os.path.exists(full_path):
                filtered_data.append({
                    "path": full_path, 
                    "caption": item['caption'], 
                    "source": item['dataset_id']
                })
                
        print("DONE!")
    return filtered_data

def shard_generator(full_list):
    for item in full_list:
        try:
            with PILImage.open(item['path']) as img:
                
                final_img = preprocess_image_to_square(img, DEFAULT_IMG_SIZE)

                yield {
                    "image": final_img,
                    "caption": item['caption'],
                    "dataset_source": item['source']
                }
                
        except Exception as e:
            # print(f"Error processing {item['path']}: {e}") # Opcjonalny debug
            continue


def process_single_dataset(path, n_workers, tmp_cache_dir):
    """
    Prepares and generates a dataset part
    """

    data_list = prepare_worker_data(path)

    if not data_list:
        return None

    features = Features({
        "image": Image(),
        "caption": Value("string"),
        "dataset_source": Value("string"),
        })


    return Dataset.from_generator(
        shard_generator,
        gen_kwargs={"full_list": data_list},
        num_proc=n_workers,
        features=features,
        cache_dir=tmp_cache_dir
        )


if __name__ == "__main__":
    
    if os.path.exists(TMP_CACHE_DIR):
        print(f"Cleaning up old cache: {TMP_CACHE_DIR}")
        shutil.rmtree(TMP_CACHE_DIR)
        
    os.makedirs(TMP_CACHE_DIR, exist_ok=True)
    print(f"Utworzono tymczasowy cache folder (aby nie zasmiecac folderu cache): {TMP_CACHE_DIR}")
    
    #List for holding database parts
    dataset_parts = []

    #Processing each dataset one by one
    for ds_name in DATABASES:
        json_path = JSON_PATHS.get(ds_name)
        
        if json_path and os.path.exists(json_path):
            print(f"\nProcessing: {ds_name}")
            ds_part = process_single_dataset(json_path, N_WORKERS, TMP_CACHE_DIR)
            if ds_part:
                dataset_parts.append(ds_part)
        else:
            print(f"Skipping {ds_name}: JSONL not found at {json_path}")

    if dataset_parts:
        print("\nFinalizing Database")
        #Concatenate combines them into a single virtual table
        unified_dataset = concatenate_datasets(dataset_parts)
        
        print(f"Total samples in single database: {len(unified_dataset)}")
        
        # This is where the physical file is written to DATABASE_PATH
        unified_dataset.save_to_disk(DATABASE_PATH, max_shard_size="5GB")
        print(f"Successfully saved to: {DATABASE_PATH}")
        
    if os.path.exists(TMP_CACHE_DIR):
        shutil.rmtree(TMP_CACHE_DIR)
        print(f"\nUsunięto tymczasowy cache: {TMP_CACHE_DIR}")        
        
        