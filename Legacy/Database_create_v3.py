import os
import json
import re
import math
import time
import random
import shutil
import torch
import gc
from PIL import Image as PILImage
import datasets
from datasets import Dataset, Features, Image, Value, Sequence, concatenate_datasets
from tqdm import tqdm  # Explicit import for the inner loop
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as transformers_logging
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
# --- CONFIGURATION ---
from Config import DATABASE_RAW_PATH, DATABASE_PATH, DEFAULT_IMG_SIZE
from DataBase_functions import preprocess_image_to_square

# Suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
transformers_logging.set_verbosity_error()
datasets.logging.set_verbosity_info()

#####################################################################################
#  1. GLOBAL LAZY LOADER & MODEL CLASS
#####################################################################################

# Global variable to hold the model in the worker process
_AUGMENTER_INSTANCE = None

class Caption_augmenter:
    def __init__(self, model_id="./Models/Qwen2.5-3B"):
        self.model_id = model_id
        print(f"🚀 Initializing Model: {model_id} (Safetensors BF16)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        )
        self.model.eval()

    def augment_positive_batch(self, captions: list):
        prompts = []
        target_limits = []
        
        # 1. Prepare Prompts
        for caption in captions:
            sentences = [s.strip() for s in re.split(r'[.!?]+', caption) if s.strip()]
            num_sentences = len(sentences)
            
            rand = random.random()
            if rand < 0.45:
                target_count = max(1, math.ceil(num_sentences * 0.50))
            elif rand < 0.80:
                target_count = max(1, math.ceil(num_sentences * 0.7))
            else:
                target_count = max(1, math.ceil(num_sentences * 0.3))

            summary_style = random.choice([
                "Rewrite as a natural, fluid sentence.",
                "Summarize objects and their spatial relations.",
                "Describe only the visible physical facts.",
                "Condense into a high-density, brief phrase."
            ])

            messages = [
                {
                    "role": "system", 
                    "content": "You are a caption compression tool. Output ONLY the shortened sentence. No intro, no meta-talk, no labels. Do not add any new information or intentions, motivations or narrative."
                },
                {
                    "role": "user", 
                    "content": f"Style: {summary_style}\nTarget: {target_count} sentence(s).\nOriginal: {caption}"
                }
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts.append(prompt)
            
            avg_words = len(caption.split()) / max(1, num_sentences)
            target_limits.append(int(avg_words * target_count) + 12)

        # 2. Dynamic Params
        dyn_temp = random.uniform(0.1, 0.4) 
        dyn_top_p = random.uniform(0.85, 0.95)
        dyn_rep_p = random.uniform(1.05, 1.2) 

        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
        input_ids_len = inputs.input_ids.shape[1]
        
        stop_strings=["Human:", "User:", "Original:", "\n\n", "###", "Style:", "Target:"]
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max(target_limits),
                do_sample=True,
                temperature=dyn_temp,
                top_p=dyn_top_p,
                repetition_penalty=dyn_rep_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer
            )

        decoded = self.tokenizer.batch_decode(outputs[:, input_ids_len:], skip_special_tokens=True)
        del inputs
        del outputs
        
        results = []
        
        for i, resp in enumerate(decoded):
            clean = re.split(r'Human:|User:|Original:|Style:|Result:', resp, flags=re.IGNORECASE)[0]
            clean = clean.strip().splitlines()[0] if clean.strip() else ""
            clean = re.sub(r'^\d+[\).]\s*', '', clean)
            
            if len(clean.split()) >= 3:
                results.append(clean)
            else:
                results.append(captions[i]) 

        return results




def get_augmenter():
    """
    Singleton pattern: Only loads the model if it doesn't exist yet.
    """
    global _AUGMENTER_INSTANCE
    if _AUGMENTER_INSTANCE is None:
        _AUGMENTER_INSTANCE = Caption_augmenter(model_id="./Models/Qwen2.5-3B")
    return _AUGMENTER_INSTANCE

# The mapping function MUST be at the top level
def batch_augment_fn(captions, n_variants=8):
    """
    Input: captions = ['cap1', 'cap2', ...] 
    (Because input_columns=['caption'] is used, we receive the list directly, NOT a dict)
    """
    augmenter = get_augmenter()
    
    # [FIX] 'captions' is already the list. No need to do captions['caption']
    original_captions = captions 
    batch_len = len(original_captions)
    
    augmented_batch = [ [] for _ in range(batch_len) ]
    
    # [FIX] Added gc and empty_cache inside the loop
    for _ in tqdm(range(n_variants), desc=f"Batch Variants ({batch_len} imgs)", leave=False):
        new_variants_list = augmenter.augment_positive_batch(original_captions)
        for i, variant in enumerate(new_variants_list):
            augmented_batch[i].append(variant)
            
        # FORCE CLEANUP: This resets the PyTorch memory allocator
        gc.collect()
        torch.cuda.empty_cache()
    
    if MAIN_PBAR is not None:
        MAIN_PBAR.update(batch_len)
    
    return {"caption_aug": augmented_batch}

#####################################################################################
#  2. DATASET LOGIC
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

CPU_WORKERS = 8 
TMP_CACHE_DIR =  os.path.join(DATABASE_RAW_PATH, "temp_cache")
DATABASES = ["flick30k", "ade20k", "coco"]

def get_universal_path(data):
    ds_id = data['dataset_id'].lower()
    img_id = data['image_id']
    if "coco" in ds_id:
        padded_id = f"{int(img_id):012d}"
        return os.path.join(BASE_IMG_PATHS["coco"], f"{padded_id}.jpg")
    if "ade20k" in ds_id:
        return os.path.join(BASE_IMG_PATHS["ade20k"], f"{img_id}.jpg")
    if "flick30k" in ds_id:
        return os.path.join(BASE_IMG_PATHS["flick30k"], f"{img_id}.jpg")
    return None

def prepare_worker_data(input_jsonl, limit=None):
    print(f"Filtering {input_jsonl}...")
    filtered_data = []
    with open(input_jsonl, 'r') as f:
        total = sum(1 for _ in open(input_jsonl, 'r')) if limit is None else limit
        f.seek(0)
        with tqdm(total=total, desc="Reading JSONL") as pbar:
            for line in f:
                if limit is not None and len(filtered_data) >= limit: break
                item = json.loads(line)
                full_path = get_universal_path(item)
                if full_path and os.path.exists(full_path):
                    filtered_data.append({
                        "path": full_path, 
                        "caption": item['caption'], 
                        "source": item['dataset_id']
                    })
                    pbar.update(1)
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
        except Exception:
            continue

def process_single_dataset(path, n_workers, tmp_cache_dir, limit=None):
    data_list = prepare_worker_data(path, limit)
    if not data_list: return None
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


# --- GLOBAL PROGRESS BAR VARIABLE ---
MAIN_PBAR = None


#####################################################################################
#  3. MAIN EXECUTION
#####################################################################################

if __name__ == "__main__":
    
    # --- CONFIG ---
    SCRIPT_MODE = False 
    LIMIT_COUNT = 500    # Your test count
    BATCH_SIZE_LLM = 96
    N_VARIANTS = 4      
    # --------------

    current_limit = LIMIT_COUNT if SCRIPT_MODE else None

    if os.path.exists(TMP_CACHE_DIR):
        shutil.rmtree(TMP_CACHE_DIR)
    os.makedirs(TMP_CACHE_DIR, exist_ok=True)

    dataset_parts = []
    for ds_name in DATABASES:
        json_path = JSON_PATHS.get(ds_name)
        if json_path and os.path.exists(json_path):
            ds_part = process_single_dataset(json_path, CPU_WORKERS, TMP_CACHE_DIR, limit=current_limit)
            if ds_part: dataset_parts.append(ds_part)

    if dataset_parts:
        print("\n=== Concatenating Datasets ===")
        unified_dataset = concatenate_datasets(dataset_parts)
        total_samples = len(unified_dataset)
        print(f"Total samples: {total_samples}")

        print("\n=== Starting Caption Augmentation (LLM) ===")
        print(f"Running augmentation: Batch {BATCH_SIZE_LLM} | Variants {N_VARIANTS}")
        
        # 1. DISABLE the default datasets progress bar (it fights with ours)
        disable_progress_bar()

        # 2. CREATE our manual global bar
        # This bar will stay 0/135 until we manually update it
        MAIN_PBAR = tqdm(total=total_samples, desc="Total Images Processed", unit="img")
        
        try:
            unified_dataset = unified_dataset.map(
                batch_augment_fn,
                batched=True,
                batch_size=BATCH_SIZE_LLM, 
                fn_kwargs={"n_variants": N_VARIANTS},
                num_proc=None, 
                
                # OPTIMIZATION: Only load text (Saves System RAM)
                input_columns=["caption"], 
                
                # OPTIMIZATION: Flush to disk frequently (Saves System RAM)
                writer_batch_size=BATCH_SIZE_LLM * 2 
            )
        finally:
            MAIN_PBAR.close()
            enable_progress_bar()
        
        print("Augmentation Complete.")
        
        save_path = DATABASE_PATH.replace(".arrow", "_TEST.arrow") if SCRIPT_MODE else DATABASE_PATH
        print(f"Saving to {save_path}...")
        unified_dataset.save_to_disk(save_path, max_shard_size="5GB")
        print("DONE.")

    if os.path.exists(TMP_CACHE_DIR):
        shutil.rmtree(TMP_CACHE_DIR)








