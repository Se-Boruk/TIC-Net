from datasets import load_dataset, load_from_disk
import os
import numpy as np
import torch
from queue import Queue
import threading
import random
import math
import torch.nn.functional as F
import torch
from PIL import Image as PILImage
from collections import Counter
from datasets import ClassLabel
import hashlib
import torch
import time
import random
from transformers import AutoTokenizer
import math
import re
import random
import torch
import time
import json
import Tokenizer_lib as Tok_lib
from Config import SOURCE_MAP
from Negative_map import NUMBERS_DICT, CATEGORY_POOLS, STRICT_PAIRS, SHARED_RELATIONS, SHARED_DEFINITIONS
import random
import spacy
import pyinflect

# =========================================================
# INITIALIZATION (Budowanie Indeksów - Uruchamiane raz)
# =========================================================

# 1. Reverse Map dla Shared Groups: "man" -> "MALE_HUMAN"
WORD_TO_SHARED_GROUP = {}
for group_name, words in SHARED_DEFINITIONS.items():
    for w in words:
        WORD_TO_SHARED_GROUP[w] = group_name

# 2. Reverse Map dla Pools: "red" -> "COLORS"
WORD_TO_POOL_ID = {}
for pool_name, words in CATEGORY_POOLS.items():
    for w in words:
        WORD_TO_POOL_ID[w] = pool_name

#SWAP OF NONE NEGATIVES
ALL_VALID_CONCEPTS = []
for pool in CATEGORY_POOLS.values():
    ALL_VALID_CONCEPTS.extend(pool)
for group in SHARED_DEFINITIONS.values():
    ALL_VALID_CONCEPTS.extend(group)
# Usunięcie duplikatów dla czystości puli
ALL_VALID_CONCEPTS = list(set(ALL_VALID_CONCEPTS))



# =========================================================
# 4. PROTECTED WORDS (STOPLIST)
# =========================================================
PROTECTED_WORDS = {
    "with", "from", "into", "onto", "over", "under", "next", "near", 
    "that", "this", "those", "these", "what", "where", "when", 
    "been", "being", "have", "has", "doing", "does", "done", "will", "would", 
    "could", "should", "their", "your", "some", "many", "very", "much",
    "background", "foreground", "image", "picture", "photo", "scene", "view",
    "looking", "standing", "sitting", "wearing", "holding"
}

# =========================================================
# LOAD POS MAPS 
# =========================================================
POS_MAP = {}
POS_GROUPS = {}

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("CRITICAL: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

def create_slightly_negative_caption(caption):
    """
    Creates a Hard Negative with strict morphological inheritance.
    Prevents sequence-based networks (LSTMs) from detecting grammatical artifacts.
    """
    if not caption or nlp is None: 
        return None, False

    # Process sentence to establish syntax dependencies and exact POS tags
    doc = nlp(caption)
    candidates = []
    
    # 1. Find High-Quality Candidates using contextual lemmas
    for i, token in enumerate(doc):
        if token.lower_ in PROTECTED_WORDS or token.is_punct: 
            continue
            
        lemma = token.lemma_.lower()

        # Route to strategies
        if lemma in NUMBERS_DICT or token.pos_ == "NUM":
            candidates.append((i, "NUMBER", lemma, token))
        elif lemma in STRICT_PAIRS:
            candidates.append((i, "STRICT", lemma, token))
        elif lemma in WORD_TO_POOL_ID:
            candidates.append((i, "POOL", lemma, token))
        elif lemma in WORD_TO_SHARED_GROUP:
            source_group = WORD_TO_SHARED_GROUP[lemma]
            if source_group in SHARED_RELATIONS:
                candidates.append((i, "SHARED", source_group, token))
        # Dynamic POS Fallback: Rely on spaCy's tags, not the static POS_MAP
        elif token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and len(lemma) > 3:
            candidates.append((i, "POS_FALLBACK", token.pos_, token))

    if not candidates:
        return None, False

    # 2. Select Targets (Updated for equal 1, 2, or 3 swap probability)
    desired_swaps = random.choice([1, 2, 3])
    num_swaps = min(desired_swaps, len(candidates))
    targets = random.sample(candidates, num_swaps)

    # Convert document to a list of strings preserving original whitespace
    final_tokens = [t.text_with_ws for t in doc]
    swaps_performed = 0

    for target_idx, strategy, meta, original_token in targets:
        new_word_lemma = None

        # Execute Swap Strategy
        if strategy == "NUMBER":
            val = int(meta) if meta.isdigit() else NUMBERS_DICT.get(meta, 1)
            offset = random.choice([-2, -1, 1, 2])
            new_val = max(1, val + offset)
            if new_val == val: new_val += 1
            new_word_lemma = str(new_val)

        elif strategy == "STRICT":
            new_word_lemma = random.choice(STRICT_PAIRS[meta])

        elif strategy == "POOL":
            pool_id = WORD_TO_POOL_ID[meta]
            valid_options = [w for w in CATEGORY_POOLS[pool_id] if w != meta]
            if valid_options:
                new_word_lemma = random.choice(valid_options)

        elif strategy == "SHARED":
            target_group_name = SHARED_RELATIONS[meta]
            if target_group_name in SHARED_DEFINITIONS:
                new_word_lemma = random.choice(SHARED_DEFINITIONS[target_group_name])
                
        elif strategy == "POS_FALLBACK":
            if meta in ["NOUN", "PROPN"] and random.random() < 0.8:
                if ALL_VALID_CONCEPTS:
                    new_word_lemma = random.choice(ALL_VALID_CONCEPTS)
            elif meta in POS_GROUPS and POS_GROUPS[meta]:
                new_word_lemma = random.choice(POS_GROUPS[meta])

        # 3. Morphological Inheritance & String Reconstruction
        if new_word_lemma:
            # Use module-level getInflection to inflect the new word to match the old word's tag
            inflected_tuple = pyinflect.getInflection(new_word_lemma, original_token.tag_)
            
            # Extract the first valid inflection, or fallback to the raw lemma if unavailable
            final_word = inflected_tuple[0] if inflected_tuple else new_word_lemma
            
            # Inherit capitalization structure
            if original_token.is_title:
                final_word = final_word.title()
            elif original_token.is_upper:
                final_word = final_word.upper()
                
            # Reconstruct string, maintaining natural spacing
            ws = original_token.whitespace_
            final_tokens[target_idx] = final_word + ws
            swaps_performed += 1

    if swaps_performed == 0:
        return None, False

    # Join the array back into a continuous string
    return "".join(final_tokens).strip(), True








def verify_splits(train, val, test):
    for name, ds in [("Train", train), ("Val", val), ("Test", test)]:
        counts = Counter(ds['dataset_source'])
        total = len(ds)
        print(f"\n{name} Distribution ({total} samples):")
        for src, count in counts.items():
            print(f" - {src}: {count} ({count/total:.2%})")

def verify_dataset_integrity(train_ds, val_ds, test_ds, expected_hashes=None):

    
    samples = {
        "train": str(train_ds[0]['caption']),
        "val": str(val_ds[0]['caption']),
        "test": str(test_ds[0]['caption'])
    }
    
    current_hashes = {}
    
    for split_name, text in samples.items():
        #hASH FOR CAPTION
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        current_hashes[split_name] = h

        
    if expected_hashes:
        mismatch = False
        for split in ["train", "val", "test"]:
            if current_hashes[split] != expected_hashes.get(split):
                print(f"ERROR: {split} hash mismatch!")
                mismatch = True
        
        if not mismatch:
            print("All hashes match. Dataset splits are consistent.")
        else:
            raise ValueError("Integrity check failed! Splits are different than expected.")
    else:
        print("No expected hashes provided. Copy the values above to Config")
        for split, h in current_hashes.items():
            print(f"{split.capitalize()} first sample hash: {h}")
        



def preprocess_image_to_square(img_input, target_size=224):
    """
    Universal preprocessor for PIL, NumPy, or Torch inputs.
    Resizes maintaining aspect ratio and pads to a square.
    """
    
    #COnversion logic
    if isinstance(img_input, torch.Tensor):
        # Handle (C, H, W) or (H, W, C) tensors
        
        if img_input.ndimension() == 3 and img_input.shape[0] in [1, 3]:
            img_input = img_input.permute(1, 2, 0)
        img_input = img_input.cpu().detach().numpy()

    if isinstance(img_input, np.ndarray):
        #Ensure uint8 for PIL conversion
        if img_input.max() <= 1.0:
            img_input = (img_input * 255).astype(np.uint8)
        img = PILImage.fromarray(img_input)
    else:
        # Assume already a PIL image or try to force it
        img = img_input

    #Standardization to RGB
    img = img.convert("RGB")
    
    #Geometry Logic
    width, height = img.size
    scale = target_size / max(width, height)
    new_width, new_height = int(width * scale), int(height * scale)
    
    img = img.resize((new_width, new_height), PILImage.LANCZOS)
    
    #Canvas Creation
    final_img = PILImage.new("RGB", (target_size, target_size), (0, 0, 0))
    upper = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    final_img.paste(img, (left, upper))
    
    return final_img



class Custom_DataSet_Manager():
    
    #Checks if there is dataset folder present, if not it creates it
    def __init__(self, DataSet_path, train_split, val_split, test_split, random_state):
        self.dataset_path = DataSet_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        

    def load_dataset_from_disk(self):  
        #Load it to split it on run
        Dataset = load_from_disk(self.dataset_path)
        
        train, val, test = self.split_dataset(Dataset)
        return train, val, test
    
    def split_dataset(self,dataset):

        
        #takint unique classes and print them
        unique_sources = dataset.unique("origin")
        print(f"Detected unique sources: {unique_sources}")
        
        #Encoding data source to number so it can be used in stratification
        dataset = dataset.map(
                lambda x: {"dataset_source": [s.lower() for s in x["origin"]]},
                batched=True,
                desc="dataset_source casing to "
            )
        
        sorted_names = [name.lower() for name, _ in sorted(SOURCE_MAP.items(), key=lambda x: x[1])]
        
        source_feature = ClassLabel(names=sorted_names)
        dataset = dataset.cast_column("dataset_source", source_feature)
        
        print(f"Mapowanie klas : {dataset.features['origin']}")
        
        ####
        Data =  dataset.shuffle(seed=self.random_state)
        
        #Split it into train and subset
        split_dataset = Data.train_test_split(test_size= (1 -self.train_split) , seed=self.random_state, stratify_by_column="dataset_source")
        
        train_subset = split_dataset['train']
        subset = split_dataset['test']
        
        #Split the subset into the val and test 
        test_fraction = self.val_split / ((self.val_split + self.test_split))
        
        split_dataset_1 = subset.train_test_split(test_size= test_fraction , seed=self.random_state, stratify_by_column="dataset_source")
        
        val_subset = split_dataset_1['train']
        test_subset = split_dataset_1['test']
        
        
        return train_subset, val_subset, test_subset
        

    
##########################################################################    
    
    
class Async_DataLoader():
    def __init__(self, dataset, batch_size=32, sequence_length = 224, num_workers=2, device='cuda', max_queue=10, image_augmentation = True , fraction = None):
        self.dataset = dataset
        #Taking sample of from dataset to initialize the shape of images
        sample_img = np.array(dataset[0]["image"], dtype=np.uint8)
        self.C, self.H, self.W = sample_img.shape[2], sample_img.shape[0], sample_img.shape[1]
        
        self.fraction = fraction # Fraction of images taken in the epoch (randomized each epoch)
        
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.queue = Queue(maxsize=max_queue)
        self.num_workers = num_workers

        #Epoch control
        self.next_idx = 0               #Next step (batch) idx
        self.idx_lock = threading.Lock()
        self.active_workers = 0 
        self.threads = []
        self.epoch_event = threading.Event()
        self.indices = list(range(len(self.dataset)))
        

        self.sequence_length = sequence_length
        #Preallocate pinned buffers
        self.pinned_bufs = [torch.empty((self.batch_size, self.C, self.H, self.W), 
                                        dtype=torch.float32).pin_memory() 
                            for _ in range(num_workers)]
        
        self.positive_caption_bufs = [torch.empty((self.batch_size,self.sequence_length), 
                                        dtype=torch.long).pin_memory() 
                            for _ in range(num_workers)]
        

        self.negative_caption_bufs = [torch.empty((self.batch_size,self.sequence_length), 
                                        dtype=torch.long).pin_memory() 
                            for _ in range(num_workers)]
        
        
        self.origin_bufs = [torch.empty((self.batch_size,1), 
                                        dtype=torch.float32).pin_memory() 
                            for _ in range(num_workers)]
        
        self.image_augmentation = image_augmentation
        

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        

        
        self.tokenizers = [Tok_lib.SimpleTokenizer("vocab.json", "lemma_map.json", self.sequence_length) for _ in range(self.num_workers)]
        self.aug_counter = 0
        
        # threads will be started lazily in start_epoch (safer on Windows/spawn)
        self.threads_started = False
        
        # do not start prefetch here
        # self._start_prefetch()

    def _augment_images(self, image_batch, flip_allowed=None,
                        brightness=0.2, contrast=0.2, saturation=0.2,
                        flip_prob=0.5, max_rot=15, crop_ratio=0.85, 
                        p_spatial=0.8, p_color=0.8, p_gray=0.15):
        """
        Zoptymalizowana augmentacja z flagami per-obrazek (flip_allowed).
        
        Args:
            image_batch (Tensor): [B, C, H, W] - batch obrazów (0-1).
            flip_allowed (Tensor Bool): [B] - maska określająca czy dany obrazek MOŻE być odbity.
                                      Jeśli None, wszystkie mogą być odbite.
        """
        B, C, H, W = image_batch.shape
        device = image_batch.device
        dtype = image_batch.dtype
        
        # Klonowanie, aby nie modyfikować oryginału w pamięci pinned
        image_batch = image_batch.clone()

        # Domyślnie pozwalamy na flip wszystkim, jeśli nie podano flag
        if flip_allowed is None:
            flip_allowed = torch.ones(B, dtype=torch.bool, device=device)

        # ==================================================================
        # 1. TRANSFORMACJE PRZESTRZENNE (Single-Pass Affine)
        # ==================================================================
        # Losujemy, które obrazki w ogóle podlegają transformacji przestrzennej
        spatial_mask = torch.rand(B, device=device) < p_spatial
        
        if spatial_mask.any():
            # Liczba obrazków do przetworzenia
            B_sub = spatial_mask.sum().item()
            
            # Wyciągamy flagi flipa TYLKO dla przetwarzanych obrazków
            # To krytyczne: mapujemy [B] -> [B_sub]
            flip_allowed_sub = flip_allowed[spatial_mask]

            # --- A. Parametry losowe dla każdego obrazka ---
            
            # Rotacja (w radianach)
            angles = (torch.rand(B_sub, device=device) * 2 - 1) * max_rot
            radians = angles * (3.14159265 / 180)
            
            # Skalowanie (Zoom in)
            # Randomizacja zoomu: od crop_ratio do 1.0
            curr_crop = crop_ratio + (torch.rand(B_sub, device=device) * (1.0 - crop_ratio))
            scale = 1.0 / curr_crop
            
            # Przesunięcie (Shift)
            # Maksymalne przesunięcie zależy od tego, jak mocno przybliżyliśmy
            max_shift = 1.0 - curr_crop
            tx = (torch.rand(B_sub, device=device) * 2 - 1) * max_shift
            ty = (torch.rand(B_sub, device=device) * 2 - 1) * max_shift

            # --- B. Logika Inteligentnego Flipa ---
            
            # 1. "Chęć" flipa (losowa szansa)
            wants_flip = torch.rand(B_sub, device=device) < flip_prob
            
            # 2. Ostateczna decyzja (AND logiczny): Chce flipa ORAZ ma pozwolenie
            actual_flip_mask = wants_flip & flip_allowed_sub
            
            # 3. Tworzymy wektor mnożnika: 1.0 (brak flipa) lub -1.0 (flip)
            flip_factor = torch.ones(B_sub, device=device)
            flip_factor[actual_flip_mask] = -1.0
            
            # --- C. Konstrukcja Macierzy Afinicznej ---
            
            cos = torch.cos(radians) * scale
            sin = torch.sin(radians) * scale
            
            affine_mats = torch.zeros(B_sub, 2, 3, device=device, dtype=dtype)
            
            # Oś X (mnożymy przez flip_factor, aby uzyskać odbicie lustrzane)
            affine_mats[:, 0, 0] = cos * flip_factor 
            affine_mats[:, 0, 1] = -sin
            affine_mats[:, 0, 2] = tx
            
            # Oś Y
            affine_mats[:, 1, 0] = sin * flip_factor # Rotacja musi uwzględniać zmianę układu
            affine_mats[:, 1, 1] = cos
            affine_mats[:, 1, 2] = ty

            # --- D. Aplikacja (Grid Sample) ---
            grid = F.affine_grid(affine_mats, [B_sub, C, H, W], align_corners=False)
            
            # Używamy trybu 'reflection', żeby nie było czarnych ramek przy obrocie
            image_batch[spatial_mask] = F.grid_sample(
                image_batch[spatial_mask], grid, 
                mode='bicubic', padding_mode='reflection', align_corners=False
            )

        # ==================================================================
        # 2. GRAYSCALE (Wymuszanie semantyki kształtu)
        # ==================================================================
        if C == 3:
            gray_mask = torch.rand(B, 1, 1, 1, device=device) < p_gray
            if gray_mask.any():
                # ITU-R 601-2 luma transform
                luma = (image_batch[:, 0:1] * 0.299 + 
                        image_batch[:, 1:2] * 0.587 + 
                        image_batch[:, 2:3] * 0.114)
                # Powielamy kanał 3 razy, żeby zachować wymiar [B, 3, H, W]
                image_batch = torch.where(gray_mask, luma.repeat(1, 3, 1, 1), image_batch)

        # ==================================================================
        # 3. COLOR JITTERING (Jasność, Kontrast, Nasycenie)
        # ==================================================================
        color_mask = torch.rand(B, 1, 1, 1, device=device) < p_color
        if color_mask.any():
            # Brightness
            b_factors = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * brightness
            image_batch = torch.where(color_mask, image_batch * b_factors, image_batch)

            # Contrast
            mean = image_batch.mean(dim=[2, 3], keepdim=True)
            c_factors = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * contrast
            image_batch = torch.where(color_mask, (image_batch - mean) * c_factors + mean, image_batch)

            # Saturation (Tylko RGB)
            if C == 3:
                gray = image_batch.mean(dim=1, keepdim=True)
                s_factors = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * saturation
                image_batch = torch.where(color_mask, (image_batch - gray) * s_factors + gray, image_batch)

        return image_batch.clamp(0, 1)

    def _start_prefetch(self):
        
        def get_chunk():
            with self.idx_lock:
                start = self.next_idx
                if start >= self.effective_len:  # use effective length
                    return None, None
                
                end = min(start + self.batch_size, self.effective_len)  # use effective length
                self.next_idx = end
                return start, end


        def worker(worker_id):
            pinned_buf = self.pinned_bufs[worker_id]
            positive_caption_bufs = self.positive_caption_bufs[worker_id]
            negative_caption_bufs = self.negative_caption_bufs[worker_id]
            origin_bufs = self.origin_bufs[worker_id]
            Tokenizer = self.tokenizers[worker_id]
        
            while True:
                self.epoch_event.wait()
                
                FORBIDDEN_FLIP_WORDS = {"left", "right", "port", "starboard", "east", "west"}
                while True:
                    start, end = get_chunk()
                    if start is None:
                        break
                    actual_bs = end - start
                    
                    flip_flags = []
                    negative_from_batch = None
                    for i in range(actual_bs):
                        idx = self.indices[start + i]
                        negative_from_batch = False
                        
                        # [Image Loading Logic ...] 
                        img = np.array(self.dataset[idx]["image"], dtype=np.float32) / 255.0
                        caption_list = self.dataset[idx]['captions']
                        
                        caption_positive = random.choice(caption_list)
                        rand_neg = random.random()
                        if rand_neg < 0.07:
                            random_idx = random.randint(0, len(self.dataset) - 1)
                            caption_negative = random.choice(self.dataset[random_idx]['captions'])
                            negative_from_batch = True
                        else:    
                            if random.choice([True, False]):
                                caption_negative = random.choice(caption_list)
                            else:
                                caption_negative = caption_positive


                        # FLIP MAP CREATION
                        if caption_positive:
                            # 1. Pobierz listę tokenów
                            tokens_list = Tok_lib.preprocess_text(caption_positive)
                            
                            # 2. Zamień na set, aby użyć intersection (i przyspieszyć wyszukiwanie)
                            cap_words = set(tokens_list)
                            
                            # 3. Sprawdź część wspólną
                            # Jeśli intersection nie jest puste -> len > 0 -> can_flip = False
                            can_flip = len(cap_words.intersection(FORBIDDEN_FLIP_WORDS)) == 0
                        else:
                            can_flip = True
                        
                        flip_flags.append(can_flip)



                        # --- NEGATIVE SELECTION 
                        if not negative_from_batch:
                            caption_negative, success = create_slightly_negative_caption(caption_negative)
                            if not success:
                                # FALLBACK: If hard negative failed, use an "Easy Negative" (Random caption from dataset)
                                # This ensures we never train on Pos == Neg
                                random_idx = random.randint(0, len(self.dataset) - 1)
                                # Ensure we don't pick the same image by accident
                                while random_idx == idx: 
                                    random_idx = random.randint(0, len(self.dataset) - 1)
                                caption_negative = random.choice(self.dataset[random_idx]['captions'])
   
                        
                        # [Encoding Logic ...]
                        caption_negative = Tokenizer.encode(caption_negative)
                        caption_negative = torch.tensor(caption_negative, dtype=torch.long)
                        caption_positive = Tokenizer.encode(caption_positive)
                        caption_positive = torch.tensor(caption_positive, dtype=torch.long)
                        
                        # [Origin Logic ...]
                        origin = self.dataset[idx]['origin']
                        origin_map = {"coco": 0, "ade20k": 1, "flickr30k": 2}
                        origin_val = origin_map.get(origin, 3) # Safe get

                        # [Buffer Copying ...]
                        pinned_buf[i].copy_(torch.from_numpy(img).permute(2, 0, 1))
                        positive_caption_bufs[i].copy_(caption_positive)
                        negative_caption_bufs[i].copy_(caption_negative)
                        origin_bufs[i].fill_(origin_val)
        
                    # Clone to avoid modifying pinned memory
                    origin_batch = origin_bufs[:actual_bs].to(self.device, non_blocking=True).clone()
                    original_batch = pinned_buf[:actual_bs].to(self.device, non_blocking=True).clone()
                    
                    positive_caption_batch = positive_caption_bufs[:actual_bs].to(self.device, non_blocking=True).clone()           
                    negative_caption_batch = negative_caption_bufs[:actual_bs].to(self.device, non_blocking=True).clone()
                    
        
                    # ---------------------------
                    # Prepare batch variants
                    # ---------------------------
                    batch_dict = {
                        'origin': origin_batch,
                        "image_original": original_batch,
                        "caption_positive": positive_caption_batch,
                        "caption_negative": negative_caption_batch
                    }
                    
                    # Augmented
                    if self.image_augmentation:
                        flip_tensor = torch.tensor(flip_flags, device=self.device, dtype=torch.bool)
                        aug_images = self._augment_images(original_batch, flip_tensor)
                        batch_dict["image_augmented"] = aug_images
                        
                    

                    
                    # Operacja na tablicy numpy: (Obraz - Średnia) / Odchylenie
                    batch_dict["image_original"] = (batch_dict["image_original"] - self.mean) / self.std
                    if "image_augmented" in batch_dict:
                        batch_dict["image_augmented"] = (batch_dict["image_augmented"] - self.mean) / self.std



                    # Push to queue
                    self.queue.put(batch_dict)
        
                # Epoch end handling
                with self.idx_lock:
                    self.active_workers -= 1
                    if self.active_workers == 0:
                        self.queue.put(None)
                        self.epoch_event.clear()

        # start worker threads
        for wid in range(self.num_workers):
            t = threading.Thread(target=worker, args=(wid,))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def start_epoch(self, shuffle=True):
        self.queue = Queue(maxsize=self.queue.maxsize)
        self.next_idx = 0
        self.active_workers = self.num_workers
    
        if not self.threads_started:
            self._start_prefetch()
            self.threads_started = True
    
        # Shuffle and optionally reduce dataset fraction
        indices = np.arange(len(self.dataset))
        if shuffle:
            np.random.shuffle(indices)
    
        if self.fraction is not None and 0 < self.fraction < 1:
            reduced_size = int(len(indices) * self.fraction)
            
            indices = np.random.choice(indices, size=reduced_size, replace=False)
    
        self.indices = list(indices)
        self.effective_len = len(self.indices)  #store effective lenght
    
        self.epoch_event.set()
        
        


    def get_batch(self):
        batch = self.queue.get()
        if batch is None:
            return None
    
        # Move all tensors in dict to device (non-blocking)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def get_num_batches(self):
        if hasattr(self, "effective_len"):
            effective_len = self.effective_len
        else:
            effective_len = len(self.dataset)
        steps = (effective_len + self.batch_size - 1) // self.batch_size
        return steps


    def get_random_batch(self, batch_size=None, shuffle=True, random_state=None):
        """
        Returns a random batch of original images from the dataset, reproducible if rng is provided.
        
        Args:
            batch_size: int, optional
            shuffle: bool, whether to pick random indices
            rng: np.random.Generator, optional — if given, sampling becomes deterministic
    
        Returns:
            batch tensor of shape (B, C, H, W)
        """
        bs = batch_size or self.batch_size
    
        #Random state for reproducibility
        if random_state is None:
            random_state = np.random.default_rng()
        else:
            random_state = np.random.default_rng(random_state)

        
        # pick indices
        if shuffle:
            indices = random_state.choice(len(self.dataset), bs, replace=False)
        else:
            indices = np.arange(bs)
    
        #preallocate pinned buffer
        pinned_buf = torch.empty((bs, self.C, self.H, self.W),
                                 dtype=torch.float32).pin_memory()
    
        #load images
        for i, idx in enumerate(indices):
            img = np.array(self.dataset[idx]["image"], dtype=np.float32) / 255.0
            pinned_buf[i] = torch.from_numpy(img).permute(2, 0, 1)
    
        #move to device
        batch = pinned_buf.to(self.device, non_blocking=True)
        
        return batch
    
    
##########################################################################


    