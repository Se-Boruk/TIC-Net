import torch
from datasets import load_from_disk 
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing
import collections
import Config 

#===========================================================================
# Functions directly for the Tokenizer and text processing
#===========================================================================

# ---------------------------------------------------------
# 1. SHARED PREPROCESSING FUNCTION
# ---------------------------------------------------------
def preprocess_text(text):
    """
    Function takes the text in the string format and performs these operations:
    1. Undercapitalize
    2. REMOVE NON-ASCII (Chinese, Emoji, Special Symbols)
    3. Separate EXTENDED punctuation
    4. Collapse repeated punctuation
    5. Split by whitespace
    """
    
    if not text:
        return []

    # 1. Undercapitalize
    text = text.lower()
    
    # 2. REMOVE NON-ASCII (Chinese Fix)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # 3. EXPANDED PUNCTUATION SPLITTING (Slash Fix)
    text = re.sub(r'([.,!?;:()"\-+=_/—|<>@#%&*\[\]\'])', r' \1 ', text)
    
    # 4. Collapse repeated punctuation
    text = re.sub(r'([.,!?;:()"\-+=_/—|<>@#%&*\[\]\'])\1+', r'\1', text)
    
    # 5. Split
    return text.split()


class SimpleTokenizer:
    """
    Tokenizer which, well tokenizes the words into the numbers
    
    We can use it to encode string-sequence to the token list and vice versa
    to decode the token list into the string sequence
    """
    
    def __init__(self, vocab_path, lemma_map_path=None, max_length=None):
        
        # Opening created vocab.json file
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        # Load Lemma Map
        self.lemma_map = {}
        if lemma_map_path and os.path.exists(lemma_map_path):
            with open(lemma_map_path, 'r', encoding='utf-8') as f:
                self.lemma_map = json.load(f)
        
        self.max_length = max_length
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        
        self.unk_id = self.vocab.get("<UNK>", 1)
        self.pad_id = self.vocab.get("<PAD>", 0)

    def encode(self, text):
        # 1. Preprocess
        words = preprocess_text(text)
        
        # 2. Offline Lemmatization & Expansion
        # The map might return "dog <MULTIPLE>", so we need to handle splitting.
        expanded_tokens = []
        if self.lemma_map:
            for w in words:
                mapped = self.lemma_map.get(w, w)
                # Split by space just in case mapped value is "base <MULTIPLE>"
                expanded_tokens.extend(mapped.split())
        else:
            expanded_tokens = words

        # 3. Map to IDs
        tokens = [self.vocab.get(w, self.unk_id) for w in expanded_tokens]
        
        # 4. Padding/Cropping
        if self.max_length:
            tokens = tokens[:self.max_length]
            tokens += [self.pad_id] * (self.max_length - len(tokens))
            
        return tokens

    def decode(self, token_ids, skip_special=True):
        if torch.is_tensor(token_ids):
            token_ids = token_ids.detach().cpu().tolist()
        
        if isinstance(token_ids[0], list):
            return [self._decode_single(seq, skip_special) for seq in token_ids]
        
        return self._decode_single(token_ids, skip_special)

    def _decode_single(self, ids, skip_special):
        words = []
        for idx in ids:
            idx = int(idx)
            word = self.inverse_vocab.get(idx, "<UNK>")
            
            if skip_special:
                if idx == self.pad_id:
                    continue
            
            words.append(word)
        
        return " ".join(words)


#===========================================================================
# Functions to create the vocabulary
#===========================================================================

def _worker_scan(args):
    dataset_path, start_idx, end_idx, batch_size = args
    dataset = load_from_disk(dataset_path)
    
    local_word_counts = collections.Counter()
    local_len_counts = collections.Counter()
    
    # We only scan raw words here; expansion happens in Phase 3
    for i in range(start_idx, end_idx, batch_size):
        slice_end = min(i + batch_size, end_idx)
        batch = dataset[i : slice_end]
        
        batch_tokens = []
        batch_lengths = []

        def process_entry(txt):
            if txt:
                tokens = preprocess_text(txt)
                batch_tokens.extend(tokens)
                batch_lengths.append(len(tokens))

        for t_list in batch['captions']:
            if t_list:
                for text in t_list:
                    process_entry(text)
           
        local_word_counts.update(batch_tokens)
        local_len_counts.update(batch_lengths)
        
    return local_word_counts, local_len_counts

def plot_rank_frequency_comparison(raw_counts, lemma_counts, min_frequency, save_path):
    sorted_raw = sorted(raw_counts.values(), reverse=True)
    sorted_lemma = sorted(lemma_counts.values(), reverse=True)
    
    plt.figure(figsize=(12, 8))
    plt.plot(sorted_raw, color='gray', linestyle=':', linewidth=1.5, label='Raw Words')
    plt.plot(sorted_lemma, color='darkblue', linewidth=2.0, label='Lemmatized + <MULTIPLE>')
    plt.axhline(y=min_frequency, color='red', linestyle='--', label=f'Cutoff ({min_frequency})')
    plt.yscale('log')
    plt.legend()
    plt.title('Vocabulary Compression Analysis')
    plt.savefig(save_path)
    plt.close()

def plot_caption_lengths(length_counts, save_path):
    lengths = sorted(length_counts.keys())
    counts = [length_counts[l] for l in lengths]
    plt.figure(figsize=(10, 6))
    plt.bar(lengths, counts, color='forestgreen', alpha=0.7)
    plt.savefig(save_path)
    plt.close()

def create_vocabulary(arrow_path, min_frequency=5, save_vocab_path="vocab.json", save_map_path="lemma_map.json", num_workers=None):
    print(f"Loading dataset from {arrow_path}...")
    dataset = load_from_disk(arrow_path)
    total_len = len(dataset)
    
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    # --- PHASE 1: Scan Raw Words ---
    print(f"Phase 1: Scanning Raw Words with {num_workers} workers...")
    chunk_size = total_len // num_workers
    worker_args = []
    
    for i in range(num_workers):
        start = i * chunk_size
        end = total_len if i == num_workers - 1 else (i + 1) * chunk_size
        worker_args.append((arrow_path, start, end, 5000))

    global_raw_words = collections.Counter()
    global_lengths = collections.Counter()
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(_worker_scan, worker_args), total=num_workers, desc="Scanning Raw"))
        print("Merging raw results...")
        for w_count, l_count in results:
            global_raw_words.update(w_count)
            global_lengths.update(l_count)

    # --- PHASE 2: Build Lemma Map (Spacy) with <MULTIPLE> ---
    print("\nPhase 2: Building Lemma Map (Spacy) with Plural Detection...")
    try:
        import spacy
        # Note: We need the tagger for plural detection, so only disable NER/Parser
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except ImportError:
        print("Error: SpaCy not installed.")
        return
    except OSError:
        print("Error: 'en_core_web_sm' not found.")
        return

    unique_raw_words = list(global_raw_words.keys())
    lemma_map = {}
    
    batch_size = 5000
    for i in tqdm(range(0, len(unique_raw_words), batch_size), desc="Lemmatizing"):
        batch = unique_raw_words[i : i + batch_size]
        docs = list(nlp.pipe(batch))
        
        for original, doc in zip(batch, docs):
            token = doc[0]
            lemma = token.lemma_.lower().strip()
            tag = token.tag_  # Part-of-Speech Tag
            
            # --- SKIP INVALID ---
            if not lemma or lemma == original:
                # Even if it's identical ("dogs"->"dogs" error?), 
                # we might still want to catch plurals if Spacy tagged it NNS.
                # But usually "dogs"->"dog".
                # If lemma == original, we skip mapping UNLESS it's a plural that wasn't lemmatized?
                # Let's keep it simple: only map if changed OR if plural.
                if tag not in ['NNS', 'NNPS']: 
                    continue
            
            # --- PLURAL HANDLING ---
            # If word is Plural Noun (NNS) or Proper Plural (NNPS)
            # Append <MULTIPLE> token
            if tag in ['NNS', 'NNPS']:
                lemma = f"{lemma} <MULTIPLE>"

            # --- SAFETY FILTERS ---
            if original.endswith('s') and lemma.split()[0] == original[:-1] and len(original) > 4:
                # "Olympus" -> "Olympu" protection (only if not tagged plural)
                if tag not in ['NNS', 'NNPS']:
                    continue

            if original == "icing": continue 

            lemma_map[original] = lemma

    with open(save_map_path, 'w', encoding='utf-8') as f:
        json.dump(lemma_map, f, ensure_ascii=False, indent=4)
    print(f"Lemma map saved. Size: {len(lemma_map)}")

    # --- PHASE 3: Build Final Vocab ---
    print("\nPhase 3: Building Final Vocab (Expanding Tokens)...")
    global_final_counts = collections.Counter()
    
    for raw_word, count in global_raw_words.items():
        # Get lemma (e.g., "dogs" -> "dog <MULTIPLE>")
        final_string = lemma_map.get(raw_word, raw_word)
        
        # ASCII Check
        if not re.match(r'^[\x00-\x7F]+$', final_string):
            continue

        # Split! "dog <MULTIPLE>" -> ["dog", "<MULTIPLE>"]
        # Both "dog" and "<MULTIPLE>" get the count increment
        sub_tokens = final_string.split()
        for t in sub_tokens:
            global_final_counts[t] += count

    os.makedirs("Plots", exist_ok=True)
    # Use raw vs final for plot
    plot_rank_frequency_comparison(global_raw_words, global_final_counts, min_frequency, "Plots/vocab_rank_freq.png")
    plot_caption_lengths(global_lengths, "Plots/caption_length_hist.png")

    print(f"Filtering words < {min_frequency}...")
    filtered_vocab = {w: c for w, c in global_final_counts.items() if c >= min_frequency}
    sorted_words = sorted(filtered_vocab.items(), key=lambda item: item[1], reverse=True)
    
    # --- SPECIAL TOKENS ---
    vocab = {}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    vocab["<MULTIPLE>"] = 2  # Explicit ID for our new token
    
    # Start words at 3
    # Check if <MULTIPLE> is in sorted_words to avoid duplicate
    # It likely is, so we skip it in the loop
    current_idx = 3
    for word, _ in sorted_words:
        if word in ["<PAD>", "<UNK>", "<MULTIPLE>"]:
            continue
        vocab[word] = current_idx
        current_idx += 1
    
    with open(save_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
        
    print(f"Done. Final Vocab Size: {len(vocab)}")
    return vocab

if __name__ == "__main__":
    path = Config.DATABASE_PATH
    create_vocabulary(path, min_frequency=2, num_workers=5)