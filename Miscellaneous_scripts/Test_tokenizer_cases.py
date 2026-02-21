import os
import Tokenizer_lib as Tok_lib

def run_tests():
    print("Loading Tokenizer...")
    
    # Ensure files exist
    if not os.path.exists("vocab.json") or not os.path.exists("lemma_map.json"):
        print("❌ Error: vocab.json or lemma_map.json not found!")
        return

    # Initialize Tokenizer
    tokenizer = Tok_lib.SimpleTokenizer("vocab.json", "lemma_map.json", max_length=128)
    
    # Define Test Cases
    test_cases = [
        # --- GROUP 1: PLURALS ---
        "Two dogs are running",
        "The leaves fell",
        "Three men and women",
        "I saw mice",
        
        # --- GROUP 2: SLASH & PUNCTUATION ---
        "trash/grass/people",
        "mix—black",
        "weird|pipe",
        "semi;colon",
        "(brackets)",
        "channel>",
        "boards'",
        
        # --- GROUP 3: CHINESE & ARTIFACTS ---
        "line左侧",
        "horse左侧，岩石在左下角",
        "100% pure",
        
        # --- GROUP 4: SAFETY FILTERS ---
        "Olympus camera",
        "Paris lights",
        "I am icing the cake",
        
        # --- GROUP 5: COMPLEX SENTENCES ---
        "A large wild cat is pursuing a horse across a meadow.",
        "Three men sit on the carpet.",

        # --- GROUP 6: LONG & MIXED SCENARIOS (Stress Tests) ---
        "A group of young men are playing basketball on an outdoor court with a chain-link fence.",
        "Several cars/trucks are parked near the building's entrance; some people are walking by.",
        "The chef is icing a chocolate cake while two dogs watch from the floor.",
        "An old photo of a street in Paris, 1920s, with horse-drawn carriages.",
        "A panoramic view: tree/grass/sky左侧 and mountains in the background."
    ]

    print(f"\n{'='*60}")
    print(f"TOKENIZER TEST RESULTS")
    print(f"{'='*60}\n")

    for text in test_cases:
        # 1. Encode
        token_ids = tokenizer.encode(text)
        
        # 2. Decode
        decoded = tokenizer.decode(token_ids, skip_special=True)
        clean_decoded = decoded.replace(" <PAD>", "")
        
        # 3. Formatted Output
        print(f"INPUT:   {text}")
        print(f"DECODED: {clean_decoded}")
        print("-" * 20)

if __name__ == "__main__":
    run_tests()