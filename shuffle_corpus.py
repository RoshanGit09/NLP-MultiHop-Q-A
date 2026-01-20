"""
Shuffle corpus file for balanced tokenizer training
Run this before 1_train_tokenizer.py
"""

import random
import os

CORPUS_FILE = 'data/combined_corpus-1.txt'
OUTPUT_FILE = 'data/combined_corpus-1.txt'  # Overwrite same file

def shuffle_corpus():
    print(f"Loading corpus from {CORPUS_FILE}...")
    
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"  Loaded {len(lines):,} lines")
    
    print("Shuffling...")
    random.seed(42)  # For reproducibility
    random.shuffle(lines)
    
    print(f"Saving shuffled corpus to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✓ Corpus shuffled! {len(lines):,} lines")
    
    # Show sample of shuffled data (first 10 lines)
    print("\nSample of shuffled corpus (first 5 lines):")
    for i, line in enumerate(lines[:5]):
        print(f"  {i+1}: {line[:60].strip()}...")

if __name__ == '__main__':
    shuffle_corpus()
