"""
Step 2: Prepare Translation Training Data
- Downloads paired data from Samanantar (EN ↔ Indic)
- Tokenizes BOTH source (English) and target (Indic)
- Saves as translation pairs for seq2seq training
"""

from datasets import load_dataset, Dataset
import sentencepiece as spm
import numpy as np
from tqdm import tqdm
import os

# Configuration
TOKENIZER_PATH = 'tokenizer/multilingual_indic-1.model'
OUTPUT_DIR = 'data/tokenized_translation'
NUM_SAMPLES = 500_000  # Samples per language pair
MAX_LENGTH = 512

# Languages to include (Samanantar format)
LANGUAGES = ['hi', 'ta', 'te', 'mr', 'kn', 'ml']


class TranslationDataProcessor:
    """Handles parallel data loading and tokenization for translation"""
    
    def __init__(self, tokenizer_path=TOKENIZER_PATH, max_length=MAX_LENGTH):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.max_length = max_length
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3
        
        print(f"✓ Tokenizer loaded: {tokenizer_path}")
        print(f"  Vocabulary size: {self.sp.get_piece_size():,}")
    
    def tokenize_pair(self, src_text, tgt_text):
        """Tokenize source and target texts"""
        # Source (English)
        src_ids = self.sp.encode_as_ids(src_text)
        if len(src_ids) > self.max_length:
            src_ids = src_ids[:self.max_length]
        
        # Target (Indic) - add BOS/EOS
        tgt_ids = self.sp.encode_as_ids(tgt_text)
        if len(tgt_ids) > self.max_length - 2:
            tgt_ids = tgt_ids[:self.max_length - 2]
        tgt_ids = [self.bos_id] + tgt_ids + [self.eos_id]
        
        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
        }


def download_translation_pairs(languages, samples_per_lang, output_dir):
    """
    Download parallel translation data from Samanantar
    
    Returns Dataset with 'src_ids' and 'tgt_ids' fields
    """
    
    print("="*60)
    print("DOWNLOADING TRANSLATION DATA FROM SAMANANTAR")
    print("="*60)
    print(f"Languages: {len(languages)}")
    print(f"Samples per language: {samples_per_lang:,}")
    
    all_pairs = []
    
    for lang in tqdm(languages, desc="Downloading languages"):
        try:
            print(f"\n  Loading en-{lang} translation pairs...")
            
            # Load Samanantar dataset
            dataset = load_dataset(
                "ai4bharat/samanantar",
                lang,
                split="train",
                streaming=True
            )
            
            count = 0
            for example in dataset:
                src = example.get('src', '')  # English
                tgt = example.get('tgt', '')  # Indic
                
                # Validate both texts
                if (src and tgt and 
                    len(src.strip()) > 20 and 
                    len(tgt.strip()) > 20 and
                    not tgt.isascii()):
                    
                    # Add forward pair (EN → Indic)
                    all_pairs.append({
                        'src': src.strip(),
                        'tgt': tgt.strip(),
                        'direction': f'en→{lang}'
                    })
                    
                    # Add reverse pair (Indic → EN)
                    all_pairs.append({
                        'src': tgt.strip(),
                        'tgt': src.strip(),
                        'direction': f'{lang}→en'
                    })
                    
                    count += 1
                    
                    if count >= samples_per_lang:
                        break
            
            print(f"    {lang}: {count:,} pairs ({count*2:,} with reverse)")
            
        except Exception as e:
            print(f"    {lang}: Error - {e}")
    
    print(f"\n✓ Downloaded {len(all_pairs):,} total translation pairs (bidirectional)")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict({
        'src': [p['src'] for p in all_pairs],
        'tgt': [p['tgt'] for p in all_pairs],
        'direction': [p['direction'] for p in all_pairs],
    })
    
    return dataset


def prepare_translation_data(
    tokenizer_path=TOKENIZER_PATH,
    output_dir=OUTPUT_DIR,
    languages=LANGUAGES,
    num_samples=NUM_SAMPLES,
):
    """
    Prepare complete translation training dataset
    """
    
    print("="*60)
    print("PREPARING TRANSLATION DATA")
    print("="*60)
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Output: {output_dir}")
    print(f"  Languages: {len(languages)}")
    print(f"  Samples per language: {num_samples:,}")
    
    # Step 1: Download paired data
    dataset = download_translation_pairs(languages, num_samples, output_dir)
    
    # Step 2: Initialize tokenizer
    processor = TranslationDataProcessor(tokenizer_path)
    
    # Step 3: Tokenize pairs
    print(f"\nTokenizing {len(dataset):,} translation pairs...")
    
    def tokenize_function(examples):
        results = {
            'src_ids': [],
            'tgt_ids': [],
        }
        
        for src, tgt in zip(examples['src'], examples['tgt']):
            pair = processor.tokenize_pair(src, tgt)
            results['src_ids'].append(pair['src_ids'])
            results['tgt_ids'].append(pair['tgt_ids'])
        
        return results
    
    # Tokenize with batching
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=['src', 'tgt'],  # Remove text to save space
        desc="Tokenizing pairs"
    )
    
    # Step 4: Split into train/eval
    print("\nSplitting into train/validation...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
    
    print(f"  Training: {len(split_dataset['train']):,} pairs")
    print(f"  Validation: {len(split_dataset['test']):,} pairs")
    
    # Step 5: Save
    print(f"\nSaving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    split_dataset.save_to_disk(output_dir)
    
    print(f"\n✓ Translation data prepared!")
    print(f"  Train: {len(split_dataset['train']):,} pairs")
    print(f"  Val: {len(split_dataset['test']):,} pairs")
    
    # Print sample
    sample = split_dataset['train'][0]
    print(f"\nSample translation pair:")
    print(f"  Source IDs (EN): {sample['src_ids'][:20]}...")
    print(f"  Target IDs (Indic): {sample['tgt_ids'][:20]}...")
    
    return split_dataset


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare translation training data')
    parser.add_argument('--tokenizer', default=TOKENIZER_PATH,
                       help='Path to tokenizer model')
    parser.add_argument('--output', default=OUTPUT_DIR,
                       help='Output directory')
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES,
                       help='Samples per language')
    
    args = parser.parse_args()
    
    dataset = prepare_translation_data(
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        num_samples=args.samples,
    )
    
    print("\n" + "="*60)
    print("✓ DATA READY FOR TRANSLATION TRAINING!")
    print("Next: Run 'python 3_train_model.py'")
    print("="*60)
