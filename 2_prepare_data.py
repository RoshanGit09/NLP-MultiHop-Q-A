"""
Step 2: Prepare Training Data from Sangraha
- Loads Sangraha dataset from HuggingFace Hub
- Tokenizes using SentencePiece
- Prepares batches for training
"""

from datasets import load_dataset, Dataset
import sentencepiece as spm
import numpy as np
from tqdm import tqdm
import os

class MultilingualDataProcessor:
    """Handles data loading and tokenization"""
    
    def __init__(self, tokenizer_path='tokenizer/multilingual_indic.model', 
                 max_length=512):
        """
        Args:
            tokenizer_path: Path to SentencePiece model
            max_length: Maximum sequence length
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.max_length = max_length
        self.pad_id = 0
        self.eos_id = 3
        
        print(f"✓ Tokenizer loaded: {tokenizer_path}")
        print(f"  Vocabulary size: {self.sp.get_piece_size()}")
    
    def tokenize_text(self, text):
        """
        Tokenize a single text
        
        Args:
            text: Input text string
            
        Returns:
            list: Token IDs
        """
        token_ids = self.sp.encode_as_ids(text)
        
        # Pad or truncate to max_length
        if len(token_ids) < self.max_length:
            token_ids += [self.pad_id] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        
        return token_ids
    
    def process_batch(self, texts):
        """
        Process a batch of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            dict: {'input_ids': [...]}
        """
        input_ids = [self.tokenize_text(text) for text in texts]
        return {'input_ids': input_ids}


def load_sangraha_data(subset='verified', num_samples=None):
    """
    Load Sangraha dataset from HuggingFace Hub
    
    Args:
        subset: 'verified' (high quality), 'verified+synthetic' (all quality levels), etc.
        num_samples: Number of samples to load (None = all)
        
    Returns:
        Dataset: HuggingFace dataset with 'text' field
    """
    
    print(f"Loading Sangraha dataset (subset: {subset})...")
    
    # Load from HuggingFace Hub
    dataset = load_dataset("ai4bharat/sangraha", data_dir=subset, split='train')
    
    print(f"  Loaded {len(dataset)} examples")
    
    # Sample if specified
    if num_samples:
        indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
        dataset = dataset.select(indices)
        print(f"  Sampled to {len(dataset)} examples")
    
    return dataset


def prepare_training_data(tokenizer_path='tokenizer/multilingual_indic.model',
                         output_dir='data/tokenized_sangraha',
                         num_samples=5_000_000,
                         batch_size=1000):
    """
    Prepare complete training dataset
    
    Args:
        tokenizer_path: Path to SentencePiece model
        output_dir: Output directory for tokenized data
        num_samples: Number of samples to use
        batch_size: Batch size for tokenization
    """
    
    print("="*60)
    print("PREPARING TRAINING DATA")
    print("="*60)
    
    # Step 1: Load Sangraha
    dataset = load_sangraha_data(subset='verified', num_samples=num_samples)
    
    # Step 2: Initialize processor
    processor = MultilingualDataProcessor(tokenizer_path)
    
    # Step 3: Tokenize dataset
    print(f"\nTokenizing {len(dataset)} examples...")
    
    def tokenize_function(examples):
        return processor.process_batch(examples['text'])
    
    # Tokenize with batching for speed
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=['text'],  # Remove original text to save space
        desc="Tokenizing"
    )
    
    # Step 4: Split into train/eval
    print("\nSplitting into train/validation...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"  Training: {len(split_dataset['train'])} examples")
    print(f"  Validation: {len(split_dataset['test'])} examples")
    
    # Step 5: Save
    print(f"\nSaving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    split_dataset.save_to_disk(output_dir)
    
    print(f"✓ Data prepared and saved!")
    print(f"  Train: {len(split_dataset['train'])} examples")
    print(f"  Val: {len(split_dataset['test'])} examples")
    
    # Print sample
    print(f"\nSample tokenized example:")
    sample = tokenized_dataset[0]
    print(f"  Input IDs (first 10): {sample['input_ids'][:10]}")
    print(f"  Length: {len(sample['input_ids'])}")
    
    return split_dataset


if __name__ == '__main__':
    # Configuration
    TOKENIZER_PATH = 'tokenizer/multilingual_indic.model'
    OUTPUT_DIR = 'data/tokenized_sangraha'
    NUM_SAMPLES = 5_000_000  # 5M samples ≈ 10B tokens
    
    print(f"Configuration:")
    print(f"  Tokenizer: {TOKENIZER_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Samples: {NUM_SAMPLES:,}")
    
    # Prepare data
    dataset = prepare_training_data(
        tokenizer_path=TOKENIZER_PATH,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        batch_size=1000
    )
    
    print("\n" + "="*60)
    print("✓ DATA READY!")
    print("Next: Run 3_train_model.py")
    print("="*60)
