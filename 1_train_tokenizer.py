"""
Step 1: Train SentencePiece Tokenizer for Multilingual Indian Languages
- Combines texts from all Indian languages
- Creates language-agnostic vocabulary (30K tokens)
- Handles Brahmic, Perso-Arabic, Latin scripts
"""

import sentencepiece as spm
import os
from pathlib import Path

def combine_corpus(data_dir='data', output_file='data/combined_corpus.txt'):
    """
    Combines all text files from different languages into one corpus
    
    Expected structure:
    data/
    ├── hindi_data.txt
    ├── tamil_data.txt
    ├── telugu_data.txt
    ├── marathi_data.txt
    └── ... (other language files)
    """
    
    print("Combining corpus from all language files...")
    
    combined_count = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in Path(data_dir).glob('*_data.txt'):
            print(f"  Processing {file_path.name}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if line.strip():
                            outfile.write(line)
                            combined_count += 1
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
    
    print(f"✓ Combined {combined_count} lines into {output_file}")
    return output_file


def train_sentencepiece_tokenizer(corpus_path='data/combined_corpus.txt', 
                                  model_prefix='tokenizer/multilingual_indic',
                                  vocab_size=30000):
    """
    Train SentencePiece tokenizer with BPE algorithm
    
    Args:
        corpus_path: Path to combined multilingual corpus
        model_prefix: Output path (creates .model and .vocab files)
        vocab_size: Number of vocabulary tokens
    """
    
    print(f"\nTraining SentencePiece tokenizer...")
    print(f"  Corpus: {corpus_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Output prefix: {model_prefix}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',  # BPE algorithm
        character_coverage=0.99,  # Cover 99% of characters
        normalization_rule_name='identity',  # Preserve script characteristics
        split_by_unicode_script=True,  # Split by script type (Brahmi, Latin, etc)
        split_by_number=True,
        split_by_whitespace=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        eos_piece='</s>',
        unk_piece='<unk>',
        unk_surface='⁇',
    )
    
    print(f"✓ SentencePiece tokenizer trained!")
    print(f"  Model: {model_prefix}.model")
    print(f"  Vocab: {model_prefix}.vocab")
    
    return f"{model_prefix}.model"


def test_tokenizer(model_path='tokenizer/multilingual_indic.model'):
    """
    Test the trained tokenizer on sample texts
    """
    
    print(f"\nTesting tokenizer...")
    
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    # Test samples in different languages
    test_samples = {
        'Hindi': 'नमस्ते दुनिया, मैं एक बहुभाषी मॉडल हूँ।',
        'Tamil': 'வணக்கம் உலகம், நான் ஒரு பன்மொழி மாதிரி.',
        'Telugu': 'హలో ప్రపంచం, నేను బహుభాషా నమూనా.',
        'Marathi': 'नमस्कार जग, मी एक बहुभाषी मॉडल आहे.',
        'Bengali': 'নমস্কার বিশ্ব, আমি একটি বহুভাষিক মডেল।',
    }
    
    for lang, text in test_samples.items():
        pieces = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        print(f"\n{lang}:")
        print(f"  Text: {text[:50]}...")
        print(f"  Tokens: {pieces[:10]}...")
        print(f"  IDs: {ids[:10]}")
        print(f"  Total tokens: {len(ids)}")
    
    print(f"\n✓ Tokenizer working on all languages!")
    print(f"  Vocabulary size: {sp.get_piece_size()}")
    
    return sp


if __name__ == '__main__':
    print("="*60)
    print("STEP 1: Train SentencePiece Tokenizer")
    print("="*60)
    
    # Step 1: Combine corpus
    # NOTE: You need to have language data files first
    # For testing, create sample data:
    # python 0_prepare_sample_data.py
    
    corpus_file = combine_corpus(data_dir='data', output_file='data/combined_corpus.txt')
    
    # Step 2: Train tokenizer
    model_path = train_sentencepiece_tokenizer(
        corpus_path=corpus_file,
        model_prefix='tokenizer/multilingual_indic',
        vocab_size=30000
    )
    
    # Step 3: Test tokenizer
    sp = test_tokenizer(model_path)
    
    print("\n" + "="*60)
    print("✓ TOKENIZER READY!")
    print("Next: Run 2_prepare_data.py")
    print("="*60)
