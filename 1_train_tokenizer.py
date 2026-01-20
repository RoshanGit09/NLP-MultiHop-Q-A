# """
# Step 1: Train SentencePiece Tokenizer for Multilingual Indian Languages
# - Downloads samples from Sangraha dataset (HuggingFace)
# - Creates language-balanced vocabulary (50K tokens)
# - Handles Brahmic, Perso-Arabic, Latin scripts
# - Optimized for MLM pretraining

# Key Improvements:
# - Uses Sangraha dataset directly (no manual data needed)
# - Larger vocabulary (50K) for better multilingual coverage
# - Unigram model (better for morphologically rich languages)
# - Higher character coverage (99.95%)
# - Includes special tokens for BERT-style training
# - Balanced language sampling
# """

# import sentencepiece as spm
# import os
# from pathlib import Path
# from datasets import load_dataset
# import random
# from tqdm import tqdm


# # =============================================================================
# # CONFIGURATION - CHANGE THESE FOR BETTER RESULTS
# # =============================================================================

# CONFIG = {
#     # Vocabulary size: Higher = better coverage but slower training
#     # Recommended: 50K for multilingual, 30K for single language
#     'vocab_size': 50000,
    
#     # Model type: 'unigram' or 'bpe'
#     # Unigram is better for morphologically rich languages (Indian languages)
#     # BPE is faster but less optimal for complex scripts
#     'model_type': 'unigram',
    
#     # Character coverage: Higher = more rare characters included
#     # 0.9995 covers rare characters in Sanskrit, Manipuri, etc.
#     'character_coverage': 0.9995,
    
#     # Number of samples per language for training
#     # More samples = better vocabulary, but slower training
#     'samples_per_language': 10000,
    
#     # Languages to include (Sangraha language codes)
#     # Add/remove based on your needs
#     'languages': [
#         'hin',  # Hindi
#         'tam',  # Tamil
#     ],
    
#     # Output paths
#     'corpus_file': 'data/combined_corpus.txt',
#     'model_prefix': 'tokenizer/multilingual_indic',
# }


# def download_sangraha_corpus(languages, samples_per_lang, output_file):
#     """
#     Download and combine text from Sangraha dataset
    
#     This is the key improvement - uses real data from HuggingFace
#     instead of requiring local files.
#     """
    
#     print(f"Downloading Sangraha corpus from HuggingFace...")
#     print(f"  Languages: {len(languages)}")
#     print(f"  Samples per language: {samples_per_lang:,}")
    
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
#     total_lines = 0
    
#     with open(output_file, 'w', encoding='utf-8') as outfile:
#         for lang in tqdm(languages, desc="Downloading languages"):
#             try:
#                 # Load Sangraha dataset for this language
#                 # Using streaming to avoid downloading entire dataset
#                 dataset = load_dataset(
#                     "ai4bharat/sangraha",
#                     data_dir="verified",
#                     split="train",
#                     streaming=True
#                 )
                
#                 # Filter by language and sample
#                 count = 0
#                 for example in dataset:
#                     if example.get('lang', '') == lang or lang in str(example):
#                         text = example.get('text', '')
#                         if text and len(text.strip()) > 20:  # Skip very short texts
#                             outfile.write(text.strip() + '\n')
#                             count += 1
#                             total_lines += 1
                            
#                             if count >= samples_per_lang:
#                                 break
                
#                 print(f"  {lang}: {count:,} samples")
                
#             except Exception as e:
#                 print(f"  {lang}: Error - {e}")
#                 # Fallback: add sample sentences for this language
#                 fallback_samples = get_fallback_samples(lang)
#                 for sample in fallback_samples:
#                     outfile.write(sample + '\n')
#                     total_lines += 1
    
#     print(f"\n✓ Corpus created: {output_file}")
#     print(f"  Total lines: {total_lines:,}")
    
#     return output_file


# def get_fallback_samples(lang):
#     """
#     Fallback samples if Sangraha download fails
#     These are just for testing - real training needs more data
#     """
    
#     fallback = {
#         'hin': [
#             'नमस्ते दुनिया, मैं एक बहुभाषी मॉडल हूँ।',
#             'भारत एक विविधताओं से भरा देश है।',
#             'हिंदी भारत की राजभाषा है।',
#             'प्राकृतिक भाषा प्रसंस्करण एक रोमांचक क्षेत्र है।',
#             'मशीन लर्निंग से कंप्यूटर सीखते हैं।',
#         ] * 100,
#         'tam': [
#             'வணக்கம் உலகம், நான் ஒரு பன்மொழி மாதிரி.',
#             'தமிழ் உலகின் பழமையான மொழிகளில் ஒன்று.',
#             'இயற்கை மொழி செயலாக்கம் ஒரு சுவாரஸ்யமான துறை.',
#             'இயந்திர கற்றல் கணினிகளை கற்க உதவுகிறது.',
#             'தமிழ்நாடு அழகான மாநிலம்.',
#         ] * 100,
#         'tel': [
#             'హలో ప్రపంచం, నేను బహుభాషా నమూనా.',
#             'తెలుగు అందమైన భాష.',
#             'సహజ భాషా ప్రాసెసింగ్ ఆసక్తికరమైన రంగం.',
#             'మషిన్ లెర్నింగ్ కంప్యూటర్లకు నేర్పిస్తుంది.',
#             'ఆంధ్ర ప్రదేశ్ అందమైన రాష్ట్రం.',
#         ] * 100,
#         'mar': [
#             'नमस्कार जग, मी एक बहुभाषी मॉडेल आहे.',
#             'मराठी महाराष्ट्राची राजभाषा आहे.',
#             'नैसर्गिक भाषा प्रक्रिया रोमांचक क्षेत्र आहे.',
#             'मशीन लर्निंग कंप्युटरला शिकवते.',
#             'महाराष्ट्र समृद्ध सांस्कृतिक वारसा असलेले राज्य आहे.',
#         ] * 100,
#         'ben': [
#             'নমস্কার বিশ্ব, আমি একটি বহুভাষিক মডেল।',
#             'বাংলা পশ্চিমবঙ্গের রাজ্য ভাষা।',
#             'প্রাকৃতিক ভাষা প্রক্রিয়াকরণ একটি আকর্ষণীয় ক্ষেত্র।',
#             'মেশিন লার্নিং কম্পিউটারকে শেখায়।',
#             'পশ্চিমবঙ্গ সুন্দর রাজ্য।',
#         ] * 100,
#         'guj': [
#             'નમસ્તે વિશ્વ, હું એક બહુભાષી મોડેલ છું.',
#             'ગુજરાતી ગુજરાતની રાજ્ય ભાષા છે.',
#             'કુદરતી ભાષા પ્રક્રિયા એક રસપ્રદ ક્ષેત્ર છે.',
#             'મશીન લર્નિંગ કમ્પ્યુટરને શીખવે છે.',
#             'ગુજરાત સુંદર રાજ્ય છે.',
#         ] * 100,
#     }
    
#     return fallback.get(lang, fallback.get('hin', []))


# def create_local_corpus(data_dir='data', output_file='data/combined_corpus.txt'):
#     """
#     Alternative: Combine local text files into corpus
#     Use this if you have your own data files
    
#     Expected structure:
#     data/
#     ├── hindi_data.txt
#     ├── tamil_data.txt
#     └── ... (other language files)
#     """
    
#     print("Combining local corpus files...")
    
#     combined_count = 0
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
#     with open(output_file, 'w', encoding='utf-8') as outfile:
#         for file_path in Path(data_dir).glob('*_data.txt'):
#             print(f"  Processing {file_path.name}...")
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as infile:
#                     for line in infile:
#                         if line.strip() and len(line.strip()) > 10:
#                             outfile.write(line)
#                             combined_count += 1
#             except Exception as e:
#                 print(f"  Error reading {file_path}: {e}")
    
#     print(f"✓ Combined {combined_count:,} lines into {output_file}")
#     return output_file


# def train_sentencepiece_tokenizer(corpus_path, model_prefix, vocab_size=50000, 
#                                    model_type='unigram', character_coverage=0.9995):
#     """
#     Train SentencePiece tokenizer with optimized settings for Indian languages
    
#     Key improvements over default:
#     - Unigram model (better for morphologically rich languages)
#     - Higher character coverage (for rare scripts)
#     - Special tokens for BERT-style training
#     - Better handling of mixed scripts
#     """
    
#     print(f"\n{'='*60}")
#     print("TRAINING SENTENCEPIECE TOKENIZER")
#     print('='*60)
#     print(f"  Corpus: {corpus_path}")
#     print(f"  Vocab size: {vocab_size:,}")
#     print(f"  Model type: {model_type}")
#     print(f"  Character coverage: {character_coverage}")
#     print(f"  Output prefix: {model_prefix}")
    
#     # Create output directory
#     os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    
#     # Special tokens for BERT-style training
#     # These are critical for MLM pretraining
#     special_tokens = [
#         '[PAD]',   # Padding token
#         '[UNK]',   # Unknown token  
#         '[CLS]',   # Classification token (sentence start)
#         '[SEP]',   # Separator token (sentence end)
#         '[MASK]',  # Mask token for MLM
#     ]
    
#     # Train SentencePiece
#     spm.SentencePieceTrainer.train(
#         input=corpus_path,
#         model_prefix=model_prefix,
#         vocab_size=vocab_size,
        
#         # Model type: 'unigram' recommended for Indian languages
#         model_type=model_type,
        
#         # Character coverage: Higher = more rare characters
#         character_coverage=character_coverage,
        
#         # Normalization: Standard NFKC is better for noisy multilingual text
#         normalization_rule_name='nmt_nfkc',
        
#         # Script handling: Split by unicode script for Brahmic scripts
#         split_by_unicode_script=True,
#         split_by_number=True,
#         split_by_whitespace=True,
        
#         # REMOVED byte_fallback=True - it was causing byte-level tokenization!
#         # Instead, let unknown chars go to <unk>
#         byte_fallback=False,
        
#         # Special token IDs (BERT-compatible)
#         pad_id=0,      # [PAD]
#         unk_id=1,      # [UNK]
#         bos_id=2,      # [CLS] (beginning of sentence)
#         eos_id=3,      # [SEP] (end of sentence)
        
#         # Add user-defined special tokens (includes [PAD], [UNK], [CLS], [SEP], [MASK])
#         user_defined_symbols=special_tokens,
        
#         # Training efficiency
#         input_sentence_size=10000000,  # Max sentences to use
#         shuffle_input_sentence=True,    # Shuffle for better sampling
#         train_extremely_large_corpus=True,  # Better for 1GB+ corpus
#         num_threads=16,  # Faster training
#     )
    
#     print(f"\n✓ SentencePiece tokenizer trained!")
#     print(f"  Model: {model_prefix}.model")
#     print(f"  Vocab: {model_prefix}.vocab")
    
#     return f"{model_prefix}.model"


# def test_tokenizer(model_path):
#     """
#     Test the trained tokenizer on sample texts from different languages
#     """
    
#     print(f"\n{'='*60}")
#     print("TESTING TOKENIZER")
#     print('='*60)
    
#     sp = spm.SentencePieceProcessor()
#     sp.load(model_path)
    
#     # Print tokenizer info
#     print(f"\nTokenizer Info:")
#     print(f"  Vocabulary size: {sp.get_piece_size():,}")
#     print(f"  Special tokens:")
#     print(f"    PAD (0): {sp.id_to_piece(0)}")
#     print(f"    UNK (1): {sp.id_to_piece(1)}")
#     print(f"    BOS (2): {sp.id_to_piece(2)}")
#     print(f"    EOS (3): {sp.id_to_piece(3)}")
    
#     # Test samples in different languages
#     test_samples = {
#         'Hindi': 'नमस्ते दुनिया, मैं एक बहुभाषी मॉडल हूँ।',
#         'Tamil': 'வணக்கம் உலகம், நான் ஒரு பன்மொழி மாதிரி.',
#         'Telugu': 'హలో ప్రపంచం, నేను బహుభాషా నమూనా.',
#         'Marathi': 'नमस्कार जग, मी एक बहुभाषी मॉडेल आहे.',
#         'Bengali': 'নমস্কার বিশ্ব, আমি একটি বহুভাষিক মডেল।',
#         'Gujarati': 'નમસ્તે વિશ્વ, હું એક બહુભાષી મોડેલ છું.',
#         'Kannada': 'ಹಲೋ ಜಗತ್ತು, ನಾನು ಬಹುಭಾಷಾ ಮಾದರಿ.',
#         'Malayalam': 'ഹലോ ലോകം, ഞാൻ ഒരു ബഹുഭാഷാ മാതൃകയാണ്.',
#         'English': 'Hello world, I am a multilingual model.',
#     }
    
#     print(f"\nTokenization Results:")
#     for lang, text in test_samples.items():
#         pieces = sp.encode_as_pieces(text)
#         ids = sp.encode_as_ids(text)
        
#         # Fertility: Number of tokens per word (lower is better)
#         words = text.split()
#         fertility = len(ids) / max(len(words), 1)
        
#         print(f"\n{lang}:")
#         print(f"  Text: {text}")
#         print(f"  Tokens ({len(ids)}): {pieces[:12]}{'...' if len(pieces) > 12 else ''}")
#         print(f"  Fertility: {fertility:.2f} tokens/word")
    
#     # Summary
#     print(f"\n{'='*60}")
#     print("TOKENIZER SUMMARY")
#     print('='*60)
#     print(f"  Vocabulary size: {sp.get_piece_size():,}")
#     print(f"  Model ready for training!")
    
#     return sp


# def main():
#     """
#     Main function to train tokenizer
#     """
    
#     print("="*60)
#     print("STEP 1: TRAIN SENTENCEPIECE TOKENIZER")
#     print("="*60)
#     print(f"\nConfiguration:")
#     print(f"  Vocab size: {CONFIG['vocab_size']:,}")
#     print(f"  Model type: {CONFIG['model_type']}")
#     print(f"  Character coverage: {CONFIG['character_coverage']}")
#     print(f"  Languages: {len(CONFIG['languages'])}")
    
#     # Step 1: Prepare corpus
#     print(f"\n[1/3] Preparing corpus...")
    
#     # Check if corpus file already exists (skip download if so)
#     if os.path.exists(CONFIG['corpus_file']):
#         print(f"  ✓ Corpus file already exists: {CONFIG['corpus_file']}")
#         corpus_file = CONFIG['corpus_file']
#         # Show file size
#         file_size = os.path.getsize(corpus_file) / (1024**3)
#         print(f"  Size: {file_size:.2f} GB")
#     else:
#         # Check if local data exists
#         local_files = list(Path('data').glob('*_data.txt')) if os.path.exists('data') else []
        
#         if local_files:
#             print(f"  Found {len(local_files)} local data files, using those...")
#             corpus_file = create_local_corpus(
#                 data_dir='data',
#                 output_file=CONFIG['corpus_file']
#             )
#         else:
#             print(f"  No local data found, downloading from Sangraha...")
#             corpus_file = download_sangraha_corpus(
#                 languages=CONFIG['languages'],
#                 samples_per_lang=CONFIG['samples_per_language'],
#                 output_file=CONFIG['corpus_file']
#             )
    
#     # Step 2: Train tokenizer
#     print(f"\n[2/3] Training tokenizer...")
#     model_path = train_sentencepiece_tokenizer(
#         corpus_path=corpus_file,
#         model_prefix=CONFIG['model_prefix'],
#         vocab_size=CONFIG['vocab_size'],
#         model_type=CONFIG['model_type'],
#         character_coverage=CONFIG['character_coverage'],
#     )
    
#     # Step 3: Test tokenizer
#     print(f"\n[3/3] Testing tokenizer...")
#     sp = test_tokenizer(model_path)
    
#     print("\n" + "="*60)
#     print("✓ TOKENIZER TRAINING COMPLETE!")
#     print("="*60)
#     print(f"\nOutput files:")
#     print(f"  Model: {CONFIG['model_prefix']}.model")
#     print(f"  Vocab: {CONFIG['model_prefix']}.vocab")
#     print(f"\nNext step:")
#     print(f"  python 2_prepare_data.py")
#     print("="*60)


# if __name__ == '__main__':
#     main()


import sentencepiece as spm
import os
from pathlib import Path
from datasets import load_dataset
import random
from tqdm import tqdm


# =============================================================================
# CONFIGURATION - CHANGE THESE FOR BETTER RESULTS
# =============================================================================

CONFIG = {
    # Vocabulary size: Higher = better coverage but slower training
    # 50K is NOT enough for 6+ languages with different scripts!
    # Recommended: 100K-128K for multilingual Indian languages
    'vocab_size': 100000,  # INCREASED from 50K to accommodate all languages
    
    # Model type: 'unigram' or 'bpe'
    # Unigram is better for morphologically rich languages (Indian languages)
    # BPE is faster but less optimal for complex scripts
    'model_type': 'unigram',
    
    # Character coverage: Higher = more rare characters included
    # 0.9995 covers rare characters in Sanskrit, Manipuri, etc.
    'character_coverage': 0.9995,
    
    # Number of samples per language for training
    # More samples = better vocabulary, but slower training
    # INCREASED to ensure good representation of all languages
    'samples_per_language': 20000,
    
    # Dataset source: 'samanantar' (cleaner) or 'sangraha' (more data)
    'dataset_source': 'samanantar',  # CHANGED: Samanantar is cleaner!
    
    # Languages to include
    # Samanantar codes: hi, ta, te, mr, kn, ml, bn, gu, pa, or
    'languages': [
        'hi',   # Hindi
        'ta',   # Tamil
        'te',   # Telugu
        'mr',   # Marathi
        'kn',   # Kannada
        'ml',   # Malayalam
    ],
    
    # Output paths
    'corpus_file': 'data/combined_corpus-1.txt',
    'model_prefix': 'tokenizer/multilingual_indic-1',
}


def download_samanantar_corpus(languages, samples_per_lang, output_file):
    """
    Download and combine text from Samanantar dataset (cleaner than Sangraha)
    
    Samanantar is a parallel corpus from AI4Bharat with high-quality translations.
    We use only the Indic language side ('tgt' field) for tokenizer training.
    
    Data structure: {'idx': int, 'src': English text, 'tgt': Indic text}
    """
    
    print(f"Downloading Samanantar corpus from HuggingFace...")
    print(f"  Languages: {len(languages)}")
    print(f"  Samples per language: {samples_per_lang:,}")
    print(f"  Dataset: ai4bharat/samanantar (high quality!)")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for lang in tqdm(languages, desc="Downloading languages"):
            try:
                # Samanantar uses just the language code (e.g., 'hi', 'ta')
                print(f"  Loading {lang}...")
                dataset = load_dataset(
                    "ai4bharat/samanantar",
                    lang,  # Just 'hi', 'ta', 'te', etc.
                    split="train",
                    streaming=True
                )
                
                count = 0
                for example in dataset:
                    # Get both English and Indic text
                    src_text = example.get('src', '')  # English
                    tgt_text = example.get('tgt', '')  # Indic
                    
                    # Save BOTH English and Indic (for multilingual tokenizer)
                    # English text
                    if src_text and len(src_text.strip()) > 20:
                        outfile.write(src_text.strip() + '\n')
                        total_lines += 1
                    
                    # Indic text (must have non-ASCII chars)
                    if tgt_text and len(tgt_text.strip()) > 20 and not tgt_text.isascii():
                        outfile.write(tgt_text.strip() + '\n')
                        count += 1
                        total_lines += 1
                        
                        if count >= samples_per_lang:
                            break
                
                print(f"    {lang}: {count:,} Indic + English samples")
                
            except Exception as e:
                print(f"    {lang}: Error - {e}")
                # Try IndicCorp as fallback
                try:
                    print(f"    Trying IndicCorp for {lang}...")
                    dataset = load_dataset(
                        "ai4bharat/IndicCorp",
                        lang,
                        split="train",
                        streaming=True
                    )
                    count = 0
                    for example in dataset:
                        text = example.get('text', '')
                        if text and len(text.strip()) > 20 and not text.isascii():
                            outfile.write(text.strip() + '\n')
                            count += 1
                            total_lines += 1
                            if count >= samples_per_lang:
                                break
                    print(f"    {lang} (IndicCorp): {count:,} samples")
                except:
                    print(f"    {lang}: No data available, using fallback")
                    fallback_samples = get_fallback_samples_2char(lang)
                    for sample in fallback_samples:
                        outfile.write(sample + '\n')
                        total_lines += 1
    
    print(f"\n✓ Corpus created: {output_file}")
    print(f"  Total lines: {total_lines:,}")
    
    return output_file


def get_fallback_samples_2char(lang):
    """Fallback samples for 2-character language codes"""
    fallback = {
        'hi': ['नमस्ते, यह एक परीक्षण वाक्य है।', 'भारत एक महान देश है।'] * 100,
        'ta': ['வணக்கம், இது ஒரு சோதனை வாக்கியம்.', 'தமிழ்நாடு அழகான மாநிலம்.'] * 100,
        'te': ['నమస్కారం, ఇది ఒక పరీక్ష వాక్యం.', 'తెలుగు మా మాతృభాష.'] * 100,
        'mr': ['नमस्कार, हे एक चाचणी वाक्य आहे.', 'महाराष्ट्र सुंदर राज्य आहे.'] * 100,
        'kn': ['ನಮಸ್ಕಾರ, ಇದು ಒಂದು ಪರೀಕ್ಷಾ ವಾಕ್ಯ.', 'ಕರ್ನಾಟಕ ಸುಂದರ ರಾಜ್ಯ.'] * 100,
        'ml': ['നമസ്കാരം, ഇത് ഒരു പരീക്ഷണ വാക്യമാണ്.', 'കേരളം മനോഹരമായ സംസ്ഥാനമാണ്.'] * 100,
    }
    return fallback.get(lang, fallback.get('hi', []))


def download_sangraha_corpus(languages, samples_per_lang, output_file):
    """
    Download and combine text from Sangraha dataset
    
    This is the key improvement - uses real data from HuggingFace
    instead of requiring local files.
    """
    
    print(f"Downloading Sangraha corpus from HuggingFace...")
    print(f"  Languages: {len(languages)}")
    print(f"  Samples per language: {samples_per_lang:,}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for lang in tqdm(languages, desc="Downloading languages"):
            try:
                # Load Sangraha dataset - try language-specific loading first
                try:
                    # Try loading with language-specific config
                    dataset = load_dataset(
                        "ai4bharat/sangraha",
                        name=lang,  # Language-specific subset
                        split="train",
                        streaming=True
                    )
                except:
                    # Fallback to verified dataset
                    dataset = load_dataset(
                        "ai4bharat/sangraha",
                        data_dir="verified",
                        split="train",
                        streaming=True
                    )
                
                # Stricter filtering by language
                count = 0
                for example in dataset:
                    # Only accept EXACT language code match
                    example_lang = example.get('lang', '').strip().lower()
                    if example_lang == lang.lower():
                        text = example.get('text', '')
                        # Validate: must have Indic script characters (not just ASCII)
                        if text and len(text.strip()) > 20 and not text.isascii():
                            outfile.write(text.strip() + '\n')
                            count += 1
                            total_lines += 1
                            
                            if count >= samples_per_lang:
                                break
                
                print(f"  {lang}: {count:,} samples")
                
            except Exception as e:
                print(f"  {lang}: Error - {e}")
                # Fallback: add sample sentences for this language
                fallback_samples = get_fallback_samples(lang)
                for sample in fallback_samples:
                    outfile.write(sample + '\n')
                    total_lines += 1
    
    print(f"\n✓ Corpus created: {output_file}")
    print(f"  Total lines: {total_lines:,}")
    
    return output_file


def get_fallback_samples(lang):
    """
    Fallback samples if Sangraha download fails
    These are just for testing - real training needs more data
    """
    
    fallback = {
        'hin': [
            'नमस्ते दुनिया, मैं एक बहुभाषी मॉडल हूँ।',
            'भारत एक विविधताओं से भरा देश है।',
            'हिंदी भारत की राजभाषा है।',
            'प्राकृतिक भाषा प्रसंस्करण एक रोमांचक क्षेत्र है।',
            'मशीन लर्निंग से कंप्यूटर सीखते हैं।',
        ] * 100,
        'tam': [
            'வணக்கம் உலகம், நான் ஒரு பன்மொழி மாதிரி.',
            'தமிழ் உலகின் பழமையான மொழிகளில் ஒன்று.',
            'இயற்கை மொழி செயலாக்கம் ஒரு சுவாரஸ்யமான துறை.',
            'இயந்திர கற்றல் கணினிகளை கற்க உதவுகிறது.',
            'தமிழ்நாடு அழகான மாநிலம்.',
        ] * 100,
        'tel': [
            'హలో ప్రపంచం, నేను బహుభాషా నమూనా.',
            'తెలుగు అందమైన భాష.',
            'సహజ భాషా ప్రాసెసింగ్ ఆసక్తికరమైన రంగం.',
            'మషిన్ లెర్నింగ్ కంప్యూటర్లకు నేర్పిస్తుంది.',
            'ఆంధ్ర ప్రదేశ్ అందమైన రాష్ట్రం.',
        ] * 100,
        'mar': [
            'नमस्कार जग, मी एक बहुभाषी मॉडेल आहे.',
            'मराठी महाराष्ट्राची राजभाषा आहे.',
            'नैसर्गिक भाषा प्रक्रिया रोमांचक क्षेत्र आहे.',
            'मशीन लर्निंग कंप्युटरला शिकवते.',
            'महाराष्ट्र समृद्ध सांस्कृतिक वारसा असलेले राज्य आहे.',
        ] * 100,
        'ben': [
            'নমস্কার বিশ্ব, আমি একটি বহুভাষিক মডেল।',
            'বাংলা পশ্চিমবঙ্গের রাজ্য ভাষা।',
            'প্রাকৃতিক ভাষা প্রক্রিয়াকরণ একটি আকর্ষণীয় ক্ষেত্র।',
            'মেশিন লার্নিং কম্পিউটারকে শেখায়।',
            'পশ্চিমবঙ্গ সুন্দর রাজ্য।',
        ] * 100,
        'guj': [
            'નમસ્તે વિશ્વ, હું એક બહુભાષી મોડેલ છું.',
            'ગુજરાતી ગુજરાતની રાજ્ય ભાષા છે.',
            'કુદરતી ભાષા પ્રક્રિયા એક રસપ્રદ ક્ષેત્ર છે.',
            'મશીન લર્નિંગ કમ્પ્યુટરને શીખવે છે.',
            'ગુજરાત સુંદર રાજ્ય છે.',
        ] * 100,
    }
    
    return fallback.get(lang, fallback.get('hin', []))


def create_local_corpus(data_dir='data', output_file='data/combined_corpus.txt'):
    """
    Alternative: Combine local text files into corpus
    Use this if you have your own data files
    
    Expected structure:
    data/
    ├── hindi_data.txt
    ├── tamil_data.txt
    └── ... (other language files)
    """
    
    print("Combining local corpus files...")
    
    combined_count = 0
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in Path(data_dir).glob('*_data.txt'):
            print(f"  Processing {file_path.name}...")
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if line.strip() and len(line.strip()) > 10:
                            outfile.write(line)
                            combined_count += 1
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
    
    print(f"✓ Combined {combined_count:,} lines into {output_file}")
    return output_file


def train_sentencepiece_tokenizer(corpus_path, model_prefix, vocab_size=50000, 
                                   model_type='unigram', character_coverage=0.9995):
    """
    Train SentencePiece tokenizer with optimized settings for Indian languages
    
    Key improvements over default:
    - Unigram model (better for morphologically rich languages)
    - Higher character coverage (for rare scripts)
    - Special tokens for BERT-style training
    - Better handling of mixed scripts
    """
    
    print(f"\n{'='*60}")
    print("TRAINING SENTENCEPIECE TOKENIZER")
    print('='*60)
    print(f"  Corpus: {corpus_path}")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Model type: {model_type}")
    print(f"  Character coverage: {character_coverage}")
    print(f"  Output prefix: {model_prefix}")
    
    # Create output directory
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    
    # Special tokens for BERT-style training
    # These are critical for MLM pretraining
    special_tokens = [
        '[PAD]',   # Padding token
        '[UNK]',   # Unknown token  
        '[CLS]',   # Classification token (sentence start)
        '[SEP]',   # Separator token (sentence end)
        '[MASK]',  # Mask token for MLM
    ]
    
    # Train SentencePiece
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        
        # Model type: 'unigram' recommended for Indian languages
        model_type=model_type,
        
        # Character coverage: Higher = more rare characters
        character_coverage=character_coverage,
        
        # Normalization: Standard NFKC is better for noisy multilingual text
        normalization_rule_name='nmt_nfkc',
        
        # Script handling: Split by unicode script for Brahmic scripts
        split_by_unicode_script=True,
        split_by_number=True,
        split_by_whitespace=True,
        
        # REMOVED byte_fallback=True - it was causing byte-level tokenization!
        # Instead, let unknown chars go to <unk>
        byte_fallback=False,
        
        # Special token IDs (BERT-compatible)
        pad_id=0,      # [PAD]
        unk_id=1,      # [UNK]
        bos_id=2,      # [CLS] (beginning of sentence)
        eos_id=3,      # [SEP] (end of sentence)
        
        # Add user-defined special tokens (includes [PAD], [UNK], [CLS], [SEP], [MASK])
        user_defined_symbols=special_tokens,
        
        # Training efficiency
        input_sentence_size=10000000,  # Max sentences to use
        shuffle_input_sentence=True,    # Shuffle for better sampling
        train_extremely_large_corpus=True,  # Better for 1GB+ corpus
        num_threads=16,  # Faster training
    )
    
    print(f"\n✓ SentencePiece tokenizer trained!")
    print(f"  Model: {model_prefix}.model")
    print(f"  Vocab: {model_prefix}.vocab")
    
    return f"{model_prefix}.model"


def test_tokenizer(model_path):
    """
    Test the trained tokenizer on sample texts from different languages
    """
    
    print(f"\n{'='*60}")
    print("TESTING TOKENIZER")
    print('='*60)
    
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    # Print tokenizer info
    print(f"\nTokenizer Info:")
    print(f"  Vocabulary size: {sp.get_piece_size():,}")
    print(f"  Special tokens:")
    print(f"    PAD (0): {sp.id_to_piece(0)}")
    print(f"    UNK (1): {sp.id_to_piece(1)}")
    print(f"    BOS (2): {sp.id_to_piece(2)}")
    print(f"    EOS (3): {sp.id_to_piece(3)}")
    
    # Test samples in different languages
    test_samples = {
        'Hindi': 'नमस्ते दुनिया, मैं एक बहुभाषी मॉडल हूँ।',
        'Tamil': 'வணக்கம் உலகம், நான் ஒரு பன்மொழி மாதிரி.',
        'Telugu': 'హలో ప్రపంచం, నేను బహుభాషా నమూనా.',
        'Marathi': 'नमस्कार जग, मी एक बहुभाषी मॉडेल आहे.',
        'Bengali': 'নমস্কার বিশ্ব, আমি একটি বহুভাষিক মডেল।',
        'Gujarati': 'નમસ્તે વિશ્વ, હું એક બહુભાષી મોડેલ છું.',
        'Kannada': 'ಹಲೋ ಜಗತ್ತು, ನಾನು ಬಹುಭಾಷಾ ಮಾದರಿ.',
        'Malayalam': 'ഹലോ ലോകം, ഞാൻ ഒരു ബഹുഭാഷാ മാതൃകയാണ്.',
        'English': 'Hello world, I am a multilingual model.',
    }
    
    print(f"\nTokenization Results:")
    for lang, text in test_samples.items():
        pieces = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        
        # Fertility: Number of tokens per word (lower is better)
        words = text.split()
        fertility = len(ids) / max(len(words), 1)
        
        print(f"\n{lang}:")
        print(f"  Text: {text}")
        print(f"  Tokens ({len(ids)}): {pieces[:12]}{'...' if len(pieces) > 12 else ''}")
        print(f"  Fertility: {fertility:.2f} tokens/word")
    
    # Summary
    print(f"\n{'='*60}")
    print("TOKENIZER SUMMARY")
    print('='*60)
    print(f"  Vocabulary size: {sp.get_piece_size():,}")
    print(f"  Model ready for training!")
    
    return sp


def main():
    """
    Main function to train tokenizer
    """
    
    print("="*60)
    print("STEP 1: TRAIN SENTENCEPIECE TOKENIZER")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Vocab size: {CONFIG['vocab_size']:,}")
    print(f"  Model type: {CONFIG['model_type']}")
    print(f"  Character coverage: {CONFIG['character_coverage']}")
    print(f"  Languages: {len(CONFIG['languages'])}")
    
    # Step 1: Prepare corpus
    print(f"\n[1/3] Preparing corpus...")
    
    # Check if corpus file already exists (skip download if so)
    if os.path.exists(CONFIG['corpus_file']):
        print(f"  ✓ Corpus file already exists: {CONFIG['corpus_file']}")
        corpus_file = CONFIG['corpus_file']
        # Show file size
        file_size = os.path.getsize(corpus_file) / (1024**3)
        print(f"  Size: {file_size:.2f} GB")
    else:
        # Check if local data exists
        local_files = list(Path('data').glob('*_data.txt')) if os.path.exists('data') else []
        
        if local_files:
            print(f"  Found {len(local_files)} local data files, using those...")
            corpus_file = create_local_corpus(
                data_dir='data',
                output_file=CONFIG['corpus_file']
            )
        else:
            # Choose dataset based on config
            dataset_source = CONFIG.get('dataset_source', 'samanantar')
            
            if dataset_source == 'samanantar':
                print(f"  Downloading from Samanantar (high quality)...")
                corpus_file = download_samanantar_corpus(
                    languages=CONFIG['languages'],
                    samples_per_lang=CONFIG['samples_per_language'],
                    output_file=CONFIG['corpus_file']
                )
            else:
                print(f"  Downloading from Sangraha...")
                corpus_file = download_sangraha_corpus(
                    languages=CONFIG['languages'],
                    samples_per_lang=CONFIG['samples_per_language'],
                    output_file=CONFIG['corpus_file']
                )
    
    # Step 2: Train tokenizer
    print(f"\n[2/3] Training tokenizer...")
    model_path = train_sentencepiece_tokenizer(
        corpus_path=corpus_file,
        model_prefix=CONFIG['model_prefix'],
        vocab_size=CONFIG['vocab_size'],
        model_type=CONFIG['model_type'],
        character_coverage=CONFIG['character_coverage'],
    )
    
    # Step 3: Test tokenizer
    print(f"\n[3/3] Testing tokenizer...")
    sp = test_tokenizer(model_path)
    
    print("\n" + "="*60)
    print("✓ TOKENIZER TRAINING COMPLETE!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  Model: {CONFIG['model_prefix']}.model")
    print(f"  Vocab: {CONFIG['model_prefix']}.vocab")
    print(f"\nNext step:")
    print(f"  python 2_prepare_data.py")
    print("="*60)


if __name__ == '__main__':
    main()
