"""
Test script to preview Samanantar dataset
Run this to see what the data looks like before full download
"""

from datasets import load_dataset

def preview_samanantar():
    """Download and display a few samples from Samanantar for each language"""
    
    # Correct config names (just language codes, not en-{lang})
    languages = ['hi', 'ta', 'te', 'mr', 'kn', 'ml']
    samples_per_lang = 5  # Just 5 samples for preview
    
    print("="*70)
    print("SAMANANTAR DATASET PREVIEW")
    print("="*70)
    print("Dataset: ai4bharat/samanantar")
    print("This is a parallel corpus (English ↔ Indic translations)")
    print("="*70)
    
    for lang in languages:
        print(f"\n{'='*70}")
        print(f"LANGUAGE: {lang.upper()}")
        print("="*70)
        
        try:
            # Load dataset with just the language code
            dataset = load_dataset(
                "ai4bharat/samanantar",
                lang,  # Just 'hi', 'ta', etc.
                split="train",
                streaming=True
            )
            
            count = 0
            for example in dataset:
                # Print the example keys first to understand structure
                if count == 0:
                    print(f"Data keys: {list(example.keys())}")
                
                # Try different possible key names
                src = example.get('src', example.get('en', example.get('source', 'N/A')))
                tgt = example.get('tgt', example.get(lang, example.get('target', 'N/A')))
                
                # If src/tgt are in a nested structure
                if isinstance(example.get('translation', None), dict):
                    src = example['translation'].get('en', 'N/A')
                    tgt = example['translation'].get(lang, 'N/A')
                
                print(f"\n--- Sample {count + 1} ---")
                if src != 'N/A':
                    print(f"English: {str(src)[:100]}{'...' if len(str(src)) > 100 else ''}")
                if tgt != 'N/A':
                    print(f"Indic:   {str(tgt)[:100]}{'...' if len(str(tgt)) > 100 else ''}")
                
                # Also show raw example for debugging
                if count == 0:
                    print(f"Raw example: {example}")
                
                count += 1
                if count >= samples_per_lang:
                    break
            
            print(f"\n✓ {lang}: Successfully loaded {count} samples")
            
        except Exception as e:
            print(f"✗ {lang}: Error - {e}")
    
    print("\n" + "="*70)
    print("PREVIEW COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    preview_samanantar()
