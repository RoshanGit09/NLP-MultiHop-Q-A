import torch
from transformers import BertForMaskedLM, PreTrainedTokenizerFast
import argparse
import os
import sentencepiece as spm

class SimpleTokenizer(PreTrainedTokenizerFast):
    """
    Wrapper for SentencePiece to make it compatible with HuggingFace pipeline
    """
    def __init__(self, tokenizer_path, **kwargs):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        
        # Load vocab manually since we don't have a vocab.json
        vocab = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
        
        super().__init__(tokenizer_object=self.sp, vocab=vocab, **kwargs)
        
        # Set special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'
        
        # Correct IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 6
        self.sep_token_id = 7
        self.mask_token_id = 8

    def _convert_token_to_id(self, token):
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp.id_to_piece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp.decode_pieces(tokens)

def test_checkpoint(checkpoint_path, tokenizer_path='tokenizer/multilingual_indic.model', use_gpu=False):
    """
    Test a model checkpoint with Fill-Mask
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}...")
    
    # Load model
    model = BertForMaskedLM.from_pretrained(checkpoint_path)
    model.eval()
    
    # Load tokenizer (SentencePiece)
    # We use the raw SentencePiece model because our custom wrapper isn't saved in the checkpoint
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    
    # Test sentences
    test_sentences = [
        # Hindi
        "भारत एक [MASK] देश है।",  # India is a [MASK] country.
        "राम ने [MASK] को खाया।",   # Ram ate [MASK].
        
        # Tamil
        "இந்தியா ஒரு [MASK] நாடு.", # India is a [MASK] country.
        
        # English
        "The quick brown [MASK] jumps over the lazy dog.",
        "I want to [MASK] some water.",
    ]
    
    mask_id = 8  # [MASK] ID from your config
    
    print("\n" + "="*50)
    print("TESTING PREDICTIONS (Top 5)")
    print("="*50)
    
    # Use CPU by default to avoid OOM while training is running
    device = "cpu"
    if torch.cuda.is_available() and use_gpu:
        device = "cuda"
        
    print(f"Using device: {device} (Use --gpu to force CUDA if memory allows)\n")
    model.to(device)
    
    for text in test_sentences:
        # Tokenize
        tokens = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        
        # Find mask index - Look for '[MASK]' string or ID 8
        # Since sp.encode splits '[MASK]' into pieces if not user_defined, we need to be careful
        # But we added it as user_defined_symbol, so it should be a single token with ID 8
        try:
            mask_idx = ids.index(mask_id)
        except ValueError:
            print(f"⚠️  Skipping: '{text}' (No [MASK] token found in ids: {ids})")
            continue
            
        # Create inputs
        input_ids = torch.tensor([ids]).to(device)
        
        print(f"DEBUG: IDs: {ids}")
        print(f"DEBUG: Mask indices: {[i for i, x in enumerate(ids) if x == mask_id]}")
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits[0, mask_idx].topk(5)
            
        print(f"Input: {text}")
        print("-" * 20)
        
        for i, (score, token_id) in enumerate(zip(predictions.values, predictions.indices)):
            token = sp.id_to_piece(token_id.item())
            prob = torch.softmax(outputs.logits[0, mask_idx], dim=0)[token_id].item()
            print(f"  {i+1}. {token:<15} ({prob:.2%})")
        print()

    # Interactive mode
    print("="*50)
    print("INTERACTIVE MODE (Type 'q' to quit)")
    print("Format: Your sentence with [MASK]")
    print("="*50)
    
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'q':
            break
            
        if "[MASK]" not in text:
            print("⚠️  Please include [MASK] in your sentence!")
            continue
            
        # Tokenize
        ids = sp.encode_as_ids(text)
        
        try:
            mask_idx = ids.index(mask_id)
        except ValueError:
             # Fallback: maybe tokenizer output pieces for [MASK] differently?
             # Let's force ID 8 if we can't find it
             print(f"Debug: Token IDs: {ids}")
             print("Could not find [MASK] ID (8). Ensure tokenizer handles it correctly.")
             continue

        input_ids = torch.tensor([ids]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits[0, mask_idx].topk(5)
            
        print("-" * 20)
        for i, (score, token_id) in enumerate(zip(predictions.values, predictions.indices)):
            token = sp.id_to_piece(token_id.item())
            prob = torch.softmax(outputs.logits[0, mask_idx], dim=0)[token_id].item()
            print(f"  {i+1}. {token:<15} ({prob:.2%})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint folder (e.g. ./multilingual_model_output/checkpoint-500)')
    parser.add_argument('--tokenizer', default='tokenizer/multilingual_indic.model', help='Path to tokenizer model')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference (default is CPU)')
    args = parser.parse_args()
    
    test_checkpoint(args.checkpoint, args.tokenizer, args.gpu)
