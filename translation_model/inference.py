"""
Inference Script for Multilingual Transformer Model
- Load trained encoder-decoder model
- Perform translation/generation on custom inputs
- Support for batch inference
- Beam search decoding
"""

import torch
import torch.nn as nn
import sentencepiece as spm
import argparse
import sys
import os
from typing import List, Optional

# Language codes mapping for target language specification
LANG_CODES = {
    'en': '<2en>', 'english': '<2en>',
    'hi': '<2hi>', 'hindi': '<2hi>',
    'ta': '<2ta>', 'tamil': '<2ta>',
    'te': '<2te>', 'telugu': '<2te>',
    'mr': '<2mr>', 'marathi': '<2mr>',
    'kn': '<2kn>', 'kannada': '<2kn>',
    'ml': '<2ml>', 'malayalam': '<2ml>',
}

# Add models directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.transformer import TransformerEncoderDecoder


class MultilingualInference:
    """Inference class for multilingual transformer model"""
    
    def __init__(
        self,
        model_path: str = './transformer_model_output-final/final_model.pt',
        tokenizer_path: str = 'tokenizer/multilingual_indic-3.model',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        vocab_size: int = 100000
    ):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            tokenizer_path: Path to SentencePiece tokenizer
            device: Device to run inference on ('cuda' or 'cpu')
            vocab_size: Vocabulary size (must match training)
        """
        
        self.device = device
        self.vocab_size = vocab_size
        
        print("="*60)
        print("INITIALIZING MULTILINGUAL TRANSFORMER INFERENCE")
        print("="*60)
        
        # Load tokenizer
        print(f"\n[1/2] Loading tokenizer from: {tokenizer_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        print(f"  ✓ Tokenizer loaded")
        print(f"  Vocabulary size: {self.sp.get_piece_size()}")
        
        # Special token IDs
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3
        
        # Load model
        print(f"\n[2/2] Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"  ✓ Model loaded successfully")
        print(f"  Device: {self.device}")
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        
        print("\n" + "="*60)
        print("✓ READY FOR INFERENCE!")
        print("="*60 + "\n")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from checkpoint"""
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model instance
        from models.transformer import create_model
        model = create_model(vocab_size=self.vocab_size)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check if checkpoint has 'model_state_dict' key
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel wrapper (keys start with 'module.')
            if any(k.startswith('module.') for k in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
                    new_state_dict[name] = v
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict)
        else:
            # Direct state dict (shouldn't happen, but handle it)
            model.load_state_dict(checkpoint)
        
        return model
    
    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Token IDs tensor (1, seq_len)
        """
        # Tokenize
        token_ids = self.sp.encode_as_ids(text)
        
        # Truncate if too long
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Convert to tensor
        return torch.tensor([token_ids], dtype=torch.long).to(self.device)
    
    def decode_ids(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        # Remove special tokens
        filtered_ids = [
            tid for tid in token_ids 
            if tid not in [self.pad_id, self.bos_id, self.eos_id]
        ]
        
        return self.sp.decode_ids(filtered_ids)
    
    def _get_lang_token_id(self, target_lang: str) -> int:
        """
        Get the token ID for a target language
        
        Args:
            target_lang: Language code ('hi', 'ta', 'en') or name ('hindi', 'tamil')
            
        Returns:
            Token ID for the language token, or BOS ID if not found
        """
        lang_token = LANG_CODES.get(target_lang.lower())
        if lang_token:
            token_id = self.sp.piece_to_id(lang_token)
            if token_id != self.sp.unk_id():
                return token_id
        # Fallback to BOS if language token not found
        return self.bos_id
    
    def _apply_repetition_penalty(self, logits, generated_ids, penalty=1.2):
        """Apply repetition penalty to already-generated tokens"""
        for token_id in set(generated_ids):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
        return logits
    
    def _check_ngram_repeat(self, token_ids, ngram_size=3):
        """Check if the last ngram_size tokens have appeared before in the sequence"""
        if len(token_ids) < ngram_size:
            return False
        last_ngram = tuple(token_ids[-ngram_size:])
        for i in range(len(token_ids) - ngram_size):
            if tuple(token_ids[i:i + ngram_size]) == last_ngram:
                return True
        return False
    
    @torch.no_grad()
    def greedy_decode(
        self,
        src_text: str,
        target_lang: str = 'hi',
        max_length: int = 512,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        verbose: bool = True
    ) -> str:
        """
        Greedy decoding with repetition penalty and temperature
        
        Args:
            src_text: Source text
            target_lang: Target language code ('hi', 'ta', 'te', 'mr', 'kn', 'ml', 'en')
            max_length: Maximum generation length
            temperature: Temperature for softmax (lower = more deterministic)
            repetition_penalty: Penalty for repeating tokens (>1.0 penalizes repetition)
            no_repeat_ngram_size: Block n-grams of this size from repeating
            verbose: Print decoding steps
            
        Returns:
            Generated text
        """
        
        if verbose:
            lang_name = target_lang.upper()
            print(f"Input: {src_text}")
            print(f"Target language: {lang_name}")
            print("Decoding (greedy)...")
        
        # Encode source
        src_ids = self.encode_text(src_text, max_length)
        
        # Start with language token instead of BOS for language-directed translation
        lang_token_id = self._get_lang_token_id(target_lang)
        tgt_ids = torch.tensor([[lang_token_id]], dtype=torch.long).to(self.device)
        generated_list = [lang_token_id]
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Forward pass
            logits = self.model(src_ids, tgt_ids)
            next_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, generated_list, repetition_penalty
                )
            
            # Block repeated n-grams
            if no_repeat_ngram_size > 0 and len(generated_list) >= no_repeat_ngram_size - 1:
                # Check what tokens would create a repeated n-gram
                prefix = tuple(generated_list[-(no_repeat_ngram_size - 1):])
                for i in range(len(generated_list) - no_repeat_ngram_size + 1):
                    if tuple(generated_list[i:i + no_repeat_ngram_size - 1]) == prefix:
                        # Block the token that would complete the repeated n-gram
                        blocked_token = generated_list[i + no_repeat_ngram_size - 1]
                        next_logits[0, blocked_token] = float('-inf')
            
            # Get next token (greedy)
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
            generated_list.append(next_token.item())
            
            # Stop if EOS generated
            if next_token.item() == self.eos_id:
                break
        
        # Decode
        output_text = self.decode_ids(tgt_ids[0].tolist())
        
        if verbose:
            print(f"Output: {output_text}\n")
        
        return output_text
    
    @torch.no_grad()
    def beam_search_decode(
        self,
        src_text: str,
        target_lang: str = 'hi',
        beam_size: int = 5,
        max_length: int = 512,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        verbose: bool = True
    ) -> str:
        """
        Beam search decoding with repetition penalty (higher quality)
        
        Args:
            src_text: Source text
            target_lang: Target language code ('hi', 'ta', 'te', 'mr', 'kn', 'ml', 'en')
            beam_size: Number of beams
            max_length: Maximum generation length
            length_penalty: Length penalty (> 1.0 favors longer sequences)
            repetition_penalty: Penalty for repeating tokens
            no_repeat_ngram_size: Block n-grams of this size from repeating
            verbose: Print decoding steps
            
        Returns:
            Generated text
        """
        
        if verbose:
            lang_name = target_lang.upper()
            print(f"Input: {src_text}")
            print(f"Target language: {lang_name}")
            print(f"Decoding (beam search, beam_size={beam_size})...")
        
        # Encode source
        src_ids = self.encode_text(src_text, max_length)
        batch_size = src_ids.size(0)
        
        # Expand source for beam search
        src_ids = src_ids.repeat(beam_size, 1)
        
        # Initialize beams with language token instead of BOS
        lang_token_id = self._get_lang_token_id(target_lang)
        beams = torch.full((beam_size, 1), lang_token_id, dtype=torch.long).to(self.device)
        beam_scores = torch.zeros(beam_size).to(self.device)
        
        # Track completed sequences
        completed_beams = []
        
        for step in range(max_length):
            # Forward pass
            logits = self.model(src_ids, beams)
            
            # Get log probabilities for next token
            next_logits = logits[:, -1, :]
            
            # Apply repetition penalty per beam
            if repetition_penalty != 1.0:
                for beam_idx in range(next_logits.size(0)):
                    beam_tokens = beams[beam_idx].tolist()
                    for token_id in set(beam_tokens):
                        if next_logits[beam_idx, token_id] > 0:
                            next_logits[beam_idx, token_id] /= repetition_penalty
                        else:
                            next_logits[beam_idx, token_id] *= repetition_penalty
            
            # Block repeated n-grams per beam
            if no_repeat_ngram_size > 0:
                for beam_idx in range(next_logits.size(0)):
                    beam_tokens = beams[beam_idx].tolist()
                    # Remove padding from token list
                    beam_tokens = [t for t in beam_tokens if t != self.pad_id]
                    if len(beam_tokens) >= no_repeat_ngram_size - 1:
                        prefix = tuple(beam_tokens[-(no_repeat_ngram_size - 1):])
                        for i in range(len(beam_tokens) - no_repeat_ngram_size + 1):
                            if tuple(beam_tokens[i:i + no_repeat_ngram_size - 1]) == prefix:
                                blocked_token = beam_tokens[i + no_repeat_ngram_size - 1]
                                next_logits[beam_idx, blocked_token] = float('-inf')
            
            log_probs = torch.log_softmax(next_logits, dim=-1)
            
            # Add to beam scores
            if step == 0:
                # First step: only use first beam
                scores = log_probs[0]
            else:
                # Add to existing beam scores
                scores = beam_scores.unsqueeze(1) + log_probs
            
            # Flatten scores
            scores_flat = scores.view(-1)
            
            # Get top beam_size candidates
            top_scores, top_indices = torch.topk(scores_flat, beam_size)
            
            # Convert flat indices to (beam_idx, token_idx)
            beam_indices = top_indices // self.vocab_size
            token_indices = top_indices % self.vocab_size
            
            # Update beams
            new_beams = []
            new_scores = []
            
            for i, (beam_idx, token_idx, score) in enumerate(
                zip(beam_indices, token_indices, top_scores)
            ):
                # Get previous beam
                prev_beam = beams[beam_idx]
                
                # Append new token
                new_beam = torch.cat([prev_beam, token_idx.unsqueeze(0)])
                
                # Check if EOS
                if token_idx.item() == self.eos_id:
                    # Apply length penalty
                    final_score = score / (len(new_beam) ** length_penalty)
                    completed_beams.append((new_beam, final_score))
                else:
                    new_beams.append(new_beam)
                    new_scores.append(score)
            
            # Stop if all beams completed
            if len(new_beams) == 0:
                break
            
            # Pad beams to same length
            max_len = max(len(b) for b in new_beams)
            beams = torch.stack([
                torch.cat([b, torch.full((max_len - len(b),), self.pad_id).to(self.device)])
                for b in new_beams
            ])
            beam_scores = torch.tensor(new_scores).to(self.device)
        
        # Add remaining beams to completed
        for beam, score in zip(beams, beam_scores):
            final_score = score / (len(beam) ** length_penalty)
            completed_beams.append((beam, final_score))
        
        # Get best beam
        if completed_beams:
            best_beam = max(completed_beams, key=lambda x: x[1])[0]
        else:
            best_beam = beams[0]
        
        # Decode
        output_text = self.decode_ids(best_beam.tolist())
        
        if verbose:
            print(f"Output: {output_text}\n")
        
        return output_text
    
    def translate_batch(
        self,
        texts: List[str],
        target_lang: str = 'hi',
        method: str = 'greedy',
        beam_size: int = 5,
        max_length: int = 512
    ) -> List[str]:
        """
        Translate a batch of texts
        
        Args:
            texts: List of input texts
            target_lang: Target language code ('hi', 'ta', 'te', 'mr', 'kn', 'ml', 'en')
            method: 'greedy' or 'beam'
            beam_size: Beam size for beam search
            max_length: Maximum generation length
            
        Returns:
            List of translated texts
        """
        
        print(f"\nTranslating {len(texts)} texts to {target_lang.upper()} using {method} decoding...")
        print("-" * 60)
        
        results = []
        for i, text in enumerate(texts, 1):
            print(f"\n[{i}/{len(texts)}]")
            
            if method == 'beam':
                output = self.beam_search_decode(
                    text,
                    target_lang=target_lang,
                    beam_size=beam_size, 
                    max_length=max_length,
                    verbose=True
                )
            else:
                output = self.greedy_decode(
                    text,
                    target_lang=target_lang,
                    max_length=max_length,
                    verbose=True
                )
            
            results.append(output)
        
        print("-" * 60)
        print(f"✓ Completed {len(texts)} translations\n")
        
        return results


def main():
    """Main inference function"""
    
    parser = argparse.ArgumentParser(
        description='Inference for Multilingual Transformer Model'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_path',
        type=str,
        default='./transformer_model_output-final/final_model.pt',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default='tokenizer/multilingual_indic-3.model',
        help='Path to SentencePiece tokenizer'
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=100000,
        help='Vocabulary size (must match training)'
    )
    
    # Inference arguments
    parser.add_argument(
        '--input',
        type=str,
        help='Input text to translate'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='File with input texts (one per line)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output file for translations'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['greedy', 'beam'],
        default='greedy',
        help='Decoding method'
    )
    parser.add_argument(
        '--beam_size',
        type=int,
        default=5,
        help='Beam size for beam search'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum generation length'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode'
    )
    parser.add_argument(
        '--target_lang',
        type=str,
        default='hi',
        choices=['en', 'hi', 'ta', 'te', 'mr', 'kn', 'ml'],
        help='Target language: en (English), hi (Hindi), ta (Tamil), te (Telugu), mr (Marathi), kn (Kannada), ml (Malayalam)'
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = MultilingualInference(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        vocab_size=args.vocab_size
    )
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*60)
        print("INTERACTIVE TRANSLATION MODE")
        print("="*60)
        print("Enter text to translate (or 'quit' to exit)")
        print(f"Target language: {args.target_lang.upper()}")
        print(f"Method: {args.method}" + (f" (beam_size={args.beam_size})" if args.method == 'beam' else ""))
        print("TIP: Use --target_lang hi/ta/te/mr/kn/ml/en to change target")
        print("="*60 + "\n")
        
        while True:
            try:
                text = input("Input: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not text:
                    continue
                
                if args.method == 'beam':
                    output = inference.beam_search_decode(
                        text,
                        target_lang=args.target_lang,
                        beam_size=args.beam_size,
                        max_length=args.max_length,
                        verbose=False
                    )
                else:
                    output = inference.greedy_decode(
                        text,
                        target_lang=args.target_lang,
                        max_length=args.max_length,
                        verbose=False
                    )
                
                print(f"Output: {output}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
    
    # Single input
    elif args.input:
        if args.method == 'beam':
            output = inference.beam_search_decode(
                args.input,
                target_lang=args.target_lang,
                beam_size=args.beam_size,
                max_length=args.max_length
            )
        else:
            output = inference.greedy_decode(
                args.input,
                target_lang=args.target_lang,
                max_length=args.max_length
            )
        
        print(f"\nFinal Output: {output}")
    
    # Batch from file
    elif args.input_file:
        print(f"\nReading inputs from: {args.input_file}")
        
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        outputs = inference.translate_batch(
            texts,
            target_lang=args.target_lang,
            method=args.method,
            beam_size=args.beam_size,
            max_length=args.max_length
        )
        
        # Save outputs
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for output in outputs:
                    f.write(output + '\n')
            print(f"✓ Outputs saved to: {args.output_file}")
        else:
            print("\nAll Outputs:")
            print("="*60)
            for i, output in enumerate(outputs, 1):
                print(f"{i}. {output}")
    
    # Demo mode
    else:
        print("\n" + "="*60)
        print("DEMO MODE - Sample Translations")
        print("="*60 + "\n")
        
        # Sample texts (adjust based on your use case)
        demo_texts = [
            "Hello, how are you?",
            "What is your name?",
            "Thank you very much!",
        ]
        
        print(f"Running demo translations to {args.target_lang.upper()}...")
        print(f"Method: {args.method}\n")
        
        for text in demo_texts:
            if args.method == 'beam':
                inference.beam_search_decode(
                    text,
                    target_lang=args.target_lang,
                    beam_size=args.beam_size,
                    max_length=args.max_length
                )
            else:
                inference.greedy_decode(
                    text,
                    target_lang=args.target_lang,
                    max_length=args.max_length
                )
        
        print("\n" + "="*60)
        print("TIP: Use --interactive for interactive mode")
        print("     Use --input 'your text' for single translation")
        print("     Use --target_lang hi/ta/te/mr/kn/ml/en to specify target")
        print("="*60)


if __name__ == '__main__':
    main()
