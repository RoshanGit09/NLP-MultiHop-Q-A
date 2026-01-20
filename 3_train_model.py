"""
Step 3: Train Encoder-Decoder Transformer Model
- Uses custom Transformer architecture (6+6 layers, 768 dim, ~250M params)
- Trains with Seq2Seq objective for Multi-Hop Q&A
- Uses HuggingFace Trainer for distributed training
"""

import os
# ===== GPU SELECTION =====
# Set which GPU to use (0 or 1). Comment out to use all GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use only GPU 1
# ==========================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
import sys
from datetime import datetime

# Add models directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.transformer import create_model, TransformerEncoderDecoder


# Configuration
YOUR_VOCAB_SIZE = 100000  # Your SentencePiece tokenizer vocab size (increased for multilingual)


class TranslationDataCollator:
    """
    Data collator for translation training (EN → Indic)
    Handles padding and creates proper encoder/decoder inputs
    """
    
    def __init__(self, pad_token_id=0, max_length=512):
        self.pad_token_id = pad_token_id
        self.bos_id = 2
        self.max_length = max_length
    
    def __call__(self, examples):
        """
        Collate batch of translation pairs
        
        Input format: {'src_ids': [...], 'tgt_ids': [...]}
        Output: {'src_ids', 'tgt_ids', 'labels'}
        """
        
        # Extract source and target IDs
        src_ids = [torch.tensor(e['src_ids']) for e in examples]
        tgt_ids = [torch.tensor(e['tgt_ids']) for e in examples]
        
        # Find max lengths
        max_src_len = min(max(len(seq) for seq in src_ids), self.max_length)
        max_tgt_len = min(max(len(seq) for seq in tgt_ids), self.max_length)
        
        # Pad source sequences (encoder input)
        padded_src = []
        for seq in src_ids:
            if len(seq) < max_src_len:
                padded = torch.cat([seq, torch.full((max_src_len - len(seq),), self.pad_token_id)])
            else:
                padded = seq[:max_src_len]
            padded_src.append(padded)
        
        # Pad target sequences (decoder input)
        # Target for decoder input excludes last token
        padded_tgt = []
        labels = []
        
        for seq in tgt_ids:
            # Decoder input: [BOS, tok1, tok2, ..., tokN-1]
            dec_input = seq[:-1] if len(seq) > 1 else seq
            if len(dec_input) < max_tgt_len:
                dec_input = torch.cat([dec_input, torch.full((max_tgt_len - len(dec_input),), self.pad_token_id)])
            else:
                dec_input = dec_input[:max_tgt_len]
            padded_tgt.append(dec_input)
            
            # Labels: [tok1, tok2, ..., tokN, EOS]
            # Shift by 1 to predict next token
            label = seq[1:] if len(seq) > 1 else seq
            if len(label) < max_tgt_len:
                # Pad with -100 (ignored in loss)
                label = torch.cat([label, torch.full((max_tgt_len - len(label),), -100)])
            else:
                label = label[:max_tgt_len]
            # Mask padding in original sequence
            label[label == self.pad_token_id] = -100
            labels.append(label)
        
        return {
            'src_ids': torch.stack(padded_src).long(),
            'tgt_ids': torch.stack(padded_tgt).long(),
            'labels': torch.stack(labels).long(),
        }


class TranslationTrainer(Trainer):
    """Custom trainer for translation (encoder-decoder model)"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        src_ids = inputs['src_ids']
        tgt_ids = inputs['tgt_ids']
        labels = inputs['labels']
        
        # Forward pass
        logits = model(src_ids, tgt_ids)
        
        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        if return_outputs:
            return loss, logits
        return loss


def train_model(
    data_dir='data/tokenized_translation',  # Updated for translation data
    model_output_dir='./transformer_model_output',
    num_epochs=3,
    batch_size=16,  # Reduced from 32 due to larger vocab embeddings
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
):
    """
    Main training function
    
    Args:
        data_dir: Path to tokenized dataset
        model_output_dir: Output directory for trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
    """
    
    print("="*60)
    print("TRAINING ENCODER-DECODER TRANSFORMER")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Create model
    print("\n[1/4] Creating Transformer Encoder-Decoder model...")
    model = create_model(vocab_size=YOUR_VOCAB_SIZE)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"  Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
        
        # Use DataParallel to distribute across all GPUs
        if num_gpus > 1:
            print(f"  → Using DataParallel across {num_gpus} GPUs!")
            model = nn.DataParallel(model)
    
    model.to(device)
    
    # Step 2: Load data
    print("\n[2/4] Loading tokenized dataset...")
    dataset = load_from_disk(data_dir)
    print(f"  Training examples: {len(dataset['train'])}")
    print(f"  Validation examples: {len(dataset['test'])}")
    
    # Step 3: Setup data collator
    print("\n[3/4] Setting up data collator...")
    data_collator = TranslationDataCollator(pad_token_id=0, max_length=512)
    
    # Step 4: Setup training arguments
    print("\n[4/4] Setting up training...")
    
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        
        # Training parameters
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,  # Our model doesn't support this yet
        
        # Optimizer
        learning_rate=learning_rate,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,  # As per Transformer paper
        adam_epsilon=1e-9,
        max_grad_norm=1.0,
        
        # Learning rate schedule (warmup + linear decay)
        warmup_steps=4000,  # Transformer paper uses 4000
        lr_scheduler_type="linear",
        
        # Logging and saving
        logging_steps=100,
        logging_dir='./logs',
        save_steps=1000,
        save_total_limit=3,
        eval_steps=1000,
        eval_strategy="steps",
        
        # Performance
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        
        # Precision
        fp16=False,
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        
        # Misc
        seed=42,
        report_to="none",
        remove_unused_columns=False,
    )
    
    print(f"  Output dir: {model_output_dir}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create trainer
    trainer = TranslationTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )
    
    # Train
    print("\n" + "="*60)
    print("TRAINING IN PROGRESS...")
    print("="*60)
    
    trainer.train()
    
    # Save final model
    print("\n[Saving] Saving final model...")
    
    # Save the PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': YOUR_VOCAB_SIZE,
            'd_model': 768,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'num_heads': 12,
            'd_ff': 3072,
        }
    }, f'{model_output_dir}/final_model.pt')
    
    print(f"✓ Model saved to: {model_output_dir}/final_model.pt")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Encoder-Decoder Transformer')
    parser.add_argument('--data_dir', default='data/tokenized_translation',
                       help='Path to tokenized dataset')
    parser.add_argument('--output_dir', default='./transformer_model_output',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size per device')
    parser.add_argument('--grad_accum', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        model_output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
    )
