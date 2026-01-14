"""
Step 3: Train Multilingual Transformer Model
- Creates BertForMaskedLM model (84M parameters)
- Trains with Masked Language Modeling objective
- Uses HuggingFace Trainer for distributed training
"""

import torch
from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import os
from datetime import datetime

def create_model_config():
    """
    Create BertConfig for small multilingual transformer
    Architecture:
    - 12 layers
    - 768 hidden dimensions
    - 12 attention heads
    - ~84M total parameters
    """
    
    config = BertConfig(
        vocab_size=30000,              # SentencePiece vocabulary
        hidden_size=768,               # Embedding and hidden state dimension
        num_hidden_layers=12,          # Number of transformer blocks
        num_attention_heads=12,        # Number of attention heads
        intermediate_size=3072,        # FFN intermediate (768 * 4)
        hidden_act="gelu",             # Activation function
        hidden_dropout_prob=0.1,       # Dropout in transformer
        attention_probs_dropout_prob=0.1,  # Attention dropout
        max_position_embeddings=512,   # Maximum sequence length
        type_vocab_size=2,             # Segment embeddings (for NSP, not used in MLM)
        initializer_range=0.02,        # Weight initialization std
        layer_norm_eps=1e-12,          # Layer normalization epsilon
        pad_token_id=0,                # Padding token
        gradient_checkpointing=False,  # Gradient checkpointing (memory vs speed)
    )
    
    return config


def create_model_and_tokenizer(config_dict=None):
    """
    Create BertForMaskedLM model with random initialization
    
    Args:
        config_dict: Custom config dict (uses default if None)
        
    Returns:
        model: BertForMaskedLM with random init (NOT pretrained)
    """
    
    if config_dict:
        config = BertConfig(**config_dict)
    else:
        config = create_model_config()
    
    # Initialize model with RANDOM weights (NOT pretrained)
    model = BertForMaskedLM(config)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")
    print(f"  Model size: {num_params * 4 / (1024**2):.2f} MB")
    
    return model, config


def setup_training_args(output_dir='./multilingual_model_output',
                       num_train_epochs=3,
                       per_device_batch_size=32,
                       learning_rate=5e-4,
                       warmup_steps=1000,
                       save_steps=500,
                       eval_steps=500,
                       use_gpu=True):
    """
    Setup training arguments
    
    Args:
        output_dir: Output directory for checkpoints
        num_train_epochs: Number of training epochs
        per_device_batch_size: Batch size per GPU
        learning_rate: Initial learning rate
        warmup_steps: Number of warmup steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        use_gpu: Use GPU if available
        
    Returns:
        TrainingArguments: Configured training arguments
    """
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=1,
        
        # Optimizer
        learning_rate=learning_rate,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        
        # Learning rate schedule
        warmup_steps=warmup_steps,
        warmup_ratio=0.0,
        lr_scheduler_type="linear",
        
        # Logging and saving
        logging_steps=100,
        logging_dir='./logs',
        save_steps=save_steps,
        save_total_limit=3,  # Keep only 3 recent checkpoints
        eval_steps=eval_steps,
        eval_strategy="steps",
        
        # Performance
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        
        # Misc
        seed=42,
        mixed_precision="no",  # Set to "fp16" for faster training if supported
        report_to=["tensorboard"],  # Can add "wandb" if using weights & biases
        remove_unused_columns=False,
    )
    
    return training_args


def train_model(data_dir='data/tokenized_sangraha',
               model_output_dir='./multilingual_model_output',
               num_epochs=3,
               batch_size=32):
    """
    Main training function
    
    Args:
        data_dir: Path to tokenized dataset
        model_output_dir: Output directory for trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
    """
    
    print("="*60)
    print("TRAINING MULTILINGUAL TRANSFORMER")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Create model
    print("\n[1/5] Creating model with random initialization...")
    model, config = create_model_and_tokenizer()
    
    # Step 2: Load data
    print("\n[2/5] Loading tokenized dataset...")
    dataset = load_from_disk(data_dir)
    print(f"  Training examples: {len(dataset['train'])}")
    print(f"  Validation examples: {len(dataset['test'])}")
    
    # Step 3: Data collator for MLM
    print("\n[3/5] Setting up MLM data collator...")
    # For MLM, we need a tokenizer wrapper
    # Since we use SentencePiece directly, we create a minimal wrapper
    class SimpleTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.vocab_size = 30000
        
        def __call__(self, examples):
            return examples
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=SimpleTokenizer(),
        mlm=True,
        mlm_probability=0.15,  # Mask 15% of tokens
    )
    
    # Step 4: Setup training arguments
    print("\n[4/5] Setting up training arguments...")
    training_args = setup_training_args(
        output_dir=model_output_dir,
        num_train_epochs=num_epochs,
        per_device_batch_size=batch_size,
        learning_rate=5e-4,
        warmup_steps=1000,
        save_steps=500,
        eval_steps=500,
    )
    
    print(f"  Output dir: {model_output_dir}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: 5e-4")
    print(f"  Warmup steps: 1000")
    
    # Step 5: Create trainer and train
    print("\n[5/5] Creating trainer and starting training...")
    
    trainer = Trainer(
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
    model.save_pretrained(f'{model_output_dir}/final_model')
    print(f"✓ Model saved to: {model_output_dir}/final_model")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\nNext: Run 4_test_model.py")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multilingual transformer')
    parser.add_argument('--data_dir', default='data/tokenized_sangraha',
                       help='Path to tokenized dataset')
    parser.add_argument('--output_dir', default='./multilingual_model_output',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        model_output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
