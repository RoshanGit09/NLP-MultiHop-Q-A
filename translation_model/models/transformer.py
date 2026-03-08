"""
Encoder-Decoder Transformer for Multilingual Indian Languages
Architecture: Base Model (6+6 layers, 768 dim, ~250M params)

Based on "Attention Is All You Need" (Vaswani et al., 2017)
With modern improvements:
- Pre-layer normalization (pre-norm)
- GELU activation
- Scaled initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from original Transformer paper"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: Optional attention mask
            is_causal: If True, apply causal (autoregressive) mask
        Returns:
            (batch_size, seq_len_q, d_model)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Apply causal mask for decoder self-attention
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=query.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply padding mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # Shape: (batch_size, num_heads, seq_len_q, head_dim)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer with Pre-Norm"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask=mask)
        x = self.dropout(x)
        x = residual + x
        
        # Pre-norm feed-forward
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer with Pre-Norm"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm masked self-attention (causal)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask=tgt_mask, is_causal=True)
        x = self.dropout(x)
        x = residual + x
        
        # Pre-norm cross-attention
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Pre-norm feed-forward
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder Stack"""
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder Stack"""
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        x = self.norm(x)
        return x


class TransformerEncoderDecoder(nn.Module):
    """
    Full Encoder-Decoder Transformer for Seq2Seq tasks
    
    Architecture: Base Model
    - 6 encoder layers
    - 6 decoder layers  
    - 768 model dimension
    - 12 attention heads
    - 3072 FFN dimension
    - ~250M parameters
    """
    
    def __init__(
        self,
        vocab_size: int = 100000,
        d_model: int = 768,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Shared embedding for encoder and decoder
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder and Decoder stacks
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        
        # Output projection (tied with embedding weights)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight  # Weight tying
        
        # Initialize weights
        self._init_weights()
        
        # Print model info
        self._print_model_info()
    
    def _init_weights(self):
        """Initialize weights with scaled initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _print_model_info(self):
        """Print model architecture info"""
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print("TRANSFORMER ENCODER-DECODER MODEL")
        print('='*60)
        print(f"  Architecture: Base Model (6+6 layers)")
        print(f"  Model dimension: {self.d_model}")
        print(f"  Total parameters: {num_params:,}")
        print(f"  Trainable parameters: {num_trainable:,}")
        print(f"  Model size: {num_params * 4 / (1024**2):.2f} MB")
        print('='*60)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create mask for padding tokens"""
        return (x == self.pad_token_id)
    
    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,  # Added for HF Trainer compatibility
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            src_ids: Source token IDs (batch_size, src_len)
            tgt_ids: Target token IDs (batch_size, tgt_len)
            src_mask: Optional source padding mask
            tgt_mask: Optional target padding mask
            labels: Optional labels (ignored, used by HuggingFace Trainer)
            
        Returns:
            logits: (batch_size, tgt_len, vocab_size)
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src_ids)
        if tgt_mask is None:
            tgt_mask = self.create_padding_mask(tgt_ids)
        
        # Embed and add positional encoding
        src_emb = self.embedding(src_ids) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        
        tgt_emb = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Encode source
        encoder_output = self.encoder(src_emb, src_mask)
        
        # Decode target
        decoder_output = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        
        return logits
    
    def generate(
        self,
        src_ids: torch.Tensor,
        max_length: int = 128,
        eos_token_id: int = 3,
        bos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Autoregressive generation (greedy decoding)
        
        Args:
            src_ids: Source token IDs (batch_size, src_len)
            max_length: Maximum generation length
            eos_token_id: End of sequence token ID
            bos_token_id: Beginning of sequence token ID
            
        Returns:
            generated_ids: (batch_size, generated_len)
        """
        self.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device
        
        with torch.no_grad():
            # Encode source
            src_mask = self.create_padding_mask(src_ids)
            src_emb = self.embedding(src_ids) * math.sqrt(self.d_model)
            src_emb = self.pos_encoding(src_emb)
            encoder_output = self.encoder(src_emb, src_mask)
            
            # Initialize with BOS token
            generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
            
            for _ in range(max_length - 1):
                # Decode current sequence
                tgt_emb = self.embedding(generated) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoding(tgt_emb)
                decoder_output = self.decoder(tgt_emb, encoder_output, src_mask)
                
                # Get next token prediction
                logits = self.output_proj(decoder_output[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences have EOS
                if (next_token == eos_token_id).all():
                    break
            
        return generated


def create_model(vocab_size: int = 100000) -> TransformerEncoderDecoder:
    """
    Create the Base Transformer Encoder-Decoder model
    
    Configuration:
    - 6 encoder layers
    - 6 decoder layers
    - 768 model dimension
    - 12 attention heads
    - 3072 FFN dimension
    - ~250M parameters
    """
    model = TransformerEncoderDecoder(
        vocab_size=vocab_size,
        d_model=768,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=12,
        d_ff=3072,
        max_seq_len=512,
        dropout=0.1,
        pad_token_id=0,
    )
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Transformer Encoder-Decoder...")
    
    model = create_model(vocab_size=100000)
    
    # Sample input
    batch_size = 4
    src_len = 64
    tgt_len = 32
    
    src_ids = torch.randint(0, 100000, (batch_size, src_len))
    tgt_ids = torch.randint(0, 100000, (batch_size, tgt_len))
    
    # Forward pass
    logits = model(src_ids, tgt_ids)
    print(f"\nInput shapes:")
    print(f"  src_ids: {src_ids.shape}")
    print(f"  tgt_ids: {tgt_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(src_ids[:1], max_length=20)
    print(f"Generated shape: {generated.shape}")
    
    print("\n✓ Model test passed!")
