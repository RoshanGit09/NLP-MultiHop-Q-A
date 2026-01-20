# Translation Training Setup

## Overview

Your model is now configured for **English ↔ Indic translation** using:
- **Samanantar dataset** (high-quality parallel corpus)
- **Custom Transformer** (6+6 layers, 768 dim, ~250M params)
- **100K vocab tokenizer** (supports EN + 6 Indic languages)

---

## Training Pipeline

```bash
#Step 1: Train tokenizer (if not done)
python 1_train_tokenizer.py

# Step 2: Download & prepare translation pairs
python 2_prepare_data.py

# Step 3: Train translation model
python 3_train_model.py --batch_size 32 --grad_accum 4 --epochs 3

# OR submit SLURM job
sbatch train_job.sh
```

---

## Data Format

### Input (from Samanantar):
```json
{"src": "The capital of India is New Delhi", 
 "tgt": "भारत की राजधानी नई दिल्ली है"}
```

### After Tokenization:
```python
{
  "src_ids": [156, 892, 34, 52, ...],  # English tokens
  "tgt_ids": [2, 1204, 567, 89, ...],  # [BOS] + Hindi tokens + [EOS]
}
```

### During Training:
```
Encoder Input:  src_ids  (English sentence)
Decoder Input:  tgt_ids[:-1]  ([BOS] + Hindi tokens, excluding EOS)
Labels:         tgt_ids[1:]   (Hindi tokens + [EOS], shifted by 1)
```

---

## Model Architecture

```
English Text → [Encoder (6 layers)] → Hidden States
                                          ↓
Hindi Text ← [Decoder (6 layers)] ← Cross-Attention
```

**Training Objective**: Given English, generate Hindi translation token-by-token.

---

## Languages Supported

| Code | Language | Samples |
|------|----------|---------|
| `hi` | Hindi | 500K+ pairs |
| `ta` | Tamil | 500K+ pairs |
| `te` | Telugu | 500K+ pairs |
| `mr` | Marathi | 500K+ pairs |
| `kn` | Kannada | 500K+ pairs |
| `ml` | Malayalam | 500K+ pairs |

**Total**: ~3M translation pairs

---

## Key Differences from Autoregressive LM

| Autoregressive LM | Translation |
|-------------------|-------------|
| Same language in/out | Different languages |
| "भारत की" → "राजधानी" | "Capital of" → "राजधानी" |
| Text completion | Language conversion |
| Monolingual corpus | Parallel corpus |

---

## Expected Training Time

| GPUs | Batch Size | Time (3 epochs) |
|------|------------|-----------------|
| 1x RTX 6000 Ada | 32 | ~18-24 hours |
| 2x RTX 6000 Ada | 64 | ~10-12 hours |

---

## After Training - Usage

```python
from models.transformer import create_model
import torch

# Load model
model = create_model(vocab_size=100000)
model.load_state_dict(torch.load('transformer_model_output/final_model.pt')['model_state_dict'])
model.eval()

# Translate
src_text = "Hello, how are you?"
src_ids = tokenizer.encode(src_text)  # Your SentencePiece tokenizer
translated_ids = model.generate(torch.tensor([src_ids]))
translation = tokenizer.decode(translated_ids[0].tolist())
# Output: "नमस्ते, आप कैसे हैं?"
```

---

## Files Changed

1. ✅ `2_prepare_data.py` - Now downloads paired EN↔Indic data
2. ✅ `3_train_model.py` - Uses TranslationDataCollator & TranslationTrainer
3. ✅ `train_job.sh` - Ready for SLURM submission

---

## Next Steps

1. Run tokenizer training (if not done):
   ```bash
   python 1_train_tokenizer.py
   ```

2. Prepare translation data:
   ```bash
   python 2_prepare_data.py
   ```

3. Submit training job:
   ```bash
   sbatch train_job.sh
   ```

4. Monitor:
   ```bash
   squeue -u $USER
   tail -f logs/multilingual-transformer_*.out
   ```
