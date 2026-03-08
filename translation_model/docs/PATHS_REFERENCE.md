# File Paths Reference

## All paths are now consistent across the pipeline

### Tokenizer
```
Input:  Download from Samanantar
Output: tokenizer/multilingual_indic-1.model
        tokenizer/multilingual_indic-1.vocab
Corpus: data/combined_corpus-1.txt
```

### Data Preparation (2_prepare_data.py)
```
Input:  Download from Samanantar (bidirectional pairs)
        Uses: tokenizer/multilingual_indic-1.model
Output: data/tokenized_translation/
        ├── train/
        └── test/
```

### Model Training (3_train_model.py)
```
Input:  data/tokenized_translation/  ✅
Output: transformer_model_output/
        ├── checkpoint-1000/
        ├── checkpoint-2000/
        ├── ...
        └── final_model.pt
```

### Logs (train_job.sh)
```
Output: /dist_home/nooglers/nooglers/Roshan/NLP-MultiHop-Q-A/logs/
        ├── multilingual-transformer_<JOB_ID>.out
        └── multilingual-transformer_<JOB_ID>.err
```

## Default Arguments

### 1_train_tokenizer.py
```python
# No CLI args, uses CONFIG dict
CONFIG = {
    'vocab_size': 100000,
    'corpus_file': 'data/combined_corpus-1.txt',
    'model_prefix': 'tokenizer/multilingual_indic-1',
    ...
}
```

### 2_prepare_data.py
```bash
python 2_prepare_data.py \
  --tokenizer tokenizer/multilingual_indic-1.model  # default ✅
  --output data/tokenized_translation               # default ✅
  --samples 500000                                  # default ✅
```

### 3_train_model.py
```bash
python 3_train_model.py \
  --data_dir data/tokenized_translation   # default ✅ FIXED
  --output_dir ./transformer_model_output # default ✅
  --epochs 3                              # default ✅
  --batch_size 32                         # CLI override
  --grad_accum 4                          # default ✅
  --lr 5e-4                               # default ✅
```

## Path Consistency Check

| File | Writes To | Reads From | Match? |
|------|-----------|------------|--------|
| `1_train_tokenizer.py` | `tokenizer/multilingual_indic-1.*` | - | - |
| `2_prepare_data.py` | `data/tokenized_translation/` | `tokenizer/multilingual_indic-1.model` | ✅ |
| `3_train_model.py` (default) | `transformer_model_output/` | `data/tokenized_translation/` | ✅ |
| `3_train_model.py` (argparse) | `args.output_dir` | `args.data_dir` | ✅ |

## Full Pipeline Command

```bash
# Step 1: Train tokenizer
python 1_train_tokenizer.py
# Creates: tokenizer/multilingual_indic-1.{model,vocab}
#          data/combined_corpus-1.txt

# Step 2: Prepare translation pairs
python 2_prepare_data.py
# Creates: data/tokenized_translation/train/
#          data/tokenized_translation/test/

# Step 3: Train model
python 3_train_model.py --batch_size 32 --grad_accum 4
# Creates: transformer_model_output/final_model.pt

# OR submit SLURM job (runs all steps)
sbatch train_job.sh
```

## All Paths Are Correct! ✅
