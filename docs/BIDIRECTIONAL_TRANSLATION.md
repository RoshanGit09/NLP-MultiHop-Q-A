# Bidirectional Translation Training

## What Changed

Your model now trains on **BOTH directions**:

## Data Structure

### Before (Unidirectional):
```
500K pairs × 6 languages = 3M examples
Direction: EN → Indic only
```

### After (Bidirectional):
```
500K pairs × 6 languages × 2 directions = 6M examples
Directions: EN ↔ Indic
```

## Training Examples

### Forward (EN → Hindi):
```
src: "The capital of India is New Delhi"
tgt: "भारत की राजधानी नई दिल्ली है"
direction: en→hi
```

### Reverse (Hindi → EN):
```
src: "भारत की राजधानी नई दिल्ली है"
tgt: "The capital of India is New Delhi"
direction: hi→en
```

## How It Works

```
Training Batch (mixed):
├── EN → HI: "Hello" → "नमस्ते"
├── HI → EN: "नमस्ते" → "Hello"
├── EN → TA: "Thank you" → "நன்றி"
├── TA → EN: "நன்றி" → "Thank you"
└── ... (all 12 directions)
```

The model learns to:
1. **Understand** which language is input
2. **Translate** to the target language
3. Work **bidirectionally**

## Translation Directions Supported

| From ↓ To → | EN | HI | TA | TE | MR | KN | ML |
|-------------|----|----|----|----|----|----|---|
| **EN** | - | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **HI** | ✅ | - | ❌ | ❌ | ❌ | ❌ | ❌ |
| **TA** | ✅ | - | - | ❌ | ❌ | ❌ | ❌ |
| **TE** | ✅ | - | - | - | ❌ | ❌ | ❌ |
| **MR** | ✅ | - | - | - | - | ❌ | ❌ |
| **KN** | ✅ | - | - | - | - | - | ❌ |
| **ML** | ✅ | - | - | - | - | - | - |

**Note**: Direct Indic↔Indic translation (e.g., HI→TA) requires English as pivot:
```
HI → EN → TA (2-step translation)
```

## Total Training Data

```
Language Pairs:
  hi: 500K × 2 = 1M examples
  ta: 500K × 2 = 1M examples  
  te: 500K × 2 = 1M examples
  mr: 500K × 2 = 1M examples
  kn: 500K × 2 = 1M examples
  ml: 500K × 2 = 1M examples
  ────────────────────────────
  Total: 6M training examples
```

## Usage After Training

```python
# English → Hindi
translate("Hello, how are you?", src_lang="en", tgt_lang="hi")
# → "नमस्ते, आप कैसे हैं?"

# Hindi → English  
translate("नमस्ते, आप कैसे हैं?", src_lang="hi", tgt_lang="en")
# → "Hello, how are you?"

# English → Tamil
translate("Thank you very much", src_lang="en", tgt_lang="ta")
# → "மிக்க நன்றி"

# Tamil → English
translate("மிக்க நன்றி", src_lang="ta", tgt_lang="en")
# → "Thank you very much"
```

## Training Time Impact

| Setup | Examples | Time (3 epochs) |
|-------|----------|-----------------|
| Unidirectional | 3M | ~12 hours |
| **Bidirectional** | **6M** | **~24 hours** |

Worth it for 2x functionality! 🚀
