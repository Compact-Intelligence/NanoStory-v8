# NanoStory v8

A 148K parameter story generation model combining GRU sequential processing with Transformer attention — built by [Compact Intelligence](https://github.com/Compact-Intelligence).

## Architecture

```
Token → Embed(256×64) → [GRU×2 h=64] → [Attn 4h + SwiGLU ff=128]×2 → lm_head → Token
```

**Parameter budget:**
- Embedding: 16,384 (11.1%)
- GRU blocks: 49,536 (33.4%)
- Transformer: 82,176 (55.4%)
- **Total: ~148,224 params**
- Compute ratio: 88.9% (target >65%)

**Key design choices:**
- **GRU backbone** — provides strong sequential bias for narrative coherence
- **ALiBi position biases** — eliminates learned position embeddings
- **SwiGLU FFN** — modern feed-forward with gated activation
- **RMSNorm** — efficient layer normalization
- **Weight-tied lm_head** — embedding and output projection share weights
- **BPE subword tokenization** — 256-token vocabulary with Metaspace pre-tokenizer

## Requirements

- Python 3.9+
- PyTorch >= 2.0
- Hugging Face `tokenizers` >= 0.15
- NumPy >= 1.24

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

NanoStory v8 trains on synthetic children's stories. You need a JSONL file where each line has a `"story"` field:

```jsonl
{"story": "once upon a time a little cat sat on the wall..."}
{"story": "ben and lily went to the park and played..."}
```

A curated dataset is available at [Compact-Intelligence/NanoStory-Dataset](https://github.com/Compact-Intelligence/NanoStory-Dataset).

## Training

### 1. Prepare the data

Edit the `STORIES_PATH` in `prepare_data_v8.py` to point to your JSONL dataset, then run:

```bash
python prepare_data_v8.py
```

This trains a BPE tokenizer (if one doesn't exist) and writes tokenized sequences to `cache_v8/`:
- `bpe_tokenizer.json` — BPE tokenizer model
- `train.bin` — training sequences (uint16)
- `val.bin` — validation sequences (uint16)
- `meta.pkl` — metadata (vocab size, block size, etc.)

### 2. Train the model

```bash
python train_v8.py
```

**Default training configuration:**
- Max iterations: 80,000
- Batch size: 24
- Block size: 128 tokens
- Learning rate: 3e-4 → 3e-5 (cosine decay)
- Warmup: 500 iterations
- Weight decay: 0.01
- Gradient clipping: 1.0
- Early stopping patience: 8,000 iterations
- Evaluation every 500 iterations
- Sample generation every 2,000 iterations

Checkpoints are saved to `checkpoints_v8/best_model.pt` whenever validation loss improves.

### 3. Generate stories

```python
import torch
from tokenizers import Tokenizer
from model_v8 import NanoStoryV8

# Load model
ckpt = torch.load("checkpoints_v8/best_model.pt", map_location="cpu")
model = NanoStoryV8(vocab_size=256)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Load tokenizer
tokenizer = Tokenizer.from_file("cache_v8/bpe_tokenizer.json")

# Generate
prompt = "once upon a time"
encoding = tokenizer.encode(prompt)
prompt_ids = [1] + encoding.ids  # BOS + prompt tokens
prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)

output = model.generate(prompt_tensor, max_new_tokens=60, temperature=0.8, top_k=40)

# Decode
vocab = tokenizer.get_vocab()
id_to_token = {v: k for k, v in vocab.items()}
tokens = [id_to_token.get(i, "<UNK>") for i in output[0].tolist()]
text = "".join(t.replace("\u0120", " ") for t in tokens).strip()
print(text)
```

## Pre-trained Checkpoint

A pre-trained checkpoint is included at `checkpoints/best_model.pt` trained on ~5,000 synthetic children's stories. This serves as a starting point — for best results, retrain on a larger corpus.

## Project Structure

```
├── model_v8.py           # Model architecture (NanoStoryV8)
├── prepare_data_v8.py    # Data preparation & BPE tokenizer training
├── train_v8.py           # Training loop with evaluation & sampling
├── requirements.txt      # Python dependencies
├── vocab_v3.json         # Reference vocabulary
├── checkpoints/
│   └── best_model.pt     # Pre-trained model checkpoint
└── README.md
```

## License

MIT
