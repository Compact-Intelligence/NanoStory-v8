#!/usr/bin/env python3
"""
NanoStory v8 — BPE Subword Data Preparation
=============================================
Same BPE tokenization as v7 (256 tokens) but with updated corpus.
v8 targets ~148K params — needs block_size tuning for longer sequences.
"""
import os
import json
import pickle
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import BPEDecoder

# ─── Configuration ────────────────────────────────────────
STORIES_PATH = "/home/joshua/tinystories-ai/stories.jsonl"
CACHE_DIR = "cache_v8"
VOCAB_SIZE = 256
BLOCK_SIZE = 128
VAL_FRAC = 0.05
SEED = 42

# Special token IDs
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

os.makedirs(CACHE_DIR, exist_ok=True)

# ─── Load stories ─────────────────────────────────────────
print("Loading stories...")
stories = []
bad = 0
with open(STORIES_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            text = obj["story"].strip()
            if len(text) >= 20:
                stories.append(text)
        except (json.JSONDecodeError, KeyError):
            bad += 1

print(f"Loaded {len(stories)} stories ({bad} bad lines skipped)")
lengths = [len(s) for s in stories]
print(f"Story lengths: avg={np.mean(lengths):.0f} chars, median={np.median(lengths):.0f} chars")

# ─── Train BPE tokenizer ─────────────────────────────────
tokenizer_path = os.path.join(CACHE_DIR, "bpe_tokenizer.json")

if os.path.exists(tokenizer_path):
    print(f"Loading existing tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    print(f"\nTraining BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Metaspace(replacement="Ġ")
    tokenizer.decoder = BPEDecoder(suffix="")

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(stories, trainer=trainer, length=len(stories))
    tokenizer.save(tokenizer_path)

print(f"Vocabulary size: {tokenizer.get_vocab_size()} tokens")
print(f"Tokenizer saved to {tokenizer_path}")

# ─── Tokenize stories ─────────────────────────────────────
print("\nTokenizing stories...")

all_token_ids = []
total_chars = 0
total_tokens = 0
unk_count = 0
story_token_counts = []

for story in stories:
    encoding = tokenizer.encode(story)
    ids = encoding.ids
    total_chars += len(story)
    total_tokens += len(ids)
    unk_count += ids.count(UNK_ID)
    story_token_counts.append(len(ids))
    all_token_ids.append(ids)

non_pad_total = sum(len(ids) for ids in all_token_ids)
print(f"\n{'='*60}")
print(f"TOKENIZATION RESULTS")
print(f"{'='*60}")
print(f"Total stories: {len(stories):,}")
print(f"Total characters: {total_chars:,}")
print(f"Total BPE tokens: {total_tokens:,}")
print(f"UNK tokens: {unk_count} ({unk_count/total_tokens*100:.2f}%)")
print(f"Avg tokens/story: {np.mean(story_token_counts):.1f}")
print(f"Median tokens/story: {np.median(story_token_counts):.0f}")
print(f"95th percentile: {np.percentile(story_token_counts, 95):.0f}")
print(f"99th percentile: {np.percentile(story_token_counts, 99):.0f}")
print(f"Block size: {BLOCK_SIZE}")

truncated = sum(1 for c in story_token_counts if c > BLOCK_SIZE - 2)
print(f"Stories truncated: {truncated} ({truncated/len(stories)*100:.1f}%)")

# ─── Create sequences ─────────────────────────────────────
np.random.seed(SEED)
indices = np.random.permutation(len(stories))
val_count = int(len(stories) * VAL_FRAC)
val_indices = set(indices[:val_count].tolist())

train_seqs = []
val_seqs = []

for i, ids in enumerate(all_token_ids):
    # Add BOS + EOS, truncate to block_size
    seq = [BOS_ID] + ids[:BLOCK_SIZE - 2] + [EOS_ID]
    # Pad to block_size
    seq = seq + [PAD_ID] * (BLOCK_SIZE - len(seq))
    if i in val_indices:
        val_seqs.append(seq)
    else:
        train_seqs.append(seq)

train_data = np.array(train_seqs, dtype=np.uint16)
val_data = np.array(val_seqs, dtype=np.uint16)

print(f"\nTrain sequences: {len(train_seqs):,}")
print(f"Val sequences: {len(val_seqs):,}")

# Count non-PAD tokens for data ratio
non_pad_train = np.count_nonzero(train_data != PAD_ID)
non_pad_val = np.count_nonzero(val_data != PAD_ID)
non_pad_total = non_pad_train + non_pad_val
print(f"Non-PAD tokens: {non_pad_total:,}")
print(f"Data ratio: {non_pad_total/148224:.0f}:1 (vs ~148K params)")

# ─── Save ─────────────────────────────────────────────────
train_path = os.path.join(CACHE_DIR, "train.bin")
val_path = os.path.join(CACHE_DIR, "val.bin")

train_data.tofile(train_path)
val_data.tofile(val_path)

meta = {
    "vocab_size": VOCAB_SIZE,
    "block_size": BLOCK_SIZE,
    "pad_id": PAD_ID,
    "bos_id": BOS_ID,
    "eos_id": EOS_ID,
    "unk_id": UNK_ID,
    "n_train": len(train_seqs),
    "n_val": len(val_seqs),
}
with open(os.path.join(CACHE_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"\nSaved train.bin: {train_data.size:,} elements ({len(train_seqs)} sequences)")
print(f"Saved val.bin: {val_data.size:,} elements ({len(val_seqs)} sequences)")

# ─── Verification ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"VERIFICATION SAMPLES")
print(f"{'='*60}")
vocab = tokenizer.get_vocab()
id_to_token = {v: k for k, v in vocab.items()}

for i in range(min(3, len(stories))):
    story = stories[i]
    enc = tokenizer.encode(story)
    ids = enc.ids[:BLOCK_SIZE - 2]
    tokens = [id_to_token.get(tid, "<UNK>") for tid in ids]
    decoded = "".join(t.replace("Ġ", " ") for t in tokens).strip()
    print(f"\nOriginal: {story[:80]}...")
    print(f"Tokens: {ids[:20]}... ({len(ids)} total)")
    print(f"Decoded: {decoded[:80]}...")

print(f"\n{'='*60}")
print(f"READY FOR TRAINING")
print(f"{'='*60}")
print(f"Cache directory: {CACHE_DIR}/")
print(f"  bpe_tokenizer.json — BPE tokenizer model")
print(f"  train.bin — {len(train_seqs):,} training sequences")
print(f"  val.bin — {len(val_seqs):,} validation sequences")
print(f"\nArchitecture target: ~148K params")
print(f"  vocab=256, d_model=64, GRU×2 h=64, Attn 4h×2 blocks, SwiGLU ff=128")
