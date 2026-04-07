"""Data preparation: stream from The Pile, tokenize, and split by domain."""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Domain partitioning for the experiment
CORE_DOMAINS = [
    "Pile-CC", "PubMed Abstracts", "StackExchange", "Github",
    "Wikipedia (en)", "USPTO Backgrounds", "HackerNews",
    "NIH ExPorter", "Enron Emails",
]
SHIFT_DOMAINS = ["FreeLaw", "DM Mathematics", "PubMed Central"]
HELD_OUT_DOMAINS = ["ArXiv"]

ALL_TRAIN_DOMAINS = CORE_DOMAINS + SHIFT_DOMAINS
ALL_DOMAINS = ALL_TRAIN_DOMAINS + HELD_OUT_DOMAINS

SEQ_LEN = 512
TOKENIZER_NAME = "gpt2"


def stream_and_tokenize(
    target_tokens_per_domain: int = 500_000,
    val_tokens_per_domain: int = 20_000,
    output_dir: str = "datasets/prepared",
    seed: int = 42,
):
    """Stream from The Pile, tokenize, and save domain-split tensors."""
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Track tokens collected per domain
    domain_tokens = defaultdict(list)
    domain_counts = defaultdict(int)
    target_per_domain = target_tokens_per_domain + val_tokens_per_domain

    print(f"Streaming from monology/pile-uncopyrighted...")
    print(f"Target: {target_tokens_per_domain} train + {val_tokens_per_domain} val tokens per domain")

    ds = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")

    domains_done = set()
    total_seen = 0

    for example in ds:
        domain = example["meta"]["pile_set_name"]

        if domain not in ALL_DOMAINS:
            continue
        if domain in domains_done:
            total_seen += 1
            continue

        text = example["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        domain_tokens[domain].extend(tokens)
        domain_counts[domain] += 1
        total_seen += 1

        if len(domain_tokens[domain]) >= target_per_domain:
            domains_done.add(domain)
            print(f"  {domain}: {len(domain_tokens[domain])} tokens from {domain_counts[domain]} docs")

        if total_seen % 10000 == 0:
            collected = {d: len(t) for d, t in domain_tokens.items()}
            done = len(domains_done)
            print(f"  Seen {total_seen} docs, {done}/{len(ALL_DOMAINS)} domains complete. {collected}")

        if len(domains_done) >= len(ALL_DOMAINS):
            break

    # Save tokenized data as tensors
    print("\nSaving tokenized data...")
    for domain in ALL_DOMAINS:
        tokens = domain_tokens[domain]
        if len(tokens) < val_tokens_per_domain + SEQ_LEN:
            print(f"  WARNING: {domain} only has {len(tokens)} tokens, may be insufficient")
            if len(tokens) == 0:
                continue

        # Truncate to target
        tokens = tokens[:target_per_domain]

        # Split into train and val
        val_tokens = tokens[:val_tokens_per_domain]
        train_tokens = tokens[val_tokens_per_domain:]

        # Create sequences of SEQ_LEN
        def make_sequences(tok_list):
            n = len(tok_list) // SEQ_LEN
            if n == 0:
                return torch.tensor([], dtype=torch.long)
            arr = torch.tensor(tok_list[: n * SEQ_LEN], dtype=torch.long)
            return arr.view(n, SEQ_LEN)

        train_seqs = make_sequences(train_tokens)
        val_seqs = make_sequences(val_tokens)

        domain_safe = domain.replace(" ", "_").replace("(", "").replace(")", "")
        torch.save(train_seqs, os.path.join(output_dir, f"{domain_safe}_train.pt"))
        torch.save(val_seqs, os.path.join(output_dir, f"{domain_safe}_val.pt"))
        print(f"  {domain}: train={train_seqs.shape}, val={val_seqs.shape}")

    print(f"\nData saved to {output_dir}/")
    return output_dir


if __name__ == "__main__":
    stream_and_tokenize()
