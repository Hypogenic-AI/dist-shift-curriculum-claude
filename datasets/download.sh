#!/usr/bin/env bash
# Download script for Distribution Shift Curriculum datasets
# Usage: bash datasets/download.sh [pile|slimpajama|both] [num_examples]
#
# Examples:
#   bash datasets/download.sh pile 100000        # 100k examples from The Pile
#   bash datasets/download.sh slimpajama 50000   # 50k examples from SlimPajama-6B
#   bash datasets/download.sh both 100000        # Both datasets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_ROOT/.venv"
OUTPUT_DIR="$SCRIPT_DIR"

DATASET="${1:-pile}"
NUM_EXAMPLES="${2:-100000}"

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Create it with: python3 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && uv pip install datasets huggingface_hub zstandard"
    exit 1
fi

# Check dependencies
python3 -c "import datasets, zstandard" 2>/dev/null || {
    echo "Installing required packages..."
    uv pip install datasets huggingface_hub zstandard
}

download_pile() {
    local n="$1"
    echo "=========================================="
    echo "Downloading $n examples from monology/pile-uncopyrighted"
    echo "=========================================="

    python3 << PYEOF
import json
import os
from collections import Counter
from datasets import load_dataset

n = $n
output_dir = "$OUTPUT_DIR/pile"
os.makedirs(output_dir, exist_ok=True)

print(f"Streaming {n} examples from monology/pile-uncopyrighted...")
ds = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")

domains = Counter()
output_path = os.path.join(output_dir, f"train_{n}.jsonl")

with open(output_path, "w") as f:
    for i, ex in enumerate(ds):
        if i >= n:
            break
        domains[ex['meta']['pile_set_name']] += 1
        f.write(json.dumps(ex) + "\n")
        if (i + 1) % 10000 == 0:
            print(f"  Downloaded {i+1}/{n} examples...")

print(f"\nSaved {n} examples to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1e6:.1f} MB")
print(f"\nDomain distribution:")
for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
    print(f"  {domain}: {count} ({count/n*100:.1f}%)")

# Also save domain-separated files for easy curriculum construction
domain_dir = os.path.join(output_dir, "by_domain")
os.makedirs(domain_dir, exist_ok=True)

# Re-read and split by domain
print(f"\nSplitting by domain into {domain_dir}/...")
domain_files = {}
with open(output_path) as f:
    for line in f:
        ex = json.loads(line)
        domain = ex['meta']['pile_set_name'].replace(' ', '_').replace('(', '').replace(')', '')
        if domain not in domain_files:
            domain_files[domain] = open(os.path.join(domain_dir, f"{domain}.jsonl"), "w")
        domain_files[domain].write(line)

for fh in domain_files.values():
    fh.close()

print("Done! Domain files created:")
for domain in sorted(domain_files.keys()):
    path = os.path.join(domain_dir, f"{domain}.jsonl")
    size = os.path.getsize(path) / 1e6
    print(f"  {domain}.jsonl: {size:.1f} MB")
PYEOF
}

download_slimpajama() {
    local n="$1"
    echo "=========================================="
    echo "Downloading $n examples from DKYoon/SlimPajama-6B"
    echo "=========================================="

    python3 << PYEOF
import json
import os
from collections import Counter
from datasets import load_dataset

n = $n
output_dir = "$OUTPUT_DIR/slimpajama"
os.makedirs(output_dir, exist_ok=True)

print(f"Streaming {n} examples from DKYoon/SlimPajama-6B...")
ds = load_dataset("DKYoon/SlimPajama-6B", streaming=True, split="train")

domains = Counter()
output_path = os.path.join(output_dir, f"train_{n}.jsonl")

with open(output_path, "w") as f:
    for i, ex in enumerate(ds):
        if i >= n:
            break
        # Normalize: remove __index_level_0__ field, keep text and meta
        record = {"text": ex["text"], "meta": ex["meta"]}
        domains[ex['meta']['redpajama_set_name']] += 1
        f.write(json.dumps(record) + "\n")
        if (i + 1) % 10000 == 0:
            print(f"  Downloaded {i+1}/{n} examples...")

print(f"\nSaved {n} examples to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1e6:.1f} MB")
print(f"\nDomain distribution:")
for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
    print(f"  {domain}: {count} ({count/n*100:.1f}%)")

# Split by domain
domain_dir = os.path.join(output_dir, "by_domain")
os.makedirs(domain_dir, exist_ok=True)

print(f"\nSplitting by domain into {domain_dir}/...")
domain_files = {}
with open(output_path) as f:
    for line in f:
        ex = json.loads(line)
        domain = ex['meta']['redpajama_set_name']
        if domain not in domain_files:
            domain_files[domain] = open(os.path.join(domain_dir, f"{domain}.jsonl"), "w")
        domain_files[domain].write(line)

for fh in domain_files.values():
    fh.close()

print("Done! Domain files created:")
for domain in sorted(domain_files.keys()):
    path = os.path.join(domain_dir, f"{domain}.jsonl")
    size = os.path.getsize(path) / 1e6
    print(f"  {domain}.jsonl: {size:.1f} MB")
PYEOF
}

case "$DATASET" in
    pile)
        download_pile "$NUM_EXAMPLES"
        ;;
    slimpajama)
        download_slimpajama "$NUM_EXAMPLES"
        ;;
    both)
        download_pile "$NUM_EXAMPLES"
        download_slimpajama "$NUM_EXAMPLES"
        ;;
    *)
        echo "Usage: $0 [pile|slimpajama|both] [num_examples]"
        echo "  pile        - Download from monology/pile-uncopyrighted (17 domains)"
        echo "  slimpajama  - Download from DKYoon/SlimPajama-6B (7 domains)"
        echo "  both        - Download both datasets"
        exit 1
        ;;
esac

echo ""
echo "Download complete. Data saved to $OUTPUT_DIR/"
