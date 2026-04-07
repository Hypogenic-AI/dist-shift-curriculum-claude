# Dataset Evaluation for Distribution Shift Curriculum Experiments

## Summary

For distribution shift curriculum experiments, we need multi-domain text corpora with
**explicit domain labels** so we can withhold and introduce domains during pre-training.

### Recommendation

**Primary: monology/pile-uncopyrighted** -- Best domain diversity (17 labeled domains),
well-studied in curriculum learning papers, streamable for flexible subset creation.

**Secondary: DKYoon/SlimPajama-6B** -- Smaller (14 GB), 7 domains, easier to download
in full. Good for quick iteration.

---

## Dataset Evaluation Results

### 1. monology/pile-uncopyrighted (RECOMMENDED)

- **Status**: Available, no authentication required
- **Format**: JSONL with zstd compression
- **Size**: 335 GB compressed (train), 0.34 GB (val), 0.34 GB (test)
- **Domain label field**: `meta.pile_set_name`
- **Number of domains**: 17 (in first 10k samples; full set has up to 22)
- **Domains observed** (approximate distribution from 10k sample):
  - Pile-CC: 29.9%
  - PubMed Abstracts: 17.1%
  - StackExchange: 16.4%
  - Github: 10.1%
  - Wikipedia (en): 9.3%
  - USPTO Backgrounds: 6.0%
  - PubMed Central: 3.1%
  - FreeLaw: 2.8%
  - NIH ExPorter: 1.2%
  - DM Mathematics: 1.2%
  - ArXiv: 1.1%
  - HackerNews: 1.0%
  - Enron Emails: 0.6%
  - EuroParl: 0.1%
  - PhilPapers: 0.1%
  - Ubuntu IRC: <0.1%
  - Gutenberg (PG-19): <0.1%
- **Pros**: Highest domain diversity, well-studied, supports streaming
- **Cons**: Full dataset is 335 GB; must stream or download subsets
- **Usage**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")
  for example in ds:
      domain = example['meta']['pile_set_name']
      text = example['text']
  ```

### 2. DKYoon/SlimPajama-6B (RECOMMENDED - smaller alternative)

- **Status**: Available, no authentication required
- **Format**: Parquet
- **Size**: ~14 GB total (48 train shards + test + validation)
- **Domain label field**: `meta.redpajama_set_name`
- **Number of domains**: 7
- **Domains observed** (from 10k sample):
  - RedPajamaC4: 54.2%
  - RedPajamaCommonCrawl: 31.3%
  - RedPajamaStackExchange: 5.3%
  - RedPajamaWikipedia: 5.0%
  - RedPajamaGithub: 3.9%
  - RedPajamaArXiv: 0.2%
  - RedPajamaBook: <0.1%
- **Pros**: Manageable size (14 GB), parquet format, same domain labels as RedPajama
- **Cons**: Only 7 domains, heavily dominated by C4 + CommonCrawl (85%)
- **Usage**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("DKYoon/SlimPajama-6B", streaming=True, split="train")
  for example in ds:
      domain = example['meta']['redpajama_set_name']
      text = example['text']
  ```

### 3. togethercomputer/RedPajama-Data-1T-Sample (UNAVAILABLE)

- **Status**: Returns 401 / Repository Not Found. Appears to have been removed or
  made private. The original RedPajama-Data-1T is also gated.
- **Alternatives found**:
  - `konwoo/RedPajama-Data-1T-Sample-subset1000` (very small, 1000 examples)
  - `DKYoon/SlimPajama-6B` (best replacement -- same domain labels)

### 4. mlfoundations/dclm-baseline-1.0 (NOT SUITABLE)

- **Status**: Available, no authentication required
- **Format**: JSONL with zstd compression
- **Size**: ~27,838 files, multi-TB scale
- **Domain labels**: NONE -- this is filtered web crawl (Common Crawl) without
  domain categorization. Has URL metadata but no domain taxonomy.
- **Verdict**: Not suitable for distribution shift experiments. No domain labels.

### 5. cerebras/SlimPajama-627B (UNAVAILABLE)

- **Status**: Returns 401 / Repository Not Found. May have been removed.
- **Alternative**: Use DKYoon/SlimPajama-6B instead (6B token subset with same schema).

### 6. allenai/dolma (PARTIALLY SUITABLE)

- **Status**: Available but hosted on olmo-data.org (not native HF storage)
- **Format**: Gzipped JSONL, downloaded from external URLs
- **Size**: Full v1.6 has 3,224 files across 9 domain categories
- **Domain labels**: Encoded in file paths only (not in document metadata):
  - cc_en_tail (1491 files)
  - cc_en_middle (777 files)
  - cc_en_head (612 files)
  - stack (149 files)
  - c4 (86 files)
  - reddit (78 files)
  - pes2o (26 files) -- scientific papers
  - books (3 files)
  - wiki (2 files)
- **Cons**: Domain labels not in document metadata (only in file paths), requires
  downloading from external server, sample version has no domain separation,
  full version is very large.
- **Verdict**: Usable but requires extra work to associate domain labels with documents.

---

## Practical Recommendations for Experiments

### Small-scale experiments (<10 GB)

Use **DKYoon/SlimPajama-6B**:
- Download the full dataset (~14 GB) or stream a subset
- 7 clearly labeled domains
- Can withhold ArXiv, Github, Books, etc. and introduce them later

### Medium-scale experiments (10-50 GB)

Use **monology/pile-uncopyrighted** with streaming:
- Stream and filter by domain to create custom subsets
- 17 domains gives much richer distribution shift possibilities
- Can create interesting held-out domains: FreeLaw (legal), EuroParl (parliamentary),
  DM Mathematics (math), PhilPapers (philosophy)

### Domain withholding strategy

For the distribution shift curriculum hypothesis, good candidates for withholding are
domains that are:
1. **Distinct** from the majority of training data (not generic web text)
2. **Evaluable** -- you can measure performance on them
3. **Small enough** to not destabilize training when removed

From The Pile, strong candidates for withholding:
- FreeLaw (legal text) -- distinct vocabulary, evaluable
- DM Mathematics -- formal/symbolic, very different from natural language
- ArXiv -- academic/scientific writing
- EuroParl -- parliamentary proceedings, formal language
- Enron Emails -- conversational business email

## Sample Data

A 100-example sample from The Pile is saved in `pile_sample/sample_100.jsonl` for
format verification.

## Dependencies

```bash
uv pip install datasets huggingface_hub zstandard
```
