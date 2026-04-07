# Resources Catalog

## Summary

This document catalogs all resources gathered for the "Distribution Shift Curriculum" research project, which investigates whether introducing regular distribution shifts during LLM pre-training encourages better generalization to unseen distributions.

---

## Papers

**Total papers downloaded: 29**

### Core Curriculum Learning for LLMs (Most Relevant)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Curriculum Learning | Bengio et al. | 2009 | papers/0904.2684_*.pdf | Foundational paper; 6800+ citations |
| On the Role of Corpus Ordering in LM | Campos et al. | 2021 | papers/2104.06127_*.pdf | Negative result for naive curricula |
| Curriculum Learning for Language Modeling | Nagatsuka et al. | 2021 | papers/2108.02170_*.pdf | CBC with difficulty proxies |
| Pre-training BERT with Curriculum (Block-Size) | Campos et al. | 2021 | papers/2012.10839_*.pdf | Increasing block size curriculum |
| Length-Based Curriculum for LM Pretraining | Choe et al. | 2022 | papers/2206.13697_*.pdf | Sequence length curriculum |
| DeepSpeed Data Efficiency | Microsoft | 2022 | papers/2212.03597_*.pdf | Production curriculum learning |
| Skill-it! Data-Driven Skills Framework | Chen et al. | 2023 | papers/2307.14430_*.pdf | **Skills graph + dynamic reweighting; 3x efficiency** |
| Irreducible Curriculum for LM Pretraining | Fan & Jaggi | 2023 | papers/2310.15389_*.pdf | **Proxy-model learnability scoring** |
| Strategic Data Ordering for LLMs | Kim & Lee | 2024 | papers/2405.07490_*.pdf | Attention-based difficulty; modest gains |
| Code-Switching Curriculum for Multilingual LLMs | Iyer et al. | 2024 | papers/2407.16974_*.pdf | Multilingual transfer curriculum |
| Preference Curriculum for LLM Pretraining | Ge et al. | 2024 | papers/2412.10369_*.pdf | Pretrain on preferred data first |
| Scaling LLM Pretraining with Vocabulary Curriculum | Chen et al. | 2024 | papers/2412.16467_*.pdf | Vocabulary expansion curriculum |
| Beyond Random Sampling: Curriculum LM Pretraining | Jiang et al. | 2025 | papers/2502.18517_*.pdf | Recent curriculum survey/method |
| How LR Decay Wastes Your Best Data | Luo et al. | 2025 | papers/2511.18903_*.pdf | **Critical: LR schedule must co-design with curriculum** |
| Curriculum-Guided Layer Scaling | Singh et al. | 2025 | papers/2506.11389_*.pdf | Layer-wise curriculum scaling |
| Impact of Pretraining Data Ordering | Xie et al. | 2026 | papers/2503.06868_*.pdf | Encoder vs decoder ordering effects |
| Curriculum LLM Pretraining: Learning Dynamics | Singh et al. | 2026 | papers/2503.15862_*.pdf | Analysis of learning dynamics |

### Distribution Shift Robustness

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Invariant Risk Minimization (IRM) | Arjovsky et al. | 2019 | papers/1907.02893_*.pdf | Invariant features; 2600+ citations |
| Group DRO | Sagawa et al. | 2019 | papers/1911.08731_*.pdf | Worst-group robustness; 1500+ citations |
| Risk Extrapolation (REx) | Krueger et al. | 2021 | papers/2003.00688_*.pdf | Risk variance penalty; 1100+ citations |
| OOD via Multi-Task Self-Supervised Pretraining | Yi et al. | 2020 | papers/2003.13525_*.pdf | Self-supervised pretraining for OOD |
| Robust Learning via Random Convolutions | Xu et al. | 2020 | papers/2007.13003_*.pdf | Random augmentation for robustness |
| Focus on the Common Good (CGD) | Piratla et al. | 2022 | papers/2110.02619_*.pdf | Common gradient descent |
| Distribution Shift Robustness From Pretraining | Yi et al. | 2022 | papers/2205.12392_*.pdf | **Pretraining diversity = robustness** |
| WILDS Benchmark | Koh et al. | 2021 | papers/2012.07421_*.pdf | Distribution shift benchmark; 1700+ citations |
| Generalizing to Unseen Domains: Survey | Wang et al. | 2021 | papers/2103.02503_*.pdf | Domain generalization survey |
| Domain Generalization: Survey | Zhou et al. | 2021 | papers/2103.03097_*.pdf | Comprehensive DG survey |

### Developmentally-Inspired Pretraining

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| BabyLM Challenge | Warstadt et al. | 2023 | papers/2311.05169_*.pdf | Sample-efficient pretraining |
| CLIMB: Curriculum Learning for Infant-inspired Models | Huebner et al. | 2023 | papers/2305.18441_*.pdf | Infant-inspired curriculum |

**Note:** Some downloaded PDFs may have content mismatches (arxiv serving issues). Key papers were verified through agent-based deep reading.

See `papers/` directory for all PDFs and `papers/pages/` for chunked versions of deeply-read papers.

---

## Datasets

**Total datasets evaluated: 6 | Recommended: 2**

| Name | Source | Size | Domains | Location | Status |
|------|--------|------|---------|----------|--------|
| The Pile (uncopyrighted) | monology/pile-uncopyrighted | 335 GB | 17 | datasets/pile/ | **Primary - sample downloaded** |
| SlimPajama-6B | DKYoon/SlimPajama-6B | 14 GB | 7 | (stream) | **Secondary - verified** |
| RedPajama-1T-Sample | togethercomputer | N/A | 7 | N/A | Unavailable (removed) |
| DCLM-Baseline | mlfoundations | Multi-TB | 0 | N/A | Not suitable (no domain labels) |
| SlimPajama-627B | cerebras | 627B tokens | 7 | N/A | Unavailable (removed) |
| Dolma | allenai | Multi-TB | 9 | N/A | Partially suitable |

See `datasets/README.md` for detailed evaluation, download instructions, and domain withholding strategy.

**Download script:** `datasets/download.sh` — streams data and splits by domain.

---

## Code Repositories

**Total repositories cloned: 3**

| Name | URL | Purpose | Location | Stars |
|------|-----|---------|----------|-------|
| Skill-it | github.com/HazyResearch/skill-it | Skills-graph curriculum learning | code/skill-it/ | ~48 |
| Rho-1 | github.com/microsoft/rho | Token-level selective pretraining | code/rho/ | ~465 |
| DoReMi | github.com/sangmichaelxie/doremi | Domain mixture optimization | code/doremi/ | ~351 |

**Additional repos documented (not cloned):**
- RHO-Loss (github.com/OATML/RHO-Loss) — Predecessor to Rho-1
- WILDS (github.com/p-lambda/wilds) — Distribution shift benchmark
- Megatron-DeepSpeed curriculum learning — Sequence-length curriculum
- IRM (github.com/facebookresearch/InvariantRiskMinimization)
- Group DRO (github.com/kohpangwei/group_DRO)
- CGD (github.com/vihari/CGD)

See `code/README.md` for detailed descriptions of each repository.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with 3 diligent-mode queries covering curriculum learning, distribution shift, and domain generalization
2. Aggregated 106 relevance-3 papers, ranked by keyword match to our hypothesis
3. Downloaded 29 papers with known arxiv IDs
4. Deep-read 8 most critical papers using PDF chunker
5. Searched HuggingFace for multi-domain text corpora with domain labels
6. Searched GitHub for curriculum learning implementations

### Selection Criteria
- **Papers:** Prioritized (1) curriculum learning for LLM pretraining, (2) distribution shift robustness methods, (3) foundational works with high citations
- **Datasets:** Required explicit domain labels for distribution shift experiments; manageable size
- **Code:** Prioritized implementations of data selection/curriculum methods for LLM pretraining

### Challenges Encountered
- RedPajama-Data-1T-Sample (the most commonly cited dataset in curriculum papers) has been removed from HuggingFace. The Pile and SlimPajama-6B serve as replacements.
- Some arxiv PDF downloads resulted in mismatched content (arxiv serving issues). Key papers were verified through deep reading.
- Semantic Scholar API rate limits required careful batching.

### Gaps and Workarounds
- No existing implementation directly tests domain-withholding curricula. Experiment code will need to be written from scratch, potentially adapting DoReMi's domain reweighting infrastructure.
- The Pile's domain distribution is heavily skewed (30% Pile-CC). Experiments should control for this.

---

## Recommendations for Experiment Design

### 1. Primary Dataset
**The Pile (monology/pile-uncopyrighted)** — 17 domains with clear labels. Withhold 3-5 niche domains (FreeLaw, DM Mathematics, EuroParl, PhilPapers) for distribution shift experiments.

### 2. Baseline Methods
1. **Uniform random sampling** — Standard pretraining baseline
2. **DoReMi mixture** — Optimized static domain weights (adapt code/doremi/)
3. **Irreducible curriculum** — Proxy-model learnability scoring
4. **No-shift control** — Train with all domains from the start

### 3. Evaluation Metrics
1. **Per-domain perplexity** on withheld domains before/after introduction
2. **Adaptation speed** to withheld domains (perplexity vs. tokens seen)
3. **Overall perplexity** across all domains
4. **Worst-domain perplexity** (robustness measure)

### 4. Code to Adapt/Reuse
- **code/doremi/** — Domain reweighting infrastructure, proxy model training
- **code/skill-it/** — Dynamic data selection with dependency modeling
- **code/rho/** — Token-level scoring with reference model

### 5. Critical Design Choices
- **Use constant LR or moderate decay** (not standard cosine) per Luo et al. (2025)
- **Apply model averaging** (EMA) over final checkpoints
- **Measure per-domain metrics** — aggregate metrics hide domain-level failures
- **Use a small proxy model** for scoring if doing learnability-based curriculum
- **Start small** — 124M model, 10K steps, then scale if effects are positive
