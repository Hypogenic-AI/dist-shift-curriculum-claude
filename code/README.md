# Distribution Shift Curriculum - Code Repository Survey

Survey conducted: 2026-04-07

## Cloned Repositories (Top 3 by Relevance)

### 1. HazyResearch/skill-it
- **URL:** https://github.com/HazyResearch/skill-it
- **Stars:** ~48 | **Forks:** ~8
- **Paper:** "Skill-It! A Data-Driven Skills Framework for Understanding and Training Language Models" (Chen et al., NeurIPS 2023 Spotlight)
- **What it implements:** A curriculum learning method that defines "skills" in training data, constructs a dependency graph (skills graph) encoding how prerequisite skills help unlock advanced skills, and dynamically adjusts sampling proportions during training. Achieves higher accuracy with less data by optimizing training order.
- **Languages:** Jupyter Notebook (97.5%), Python (2.2%), Shell (0.3%)
- **Relevance:** **HIGH** - Directly implements curriculum learning for language models with skill-based data ordering. Core to our research on distribution shift curricula.
- **Last updated:** October 2023

### 2. microsoft/rho
- **URL:** https://github.com/microsoft/rho
- **Stars:** ~465 | **Forks:** ~15
- **Paper:** "Rho-1: Not All Tokens Are What You Need" (NeurIPS 2024)
- **What it implements:** Selective Language Modeling (SLM) for LLM pretraining. Trains a reference model on high-quality data, scores pretraining tokens by excess loss, then selectively trains on tokens with higher excess loss. Rho-1-1B/7B achieve SOTA on MATH with only 3% of pretraining tokens. On 80B general tokens, achieves 6.8% average improvement across 15 tasks.
- **Languages:** Python (primary)
- **Relevance:** **HIGH** - Token-level data selection is a fine-grained form of curriculum/distribution control during pretraining. Directly relevant to understanding how training data distribution affects LLM quality.
- **Last updated:** April 2024

### 3. sangmichaelxie/doremi
- **URL:** https://github.com/sangmichaelxie/doremi
- **Stars:** ~351 | **Forks:** ~35
- **Paper:** "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining" (Xie et al., NeurIPS 2023)
- **What it implements:** Domain Reweighting with Minimax Optimization. Uses a small proxy model trained with Group DRO to produce optimal domain weights (mixture proportions), then resamples the dataset with these weights for training a larger model. A 280M proxy sets weights for an 8B model, improving average few-shot accuracy by 6.5% and reaching baseline accuracy 2.6x faster.
- **Languages:** HTML (72.2%), C++ (15%), CUDA (10.6%), Python (1.8%) -- skewed by FlashAttention2 submodule; core logic is Python
- **Relevance:** **HIGH** - Optimizing data mixture distributions is central to understanding how distribution shifts during pretraining affect model quality.
- **Last updated:** September 2023

---

## Other Relevant Repositories (Not Cloned)

### 4. OATML/RHO-Loss
- **URL:** https://github.com/OATML/RHO-Loss
- **Stars:** ~213 | **Forks:** ~19
- **Paper:** "Prioritized training on points that are learnable, worth learning, and not yet learned" (Mindermann et al., ICML 2022)
- **What it implements:** Reducible Holdout Loss Selection - selects training points that most reduce generalization loss. 18x faster training on Clothing-1M with 2% higher accuracy. Supports vision (ResNets) and NLP (ALBERT).
- **Relevance:** MEDIUM - Pioneering work on loss-based data selection (predecessor to Rho-1), but focused more on vision/BERT than LLM pretraining.

### 5. p-lambda/wilds
- **URL:** https://github.com/p-lambda/wilds
- **Stars:** ~596 | **Forks:** ~135
- **Paper:** "WILDS: A Benchmark of in-the-Wild Distribution Shifts" (Koh et al., ICML 2021)
- **What it implements:** Benchmark of 10 datasets with real-world distribution shifts. Includes data loaders, evaluators, and implementations of adaptation methods (ERM, GroupDRO, CORAL, IRM, DANN, etc.).
- **Relevance:** MEDIUM - Important for evaluation methodology and understanding distribution shift, but focused on supervised learning rather than LLM pretraining.

### 6. microsoft/Megatron-DeepSpeed (curriculum learning module)
- **URL:** https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples_deepspeed/curriculum_learning
- **Stars:** ~2000+ (whole repo)
- **Paper:** "Curriculum Learning: A Regularization Method for Efficient and Stable Billion-Scale GPT Model Pre-Training"
- **What it implements:** Sequence-length-based curriculum learning for GPT pretraining. Enables 8x larger batch size and 3.3x faster pretraining. Part of DeepSpeed Data Efficiency Library.
- **Relevance:** MEDIUM - Production-grade curriculum learning but limited to sequence length as difficulty metric.

### 7. RUCAIBox/awesome-llm-pretraining
- **URL:** https://github.com/RUCAIBox/awesome-llm-pretraining
- **What it implements:** Curated list of LLM pre-training resources including data, frameworks, and methods.
- **Relevance:** LOW (reference list) - Useful as a secondary source for finding more papers/repos.

---

## Key Insights from Survey

1. **Curriculum learning for LLMs is an active area (2023-2025)** with approaches ranging from domain-level reweighting (DoReMi) to token-level selection (Rho-1) to skill-graph-based ordering (Skill-It).

2. **Recent findings (2025)** suggest that curriculum learning's advantage diminishes under standard LR decay schedules but can be recovered with moderate LR decay + model averaging (see arxiv 2601.21698).

3. **Multi-stage pretraining** (web data first, then high-quality data) has been adopted by production LLMs including OLMo 2, Phi-4, and LongCat-Flash -- this is effectively a coarse-grained curriculum.

4. **The three cloned repos represent complementary approaches:** skill-based ordering (Skill-It), token-level selection (Rho-1), and domain mixture optimization (DoReMi). Together they cover the main paradigms for distribution-aware pretraining.
