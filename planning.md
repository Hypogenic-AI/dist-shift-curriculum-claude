# Research Plan: Distribution Shift Curriculum

## Motivation & Novelty Assessment

### Why This Research Matters
Standard LLM pretraining uses uniform random sampling across all available data. This ignores the potential benefit of *strategic* distribution shifts that could teach models to adapt quickly to new domains — a form of meta-learning during pretraining. If successful, this approach could produce models that generalize better to unseen domains without domain-specific fine-tuning, which has major practical implications for deploying LLMs in specialized fields.

### Gap in Existing Work
Based on the literature review, curriculum learning for LLMs has been studied extensively (Bengio 2009, Fan & Jaggi 2023, Chen et al. 2023, Luo et al. 2025), but:
1. **No paper directly tests domain-withholding curricula** — existing work focuses on difficulty-based ordering or optimal static mixtures (DoReMi)
2. **Domain-level distribution shifts are understudied** — most curriculum work operates at sample level
3. **Sequential domain introduction** (our approach) is distinct from static mixture optimization and has not been systematically evaluated
4. **Adaptation speed** to withheld domains is not measured in any prior work

### Our Novel Contribution
We test whether deliberately withholding domains during pretraining and introducing them later (creating controlled distribution shifts) produces models that adapt more quickly to unseen domains compared to standard uniform training. This bridges curriculum learning and domain generalization in a way not previously studied.

### Experiment Justification
- **Experiment 1 (Baseline)**: Uniform training on all domains — establishes the standard reference
- **Experiment 2 (Distribution Shift Curriculum)**: Phased domain introduction — tests the core hypothesis
- **Experiment 3 (Domain Generalization Probe)**: Evaluate both models on fully withheld domains — measures true generalization

## Research Question
Does introducing regular distribution shifts during pretraining (by withholding and then introducing specific domains) improve a model's ability to rapidly adapt to new distributions, compared to standard uniform pretraining?

## Background and Motivation
Meta-learning principles suggest that exposing a learner to diverse tasks/distributions improves adaptation. We apply this to LLM pretraining by creating a curriculum of domain shifts. The literature (Luo et al. 2025) warns that LR schedules interact critically with curricula, so we use constant LR with EMA averaging.

## Hypothesis Decomposition
1. **H1**: A model trained with phased domain introduction will achieve lower perplexity on newly introduced domains *faster* (fewer tokens) than a uniformly trained model
2. **H2**: The curriculum model will have higher perplexity on withheld domains initially but converge to comparable or better perplexity
3. **H3**: The curriculum model will show better zero-shot transfer to completely unseen domains (domains never seen during training)

## Proposed Methodology

### Approach
Train a small GPT-2 architecture (~30M params) on streamed data from The Pile (monology/pile-uncopyrighted). Use 4x A6000 GPUs. Compare two training regimes:
- **Baseline**: Uniform random sampling from all domains
- **Curriculum**: Phase 1 trains on "core" domains only; Phase 2 introduces "shift" domains; Phase 3 evaluates on "held-out" domains never seen during training

### Domain Partitioning
- **Core domains** (always available): Pile-CC, PubMed Abstracts, StackExchange, Github, Wikipedia, USPTO Backgrounds, HackerNews, NIH ExPorter, Enron Emails (~90% of data)
- **Shift domains** (introduced in Phase 2): FreeLaw, DM Mathematics, PubMed Central
- **Held-out domains** (never trained on, for zero-shot evaluation): ArXiv

### Experimental Steps
1. Stream and tokenize ~50M tokens from The Pile, split by domain
2. Create domain-specific validation sets (5K tokens each)
3. Train baseline model for 10K steps on all domains (uniform)
4. Train curriculum model: 7K steps on core domains, then 3K steps with shift domains added
5. After training, measure adaptation: continue training both models on shift domains for 1K additional steps, measuring perplexity curves
6. Evaluate both on held-out ArXiv domain (zero-shot)

### Baselines
1. **Uniform**: Standard random sampling across all domains
2. **Core-only**: Train only on core domains (no shift domains ever) — ablation

### Evaluation Metrics
- Per-domain validation perplexity
- Adaptation speed: perplexity on shift domains as a function of tokens seen
- Zero-shot perplexity on held-out ArXiv domain
- Average perplexity across all domains

### Statistical Analysis Plan
- Run each configuration with 3 random seeds
- Report mean ± std for all metrics
- Paired t-test for key comparisons (p < 0.05)
- Plot learning curves with confidence bands

## Expected Outcomes
- **Supporting H1**: Curriculum model reaches target perplexity on shift domains in fewer adaptation tokens
- **Supporting H2**: Curriculum model converges to comparable final perplexity despite later introduction
- **Supporting H3**: Curriculum model achieves lower zero-shot perplexity on ArXiv
- **Refuting**: No difference or curriculum model is strictly worse

## Timeline and Milestones
1. Data preparation and tokenization: 15 min
2. Model implementation and training harness: 30 min
3. Baseline training (3 seeds): 30 min
4. Curriculum training (3 seeds): 30 min
5. Adaptation experiments: 20 min
6. Analysis and visualization: 20 min
7. Documentation: 20 min

## Potential Challenges
- Streaming data may be slow — mitigate by pre-downloading sufficient data
- Small model may not show clear curriculum effects — but literature shows effects even at 124M (Fan & Jaggi 2023)
- Domain imbalance in The Pile — control by capping tokens per domain

## Success Criteria
1. Clear, reproducible experiments with 3 seeds each
2. Statistically significant difference in adaptation speed (p < 0.05)
3. Comprehensive per-domain analysis showing where curriculum helps/hurts
