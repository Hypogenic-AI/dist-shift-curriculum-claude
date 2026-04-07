# Literature Review: Distribution Shift Curriculum for LLM Pretraining

## Research Area Overview

This review covers the intersection of **curriculum learning for language model pretraining** and **distribution shift robustness**. The central research question is whether deliberately introducing distribution shifts during pretraining—by withholding and then presenting certain data domains—can encourage models to generalize more effectively to unseen distributions.

Two largely separate literatures converge here: (1) curriculum learning, which studies how training data ordering affects learning outcomes, and (2) domain generalization/distributionally robust optimization, which studies how to train models that perform well under distribution shift. Recent work (2023–2026) has begun bridging these areas, with several papers demonstrating that strategic data ordering during LLM pretraining can yield significant efficiency and quality gains.

---

## Key Papers

### Foundational: Curriculum Learning

#### Bengio et al. (2009) — "Curriculum Learning"
- **Source:** ICML 2009 (6800+ citations)
- **Key Contribution:** Formalized curriculum learning: training on progressively harder examples improves convergence and generalization. Demonstrated on vision (shape recognition) and NLP (language modeling) tasks.
- **Methodology:** Define difficulty metrics, sort training data easy-to-hard, gradually increase the proportion of hard examples.
- **Relevance:** Foundational framework. However, subsequent work shows that simple difficulty-based orderings yield mixed results for modern LLMs.

---

### Curriculum Learning for LLM Pretraining

#### Campos et al. (2021) — "On the Role of Corpus Ordering in Language Modeling"
- **arXiv:** 2104.06127
- **Key Finding:** Competence-Based Curriculum with 8 difficulty proxies (sentence length, n-gram entropy, parse depth) produced **no compelling evidence** of benefit for LM pretraining (ELMo). Random curricula performed comparably to linguistically motivated ones. Benefits disappeared with larger corpora.
- **Code:** https://github.com/spacemanidol/CurriculumLearningForLanguageModels
- **Relevance:** Important negative result—naive difficulty-based curricula do not help LM pretraining under standard settings.

#### Fan & Jaggi (2023) — "Irreducible Curriculum for Language Model Pretraining"
- **arXiv:** 2310.15389
- **Key Contribution:** Uses a small proxy model (82M params) to compute per-sample "learnability" scores (loss gap between early and late proxy checkpoints). Training starts with the most learnable samples and gradually includes all data.
- **Datasets:** RedPajama-1B (7 domains)
- **Key Results:** Consistently lower perplexity across all 7 domains vs random baseline. MMLU 5-shot: 26.9% vs 22.9% baseline. Reduces network sharpness (flatter minima). **Critical finding:** Global curriculum (ranking across domains) hurts some domains because learnability scores are not comparable across domains—must apply intra-domain.
- **Relevance:** Directly demonstrates that controlled distribution shifts (biased→full) during pretraining improve generalization. The intra-domain requirement is a key design constraint for our hypothesis.

#### Chen et al. (2023) — "Skill-it! A Data-Driven Skills Framework"
- **arXiv:** 2307.14430 (NeurIPS 2023 Spotlight)
- **Key Contribution:** Defines "skills" as learnable data units with prerequisite dependencies. Constructs a directed skills graph and uses online mirror descent (SKILL-IT) to dynamically adjust sampling proportions during training.
- **Datasets:** LEGO synthetic, Natural Instructions, RedPajama (3B model)
- **Key Results:** 36.5 points higher accuracy than random on LEGO. On RedPajama, 1B tokens with SKILL-IT matches 3B tokens with uniform sampling (**3x data efficiency**). Skills graphs transfer across model scales (125M→1.3B).
- **Relevance:** Demonstrates that **prerequisite structure matters more than easy-to-hard ordering**. Dynamic distribution shifts guided by skill dependencies are far more effective than static curricula.

#### Luo et al. (2025) — "How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining"
- **arXiv:** 2511.18903
- **Key Contribution:** Identifies a fundamental conflict: standard LR decay (cosine, WSD) reduces learning rate near zero at the end of training, exactly when quality-ascending curricula present the best data. Proposes Curriculum Model Averaging (CMA): constant LR + EMA over final checkpoints.
- **Datasets:** DCLM-Baseline (30B tokens), Qwen2.5-1.5B architecture
- **Key Results:** Curriculum with constant LR substantially outperforms random, but advantage disappears with standard LR decay. CMA yields +1.64% average benchmark improvement. **Prior negative results about curriculum learning were artifacts of LR schedule choice.**
- **Relevance:** **Critical methodological insight for our experiments.** Any distribution shift curriculum must co-design the learning rate schedule with the data ordering. Model averaging is a key enabler.

#### Kim & Lee (2024) — "Strategic Data Ordering: Enhancing LLM Performance through Curriculum Learning"
- **arXiv:** 2405.07490
- **Key Results:** Applied curriculum learning to instruction tuning (Mistral-7B, Gemma-7B) with 3 difficulty criteria (prompt length, attention score, loss). Results are **mixed and modest** (0.1–0.6 pp improvements). The paper itself characterizes improvements as "not significantly impactful."
- **Relevance:** Suggests that for fine-tuning (vs pretraining), curriculum effects are smaller. Our hypothesis focuses on pretraining, where effects appear larger.

#### Xie et al. (2023) — "DoReMi: Optimizing Data Mixtures Speeds Up LM Pretraining"
- **NeurIPS 2023**
- **Key Contribution:** Uses a small proxy model trained with Group DRO to find optimal domain mixture weights. A 280M proxy sets weights for an 8B model.
- **Key Results:** +6.5% average few-shot accuracy, 2.6x faster to reach baseline accuracy.
- **Code:** https://github.com/sangmichaelxie/doremi
- **Relevance:** Domain mixture optimization is a form of static distribution control during pretraining. Complements curriculum approaches that change the mixture over time.

#### Lin et al. (2024) — "Rho-1: Not All Tokens Are What You Need"
- **NeurIPS 2024**
- **Key Contribution:** Token-level selective pretraining using excess loss against a reference model. Trains only on tokens with highest excess loss.
- **Key Results:** SOTA on MATH with only 3% of pretraining tokens. +6.8% average improvement across 15 tasks on 80B general tokens.
- **Code:** https://github.com/microsoft/rho
- **Relevance:** Fine-grained data selection is an extreme form of distribution shift—the model sees a very different distribution than the raw data.

#### Other Curriculum-for-LLM Papers (Reviewed Abstracts)
- **Campos (2021)** — BERT curriculum with increasing block size. Minor perplexity improvements.
- **Choe (2022)** — Length-based curriculum for LM pretraining. Small but consistent gains.
- **DeepSpeed (2022)** — Sequence-length curriculum enables 8x larger batch size.
- **BabyLM Challenge (2023)** — Sample-efficient pretraining on developmentally plausible corpora. Curriculum approaches competitive.
- **CLIMB (2023)** — Infant-inspired curriculum for model building.
- **Code-Switching Curriculum (2024)** — Multilingual transfer via code-switching curriculum.
- **Vocabulary Curriculum (2024)** — Scaling LLM pretraining by gradually expanding vocabulary.
- **Preference Curriculum (2024)** — Pretrain on model's preferred (lower-loss) data first.

---

### Distribution Shift Robustness

#### Arjovsky et al. (2019) — "Invariant Risk Minimization (IRM)"
- **arXiv:** 1907.02893 (2600+ citations)
- **Key Contribution:** Learns representations where the optimal classifier is invariant across training environments. Penalizes environment-specific feature reliance.
- **Code:** https://github.com/facebookresearch/InvariantRiskMinimization
- **Relevance:** Provides theoretical framework for invariant features. Our hypothesis implicitly assumes that distribution shifts encourage learning invariant representations.

#### Sagawa et al. (2019) — "Distributionally Robust Neural Networks for Group Shifts"
- **ICLR 2020** (1500+ citations)
- **Key Contribution:** Group DRO minimizes worst-group loss. Critical finding: DRO only works with strong regularization (large L2 penalty or early stopping) in overparameterized networks.
- **Datasets:** Waterbirds, CelebA, MultiNLI
- **Code:** https://github.com/kohpangwei/group_DRO
- **Relevance:** Demonstrates that worst-group robustness requires explicit optimization + regularization. Relevant to ensuring our curriculum doesn't sacrifice minority domain performance.

#### Krueger et al. (2021) — "Risk Extrapolation (REx)"
- **arXiv:** 2003.00688 (1100+ citations)
- **Key Contribution:** Penalizes variance of per-domain risks. Unlike IRM, also provides robustness to covariate shift. Proves that equalizing risks across environments can recover causal mechanisms.
- **Relevance:** V-REx (variance penalty) could be combined with our curriculum: penalize risk variance across domains at each training phase to encourage invariant learning during distribution shifts.

#### Piratla et al. (2022) — "Focus on the Common Good" (CGD)
- **ICLR 2022**
- **Key Contribution:** Selects training groups whose gradients most decrease average loss across ALL groups (not just worst group). Avoids Group DRO's failure mode with noisy groups.
- **Code:** https://github.com/vihari/CGD
- **Relevance:** "Common gradient" selection could inform which domain to introduce next in a distribution shift curriculum—introduce domains whose gradients are most aligned with existing knowledge.

#### Yi et al. (2022) — "Distribution Shift Robustness From the Perspective of Pre-Training"
- **arXiv:** 2205.12392
- **Key Finding:** Pre-training on diverse data provides substantial distribution shift robustness, often more than specialized OOD methods applied during fine-tuning. Data augmentation during pre-training further helps.
- **Relevance:** Directly supports our hypothesis that pretraining-phase interventions (including distribution shifts) can improve OOD robustness.

#### Koh et al. (2021) — "WILDS: A Benchmark of in-the-Wild Distribution Shifts"
- **arXiv:** 2012.07421 (1700+ citations)
- **Key Contribution:** Benchmark of 10 datasets with real-world distribution shifts across domains, subpopulations, and time.
- **Code:** https://github.com/p-lambda/wilds
- **Relevance:** Provides evaluation methodology for distribution shift robustness. Could be used for downstream evaluation of pretrained models.

---

## Common Methodologies

### Data Selection/Ordering Approaches
1. **Difficulty-based:** Sort by difficulty metrics (length, perplexity, entropy). Mixed results for LLMs.
2. **Learnability-based:** Use proxy model to score sample learnability. More effective (Irreducible Curriculum).
3. **Skill-graph-based:** Model prerequisite dependencies between data subsets. Most effective (Skill-it, 3x efficiency).
4. **Domain mixture optimization:** Find optimal static domain weights (DoReMi) or dynamic reweighting.
5. **Token-level selection:** Select individual tokens based on excess loss (Rho-1). Most fine-grained.

### Robustness Approaches
1. **ERM + diverse pretraining:** Simple but effective baseline.
2. **Group DRO:** Minimize worst-group loss with strong regularization.
3. **IRM/REx:** Penalize environment-specific features or risk variance.
4. **Common gradient descent:** Select training data whose gradients benefit all groups.

---

## Standard Baselines for Experiments

1. **Uniform random sampling** — The default pretraining approach
2. **Domain-proportional sampling** — Sample proportional to domain sizes
3. **DoReMi mixture** — Optimized static domain weights
4. **Easy-to-hard curriculum** — Traditional curriculum learning
5. **Anti-curriculum** — Hard-to-easy ordering (surprisingly competitive on some benchmarks)

---

## Evaluation Metrics

- **Perplexity** on held-out validation sets (per-domain and average)
- **MMLU** (5-shot) — General knowledge and reasoning
- **LM Evaluation Harness** benchmarks (ARC, HellaSwag, WinoGrande, etc.)
- **Per-domain performance** on withheld domains — Key metric for our hypothesis
- **Worst-domain performance** — Measures robustness
- **Data efficiency** — Tokens needed to reach a performance threshold

---

## Datasets in the Literature

| Dataset | Used By | Domains | Size |
|---------|---------|---------|------|
| RedPajama-1B | Irreducible Curriculum, Skill-it | 7 | ~1B tokens |
| The Pile | Multiple | 22 | 825 GB |
| SlimPajama-6B | Available replacement | 7 | 14 GB |
| DCLM-Baseline | LR Decay paper | Web crawl | 30B+ tokens |
| Natural Instructions | Skill-it | 23+ task categories | Varies |
| WILDS | Distribution shift benchmark | 10 datasets | Varies |

---

## Gaps and Opportunities

1. **No paper directly tests our hypothesis:** No existing work deliberately withholds specific domains during pretraining and then evaluates rapid adaptation to those domains. This is a genuine research gap.

2. **Domain-level vs. sample-level curriculum:** Most curriculum work operates at the sample level (difficulty scoring). Domain-level distribution shifts (our approach) are understudied.

3. **Interaction with optimization:** The LR decay paper shows that curriculum benefits are highly sensitive to optimization choices. This must be accounted for in experimental design.

4. **Scale dependence:** Most curriculum learning results are at small scale (<3B params, <30B tokens). Whether effects persist at larger scale is unknown.

5. **Temporal/sequential domain introduction:** Our hypothesis involves sequential introduction of withheld domains, which is distinct from static mixture optimization (DoReMi) or gradual difficulty increase. This sequential shift has not been systematically studied.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **Primary:** `monology/pile-uncopyrighted` — 17 labeled domains, well-studied, supports streaming. Withhold domains like FreeLaw, DM Mathematics, EuroParl.
2. **Alternative:** `DKYoon/SlimPajama-6B` — 7 domains, 14 GB, faster iteration.

### Recommended Baselines
1. Uniform random sampling (standard pretraining)
2. DoReMi-style optimized static mixture
3. Irreducible curriculum (learnability-based ordering)
4. Anti-curriculum (to test if any ordering helps)

### Recommended Metrics
1. Per-domain validation perplexity (especially on withheld domains)
2. Withheld-domain adaptation speed (few-shot perplexity after domain introduction)
3. Average perplexity across all domains
4. Downstream benchmarks (MMLU, ARC) if compute allows

### Methodological Considerations
1. **Use constant LR or moderate LR decay** — Standard cosine decay will mask curriculum effects (Luo et al. 2025).
2. **Apply curriculum intra-domain** if using learnability scoring — Global scoring creates domain imbalance (Fan & Jaggi 2023).
3. **Consider model averaging** (EMA over checkpoints) as a complementary technique.
4. **Use a small proxy model** (10-50M params) to score data and set curriculum, then apply to a larger model. Proxy-based approaches are computationally cheap and transfer across scales.
5. **Measure per-domain performance separately** — Average metrics can hide domain-level regressions.
