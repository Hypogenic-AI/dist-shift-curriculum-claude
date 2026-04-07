# Distribution Shift Curriculum

**Can deliberate distribution shifts during pretraining improve a model's ability to rapidly adapt to new domains?**

## Key Findings

- **2.8x faster adaptation**: Models trained with domain-withholding curriculum adapt to newly introduced domains nearly 3x faster than uniformly trained models (9.1 vs 3.3 PPL improvement in 500 steps)
- **Unexpected core-domain improvement**: The curriculum model significantly outperforms the baseline on 5/9 core domains (p < 0.05), suggesting that concentrated early training builds better representations
- **Trade-off on shift domains**: The curriculum model finishes with ~20% higher perplexity on late-introduced domains, not fully catching up within the training budget
- **No zero-shot advantage**: No significant difference on completely held-out domains (ArXiv)
- **Less catastrophic forgetting**: During adaptation, the curriculum model shows 26% less degradation on held-out domains

## Experiment Design

A 30M-parameter GPT-2 model trained on 13 domains from The Pile:
- **Baseline**: 3,000 steps on all 12 train domains (uniform)
- **Curriculum**: 1,800 steps on 9 core domains, then 1,200 steps on all 12 domains
- **Adaptation**: Both models fine-tuned for 500 additional steps on shift domains
- **3 random seeds** per condition, constant learning rate with EMA averaging

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch==2.4.0 transformers datasets tokenizers numpy matplotlib scipy tqdm

# Prepare data (streams from The Pile)
cd src && python data_prep.py

# Run experiments (3 seeds x 2 conditions)
python train.py

# Analyze and visualize
python analyze.py
```

Requires ~30 min on a single NVIDIA GPU. Results saved to `results/` and `figures/`.

## File Structure

```
├── REPORT.md              # Full research report with analysis
├── README.md              # This file
├── planning.md            # Research plan and experimental design
├── literature_review.md   # Comprehensive literature review
├── resources.md           # Catalog of datasets, papers, code
├── src/
│   ├── data_prep.py       # Data streaming, tokenization, domain splitting
│   ├── model.py           # 30M GPT-2 architecture
│   ├── train.py           # Training harness (baseline + curriculum + adaptation)
│   └── analyze.py         # Statistical analysis and visualization
├── results/
│   ├── experiment_results.json  # Raw experiment data (3 seeds)
│   ├── statistics.json          # Per-domain statistical comparisons
│   └── summary_table.md        # Results table in markdown
├── figures/
│   ├── final_comparison.png           # Bar chart of final perplexity by domain
│   ├── training_curves_core_domains.png
│   ├── training_curves_shift_domains.png
│   ├── training_curves_held-out_domains.png
│   └── adaptation_curves.png          # Adaptation speed comparison
├── datasets/              # Downloaded/streamed data
├── papers/                # 29 research papers (PDFs)
└── code/                  # Cloned repos (DoReMi, Skill-it, Rho-1)
```

See [REPORT.md](REPORT.md) for full methodology, results, and discussion.
