"""Analysis and visualization for distribution shift curriculum experiments."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FIGURES_DIR = "figures"
RESULTS_DIR = "results"


def load_results(path="results/experiment_results.json"):
    with open(path) as f:
        return json.load(f)


def extract_final_perplexity(history, domains):
    """Get final perplexity for each domain from a training history."""
    if not history:
        return {}
    final = history[-1]
    return {d: final[d]["perplexity"] for d in domains if d in final}


def plot_training_curves(all_results, save_dir=FIGURES_DIR):
    """Plot per-domain perplexity curves for baseline vs curriculum."""
    os.makedirs(save_dir, exist_ok=True)

    from data_prep import CORE_DOMAINS, SHIFT_DOMAINS, HELD_OUT_DOMAINS

    domain_groups = {
        "Core Domains": CORE_DOMAINS,
        "Shift Domains": SHIFT_DOMAINS,
        "Held-Out Domains": HELD_OUT_DOMAINS,
    }

    for group_name, domains in domain_groups.items():
        available_domains = []
        for d in domains:
            # Check if any result has this domain
            for r in all_results:
                if r["baseline_history"] and d in r["baseline_history"][-1]:
                    available_domains.append(d)
                    break

        if not available_domains:
            continue

        n_domains = len(available_domains)
        fig, axes = plt.subplots(1, n_domains, figsize=(5 * n_domains, 4), squeeze=False)

        for j, domain in enumerate(available_domains):
            ax = axes[0, j]

            for method, key, color, label in [
                ("baseline", "baseline_history", "tab:blue", "Baseline (uniform)"),
                ("curriculum", "curriculum_history", "tab:orange", "Curriculum (shift)"),
            ]:
                all_steps = []
                all_ppls = []
                for r in all_results:
                    steps = [h["_step"] for h in r[key] if domain in h]
                    ppls = [h[domain]["perplexity"] for h in r[key] if domain in h]
                    if steps:
                        all_steps.append(steps)
                        all_ppls.append(ppls)

                if not all_steps:
                    continue

                # Average across seeds
                min_len = min(len(s) for s in all_steps)
                steps_arr = np.array([s[:min_len] for s in all_steps])
                ppls_arr = np.array([p[:min_len] for p in all_ppls])

                mean_steps = steps_arr.mean(axis=0)
                mean_ppls = ppls_arr.mean(axis=0)
                std_ppls = ppls_arr.std(axis=0)

                ax.plot(mean_steps, mean_ppls, color=color, label=label, linewidth=2)
                ax.fill_between(
                    mean_steps,
                    mean_ppls - std_ppls,
                    mean_ppls + std_ppls,
                    alpha=0.2,
                    color=color,
                )

            ax.set_title(domain, fontsize=11)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Perplexity")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Perplexity During Training: {group_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        safe_name = group_name.lower().replace(" ", "_")
        fig.savefig(os.path.join(save_dir, f"training_curves_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved training_curves_{safe_name}.png")


def plot_adaptation_curves(all_results, save_dir=FIGURES_DIR):
    """Plot adaptation curves on shift domains."""
    os.makedirs(save_dir, exist_ok=True)
    from data_prep import SHIFT_DOMAINS, HELD_OUT_DOMAINS

    domains_to_plot = SHIFT_DOMAINS + HELD_OUT_DOMAINS
    available = []
    for d in domains_to_plot:
        for r in all_results:
            if r["baseline_adapt_history"] and d in r["baseline_adapt_history"][-1]:
                available.append(d)
                break

    if not available:
        print("No adaptation data to plot")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for j, domain in enumerate(available):
        ax = axes[0, j]

        for method, key, color, label in [
            ("baseline", "baseline_adapt_history", "tab:blue", "Baseline → Adapt"),
            ("curriculum", "curriculum_adapt_history", "tab:orange", "Curriculum → Adapt"),
        ]:
            all_steps = []
            all_ppls = []
            for r in all_results:
                steps = [h["_step"] for h in r[key] if domain in h]
                ppls = [h[domain]["perplexity"] for h in r[key] if domain in h]
                if steps:
                    all_steps.append(steps)
                    all_ppls.append(ppls)

            if not all_steps:
                continue

            min_len = min(len(s) for s in all_steps)
            steps_arr = np.array([s[:min_len] for s in all_steps])
            ppls_arr = np.array([p[:min_len] for p in all_ppls])

            mean_steps = steps_arr.mean(axis=0)
            mean_ppls = ppls_arr.mean(axis=0)
            std_ppls = ppls_arr.std(axis=0)

            ax.plot(mean_steps, mean_ppls, color=color, label=label, linewidth=2)
            ax.fill_between(mean_steps, mean_ppls - std_ppls, mean_ppls + std_ppls, alpha=0.2, color=color)

        ax.set_title(domain, fontsize=11)
        ax.set_xlabel("Adaptation Step")
        ax.set_ylabel("Perplexity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Adaptation to Shift/Held-Out Domains", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "adaptation_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved adaptation_curves.png")


def plot_final_comparison(all_results, save_dir=FIGURES_DIR):
    """Bar chart comparing final perplexity across domains."""
    os.makedirs(save_dir, exist_ok=True)
    from data_prep import CORE_DOMAINS, SHIFT_DOMAINS, HELD_OUT_DOMAINS

    all_domains = CORE_DOMAINS + SHIFT_DOMAINS + HELD_OUT_DOMAINS

    baseline_ppls = defaultdict(list)
    curriculum_ppls = defaultdict(list)

    for r in all_results:
        b_final = extract_final_perplexity(r["baseline_history"], all_domains)
        c_final = extract_final_perplexity(r["curriculum_history"], all_domains)
        for d in all_domains:
            if d in b_final:
                baseline_ppls[d].append(b_final[d])
            if d in c_final:
                curriculum_ppls[d].append(c_final[d])

    domains_with_data = [d for d in all_domains if d in baseline_ppls and d in curriculum_ppls]
    if not domains_with_data:
        print("No data for final comparison")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(domains_with_data) * 1.2), 6))

    x = np.arange(len(domains_with_data))
    width = 0.35

    b_means = [np.mean(baseline_ppls[d]) for d in domains_with_data]
    b_stds = [np.std(baseline_ppls[d]) for d in domains_with_data]
    c_means = [np.mean(curriculum_ppls[d]) for d in domains_with_data]
    c_stds = [np.std(curriculum_ppls[d]) for d in domains_with_data]

    bars1 = ax.bar(x - width / 2, b_means, width, yerr=b_stds, label="Baseline", color="tab:blue", alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width / 2, c_means, width, yerr=c_stds, label="Curriculum", color="tab:orange", alpha=0.8, capsize=3)

    # Color domain labels by group
    colors = []
    for d in domains_with_data:
        if d in SHIFT_DOMAINS:
            colors.append("red")
        elif d in HELD_OUT_DOMAINS:
            colors.append("purple")
        else:
            colors.append("black")

    ax.set_xticks(x)
    ax.set_xticklabels([d[:15] for d in domains_with_data], rotation=45, ha="right", fontsize=9)
    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)

    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Final Perplexity by Domain\n(Red=Shift domains, Purple=Held-out)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "final_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved final_comparison.png")


def compute_statistics(all_results):
    """Compute statistical comparisons between baseline and curriculum."""
    from data_prep import CORE_DOMAINS, SHIFT_DOMAINS, HELD_OUT_DOMAINS
    all_domains = CORE_DOMAINS + SHIFT_DOMAINS + HELD_OUT_DOMAINS

    stats_results = {}

    for domain in all_domains:
        b_ppls = []
        c_ppls = []
        for r in all_results:
            b_final = extract_final_perplexity(r["baseline_history"], [domain])
            c_final = extract_final_perplexity(r["curriculum_history"], [domain])
            if domain in b_final:
                b_ppls.append(b_final[domain])
            if domain in c_final:
                c_ppls.append(c_final[domain])

        if len(b_ppls) >= 2 and len(c_ppls) >= 2:
            t_stat, p_val = stats.ttest_rel(b_ppls[:min(len(b_ppls), len(c_ppls))],
                                             c_ppls[:min(len(b_ppls), len(c_ppls))])
            effect = (np.mean(b_ppls) - np.mean(c_ppls)) / np.sqrt(
                (np.std(b_ppls) ** 2 + np.std(c_ppls) ** 2) / 2
            ) if (np.std(b_ppls) + np.std(c_ppls)) > 0 else 0

            stats_results[domain] = {
                "baseline_mean": float(np.mean(b_ppls)),
                "baseline_std": float(np.std(b_ppls)),
                "curriculum_mean": float(np.mean(c_ppls)),
                "curriculum_std": float(np.std(c_ppls)),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(effect),
                "curriculum_better": np.mean(c_ppls) < np.mean(b_ppls),
            }

    # Adaptation speed comparison
    adapt_stats = {}
    for domain_group_name, domains in [("shift", SHIFT_DOMAINS), ("held_out", HELD_OUT_DOMAINS)]:
        b_init_ppls = []
        c_init_ppls = []
        b_final_ppls = []
        c_final_ppls = []

        for r in all_results:
            for key_init, key_final, init_list, final_list in [
                ("baseline_adapt_history", "baseline_adapt_history", b_init_ppls, b_final_ppls),
                ("curriculum_adapt_history", "curriculum_adapt_history", c_init_ppls, c_final_ppls),
            ]:
                hist = r[key_init]
                if hist:
                    init_ppls = [hist[0].get(d, {}).get("perplexity", float("nan")) for d in domains if d in hist[0]]
                    final_ppls = [hist[-1].get(d, {}).get("perplexity", float("nan")) for d in domains if d in hist[-1]]
                    if init_ppls:
                        init_list.append(np.nanmean(init_ppls))
                    if final_ppls:
                        final_list.append(np.nanmean(final_ppls))

        if b_init_ppls and c_init_ppls:
            adapt_stats[domain_group_name] = {
                "baseline_init_ppl": float(np.mean(b_init_ppls)),
                "curriculum_init_ppl": float(np.mean(c_init_ppls)),
                "baseline_final_ppl": float(np.mean(b_final_ppls)) if b_final_ppls else None,
                "curriculum_final_ppl": float(np.mean(c_final_ppls)) if c_final_ppls else None,
                "baseline_improvement": float(np.mean(b_init_ppls) - np.mean(b_final_ppls)) if b_final_ppls else None,
                "curriculum_improvement": float(np.mean(c_init_ppls) - np.mean(c_final_ppls)) if c_final_ppls else None,
            }

    return {"per_domain": stats_results, "adaptation": adapt_stats}


def generate_summary_table(stats_results):
    """Generate a markdown table of results."""
    lines = []
    lines.append("| Domain | Group | Baseline PPL | Curriculum PPL | Diff | p-value | Cohen's d | Better? |")
    lines.append("|--------|-------|-------------|----------------|------|---------|-----------|---------|")

    from data_prep import CORE_DOMAINS, SHIFT_DOMAINS, HELD_OUT_DOMAINS

    for domain, r in sorted(stats_results["per_domain"].items()):
        if domain in SHIFT_DOMAINS:
            group = "Shift"
        elif domain in HELD_OUT_DOMAINS:
            group = "Held-out"
        else:
            group = "Core"

        diff = r["baseline_mean"] - r["curriculum_mean"]
        sig = "*" if r["p_value"] < 0.05 else ""
        better = "Curriculum" if r["curriculum_better"] else "Baseline"
        lines.append(
            f"| {domain[:20]} | {group} | {r['baseline_mean']:.1f} +/- {r['baseline_std']:.1f} | "
            f"{r['curriculum_mean']:.1f} +/- {r['curriculum_std']:.1f} | {diff:+.1f} | "
            f"{r['p_value']:.3f}{sig} | {r['cohens_d']:.2f} | {better} |"
        )

    return "\n".join(lines)


def main():
    os.chdir("/workspaces/dist-shift-curriculum-claude")
    results = load_results()

    print("Generating visualizations...")
    plot_training_curves(results)
    plot_final_comparison(results)
    plot_adaptation_curves(results)

    print("\nComputing statistics...")
    stats_results = compute_statistics(results)

    def make_serializable(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open("results/statistics.json", "w") as f:
        json.dump(stats_results, f, indent=2, default=make_serializable)

    table = generate_summary_table(stats_results)
    print("\n" + table)

    with open("results/summary_table.md", "w") as f:
        f.write(table)

    print("\nAnalysis complete. Results in results/ and figures/")
    return stats_results


if __name__ == "__main__":
    main()
