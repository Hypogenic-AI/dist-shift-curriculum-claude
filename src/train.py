"""Training harness for distribution shift curriculum experiments."""
import copy
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from model import GPTConfig, SmallGPT
from data_prep import CORE_DOMAINS, SHIFT_DOMAINS, HELD_OUT_DOMAINS, ALL_TRAIN_DOMAINS, SEQ_LEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "datasets/prepared"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_domain_data(domain, split="train"):
    """Load tokenized data for a specific domain."""
    domain_safe = domain.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(DATA_DIR, f"{domain_safe}_{split}.pt")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping {domain}")
        return None
    data = torch.load(path, weights_only=True)
    if data.numel() == 0:
        return None
    return data


def make_dataloader(domains, split="train", batch_size=32, shuffle=True, max_seqs_per_domain=None):
    """Create a DataLoader combining multiple domains."""
    all_inputs = []
    for domain in domains:
        data = load_domain_data(domain, split)
        if data is not None:
            if max_seqs_per_domain and data.shape[0] > max_seqs_per_domain:
                data = data[:max_seqs_per_domain]
            # Input is tokens[:-1], target is tokens[1:]
            inputs = data[:, :-1]
            targets = data[:, 1:]
            all_inputs.append((inputs, targets))

    if not all_inputs:
        return None

    all_x = torch.cat([x for x, _ in all_inputs], dim=0)
    all_y = torch.cat([y for _, y in all_inputs], dim=0)
    dataset = TensorDataset(all_x, all_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def evaluate_per_domain(model, domains, batch_size=64):
    """Evaluate model perplexity on each domain's validation set."""
    model.eval()
    results = {}
    with torch.no_grad():
        for domain in domains:
            data = load_domain_data(domain, "val")
            if data is None:
                continue
            inputs = data[:, :-1].to(DEVICE)
            targets = data[:, 1:].to(DEVICE)
            total_loss = 0.0
            count = 0
            for i in range(0, inputs.shape[0], batch_size):
                x = inputs[i : i + batch_size]
                y = targets[i : i + batch_size]
                _, loss = model(x, y)
                total_loss += loss.item() * x.shape[0]
                count += x.shape[0]
            avg_loss = total_loss / max(count, 1)
            results[domain] = {
                "loss": avg_loss,
                "perplexity": np.exp(avg_loss),
                "n_sequences": count,
            }
    model.train()
    return results


def train_model(
    model,
    train_loader,
    optimizer,
    total_steps,
    eval_domains,
    eval_every=200,
    label="",
    ema_decay=0.999,
):
    """Train model and return history of per-domain eval results."""
    model.train()
    history = []
    step = 0
    epoch = 0

    # EMA model
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    while step < total_steps:
        epoch += 1
        for batch_x, batch_y in train_loader:
            if step >= total_steps:
                break

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            _, loss = model(batch_x, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update EMA
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            step += 1

            if step % eval_every == 0 or step == total_steps:
                eval_results = evaluate_per_domain(ema_model, eval_domains)
                avg_ppl = np.mean([r["perplexity"] for r in eval_results.values()])
                eval_results["_step"] = step
                eval_results["_train_loss"] = loss.item()
                eval_results["_avg_ppl"] = avg_ppl
                history.append(eval_results)

                domains_str = ", ".join(
                    f"{d[:8]}={eval_results[d]['perplexity']:.1f}"
                    for d in sorted(eval_results.keys())
                    if not d.startswith("_")
                )
                print(f"  [{label}] Step {step}/{total_steps}: train_loss={loss.item():.3f} avg_ppl={avg_ppl:.1f} | {domains_str}")

    return history, ema_model


def run_experiment(seed=42, total_steps=3000, curriculum_switch_frac=0.6, batch_size=32, lr=3e-4):
    """Run one full experiment: baseline + curriculum."""
    set_seed(seed)
    print(f"\n{'='*80}")
    print(f"EXPERIMENT seed={seed}, steps={total_steps}, switch_frac={curriculum_switch_frac}")
    print(f"{'='*80}")

    all_eval_domains = CORE_DOMAINS + SHIFT_DOMAINS + HELD_OUT_DOMAINS
    config = GPTConfig()

    # Cap sequences per domain for balance
    max_seqs = 800

    # === BASELINE: uniform training on all train domains ===
    print(f"\n--- BASELINE (uniform all domains) ---")
    set_seed(seed)
    baseline_model = SmallGPT(config).to(DEVICE)
    baseline_opt = torch.optim.AdamW(baseline_model.parameters(), lr=lr, weight_decay=0.01)
    baseline_loader = make_dataloader(ALL_TRAIN_DOMAINS, "train", batch_size, max_seqs_per_domain=max_seqs)

    baseline_history, baseline_ema = train_model(
        baseline_model, baseline_loader, baseline_opt, total_steps,
        all_eval_domains, eval_every=200, label="baseline"
    )

    # === CURRICULUM: core domains first, then add shift domains ===
    print(f"\n--- CURRICULUM (core → core+shift) ---")
    set_seed(seed)
    curriculum_model = SmallGPT(config).to(DEVICE)
    curriculum_opt = torch.optim.AdamW(curriculum_model.parameters(), lr=lr, weight_decay=0.01)

    switch_step = int(total_steps * curriculum_switch_frac)

    # Phase 1: core domains only
    core_loader = make_dataloader(CORE_DOMAINS, "train", batch_size, max_seqs_per_domain=max_seqs)
    curriculum_history_p1, curriculum_ema = train_model(
        curriculum_model, core_loader, curriculum_opt, switch_step,
        all_eval_domains, eval_every=200, label="curriculum-p1"
    )

    # Phase 2: all train domains (core + shift)
    all_loader = make_dataloader(ALL_TRAIN_DOMAINS, "train", batch_size, max_seqs_per_domain=max_seqs)
    curriculum_history_p2, curriculum_ema = train_model(
        curriculum_model, all_loader, curriculum_opt, total_steps - switch_step,
        all_eval_domains, eval_every=200, label="curriculum-p2"
    )

    # Fix step numbers for phase 2
    for entry in curriculum_history_p2:
        entry["_step"] += switch_step

    curriculum_history = curriculum_history_p1 + curriculum_history_p2

    # === ADAPTATION EXPERIMENT ===
    # Continue training both models on SHIFT domains only for 500 more steps
    print(f"\n--- ADAPTATION (shift domains only, 500 steps) ---")
    shift_loader = make_dataloader(SHIFT_DOMAINS, "train", batch_size, max_seqs_per_domain=max_seqs)
    if shift_loader is not None:
        # Baseline adaptation
        baseline_adapt_opt = torch.optim.AdamW(baseline_ema.parameters(), lr=lr * 0.5, weight_decay=0.01)
        for p in baseline_ema.parameters():
            p.requires_grad_(True)
        baseline_adapt_history, _ = train_model(
            baseline_ema, shift_loader, baseline_adapt_opt, 500,
            SHIFT_DOMAINS + HELD_OUT_DOMAINS, eval_every=100, label="baseline-adapt"
        )

        # Curriculum adaptation
        curriculum_adapt_opt = torch.optim.AdamW(curriculum_ema.parameters(), lr=lr * 0.5, weight_decay=0.01)
        for p in curriculum_ema.parameters():
            p.requires_grad_(True)
        curriculum_adapt_history, _ = train_model(
            curriculum_ema, shift_loader, curriculum_adapt_opt, 500,
            SHIFT_DOMAINS + HELD_OUT_DOMAINS, eval_every=100, label="curriculum-adapt"
        )
    else:
        baseline_adapt_history = []
        curriculum_adapt_history = []

    return {
        "seed": seed,
        "baseline_history": baseline_history,
        "curriculum_history": curriculum_history,
        "baseline_adapt_history": baseline_adapt_history,
        "curriculum_adapt_history": curriculum_adapt_history,
        "config": {
            "total_steps": total_steps,
            "curriculum_switch_frac": curriculum_switch_frac,
            "batch_size": batch_size,
            "lr": lr,
            "model_config": vars(config),
        },
    }


def main():
    import sys
    os.chdir("/workspaces/dist-shift-curriculum-claude")

    # Check GPU
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check data exists
    if not os.path.exists(DATA_DIR):
        print("ERROR: Prepared data not found. Run data_prep.py first.")
        sys.exit(1)

    seeds = [42, 123, 456]
    all_results = []

    for seed in seeds:
        result = run_experiment(
            seed=seed,
            total_steps=3000,
            curriculum_switch_frac=0.6,
            batch_size=32,
            lr=3e-4,
        )
        all_results.append(result)

    # Save results
    os.makedirs("results", exist_ok=True)
    # Convert numpy/tensor values to python types for JSON
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open("results/experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=to_serializable)

    print(f"\nResults saved to results/experiment_results.json")
    return all_results


if __name__ == "__main__":
    main()
