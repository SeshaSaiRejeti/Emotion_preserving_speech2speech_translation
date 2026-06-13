"""
exp3_comparison.py — Experiment 3: Strategy Comparison
=======================================================

PURPOSE
-------
Empirically compare two translation optimization strategies:

  Strategy A — Iterative Regeneration  (3 LLM calls per sentence)
  Strategy B — Candidate Generation    (3 LLM calls per sentence, equal budget)

Both compared against a no-optimization baseline (1 call per sentence).

DESIGN
------
- Equal LLM call budget: MAX_ATTEMPTS=3 vs N_CANDIDATES=3
- Stratified sample: equal emotion class representation
- Soft reference distributions from classifier (not one-hot)
- Per-row checkpointing: crash-safe, resumable

STATISTICAL TESTING
-------------------
Wilcoxon signed-rank test (paired) on D_JS between strategies.

SAMPLE     : 200 sentences, stratified, seed=42
OUTPUT     : results/exp3_raw.json, results/exp3_summary.json
CHECKPOINT : results/exp3_checkpoint.json  (delete before fresh run)
"""

import os
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
from collections import defaultdict

from shared import (
    RESULTS_DIR, EMOTION_ORDER, DriftMetric,
    get_emotion_dist, get_reference_dist, sem_sim, penalty_full,
    generate_paraphrase, run_iterative, run_candidate,
    aggregate, save_json, load_json, print_agg_table,
)

# ================================================================
# CONFIG
# ================================================================

SAMPLE_SIZE        = 200
RANDOM_SEED        = 42
RATE_DELAY         = 0.5

MAX_ATTEMPTS       = 3    # Strategy A: 1 initial + 2 feedback = 3 LLM calls
ITERATIVE_TEMP     = 0.1

N_CANDIDATES       = 3   # Strategy B: 3 independent calls  (equal budget)
CANDIDATE_TEMP     = 0.5   # must be > 0.1 for meaningful diversity

# ================================================================
# CHECKPOINT
# ================================================================

CHECKPOINT = str(RESULTS_DIR / "exp3_checkpoint.json")


def save_checkpoint(baseline, iterative, candidate):
    save_json(CHECKPOINT, {
        "baseline":  baseline,
        "iterative": iterative,
        "candidate": candidate,
    })


def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        print("[Exp 3] Checkpoint found — resuming ...")
        data = load_json(CHECKPOINT)
        b = data.get("baseline",  [])
        i = data.get("iterative", [])
        c = data.get("candidate", [])
        print(f"  baseline={len(b)}  iterative={len(i)}  candidate={len(c)}")
        return b, i, c
    return [], [], []

# ================================================================
# DATA — STRATIFIED SAMPLE
# ================================================================

df_full = pd.read_csv("golden_goemotions_7class.csv")


def stratified_sample(df, n, seed):
    classes   = df["label_7"].unique()
    per_class = max(1, n // len(classes))
    rng       = np.random.default_rng(seed)
    parts     = []
    for cls in classes:
        pool = df[df["label_7"] == cls]
        k    = min(per_class, len(pool))
        parts.append(pool.sample(k, random_state=int(rng.integers(0, 9999))))
    result = pd.concat(parts)
    if len(result) < n:
        remaining = df[~df.index.isin(result.index)]
        extra     = remaining.sample(
            min(n - len(result), len(remaining)), random_state=seed
        )
        result = pd.concat([result, extra])
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


df = stratified_sample(df_full, SAMPLE_SIZE, RANDOM_SEED)

print(f"[Exp 3] {len(df)} sentences (stratified)")
print(df["label_7"].value_counts().to_string())
print()

metric = DriftMetric()

# ================================================================
# STATISTICAL TESTS
# ================================================================

def wilcoxon_test(rows_a, rows_b, metric_key):
    vals_a = np.array([r[metric_key] for r in rows_a])
    vals_b = np.array([r[metric_key] for r in rows_b])
    stat, p = stats.wilcoxon(vals_a, vals_b, alternative="two-sided")
    return {
        "metric":           metric_key,
        "mean_A":           float(np.mean(vals_a)),
        "mean_B":           float(np.mean(vals_b)),
        "diff_B_minus_A":   float(np.mean(vals_b) - np.mean(vals_a)),
        "statistic":        float(stat),
        "p_value":          float(p),
        "significant":      bool(p < 0.05),
        "better":           "B" if np.mean(vals_b) < np.mean(vals_a) else "A",
    }

# ================================================================
# PER-EMOTION BREAKDOWN
# ================================================================

def per_emotion_breakdown(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["label"]].append(r)
    result = {}
    for emotion, sub in sorted(buckets.items()):
        result[emotion] = {
            "n":           len(sub),
            "flip_rate":   float(np.mean([r["D_flip"]   for r in sub])),
            "mean_D_JS":   float(np.mean([r["D_JS"]     for r in sub])),
            "mean_sem":    float(np.mean([r["semantic"]  for r in sub])),
            "catast_rate": float(np.mean(
                [(r["semantic"] < 0.30) or (r["D_JS"] > 0.50) for r in sub]
            )),
        }
    return result

# ================================================================
# FAILURE ANALYSIS
# ================================================================

def failure_analysis(rows, condition_name):
    failures = [r for r in rows if r["semantic"] < 0.30 or r["D_JS"] > 0.50]
    sem_only = sum(1 for r in rows if r["semantic"] < 0.30 and r["D_JS"] <= 0.50)
    emo_only = sum(1 for r in rows if r["semantic"] >= 0.30 and r["D_JS"] > 0.50)
    both     = sum(1 for r in rows if r["semantic"] < 0.30 and r["D_JS"] > 0.50)
    emo_counts = defaultdict(int)
    for r in failures:
        emo_counts[r["label"]] += 1
    print(f"\n[Failure Analysis — {condition_name}]")
    print(f"  Total catastrophic : {len(failures)}/{len(rows)}")
    print(f"  Semantic-only      : {sem_only}")
    print(f"  Emotion-only       : {emo_only}")
    print(f"  Both               : {both}")
    print(f"  By emotion class   : {dict(emo_counts)}")
    return {"total": len(failures), "sem_only": sem_only,
            "emo_only": emo_only, "both": both, "by_emotion": dict(emo_counts)}

# ================================================================
# MAIN — PER-ROW CHECKPOINTING
# ================================================================

baseline_rows, iterative_rows, candidate_rows = load_checkpoint()

already_baseline  = {r["idx"] for r in baseline_rows}
already_iterative = {r["idx"] for r in iterative_rows}
already_candidate = {r["idx"] for r in candidate_rows}

# ----------------------------------------------------------------
# BASELINE
# ----------------------------------------------------------------
if len(baseline_rows) < len(df):
    print("[Exp 3] Running BASELINE ...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="baseline"):
        if int(row.name) in already_baseline:
            continue

        x     = row["text"]
        label = row["label_7"]
        ref_dist = get_reference_dist(x, label)["dist"]

        y = generate_paraphrase(x, label, temperature=0.1)
        time.sleep(RATE_DELAY)
        if y is None:
            y = x

        dist  = get_emotion_dist(y)
        drift = metric.compute(ref_dist, dist)
        sim   = sem_sim(x, y)

        baseline_rows.append({
            "idx": int(row.name), "text": x, "label": label,
            "output": y, "semantic": sim, **drift,
        })
        save_checkpoint(baseline_rows, iterative_rows, candidate_rows)
else:
    print("[Exp 3] Baseline complete — skipping")

# ----------------------------------------------------------------
# STRATEGY A — ITERATIVE REGENERATION
# ----------------------------------------------------------------
if len(iterative_rows) < len(df):
    print("\n[Exp 3] Running ITERATIVE REGENERATION (Strategy A) ...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="iterative"):
        if int(row.name) in already_iterative:
            continue

        x     = row["text"]
        label = row["label_7"]
        ref_dist = get_reference_dist(x, label)["dist"]

        result = run_iterative(
            text=x, emotion=label, gold_dist=ref_dist, metric=metric,
            max_attempts=MAX_ATTEMPTS, temperature=ITERATIVE_TEMP,
            rate_delay=RATE_DELAY,
        )

        iterative_rows.append({
            "idx": int(row.name), "text": x, "label": label,
            "best_text": result["best_text"],
            "attempts":  result["attempts"],
            **result["best"],
        })
        save_checkpoint(baseline_rows, iterative_rows, candidate_rows)
else:
    print("[Exp 3] Iterative complete — skipping")

# ----------------------------------------------------------------
# STRATEGY B — CANDIDATE GENERATION
# ----------------------------------------------------------------
if len(candidate_rows) < len(df):
    print("\n[Exp 3] Running CANDIDATE GENERATION (Strategy B) ...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="candidate"):
        if int(row.name) in already_candidate:
            continue

        x     = row["text"]
        label = row["label_7"]
        ref_dist = get_reference_dist(x, label)["dist"]

        result = run_candidate(
            text=x, emotion=label, gold_dist=ref_dist, metric=metric,
            n_candidates=N_CANDIDATES, temperature=CANDIDATE_TEMP,
            rate_delay=RATE_DELAY,
        )

        candidate_rows.append({
            "idx": int(row.name), "text": x, "label": label,
            "best_text":  result["best_text"],
            "candidates": result["candidates"],
            "early_exit": result["early_exit"],
            **result["best"],
        })
        save_checkpoint(baseline_rows, iterative_rows, candidate_rows)
else:
    print("[Exp 3] Candidate complete — skipping")

# ================================================================
# AGGREGATE + STATS
# ================================================================

baseline_agg  = aggregate(baseline_rows)
iterative_agg = aggregate(iterative_rows)
candidate_agg = aggregate(candidate_rows)

print_agg_table("BASELINE   (no optimization)",  baseline_agg)
print_agg_table("STRATEGY A (iterative regen.)", iterative_agg)
print_agg_table("STRATEGY B (candidate gen.)",   candidate_agg)

baseline_emo  = per_emotion_breakdown(baseline_rows)
iterative_emo = per_emotion_breakdown(iterative_rows)
candidate_emo = per_emotion_breakdown(candidate_rows)

baseline_fail  = failure_analysis(baseline_rows,  "Baseline")
iterative_fail = failure_analysis(iterative_rows, "Strategy A — Iterative")
candidate_fail = failure_analysis(candidate_rows, "Strategy B — Candidate")

test_js           = wilcoxon_test(iterative_rows, candidate_rows, "D_JS")
test_sem          = wilcoxon_test(iterative_rows, candidate_rows, "semantic")
test_entropy      = wilcoxon_test(iterative_rows, candidate_rows, "D_entropy")
test_flip         = wilcoxon_test(iterative_rows, candidate_rows, "D_flip")
test_iter_vs_base = wilcoxon_test(baseline_rows,  iterative_rows, "D_JS")
test_cand_vs_base = wilcoxon_test(baseline_rows,  candidate_rows, "D_JS")

print("\n" + "=" * 62)
print(f"STATISTICAL TESTS — Wilcoxon Signed-Rank (paired, n={SAMPLE_SIZE})")
print("=" * 62)
for t in [test_js, test_sem, test_entropy, test_flip]:
    sig = ("***" if t["p_value"] < 0.001 else
           "**"  if t["p_value"] < 0.01  else
           "*"   if t["p_value"] < 0.05  else "ns")
    print(f"  {t['metric']:<14}  A={t['mean_A']:.4f}  B={t['mean_B']:.4f}  "
          f"p={t['p_value']:.4f} {sig}  better={t['better']}")
print()
print("  (A=Iterative  B=Candidate  *p<0.05  **p<0.01  ***p<0.001  ns=not sig)")

print("\n  --- vs Baseline (D_JS) ---")
for t, lbl in [(test_iter_vs_base, "Iter vs Base"),
               (test_cand_vs_base, "Cand vs Base")]:
    sig = ("***" if t["p_value"] < 0.001 else
           "**"  if t["p_value"] < 0.01  else
           "*"   if t["p_value"] < 0.05  else "ns")
    print(f"  {lbl:<16}  base={t['mean_A']:.4f}  opt={t['mean_B']:.4f}  "
          f"p={t['p_value']:.4f} {sig}")

print("\n" + "=" * 76)
print(f"TABLE 2 — STRATEGY COMPARISON (n={SAMPLE_SIZE}, stratified)")
print("=" * 76)
print(f"{'Condition':<22} {'Sem.Sim':>9} {'D_JS':>9} {'Flip%':>9} "
      f"{'Entropy':>9} {'Catast%':>9}")
print("-" * 76)
for lbl, agg in [("Baseline",  baseline_agg),
                 ("Iterative", iterative_agg),
                 ("Candidate", candidate_agg)]:
    print(f"{lbl:<22} {agg['mean_semantic']:9.4f} {agg['mean_D_JS']:9.4f} "
          f"{agg['flip_rate']*100:9.2f} {agg['mean_D_entropy']:9.4f} "
          f"{agg['catastrophic_rate']*100:9.2f}")
print("=" * 76)

early_rate = float(np.mean([r["early_exit"] for r in candidate_rows]))
print(f"\n  Candidate early-exit rate: {early_rate*100:.2f}%")

# ================================================================
# SAVE
# ================================================================

raw_output = {
    "config": {
        "sample_size":    SAMPLE_SIZE,
        "random_seed":    RANDOM_SEED,
        "max_attempts":   MAX_ATTEMPTS,
        "n_candidates":   N_CANDIDATES,
        "iterative_temp": ITERATIVE_TEMP,
        "candidate_temp": CANDIDATE_TEMP,
        "rate_delay":     RATE_DELAY,
        "sampling":       "stratified",
    },
    "baseline":  baseline_rows,
    "iterative": iterative_rows,
    "candidate": candidate_rows,
}
save_json(RESULTS_DIR / "exp3_raw.json", raw_output)

summary = {
    "config": raw_output["config"],
    "aggregate": {
        "baseline":  baseline_agg,
        "iterative": iterative_agg,
        "candidate": candidate_agg,
    },
    "per_emotion": {
        "baseline":  baseline_emo,
        "iterative": iterative_emo,
        "candidate": candidate_emo,
    },
    "statistical_tests": {
        "A_vs_B": {
            "D_JS":      test_js,
            "semantic":  test_sem,
            "D_entropy": test_entropy,
            "D_flip":    test_flip,
        },
        "vs_baseline": {
            "iterative": test_iter_vs_base,
            "candidate": test_cand_vs_base,
        },
    },
    "failure_analysis": {
        "baseline":  baseline_fail,
        "iterative": iterative_fail,
        "candidate": candidate_fail,
    },
    "candidate_early_exit_rate": early_rate,
}
save_json(RESULTS_DIR / "exp3_summary.json", summary)

print("\n[Exp 3 COMPLETE]")
print(f"  Full data  → {RESULTS_DIR}/exp3_raw.json")
print(f"  Summary    → {RESULTS_DIR}/exp3_summary.json")
print(f"  Checkpoint → {CHECKPOINT}  (safe to delete after successful run)")