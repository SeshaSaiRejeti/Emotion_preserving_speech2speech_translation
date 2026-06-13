"""
exp5_emotion_analysis.py — Experiment 5: Per-Emotion Analysis
==============================================================

PURPOSE
-------
Identify which emotion classes are hardest to preserve under each strategy.
This analysis adds depth to the paper and opens discussion on
emotion-specific translation challenges.

DESIGN
------
No new LLM calls. Pure re-analysis of exp3_raw.json.
Produces per-emotion tables for all three conditions (baseline, iterative, candidate).
Also identifies the "hardest" and "easiest" emotions by flip rate.

REQUIRES
--------
  results/exp3_raw.json   (produced by exp3_comparison.py)

OUTPUT
------
  results/exp5_emotion_analysis.json
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from shared import RESULTS_DIR, load_json, save_json

# ================================================================
# LOAD EXP3 RESULTS
# ================================================================

print("[Exp 5] Loading exp3 results ...")
data = load_json(RESULTS_DIR / "exp3_raw.json")

baseline_rows  = data["baseline"]
iterative_rows = data["iterative"]
candidate_rows = data["candidate"]

print(f"[Exp 5] Loaded {len(baseline_rows)} sentences per condition")

# ================================================================
# PER-EMOTION STATS
# ================================================================

def per_emotion_stats(rows: list[dict]) -> dict:
    """
    Detailed per-emotion breakdown.
    Returns dict keyed by emotion class.
    """
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["label"]].append(r)

    result = {}
    for emotion, sub in sorted(buckets.items()):
        sems = np.array([r["semantic"]  for r in sub])
        js   = np.array([r["D_JS"]      for r in sub])
        de   = np.array([r["D_entropy"] for r in sub])
        fl   = np.array([r["D_flip"]    for r in sub], dtype=float)

        result[emotion] = {
            "n":              len(sub),
            "mean_semantic":  float(np.mean(sems)),
            "std_semantic":   float(np.std(sems)),
            "mean_D_JS":      float(np.mean(js)),
            "median_D_JS":    float(np.median(js)),
            "p95_D_JS":       float(np.percentile(js, 95)),
            "mean_D_entropy": float(np.mean(de)),
            "flip_rate":      float(np.mean(fl)),
            "catast_rate":    float(np.mean((sems < 0.30) | (js > 0.50))),
        }
    return result

# ================================================================
# IMPROVEMENT ANALYSIS
# ================================================================

def compute_improvement(base_rows, opt_rows, label: str) -> dict:
    """
    Compute per-emotion improvement of an optimized strategy over baseline.
    delta is positive when optimized is better (lower D_JS or higher semantic).
    """
    base_by_idx = {r["idx"]: r for r in base_rows}
    opt_by_idx  = {r["idx"]: r for r in opt_rows}

    buckets = defaultdict(list)
    for idx in base_by_idx:
        if idx not in opt_by_idx:
            continue
        b = base_by_idx[idx]
        o = opt_by_idx[idx]
        buckets[b["label"]].append({
            "delta_D_JS":      b["D_JS"]     - o["D_JS"],       # positive = improved
            "delta_semantic":  o["semantic"] - b["semantic"],   # positive = improved
            "delta_D_flip":    b["D_flip"]   - o["D_flip"],     # positive = improved
        })

    result = {}
    for emotion, deltas in sorted(buckets.items()):
        djs  = np.array([d["delta_D_JS"]     for d in deltas])
        dsem = np.array([d["delta_semantic"]  for d in deltas])
        dfl  = np.array([d["delta_D_flip"]    for d in deltas])
        result[emotion] = {
            "n":                    len(deltas),
            "mean_delta_D_JS":      float(np.mean(djs)),
            "mean_delta_semantic":  float(np.mean(dsem)),
            "mean_delta_flip":      float(np.mean(dfl)),
            "improved_fraction":    float(np.mean(djs > 0)),  # fraction where D_JS dropped
        }
    return result

# ================================================================
# DIFFICULTY RANKING
# ================================================================

def rank_emotions_by_difficulty(per_emotion: dict, metric: str = "flip_rate") -> list:
    """
    Returns emotions sorted hardest-to-easiest by given metric.
    """
    return sorted(per_emotion.items(), key=lambda x: x[1][metric], reverse=True)

# ================================================================
# CROSS-STRATEGY EMOTION COMPARISON
# ================================================================

def head_to_head(iter_stats: dict, cand_stats: dict) -> dict:
    """
    For each emotion class, determines which strategy is better.
    Decision criterion: lower mean D_JS.
    """
    result = {}
    for emotion in iter_stats:
        iter_js = iter_stats[emotion]["mean_D_JS"]
        cand_js = cand_stats[emotion]["mean_D_JS"]
        result[emotion] = {
            "iterative_D_JS": iter_js,
            "candidate_D_JS": cand_js,
            "better":         "candidate" if cand_js < iter_js else "iterative",
            "delta":          float(iter_js - cand_js),  # positive = candidate better
        }
    return result

# ================================================================
# PRINT HELPERS
# ================================================================

def print_per_emotion_table(title: str, stats: dict) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    print(f"{'Emotion':<12} {'N':>5} {'Sem':>8} {'D_JS':>8} {'Flip%':>8} {'Catast%':>8}")
    print("-" * 72)
    for emotion, s in stats.items():
        print(f"{emotion:<12} {s['n']:5d} {s['mean_semantic']:8.4f} "
              f"{s['mean_D_JS']:8.4f} {s['flip_rate']*100:8.2f} "
              f"{s['catast_rate']*100:8.2f}")
    print(f"{'='*72}")

def print_improvement_table(title: str, improvements: dict) -> None:
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")
    print(f"{'Emotion':<12} {'ΔD_JS':>9} {'ΔSem':>9} {'ΔFlip':>9} {'Improved%':>10}")
    print("-" * 68)
    for emotion, d in improvements.items():
        print(f"{emotion:<12} {d['mean_delta_D_JS']:9.4f} {d['mean_delta_semantic']:9.4f} "
              f"{d['mean_delta_flip']:9.4f} {d['improved_fraction']*100:10.2f}")
    print(f"{'='*68}")
    print("  (positive ΔD_JS = D_JS decreased = improvement; positive ΔSem = improved)")

# ================================================================
# MAIN
# ================================================================

# Per-emotion stats for each condition
base_stats = per_emotion_stats(baseline_rows)
iter_stats = per_emotion_stats(iterative_rows)
cand_stats = per_emotion_stats(candidate_rows)

print_per_emotion_table("PER-EMOTION — Baseline",   base_stats)
print_per_emotion_table("PER-EMOTION — Iterative",  iter_stats)
print_per_emotion_table("PER-EMOTION — Candidate",  cand_stats)

# Difficulty rankings
print("\n--- Emotion Difficulty Ranking (by flip rate, Baseline) ---")
for rank, (emotion, stats) in enumerate(rank_emotions_by_difficulty(base_stats), 1):
    print(f"  {rank}. {emotion:<12} flip={stats['flip_rate']*100:.1f}%  D_JS={stats['mean_D_JS']:.4f}")

# Improvements over baseline
iter_improvement = compute_improvement(baseline_rows, iterative_rows, "iterative")
cand_improvement = compute_improvement(baseline_rows, candidate_rows, "candidate")

print_improvement_table("IMPROVEMENT over Baseline — Iterative Strategy", iter_improvement)
print_improvement_table("IMPROVEMENT over Baseline — Candidate Strategy", cand_improvement)

# Head-to-head per emotion
h2h = head_to_head(iter_stats, cand_stats)
print("\n--- HEAD-TO-HEAD: Iterative vs Candidate per Emotion (D_JS criterion) ---")
print(f"{'Emotion':<12} {'Iter D_JS':>10} {'Cand D_JS':>10} {'Better':>12} {'Δ':>9}")
print("-" * 58)
for emotion, result in h2h.items():
    print(f"{emotion:<12} {result['iterative_D_JS']:10.4f} {result['candidate_D_JS']:10.4f} "
          f"{result['better']:>12} {result['delta']:9.4f}")

cand_wins = sum(1 for v in h2h.values() if v["better"] == "candidate")
iter_wins = sum(1 for v in h2h.values() if v["better"] == "iterative")
print(f"\n  Candidate wins on {cand_wins} emotions, Iterative wins on {iter_wins} emotions")

# ================================================================
# SAVE
# ================================================================

output = {
    "per_emotion": {
        "baseline":  base_stats,
        "iterative": iter_stats,
        "candidate": cand_stats,
    },
    "improvement_over_baseline": {
        "iterative": iter_improvement,
        "candidate": cand_improvement,
    },
    "head_to_head": h2h,
    "difficulty_ranking": [
        {"rank": i+1, "emotion": emo, **stats}
        for i, (emo, stats) in enumerate(rank_emotions_by_difficulty(base_stats))
    ],
}
save_json(RESULTS_DIR / "exp5_emotion_analysis.json", output)

print("\n[Exp 5 COMPLETE]")
print(f"  Results → {RESULTS_DIR}/exp5_emotion_analysis.json")
