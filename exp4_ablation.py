"""
exp4_ablation.py — Experiment 4: Penalty Function Ablation
===========================================================

PURPOSE
-------
Determine which terms of the penalty function contribute most to
selection quality. Shows the metric design was not arbitrary.

DESIGN
------
No new LLM calls. Re-uses ALL candidate/attempt data from exp3_raw.json.
For the candidate generation strategy, re-ranks candidates using 5
different penalty formulations and measures resulting metric quality.

For iterative regeneration, re-selects the best attempt using each
penalty variant and evaluates the resulting selection quality.

PENALTY VARIANTS
----------------
  full        : α(1 - sem) + D_JS + D_flip + max(0, D_entropy - 0.15)
  sem_only    : α(1 - sem)
  js_only     : D_JS
  flip_only   : D_flip
  no_entropy  : α(1 - sem) + D_JS + D_flip

HYPOTHESIS
----------
The full penalty should outperform all ablated variants on at least
two of the four evaluation metrics (D_JS, semantic, flip_rate, D_entropy).

REQUIRES
--------
  results/exp3_raw.json   (produced by exp3_comparison.py)

OUTPUT
------
  results/exp4_ablation.json
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from shared import (
    RESULTS_DIR, EMOTION_ORDER,
    PENALTY_VARIANTS,
    aggregate, save_json, load_json, print_agg_table,
)

# ================================================================
# LOAD EXP3 RESULTS
# ================================================================

print("[Exp 4] Loading exp3 raw data ...")
data = load_json(RESULTS_DIR / "exp3_raw.json")

iterative_rows = data["iterative"]
candidate_rows = data["candidate"]

print(f"[Exp 4] Loaded {len(candidate_rows)} sentences")
print(f"[Exp 4] Penalty variants: {list(PENALTY_VARIANTS.keys())}")

# ================================================================
# CANDIDATE RE-RANKING
# ================================================================

def rerank_candidates(rows: list[dict], penalty_fn) -> list[dict]:
    """
    For each sentence in candidate generation results,
    select the best candidate using a different penalty function.
    Returns rows with updated best-candidate metrics.
    """
    reranked = []

    for row in rows:
        candidates = row.get("candidates", [])

        if not candidates:
            # Fallback: use whatever was stored
            reranked.append({
                "idx":      row["idx"],
                "text":     row["text"],
                "label":    row["label"],
                "semantic": row["semantic"],
                "D_JS":     row["D_JS"],
                "D_flip":   row["D_flip"],
                "D_entropy":row["D_entropy"],
                "D_conf":   row.get("D_conf", 0.0),
            })
            continue

        best_p   = float("inf")
        best_rec = None

        for cand in candidates:
            sim   = cand["semantic"]
            drift = {
                "D_JS":      cand["D_JS"],
                "D_flip":    cand["D_flip"],
                "D_entropy": cand["D_entropy"],
                "D_conf":    cand.get("D_conf", 0.0),
            }
            p = penalty_fn(sim, drift)

            if p < best_p:
                best_p   = p
                best_rec = {
                    "idx":       row["idx"],
                    "text":      row["text"],
                    "label":     row["label"],
                    "semantic":  sim,
                    **drift,
                }

        reranked.append(best_rec)

    return reranked

# ================================================================
# ITERATIVE RE-SELECTION
# ================================================================

def reselect_iterative(rows: list[dict], penalty_fn) -> list[dict]:
    """
    For each sentence in iterative regeneration results,
    select the best attempt using a different penalty function.
    """
    reselected = []

    for row in rows:
        attempts = row.get("attempts", [])

        if not attempts:
            reselected.append({
                "idx":       row["idx"],
                "text":      row["text"],
                "label":     row["label"],
                "semantic":  row["semantic"],
                "D_JS":      row["D_JS"],
                "D_flip":    row["D_flip"],
                "D_entropy": row["D_entropy"],
                "D_conf":    row.get("D_conf", 0.0),
            })
            continue

        best_p   = float("inf")
        best_rec = None

        for att in attempts:
            sim   = att["semantic"]
            drift = {
                "D_JS":      att["D_JS"],
                "D_flip":    att["D_flip"],
                "D_entropy": att["D_entropy"],
                "D_conf":    att.get("D_conf", 0.0),
            }
            p = penalty_fn(sim, drift)

            if p < best_p:
                best_p   = p
                best_rec = {
                    "idx":       row["idx"],
                    "text":      row["text"],
                    "label":     row["label"],
                    "semantic":  sim,
                    **drift,
                }

        reselected.append(best_rec)

    return reselected

# ================================================================
# PRINT ABLATION TABLE
# ================================================================

def print_ablation_table(title: str, results: dict) -> None:
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    print(f"{'Penalty Variant':<20} {'Sem.Sim':>9} {'D_JS':>9} {'Flip%':>9} {'Entropy':>9} {'Catast%':>9}")
    print("-" * 78)
    for variant, agg in results.items():
        marker = " ← full" if variant == "full" else ""
        print(f"{variant:<20} {agg['mean_semantic']:9.4f} {agg['mean_D_JS']:9.4f} "
              f"{agg['flip_rate']*100:9.2f} {agg['mean_D_entropy']:9.4f} "
              f"{agg['catastrophic_rate']*100:9.2f}{marker}")
    print(f"{'='*78}")

# ================================================================
# MAIN
# ================================================================

cand_results = {}
iter_results = {}

print("\n[Exp 4] Running ablation on CANDIDATE strategy ...")
for variant_name, penalty_fn in PENALTY_VARIANTS.items():
    rows = rerank_candidates(candidate_rows, penalty_fn)
    agg  = aggregate(rows)
    cand_results[variant_name] = agg
    print(f"  {variant_name:<20} sem={agg['mean_semantic']:.4f}  D_JS={agg['mean_D_JS']:.4f}  "
          f"flip={agg['flip_rate']*100:.2f}%")

print("\n[Exp 4] Running ablation on ITERATIVE strategy ...")
for variant_name, penalty_fn in PENALTY_VARIANTS.items():
    rows = reselect_iterative(iterative_rows, penalty_fn)
    agg  = aggregate(rows)
    iter_results[variant_name] = agg
    print(f"  {variant_name:<20} sem={agg['mean_semantic']:.4f}  D_JS={agg['mean_D_JS']:.4f}  "
          f"flip={agg['flip_rate']*100:.2f}%")

print_ablation_table("ABLATION TABLE — Candidate Generation", cand_results)
print_ablation_table("ABLATION TABLE — Iterative Regeneration", iter_results)

# ================================================================
# DETERMINE WHICH TERMS MATTER MOST
# ================================================================

print("\n--- Contribution Analysis (Candidate Strategy) ---")
print("Removing a term increases loss. Δ = full − ablated (positive = term helped).")
print()

full_cand = cand_results["full"]
print(f"{'Term removed':<20} {'ΔD_JS':>9} {'ΔSem':>9} {'ΔFlip%':>9}")
print("-" * 50)
for variant, agg in cand_results.items():
    if variant == "full":
        continue
    delta_js  = agg["mean_D_JS"]     - full_cand["mean_D_JS"]
    delta_sem = full_cand["mean_semantic"] - agg["mean_semantic"]
    delta_fl  = agg["flip_rate"]     - full_cand["flip_rate"]
    direction = "↑ bad" if delta_js > 0 else "↓ good"
    print(f"{variant:<20} {delta_js:9.4f} {delta_sem:9.4f} {delta_fl*100:9.2f}  {direction}")

print("\n  (positive ΔD_JS = removing term made things worse = term contributed)")

# ================================================================
# RANK PENALTY TERMS BY IMPORTANCE
# ================================================================

def rank_terms(results: dict) -> list:
    """Rank variants by how much worse they perform vs full penalty on D_JS."""
    full_js = results["full"]["mean_D_JS"]
    deltas  = []
    for variant, agg in results.items():
        if variant == "full":
            continue
        delta = agg["mean_D_JS"] - full_js  # positive = variant is worse = term was important
        deltas.append((variant, delta))
    return sorted(deltas, key=lambda x: -x[1])

print("\n--- Term Importance Ranking (Candidate, by D_JS impact) ---")
for rank, (variant, delta) in enumerate(rank_terms(cand_results), 1):
    print(f"  {rank}. {variant:<20}  removing this term costs +{delta:.4f} D_JS")

# ================================================================
# SAVE
# ================================================================

output = {
    "candidate_ablation": cand_results,
    "iterative_ablation": iter_results,
    "term_importance_candidate": [
        {"variant": v, "delta_D_JS": d}
        for v, d in rank_terms(cand_results)
    ],
    "term_importance_iterative": [
        {"variant": v, "delta_D_JS": d}
        for v, d in rank_terms(iter_results)
    ],
}
save_json(RESULTS_DIR / "exp4_ablation.json", output)

print("\n[Exp 4 COMPLETE]")
print(f"  Results → {RESULTS_DIR}/exp4_ablation.json")
