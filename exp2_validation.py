"""
exp2_validation.py — Experiment 2: Metric Validation (Pairwise)
================================================================

PURPOSE
-------
Validate that the drift metrics correlate with human-perceived
emotion preservation quality.

APPROACH — PAIRWISE COMPARISON
--------------------------------
Instead of absolute 1-5 rating (which produces extreme ceiling effects
with small LLM judges), we use pairwise preference judgment:

  Given: original sentence + emotion label
  Option A: baseline output (unoptimized, 1 LLM call)
  Option B: iterative output (optimized, 3 LLM calls)
  Question: Which better preserves the emotional tone?

This produces a binary outcome (A/B) with guaranteed variance.
We then check whether our penalty metric correctly predicts the
judge's preference — i.e., does lower penalty → judge prefers it?

WHAT THIS VALIDATES
--------------------
  H1: When iterative penalty < baseline penalty,
      judge should prefer iterative  (metric correctly identifies better output)

  H2: The penalty difference (Δpenalty) should correlate with
      judge confidence — larger penalty gap → clearer preference

SAMPLE
------
90 pairs: one (baseline, iterative) pair per sentence,
sampled from sentences where penalty gap is largest
(these are the most discriminative cases for the judge).

REQUIRES
--------
  results/exp3_raw.json

OUTPUT
------
  results/exp2_validation.json
"""

import time
import numpy as np
import re
from collections import defaultdict
from scipy import stats

from shared import (
    RESULTS_DIR, penalty_full, call_groq,
    save_json, load_json,
)

# ================================================================
# CONFIG
# ================================================================

N_PAIRS    = 90     # number of (baseline, iterative) pairs to judge
RATE_DELAY = 1.0
RANDOM_SEED = 0

# ================================================================
# LOAD DATA
# ================================================================

print("[Exp 2] Loading exp3 data ...")
data = load_json(RESULTS_DIR / "exp3_raw.json")

baseline_rows  = data["baseline"]
iterative_rows = data["iterative"]

# Index by idx for pairing
base_by_idx = {r["idx"]: r for r in baseline_rows}
iter_by_idx = {r["idx"]: r for r in iterative_rows}

# ================================================================
# BUILD PAIRS
# ================================================================
# For each sentence, compute penalty for both baseline and iterative output.
# Select sentences where the penalty gap is largest — these are the most
# discriminative cases where the judge should show clearest preference.

def get_penalty(row):
    return penalty_full(row["semantic"], {
        "D_JS":      row["D_JS"],
        "D_flip":    row["D_flip"],
        "D_entropy": row["D_entropy"],
        "D_conf":    row.get("D_conf", 0.0),
    })

pairs = []
for idx in base_by_idx:
    if idx not in iter_by_idx:
        continue
    b = base_by_idx[idx]
    it = iter_by_idx[idx]

    p_base = get_penalty(b)
    p_iter = get_penalty(it)

    pairs.append({
        "idx":         idx,
        "text":        b["text"],
        "label":       b["label"],
        "base_output": b.get("output", b["text"]),
        "iter_output": it.get("best_text", it["text"]),
        "penalty_base": p_base,
        "penalty_iter": p_iter,
        "delta_penalty": p_base - p_iter,  # positive = iterative is better by metric
        "base_D_JS":   b["D_JS"],
        "iter_D_JS":   it["D_JS"],
        "base_sem":    b["semantic"],
        "iter_sem":    it["semantic"],
        "base_flip":   b["D_flip"],
        "iter_flip":   it["D_flip"],
    })

# Sort by absolute penalty gap descending — most discriminative first
pairs.sort(key=lambda p: abs(p["delta_penalty"]), reverse=True)

# Take top N_PAIRS
# Also ensure some pairs where metric says they're equal (gap ≈ 0)
# by adding 10 near-zero-gap pairs at the end
top_pairs  = pairs[:N_PAIRS - 10]
zero_pairs = sorted(pairs, key=lambda p: abs(p["delta_penalty"]))[:10]
eval_pairs = top_pairs + zero_pairs

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(eval_pairs)

print(f"[Exp 2] {len(eval_pairs)} pairs selected")
print(f"  Mean |Δpenalty|: {np.mean([abs(p['delta_penalty']) for p in eval_pairs]):.4f}")
print(f"  Metric predicts iterative better: "
      f"{sum(1 for p in eval_pairs if p['delta_penalty'] > 0)}/{len(eval_pairs)}")

# ================================================================
# JUDGE PROMPT
# ================================================================

_JUDGE_SYS = """
You are a strict emotion-preservation evaluator.

You will be shown an original sentence with its emotion label,
and two paraphrases: Option A and Option B.

Your task: decide which paraphrase better preserves the emotional
tone and intensity of the original sentence.

Rules:
- Focus ONLY on emotional preservation, not grammar or fluency.
- If one option weakens, softens, or changes the emotion, prefer the other.
- If both are equally good or equally bad, output TIE.
- Output ONLY one of: A, B, or TIE
- No explanation. No preamble. Just the letter.
"""

def judge_pair(original, emotion, output_a, output_b):
    """
    Ask judge to pick which output better preserves emotion.
    Returns 'A', 'B', 'TIE', or None on failure.
    """
    user = (
        f'Original sentence: "{original}"\n'
        f'Emotion to preserve: {emotion}\n\n'
        f'Option A: "{output_a}"\n'
        f'Option B: "{output_b}"\n\n'
        "Which option better preserves the emotional tone? "
        "Output only A, B, or TIE."
    )
    out = call_groq(_JUDGE_SYS, user, temperature=0.0)
    if out is None:
        return None
    out = out.strip().upper()
    if out in ("A", "B", "TIE"):
        return out
    # Try extracting from longer response
    m = re.search(r'\b(A|B|TIE)\b', out)
    return m.group(1) if m else None

# ================================================================
# RANDOMIZE PRESENTATION ORDER
# ================================================================
# To avoid position bias, randomly swap which output is A vs B
# for each pair. Record the swap so we can recover true preference.

judged = []
print(f"\n[Exp 2] Running pairwise judge on {len(eval_pairs)} pairs ...")

for i, pair in enumerate(eval_pairs):
    # Randomly assign A/B to avoid position bias
    swap = rng.integers(0, 2) == 1  # True = swap baseline/iterative

    if swap:
        output_a = pair["iter_output"]
        output_b = pair["base_output"]
        a_is = "iterative"
        b_is = "baseline"
    else:
        output_a = pair["base_output"]
        output_b = pair["iter_output"]
        a_is = "baseline"
        b_is = "iterative"

    verdict = judge_pair(pair["text"], pair["label"], output_a, output_b)
    time.sleep(RATE_DELAY)

    if verdict is None:
        print(f"  [warn] judge returned None for idx={pair['idx']}, skipping")
        continue

    # Resolve to which condition was preferred
    if verdict == "TIE":
        preferred = "tie"
    elif verdict == "A":
        preferred = a_is
    else:
        preferred = b_is

    # Did the metric correctly predict the preference?
    # metric says iterative is better when delta_penalty > 0
    metric_says = "iterative" if pair["delta_penalty"] > 0 else "baseline"
    metric_correct = (preferred == metric_says) or preferred == "tie"

    judged.append({
        **pair,
        "swap":           bool(swap),
        "verdict":        verdict,
        "preferred":      preferred,
        "metric_says":    metric_says,
        "metric_correct": metric_correct,
    })

    if (i + 1) % 10 == 0:
        print(f"  ... {i+1}/{len(eval_pairs)} judged")

print(f"\n[Exp 2] Successfully judged: {len(judged)} pairs")

# ================================================================
# ANALYSIS
# ================================================================

# 1. Overall preference distribution
n_iter_preferred = sum(1 for r in judged if r["preferred"] == "iterative")
n_base_preferred = sum(1 for r in judged if r["preferred"] == "baseline")
n_tie            = sum(1 for r in judged if r["preferred"] == "tie")
n_total          = len(judged)

print(f"\n  Judge preferences:")
print(f"    Iterative preferred : {n_iter_preferred}/{n_total} ({n_iter_preferred/n_total*100:.1f}%)")
print(f"    Baseline preferred  : {n_base_preferred}/{n_total} ({n_base_preferred/n_total*100:.1f}%)")
print(f"    Tie                 : {n_tie}/{n_total} ({n_tie/n_total*100:.1f}%)")

# 2. Metric accuracy — when metric predicts iterative is better, is judge agreement > 50%?
metric_predicts_iter = [r for r in judged if r["metric_says"] == "iterative"]
metric_predicts_base = [r for r in judged if r["metric_says"] == "baseline"]

if metric_predicts_iter:
    iter_correct = sum(1 for r in metric_predicts_iter
                       if r["preferred"] == "iterative" or r["preferred"] == "tie")
    print(f"\n  Metric accuracy (predicts iterative better):")
    print(f"    Judge agrees : {iter_correct}/{len(metric_predicts_iter)} "
          f"({iter_correct/len(metric_predicts_iter)*100:.1f}%)")

# 3. Binomial test — is iterative preference rate > 50%?
# Exclude ties for this test
decisive = [r for r in judged if r["preferred"] != "tie"]
if decisive:
    n_iter_decisive = sum(1 for r in decisive if r["preferred"] == "iterative")
    binom_result = stats.binomtest(n_iter_decisive, len(decisive), p=0.5, alternative="greater")
    print(f"\n  Binomial test (iterative > baseline, excluding ties):")
    print(f"    {n_iter_decisive}/{len(decisive)} pairs prefer iterative")
    print(f"    p = {binom_result.pvalue:.4f} "
          f"{'***' if binom_result.pvalue < 0.001 else '**' if binom_result.pvalue < 0.01 else '*' if binom_result.pvalue < 0.05 else 'ns'}")

# 4. Correlation: does larger Δpenalty predict stronger agreement with metric?
# Encode: +1 if judge preferred what metric predicted, -1 if opposite, 0 if tie
agreement_scores = []
delta_penalties  = []
for r in judged:
    if r["preferred"] == "tie":
        agreement_scores.append(0)
    elif r["preferred"] == r["metric_says"]:
        agreement_scores.append(1)
    else:
        agreement_scores.append(-1)
    delta_penalties.append(abs(r["delta_penalty"]))

rho, p = stats.spearmanr(delta_penalties, agreement_scores)
print(f"\n  Spearman correlation (|Δpenalty| vs metric-judge agreement):")
print(f"    ρ = {rho:.4f}   p = {p:.4f} "
      f"{'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
print(f"    (positive ρ = larger penalty gap → stronger judge agreement with metric)")

# 5. Per-emotion breakdown
print(f"\n  Per-emotion preference (iterative vs baseline):")
print(f"  {'Emotion':<12} {'N':>5} {'Iter%':>8} {'Base%':>8} {'Tie%':>8}")
print(f"  {'-'*48}")
emo_buckets = defaultdict(list)
for r in judged:
    emo_buckets[r["label"]].append(r)
for emotion in sorted(emo_buckets.keys()):
    sub = emo_buckets[emotion]
    n_i = sum(1 for r in sub if r["preferred"] == "iterative")
    n_b = sum(1 for r in sub if r["preferred"] == "baseline")
    n_t = sum(1 for r in sub if r["preferred"] == "tie")
    print(f"  {emotion:<12} {len(sub):5d} {n_i/len(sub)*100:8.1f} "
          f"{n_b/len(sub)*100:8.1f} {n_t/len(sub)*100:8.1f}")

# ================================================================
# PRINT SUMMARY TABLE
# ================================================================

print(f"\n{'='*62}")
print(f"METRIC VALIDATION — Pairwise Judge (n={len(judged)} pairs)")
print(f"{'='*62}")
print(f"  Iterative preferred by judge  : {n_iter_preferred/n_total*100:.1f}%")
print(f"  Baseline preferred by judge   : {n_base_preferred/n_total*100:.1f}%")
print(f"  Tie                           : {n_tie/n_total*100:.1f}%")
if decisive:
    print(f"  Binomial test p-value         : {binom_result.pvalue:.4f}")
print(f"  Spearman ρ (|Δpenalty| vs agreement): {rho:.4f}  p={p:.4f}")
print(f"{'='*62}")

# ================================================================
# SAVE
# ================================================================

output = {
    "config": {
        "n_pairs":    N_PAIRS,
        "approach":   "pairwise_comparison",
        "judge_model": "llama-3.1-8b-instant",
        "note": ("Pairwise used instead of absolute rating due to "
                 "ceiling effect in absolute scoring with small LLM judges"),
    },
    "preferences": {
        "iterative": n_iter_preferred,
        "baseline":  n_base_preferred,
        "tie":       n_tie,
        "total":     n_total,
        "iterative_pct": float(n_iter_preferred / n_total),
    },
    "binomial_test": {
        "n_iter_decisive":  int(n_iter_decisive) if decisive else 0,
        "n_decisive":       len(decisive),
        "p_value":          float(binom_result.pvalue) if decisive else None,
        "significant":      bool(binom_result.pvalue < 0.05) if decisive else False,
    },
    "spearman_delta_vs_agreement": {
        "rho":       float(rho),
        "p_value":   float(p),
        "significant": bool(p < 0.05),
    },
    "per_emotion": {
        emotion: {
            "n": len(sub),
            "iterative_pct": float(sum(1 for r in sub if r["preferred"]=="iterative")/len(sub)),
            "baseline_pct":  float(sum(1 for r in sub if r["preferred"]=="baseline")/len(sub)),
            "tie_pct":       float(sum(1 for r in sub if r["preferred"]=="tie")/len(sub)),
        }
        for emotion, sub in emo_buckets.items()
    },
    "judged_rows": judged,
}
save_json(RESULTS_DIR / "exp2_validation.json", output)

print("\n[Exp 2 COMPLETE]")
print(f"  Results → {RESULTS_DIR}/exp2_validation.json")