"""
exp1_baseline.py — Experiment 1: Baseline Translation Characterization
=======================================================================

MEASUREMENT FIX (v2)
---------------------
Reference distribution is now obtained by running the classifier on the
ORIGINAL TEXT, not constructed as a one-hot vector.

  # WRONG (v1) — one-hot inflates all divergence metrics artificially
  gold_dist = np.zeros(7); gold_dist[idx] = 1.0

  # CORRECT (v2) — classifier output vs classifier output
  ref_dist = get_emotion_dist(original_text)

Reason: roberta-base-go_emotions never produces one-hot outputs.
Comparing one-hot vs soft distribution gives artificially high JS divergence
even for perfect translations. Classifier vs classifier is the fair comparison.

CONTROL (v2)
------------
The control now measures pure classifier noise: run the classifier TWICE on
the SAME original text with no LLM involvement.
Expected: D_JS ≈ 0, flip_rate ≈ 0%, semantic = 1.0.

PURPOSE
-------
Quantify emotional drift in standard unoptimized LLM translation.
Provides Table 1 numbers for the paper.

LANGUAGES   : Spanish, French, Hindi
SAMPLE      : 500 sentences  (golden_goemotions_7class.csv, stratified)
SEED        : 42

OUTPUT FILES
------------
  results/exp1_<lang>_raw.json
  results/exp1_summary.json
"""

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from shared import (
    RESULTS_DIR, EMOTION_ORDER, DriftMetric,
    get_emotion_dist, get_reference_dist, sem_sim,
    generate_translation, generate_back_translation,
    aggregate, save_json, print_agg_table,
)

# ================================================================
# CONFIG
# ================================================================

SAMPLE_SIZE   = 500
RANDOM_SEED   = 42
CONTROL_SIZE  = 100
RATE_DELAY    = 1.5

TARGET_LANGUAGES = {
    "Spanish": "Spanish",
    "French":  "French",
    "Hindi":   "Hindi",
}

# ================================================================
# DATA
# ================================================================

df_full = pd.read_csv("golden_goemotions_7class.csv")

def stratified_sample(df, n, seed):
    classes    = df["label_7"].unique()
    per_class  = max(1, n // len(classes))
    rng        = np.random.default_rng(seed)
    parts      = []
    for cls in classes:
        pool = df[df["label_7"] == cls]
        k    = min(per_class, len(pool))
        parts.append(pool.sample(k, random_state=int(rng.integers(0, 9999))))
    result = pd.concat(parts)
    if len(result) < n:
        remaining = df[~df.index.isin(result.index)]
        extra     = remaining.sample(min(n - len(result), len(remaining)), random_state=seed)
        result    = pd.concat([result, extra])
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)

df = stratified_sample(df_full, SAMPLE_SIZE, RANDOM_SEED)
metric = DriftMetric()

print(f"[Exp 1] {len(df)} sentences (stratified)")
print(df["label_7"].value_counts().to_string())

print("\n[Exp 1] Checking source classifier agreement ...")
agreement_count = 0
for _, row in tqdm(df.iterrows(), total=len(df), desc="source check"):
    ref = get_reference_dist(row["text"], row["label_7"])
    if ref["classifier_agrees"]:
        agreement_count += 1
print(f"  Classifier-gold agreement: {agreement_count}/{len(df)} ({agreement_count/len(df)*100:.1f}%)")

# ================================================================
# CONTROL — CLASSIFIER SELF-CONSISTENCY
# ================================================================

def run_control(df, n):
    subset = df.sample(n, random_state=0).reset_index(drop=True)
    rows   = []
    print(f"\n[Exp 1] CONTROL — classifier self-consistency (n={n})")
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="control"):
        x     = row["text"]
        label = row["label_7"]
        ref   = get_reference_dist(x, label)
        dist2 = get_emotion_dist(x)          # second pass, same input
        drift = metric.compute(ref["dist"], dist2)
        sim   = sem_sim(x, x)
        rows.append({"idx": int(row.name), "text": x, "label": label,
                     "semantic": sim, **drift,
                     "classifier_agrees": ref["classifier_agrees"],
                     "source_confidence": ref["source_confidence"]})
    return rows, aggregate(rows)

# ================================================================
# BASELINE TRANSLATION
# ================================================================

def run_baseline(lang_name, lang_str):
    print(f"\n[Exp 1] BASELINE — {lang_name}")
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=lang_name):
        x     = row["text"]
        label = row["label_7"]

        ref      = get_reference_dist(x, label)
        ref_dist = ref["dist"]

        translated = generate_translation(x, label, lang_str)
        time.sleep(RATE_DELAY)
        if translated is None:
            translated = x

        back = generate_back_translation(translated, lang_name)
        time.sleep(RATE_DELAY)
        if back is None:
            back = x

        back_dist = get_emotion_dist(back)
        drift     = metric.compute(ref_dist, back_dist)
        sim       = sem_sim(x, back)

        rows.append({
            "idx": int(row.name), "text": x, "label": label,
            "classifier_agrees": ref["classifier_agrees"],
            "source_confidence": ref["source_confidence"],
            "translated": translated, "back": back,
            "semantic": sim, **drift,
        })

    return rows, aggregate(rows), per_emotion_breakdown(rows)

def per_emotion_breakdown(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["label"]].append(r)
    result = {}
    for emotion, sub in sorted(buckets.items()):
        result[emotion] = {
            "n":                     len(sub),
            "flip_rate":             float(np.mean([r["D_flip"]  for r in sub])),
            "mean_D_JS":             float(np.mean([r["D_JS"]    for r in sub])),
            "mean_sem":              float(np.mean([r["semantic"] for r in sub])),
            "catast_rate":           float(np.mean(
                [(r["semantic"] < 0.30) or (r["D_JS"] > 0.50) for r in sub])),
            "classifier_agree_rate": float(np.mean([r["classifier_agrees"] for r in sub])),
        }
    return result

def agreement_split(rows):
    agreed    = [r for r in rows if r["classifier_agrees"]]
    disagreed = [r for r in rows if not r["classifier_agrees"]]
    return {
        "classifier_agreed":    aggregate(agreed)    if agreed    else {},
        "classifier_disagreed": aggregate(disagreed) if disagreed else {},
    }

# ================================================================
# MAIN
# ================================================================

all_results = {}

ctrl_rows, ctrl_agg = run_control(df, CONTROL_SIZE)
all_results["control"] = {"aggregate": ctrl_agg, "raw": ctrl_rows}
print_agg_table("CONTROL — Classifier Self-Consistency (noise floor)", ctrl_agg)
print("  Expected: D_JS ≈ 0.0000, flip_rate ≈ 0.00%, semantic = 1.0000")

for lang_name, lang_str in TARGET_LANGUAGES.items():
    rows, agg, emo = run_baseline(lang_name, lang_str)
    asplit         = agreement_split(rows)
    all_results[lang_name] = {"aggregate": agg, "per_emotion": emo,
                               "agreement_split": asplit, "raw": rows}
    print_agg_table(f"BASELINE — {lang_name}", agg)
    save_json(RESULTS_DIR / f"exp1_{lang_name.lower()}_raw.json", rows)

# Net drift
print("\n--- Net Drift (language − noise floor) ---")
print(f"  Control noise floor  D_JS = {ctrl_agg['mean_D_JS']:.4f}")
for lang in TARGET_LANGUAGES:
    a = all_results[lang]["aggregate"]
    print(f"  {lang:<10}  D_JS = {a['mean_D_JS']:.4f}  "
          f"net = +{a['mean_D_JS'] - ctrl_agg['mean_D_JS']:.4f}  "
          f"flip = {a['flip_rate']*100:.1f}%  "
          f"net_flip = +{(a['flip_rate'] - ctrl_agg['flip_rate'])*100:.1f}%")

# Paper table
print("\n" + "=" * 72)
print("TABLE 1 — BASELINE EMOTIONAL DRIFT")
print("Reference: classifier soft distribution on original text")
print("=" * 72)
print(f"{'Condition':<14} {'Sem.Sim':>9} {'D_JS':>9} {'Flip%':>9} {'Catast%':>9}")
print("-" * 72)
c = ctrl_agg
print(f"{'Control':14} {c['mean_semantic']:9.4f} {c['mean_D_JS']:9.4f} "
      f"{c['flip_rate']*100:9.2f} {c['catastrophic_rate']*100:9.2f}  ← noise floor")
for lang in TARGET_LANGUAGES:
    a = all_results[lang]["aggregate"]
    print(f"{lang:<14} {a['mean_semantic']:9.4f} {a['mean_D_JS']:9.4f} "
          f"{a['flip_rate']*100:9.2f} {a['catastrophic_rate']*100:9.2f}")
print("=" * 72)

# Save summary
summary = {
    "config": {
        "sample_size": SAMPLE_SIZE, "control_size": CONTROL_SIZE,
        "random_seed": RANDOM_SEED, "languages": list(TARGET_LANGUAGES.keys()),
        "reference_method": "classifier_soft_distribution",
    },
    "control": ctrl_agg,
    "languages": {
        lang: {
            "aggregate":       all_results[lang]["aggregate"],
            "per_emotion":     all_results[lang]["per_emotion"],
            "agreement_split": all_results[lang]["agreement_split"],
        }
        for lang in TARGET_LANGUAGES
    },
}
save_json(RESULTS_DIR / "exp1_summary.json", summary)

print("\n[Exp 1 COMPLETE]")
