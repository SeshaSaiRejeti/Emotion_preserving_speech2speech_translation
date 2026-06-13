"""
shared.py
=========
Centralised utilities for all experiment scripts.

Provides:
  - DriftMetric          : 4-dimensional emotion drift measurement
  - get_emotion_dist()   : roberta-base-go_emotions → 7-class vector
  - get_reference_dist() : CORRECT reference distribution (classifier on original text)
  - sem_sim()            : cosine similarity via all-mpnet-base-v2
  - Groq generation wrappers (paraphrase, translation, back-translation)
  - Penalty function variants (canonical + 4 ablation variants)
  - Iterative regeneration feedback builder
  - Result aggregation and JSON helpers

All heavy models are lazy-loaded once and cached as module-level singletons.

IMPORTANT — REFERENCE DISTRIBUTION DESIGN NOTE
----------------------------------------------
The reference distribution for drift measurement is ALWAYS obtained by running
the emotion classifier on the original text:

    ref_dist = get_emotion_dist(original_text)

NOT constructed as a one-hot vector from the gold label:

    # WRONG — never do this
    gold_dist = np.zeros(7)
    gold_dist[EMOTION_ORDER.index(gold_label)] = 1.0

Reason: the emotion classifier (roberta-base-go_emotions) never produces
one-hot distributions. It always outputs soft distributions. Computing
JS divergence between a one-hot and any soft distribution produces
artificially high divergence even when the translation is perfect.
Using classifier output vs classifier output (both soft) gives a fair,
calibrated comparison.

The gold label is still used to:
  (a) verify the classifier agrees with the gold label on the source text
  (b) filter sentences where classifier disagrees (low-confidence source)
  (c) per-emotion breakdown tables

get_reference_dist(text, gold_label) handles this correctly and returns
the classifier distribution along with an agreement flag.
"""

import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================================================================
# CONSTANTS
# ================================================================

EMOTION_ORDER = ["anger", "disgust", "fear", "sadness", "joy", "surprise", "neutral"]
LLM_MODEL     = "llama-3.1-8b-instant"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

MAPPING_27_TO_7 = {
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "sadness": "sadness", "disappointment": "sadness",
    "grief": "sadness", "remorse": "sadness",
    "joy": "joy", "amusement": "joy", "approval": "joy",
    "optimism": "joy", "gratitude": "joy", "love": "joy",
    "pride": "joy", "relief": "joy", "excitement": "joy", "caring": "joy",
    "surprise": "surprise", "realization": "surprise",
    "neutral": "neutral",
}

# ================================================================
# RESULTS DIRECTORY
# ================================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ================================================================
# DRIFT METRIC
# ================================================================

class DriftMetric:
    """
    Four-dimensional emotion drift measurement.

    All metrics operate on 7-class emotion probability distributions.
    Distributions are automatically normalised to the probability simplex.

    Metrics
    -------
    D_JS      : Jensen-Shannon divergence (distributional drift)
    D_conf    : Absolute peak confidence degradation
    D_flip    : Binary dominant-emotion flip indicator
    D_entropy : Entropy shift (positive = weakened, negative = exaggerated)
    """

    def __init__(self, epsilon: float = 1e-12):
        self.eps = epsilon

    def _norm(self, P: np.ndarray, Q: np.ndarray):
        P = np.asarray(P, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        if P.shape != (7,) or Q.shape != (7,):
            raise ValueError("Distributions must be length 7.")
        if np.any(P < 0) or np.any(Q < 0):
            raise ValueError("Probabilities must be non-negative.")
        P = P / (P.sum() + self.eps)
        Q = Q / (Q.sum() + self.eps)
        return P, Q

    def _kl(self, P: np.ndarray, Q: np.ndarray) -> float:
        return float(np.sum(P * np.log((P + self.eps) / (Q + self.eps))))

    def js(self, P: np.ndarray, Q: np.ndarray) -> float:
        M = 0.5 * (P + Q)
        return 0.5 * self._kl(P, M) + 0.5 * self._kl(Q, M)

    def H(self, P: np.ndarray) -> float:
        return float(-np.sum(P * np.log(P + self.eps)))

    def compute(self, P, Q) -> dict:
        """
        Parameters
        ----------
        P : source emotion distribution (7-vector)
        Q : generated/translated emotion distribution (7-vector)

        Returns
        -------
        dict with keys: D_JS, D_conf, D_flip, D_entropy
        """
        P, Q = self._norm(P, Q)
        return {
            "D_JS":      float(self.js(P, Q)),
            "D_conf":    float(abs(np.max(P) - np.max(Q))),
            "D_flip":    int(np.argmax(P) != np.argmax(Q)),
            "D_entropy": float(self.H(Q) - self.H(P)),
        }

# ================================================================
# LAZY MODEL SINGLETONS
# ================================================================

_cache: dict = {}


def _emotion_models():
    if "emotion" not in _cache:
        print("[shared] Loading SamLowe/roberta-base-go_emotions ...")
        tok = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        mdl = AutoModelForSequenceClassification.from_pretrained(
            "SamLowe/roberta-base-go_emotions"
        ).to(DEVICE)
        mdl.eval()
        _cache["emotion"] = (tok, mdl)
    return _cache["emotion"]


def _embedder():
    if "embed" not in _cache:
        print("[shared] Loading all-mpnet-base-v2 ...")
        _cache["embed"] = SentenceTransformer("all-mpnet-base-v2")
    return _cache["embed"]


def _groq():
    if "groq" not in _cache:
        _cache["groq"] = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _cache["groq"]

# ================================================================
# EMOTION DISTRIBUTION
# ================================================================

def get_emotion_dist(text: str) -> np.ndarray:
    """
    Returns a 7-class emotion probability distribution for text.
    Uses SamLowe/roberta-base-go_emotions with 27→7 class mapping.
    """
    tok, mdl = _emotion_models()
    labels   = mdl.config.id2label
    inp = tok(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(mdl(**inp).logits, dim=-1).cpu().numpy()[0]

    dist = np.zeros(7)
    for idx, p in enumerate(probs):
        lbl = labels[idx]
        if lbl in MAPPING_27_TO_7:
            dist[EMOTION_ORDER.index(MAPPING_27_TO_7[lbl])] += p

    return dist / (dist.sum() + 1e-12)


def get_reference_dist(text: str, gold_label: str) -> dict:
    """
    Get the CORRECT reference distribution for drift measurement.

    Runs the classifier on the original text and returns its soft distribution.
    Also returns whether the classifier agrees with the gold label,
    which can be used to filter low-confidence source sentences.

    Parameters
    ----------
    text       : original source sentence
    gold_label : gold emotion label from GoEmotions dataset

    Returns
    -------
    dict with:
      dist              : 7-class soft distribution (use this as reference)
      predicted_label   : classifier top prediction on source text
      gold_label        : the gold label passed in
      classifier_agrees : True if classifier top prediction matches gold
      source_confidence : probability mass on the gold class
    """
    dist            = get_emotion_dist(text)
    predicted_idx   = int(np.argmax(dist))
    predicted_label = EMOTION_ORDER[predicted_idx]
    source_conf     = float(dist[EMOTION_ORDER.index(gold_label)])

    return {
        "dist":               dist,
        "predicted_label":    predicted_label,
        "gold_label":         gold_label,
        "classifier_agrees":  predicted_label == gold_label,
        "source_confidence":  source_conf,
    }

# ================================================================
# SEMANTIC SIMILARITY
# ================================================================

def sem_sim(x: str, y: str) -> float:
    """Cosine similarity between sentence embeddings of x and y."""
    emb = _embedder().encode([x, y])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

# ================================================================
# PENALTY FUNCTIONS
# ================================================================
# Canonical penalty is used for strategy comparison and integration.
# Ablation variants are used in exp4_ablation.py.

def penalty_full(sim: float, drift: dict) -> float:
    """Canonical 4-term penalty (used in all strategies)."""
    return (
        max(0.0, 0.75 - sim)
        + drift["D_JS"]
        + drift["D_flip"] * 1.0
        + max(0.0, drift["D_entropy"] - 0.15)
    )

def penalty_sem_only(sim: float, drift: dict) -> float:
    return max(0.0, 0.75 - sim)

def penalty_js_only(sim: float, drift: dict) -> float:
    return drift["D_JS"]

def penalty_flip_only(sim: float, drift: dict) -> float:
    return float(drift["D_flip"])

def penalty_no_entropy(sim: float, drift: dict) -> float:
    return max(0.0, 0.75 - sim) + drift["D_JS"] + drift["D_flip"] * 1.0

# Registry used by exp4_ablation.py
PENALTY_VARIANTS = {
    "full":         penalty_full,
    "sem_only":     penalty_sem_only,
    "js_only":      penalty_js_only,
    "flip_only":    penalty_flip_only,
    "no_entropy":   penalty_no_entropy,
}

def satisfies_constraints(sim: float, drift: dict) -> bool:
    return (
        sim               >= 0.75
        and drift["D_JS"]      <= 0.10
        and drift["D_flip"]    == 0
        and drift["D_entropy"] <= 0.15
    )

# ================================================================
# GROQ CALL WRAPPER
# ================================================================

def call_groq(
    system:      str,
    user:        str,
    temperature: float,
    extra_msgs:  list | None = None,
    retries:     int   = 3,
    backoff:     float = 2.0,
) -> str | None:
    """
    Single Groq API call with exponential backoff on failure.

    Parameters
    ----------
    extra_msgs : optional list of additional messages appended after user
                 (used for iterative regeneration to pass conversation history)

    Returns None on total failure (caller should fall back to original text).
    """
    client = _groq()
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    if extra_msgs:
        messages.extend(extra_msgs)

    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=temperature,
                messages=messages,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            wait = backoff * (2 ** attempt)
            print(f"  [groq error attempt {attempt+1}/{retries}]: {e} — waiting {wait:.0f}s")
            if attempt < retries - 1:
                time.sleep(wait)
    return None

# ================================================================
# PROMPT TEMPLATES
# ================================================================

_PARAPHRASE_SYS = """
You are a controlled transformation engine.

Your task is STRICT semantic-preserving paraphrase generation.

Rules:
- You must preserve ALL information from the original sentence.
- You must NOT add new details.
- You must NOT remove any content.
- You must NOT summarize.
- You must NOT generalize.
- You must NOT explain.
- You must preserve the dominant emotion and its intensity.
- Use minimal lexical substitutions only.
- Keep sentence structure as similar as possible.
- Do not introduce extra clauses.
- Do not change speaker intent.
- Return only the rewritten sentence.
- Do not include any preamble, explanation, or quotation marks.
- The first character of your response must be the first character of the sentence.
"""

_TRANSLATION_SYS = """
You are a constrained speech translation engine.

Translate the given English sentence into the specified target language.

Rules:
- Preserve exact meaning.
- Preserve emotional intent implicitly through natural word choice.
- Use idiomatic expressions a native speaker would use.
- Do NOT add explanations, commentary, or formatting.
- Do NOT include quotation marks or preamble.
- Output ONLY the translated sentence.
- The first character of your response must be the first character of the translation.
"""

_BACK_TRANS_SYS = """
You are a translation engine.

Translate the given sentence back to English.

Rules:
- Preserve exact meaning.
- Output ONLY the English translation.
- No preamble, no explanation, no quotation marks.
"""

_JUDGE_SYS = """
You are an emotion-preservation evaluator.

Rate how well the paraphrase preserves the emotional tone of the original sentence.

Scoring:
  5 = emotion perfectly preserved
  4 = emotion mostly preserved, minor shift
  3 = emotion partially preserved, noticeable shift
  2 = emotion weakened or changed significantly
  1 = emotion completely lost or reversed

Return ONLY the single integer. No explanation. No preamble.
"""


def _para_user(text, emotion):
    return (
        f'Original sentence:\n"{text}"\n\n'
        f"Target emotion: {emotion}\n\n"
        "Rewrite the sentence as a paraphrase.\n\n"
        "STRICT REQUIREMENTS:\n"
        "- Keep meaning exactly the same.\n"
        "- Keep emotional intensity exactly the same.\n"
        "- Make minimal word changes.\n"
        "- Do not add, remove, or restructure content.\n"
        "- Keep the dominant emotion unchanged.\n"
        "- Do not soften or exaggerate emotion.\n"
        "- Keep sentence length approximately the same.\n\n"
        "Return only the paraphrased sentence."
    )

def _trans_user(text, emotion, lang):
    return (
        f'Translate to {lang}.\n'
        f'Source: "{text}"\n'
        f"Emotion context: {emotion}\n\n"
        f"Return only the {lang} translation."
    )

def _back_user(text, lang):
    return (
        f'Translate this {lang} sentence to English.\n'
        f'Sentence: "{text}"\n\n'
        "Return only the English translation."
    )

def _judge_user(original, paraphrase, emotion):
    return (
        f'Original sentence: "{original}"\n'
        f'Gold emotion: {emotion}\n'
        f'Paraphrase: "{paraphrase}"\n\n'
        "Rate emotion preservation 1-5. Return only the integer."
    )

# ================================================================
# GENERATION FUNCTIONS
# ================================================================

def generate_paraphrase(text: str, emotion: str, temperature: float = 0.1) -> str:
    out = call_groq(_PARAPHRASE_SYS, _para_user(text, emotion), temperature)
    return out if out else text

def generate_translation(text: str, emotion: str, lang: str, temperature: float = 0.1) -> str:
    out = call_groq(_TRANSLATION_SYS, _trans_user(text, emotion, lang), temperature)
    return out if out else text

def generate_back_translation(text: str, lang: str, temperature: float = 0.0) -> str:
    out = call_groq(_BACK_TRANS_SYS, _back_user(text, lang), temperature)
    return out if out else text

def llm_judge_score(original: str, paraphrase: str, emotion: str) -> int | None:
    """
    Uses the LLM as a judge to rate emotion preservation 1-5.
    Returns None if the response cannot be parsed as an integer.
    """
    out = call_groq(_JUDGE_SYS, _judge_user(original, paraphrase, emotion), temperature=0.0)
    if out is None:
        return None
    try:
        score = int(out.strip())
        if 1 <= score <= 5:
            return score
    except ValueError:
        pass
    # Try extracting digit from longer response
    import re
    m = re.search(r"\b([1-5])\b", out)
    if m:
        return int(m.group(1))
    return None

# ================================================================
# ITERATIVE REGENERATION FEEDBACK BUILDER
# ================================================================

def build_feedback(sim: float, drift: dict) -> str:
    lines = ["Your previous output violated constraints:\n"]

    if sim < 0.50:
        lines.append("- Meaning changed significantly. You altered content.")
    elif sim < 0.65:
        lines.append("- Meaning drifted noticeably. Reduce modifications.")

    if drift["D_flip"] == 1:
        lines.append("- You changed the dominant emotion. This is strictly not allowed.")

    if drift["D_JS"] > 0.30:
        lines.append("- Emotional tone changed strongly.")
    elif drift["D_JS"] > 0.15:
        lines.append("- Emotional tone shifted slightly.")

    if drift["D_entropy"] > 0.30:
        lines.append("- Emotional intensity weakened significantly.")
    elif drift["D_entropy"] < -0.20:
        lines.append("- Emotional intensity was exaggerated.")

    lines.append(
        "\nRewrite again correcting ALL issues above. "
        "Return only the improved sentence."
    )
    return "\n".join(lines)

# ================================================================
# ITERATIVE REGENERATION  (Strategy A)
# ================================================================

def run_iterative(
    text:        str,
    emotion:     str,
    gold_dist:   np.ndarray,
    metric:      DriftMetric,
    max_attempts: int = 4,
    temperature: float = 0.1,
    rate_delay:  float = 1.0,
) -> dict:
    """
    Iterative regeneration with metric-guided feedback.

    Returns
    -------
    dict with keys:
      attempts  : list of per-attempt records (text, semantic, D_JS, D_flip, D_entropy)
      best      : metrics of the lowest-penalty attempt
      best_text : text of the best attempt
    """
    current = generate_paraphrase(text, emotion, temperature)
    time.sleep(rate_delay)

    conv_history = []  # tracks (user_prompt, assistant_output) pairs
    attempts     = []
    best_p       = float("inf")
    best_record  = None
    best_text    = current

    for attempt in range(max_attempts):
        dist  = get_emotion_dist(current)
        drift = metric.compute(gold_dist, dist)
        sim   = sem_sim(text, current)
        p     = penalty_full(sim, drift)

        record = {
            "attempt":   attempt,
            "text":      current,
            "semantic":  sim,
            **drift,
            "penalty":   p,
        }
        attempts.append(record)

        if p < best_p:
            best_p      = p
            best_record = {k: v for k, v in record.items() if k != "attempt" and k != "text"}
            best_text   = current

        if satisfies_constraints(sim, drift):
            break  # early exit — constraints satisfied

        if attempt < max_attempts - 1:
            feedback = build_feedback(sim, drift)
            # Build corrective continuation
            new_output = call_groq(
                system      = _PARAPHRASE_SYS,
                user        = _para_user(text, emotion),
                temperature = 0.0,
                extra_msgs  = conv_history + [
                    {"role": "assistant", "content": current},
                    {"role": "user",      "content": feedback},
                ],
            )
            time.sleep(rate_delay)

            if new_output is None:
                break

            # Extend conversation history for next round
            conv_history.append({"role": "assistant", "content": current})
            conv_history.append({"role": "user",      "content": feedback})
            current = new_output

    return {
        "attempts":   attempts,
        "best":       best_record,
        "best_text":  best_text,
    }

# ================================================================
# CANDIDATE GENERATION  (Strategy B)
# ================================================================

def run_candidate(
    text:         str,
    emotion:      str,
    gold_dist:    np.ndarray,
    metric:       DriftMetric,
    n_candidates: int   = 5,
    temperature:  float = 0.7,
    rate_delay:   float = 1.0,
) -> dict:
    """
    Independent candidate generation + best-of-N selection.

    Returns
    -------
    dict with keys:
      candidates : list of per-candidate records
      best       : metrics of the best (lowest-penalty) candidate
      best_text  : text of the best candidate
      early_exit : whether an early exit occurred
    """
    candidates  = []
    best_p      = float("inf")
    best_record = None
    best_text   = text
    early_exit  = False

    for i in range(n_candidates):
        cand = generate_paraphrase(text, emotion, temperature)
        time.sleep(rate_delay)

        if cand is None:
            cand = text

        dist  = get_emotion_dist(cand)
        drift = metric.compute(gold_dist, dist)
        sim   = sem_sim(text, cand)
        p     = penalty_full(sim, drift)

        record = {
            "candidate": i,
            "text":      cand,
            "semantic":  sim,
            **drift,
            "penalty":   p,
        }
        candidates.append(record)

        if p < best_p:
            best_p      = p
            best_record = {k: v for k, v in record.items()
                           if k not in ("candidate", "text")}
            best_text   = cand

        if satisfies_constraints(sim, drift):
            early_exit = True
            break

    return {
        "candidates":  candidates,
        "best":        best_record,
        "best_text":   best_text,
        "early_exit":  early_exit,
    }

# ================================================================
# AGGREGATION
# ================================================================

def aggregate(rows: list[dict]) -> dict:
    """
    Aggregate per-row metric dicts into summary statistics.
    Input rows must have keys: semantic, D_JS, D_entropy, D_flip.
    """
    s  = np.array([r["semantic"]  for r in rows], dtype=float)
    js = np.array([r["D_JS"]      for r in rows], dtype=float)
    de = np.array([r["D_entropy"] for r in rows], dtype=float)
    fl = np.array([r["D_flip"]    for r in rows], dtype=float)

    return {
        "n":                 len(rows),
        "mean_semantic":     float(np.mean(s)),
        "std_semantic":      float(np.std(s)),
        "mean_D_JS":         float(np.mean(js)),
        "median_D_JS":       float(np.median(js)),
        "p95_D_JS":          float(np.percentile(js, 95)),
        "mean_D_entropy":    float(np.mean(de)),
        "flip_rate":         float(np.mean(fl)),
        "catastrophic_rate": float(np.mean((s < 0.30) | (js > 0.50))),
    }

# ================================================================
# I/O HELPERS
# ================================================================

def save_json(path: str | Path, data) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"[saved] {path}")

def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_agg_table(title: str, agg: dict) -> None:
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"{'='*58}")
    print(f"  n                   : {agg['n']}")
    print(f"  Mean semantic sim   : {agg['mean_semantic']:.4f}  (± {agg['std_semantic']:.4f})")
    print(f"  Mean D_JS           : {agg['mean_D_JS']:.4f}")
    print(f"  Median D_JS         : {agg['median_D_JS']:.4f}")
    print(f"  95th pct D_JS       : {agg['p95_D_JS']:.4f}")
    print(f"  Mean D_entropy      : {agg['mean_D_entropy']:.4f}")
    print(f"  Flip rate           : {agg['flip_rate']*100:.2f}%")
    print(f"  Catastrophic rate   : {agg['catastrophic_rate']*100:.2f}%")
    print(f"{'='*58}")