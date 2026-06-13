"""
drift_eval.py — Drift Evaluation Utilities (German → English)
=============================================================

Updated for EmoDB German → English pipeline.

Key changes vs. original:
  1. Translation direction: German → English (not English → target)
  2. Source emotion distribution: derived as one-hot from gold label
     (passed in by caller) — no text classifier on German source text
  3. Emotion classification: run directly on English translation output
     (no back-translation needed since target IS English)
  4. Semantic similarity: paraphrase-multilingual-mpnet-base-v2
     (cross-lingual; handles German source ↔ English translation)
  5. back_translate() retained for potential future use but NOT called
     in the main evaluation loop

All models are lazy-loaded and cached (singleton pattern).
"""

import os
import time
import numpy as np
import requests
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================================================================
# CONFIG
# ================================================================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL    = "llama-3.1-8b-instant"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

SOURCE_LANG  = "German"    # EmoDB
TARGET_LANG  = "English"

# j-hartmann native 7-class output — fixed order from model id2label
# {0:anger, 1:disgust, 2:fear, 3:joy, 4:neutral, 5:sadness, 6:surprise}
EMOTION_ORDER = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# ================================================================
# LAZY SINGLETONS
# ================================================================

_cache: dict = {}


def _groq_key() -> str:
    return os.getenv("GROQ_API_KEY")

def _classifier():
    """j-hartmann/emotion-english-distilroberta-base — 7 native classes."""
    if "clf" not in _cache:
        print("[drift_eval] Loading j-hartmann/emotion-english-distilroberta-base ...")
        name = "j-hartmann/emotion-english-distilroberta-base"
        tok  = AutoTokenizer.from_pretrained(name)
        mdl  = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE)
        mdl.eval()
        labels = [mdl.config.id2label[i] for i in range(len(mdl.config.id2label))]
        # Verify label order matches EMOTION_ORDER
        if labels != EMOTION_ORDER:
            print(f"[drift_eval] WARNING: model label order {labels} differs from EMOTION_ORDER {EMOTION_ORDER}")
        _cache["clf"]    = (tok, mdl)
        _cache["labels"] = labels
        print(f"[drift_eval] Classifier ready. Labels: {labels}")
    return _cache["clf"], _cache["labels"]


def _embedder():
    """
    paraphrase-multilingual-mpnet-base-v2 — supports 50+ languages.
    Used for cross-lingual semantic similarity (German source ↔ English translation).
    """
    if "emb" not in _cache:
        print("[drift_eval] Loading paraphrase-multilingual-mpnet-base-v2 ...")
        _cache["emb"] = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        print("[drift_eval] Embedder ready.")
    return _cache["emb"]


# ================================================================
# EMOTION DISTRIBUTION
# ================================================================

def get_emotion_dist(text: str) -> tuple[np.ndarray, list[str]]:
    """
    Classify English text with j-hartmann.
    Returns (probability distribution, label_order).
    ONLY call this on English text.
    """
    (tok, mdl), labels = _classifier()
    inp = tok(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(mdl(**inp).logits, dim=-1).cpu().numpy()[0]

    dist = probs / (probs.sum() + 1e-12)
    return dist, labels


def make_source_dist(emotion_label: str, label_order: list[str]) -> np.ndarray:
    """
    Derive a one-hot source distribution from a known emotion label.
    This is the correct approach when gold labels are available (EmoDB),
    avoiding the need to classify non-English source text.

    Args:
        emotion_label: e.g. "anger", "joy", "neutral"
        label_order:   list of emotion names in index order (from j-hartmann)

    Returns:
        7-dim one-hot numpy array aligned to label_order
    """
    dist = np.zeros(len(label_order), dtype=np.float64)

    norm_label = emotion_label.strip().lower()
    if norm_label in label_order:
        dist[label_order.index(norm_label)] = 1.0
    else:
        # Fallback: map SUPERB 4-class short codes to standard names
        fallback = {
            "ang": "anger",
            "hap": "joy",
            "neu": "neutral",
            "sad": "sadness",
        }
        mapped = fallback.get(norm_label)
        if mapped and mapped in label_order:
            dist[label_order.index(mapped)] = 1.0
        else:
            # Last resort: uniform (unknown emotion)
            dist[:] = 1.0 / len(label_order)
            print(f"[drift_eval] WARNING: unknown emotion label '{emotion_label}' — using uniform dist")

    return dist


def dominant_emotion(dist: np.ndarray, label_order: list[str]) -> str:
    return label_order[int(np.argmax(dist))]


# ================================================================
# SEMANTIC SIMILARITY (CROSS-LINGUAL)
# ================================================================

def sem_sim(source: str, target: str) -> float:
    """
    Cross-lingual semantic similarity using multilingual sentence embeddings.
    Correctly handles German source ↔ English target comparison.
    """
    emb = _embedder().encode([source, target])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


# ================================================================
# DRIFT METRIC
# ================================================================

class DriftMetric:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def _kl(self, P: np.ndarray, Q: np.ndarray) -> float:
        return float(np.sum(P * np.log((P + self.eps) / (Q + self.eps))))

    def js(self, P: np.ndarray, Q: np.ndarray) -> float:
        M = 0.5 * (P + Q)
        return 0.5 * self._kl(P, M) + 0.5 * self._kl(Q, M)

    def H(self, P: np.ndarray) -> float:
        return float(-np.sum(P * np.log(P + self.eps)))

    def compute(self, source_dist: np.ndarray, gen_dist: np.ndarray) -> dict:
        """
        source_dist: one-hot from gold label (make_source_dist)
        gen_dist:    j-hartmann output on English translation
        Both must be aligned to the same label_order.
        """
        P = source_dist / (source_dist.sum() + self.eps)
        Q = gen_dist    / (gen_dist.sum()    + self.eps)

        return {
            "D_JS":      float(self.js(P, Q)),
            "D_conf":    float(abs(np.max(P) - np.max(Q))),
            "D_flip":    int(np.argmax(P) != np.argmax(Q)),
            "D_entropy": float(self.H(Q) - self.H(P)),
        }


_metric = DriftMetric()


# ================================================================
# PENALTY + CONSTRAINTS
# ================================================================

def penalty(sim: float, drift: dict) -> float:
    return (
        max(0.0, 0.75 - sim)
        + drift["D_JS"]
        + drift["D_flip"] * 1.0
        + max(0.0, drift["D_entropy"] - 0.15)
    )


def satisfies_constraints(sim: float, drift: dict) -> bool:
    return (
        sim >= 0.75
        and drift["D_JS"]      <= 0.10
        and drift["D_flip"]    == 0
        and drift["D_entropy"] <= 0.15
    )


# ================================================================
# GROQ CALL (WITH RETRY + EXPONENTIAL BACKOFF)
# ================================================================

def call_groq(
    system:      str,
    messages:    list[dict],
    temperature: float = 0.0,
    retries:     int   = 3,
    backoff:     float = 3.0,
) -> str | None:
    full = [{"role": "system", "content": system}] + messages

    for attempt in range(retries):
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {_groq_key()}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       LLM_MODEL,
                    "temperature": temperature,
                    "messages":    full,
                    "max_tokens":  1024,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            else:
                print(f"  [groq {attempt+1}/{retries}] HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"  [groq {attempt+1}/{retries}] Exception: {e}")

        if attempt < retries - 1:
            wait = backoff * (2 ** attempt)
            print(f"  [groq] Retrying in {wait:.0f}s ...")
            time.sleep(wait)

    return None


# ================================================================
# TRANSLATION PROMPTS  (German → English)
# ================================================================

def _translation_system() -> str:
    return (
        f"You are a constrained speech translation engine.\n\n"
        f"Translate the given {SOURCE_LANG} sentence into {TARGET_LANG}.\n\n"
        "Rules:\n"
        "- Preserve exact meaning.\n"
        "- Convey emotional intent implicitly through natural word choice.\n"
        "- Use idiomatic expressions a native English speaker would use.\n"
        "- Do NOT add explanations, commentary, or formatting.\n"
        "- Do NOT include quotation marks or preamble.\n"
        "- Output ONLY the translated sentence.\n"
        "- The first character of your response must be the first character of the translation."
    )


def _translation_user(text: str, emotion: str) -> str:
    return (
        f'Source sentence ({SOURCE_LANG}): "{text}"\n'
        f"Emotion context: {emotion}\n\n"
        "Return only the English translation."
    )


# ================================================================
# FEEDBACK BUILDER (ITERATIVE STRATEGY)
# ================================================================

def build_feedback(sim: float, drift: dict) -> str:
    lines = ["Your previous translation has issues:\n"]

    if sim < 0.50:
        lines.append("- The meaning changed significantly. Stay much closer to the source.")
    elif sim < 0.65:
        lines.append("- The meaning drifted noticeably. Reduce paraphrasing.")

    if drift["D_flip"] == 1:
        lines.append("- The dominant emotional tone changed. Preserve the original emotion.")

    if drift["D_JS"] > 0.30:
        lines.append("- The emotional tone shifted strongly.")
    elif drift["D_JS"] > 0.15:
        lines.append("- The emotional tone shifted slightly.")

    if drift["D_entropy"] > 0.30:
        lines.append("- The emotional intensity weakened significantly.")
    elif drift["D_entropy"] < -0.20:
        lines.append("- The emotional intensity was exaggerated.")

    lines.append(
        "\nRetranslate from German, correcting ALL the above issues. "
        "Return only the improved English translation."
    )
    return "\n".join(lines)


# ================================================================
# EVALUATE A SINGLE TRANSLATION
# ================================================================

def evaluate_translation(
    source_text:  str,          # German source text
    translation:  str,          # English translation to evaluate
    source_dist:  np.ndarray,   # one-hot from gold label (already computed by caller)
    label_order:  list[str],    # j-hartmann label order
    delay:        float = 0.5,
) -> dict:
    """
    Evaluate English translation against German source.

    Since target is English:
    - NO back-translation needed
    - Classify translation directly with j-hartmann
    - Semantic similarity via multilingual embedder (cross-lingual)

    Returns: semantic, D_JS, D_flip, D_entropy, D_conf, penalty
    """
    # Cross-lingual semantic similarity
    sim = sem_sim(source_text, translation)

    # Emotion distribution on English translation (direct — no back-translation)
    gen_dist, gen_labels = get_emotion_dist(translation)

    # Drift metrics
    drift = _metric.compute(source_dist, gen_dist)
    p     = penalty(sim, drift)

    return {
        "semantic":  sim,
        "D_JS":      drift["D_JS"],
        "D_flip":    drift["D_flip"],
        "D_entropy": drift["D_entropy"],
        "D_conf":    drift["D_conf"],
        "penalty":   p,
    }


# ================================================================
# BASELINE TRANSLATION (SINGLE PASS, NO OPTIMIZATION)
# ================================================================

def translate_baseline(
    text:    str,
    emotion: str,
    delay:   float = 0.8,
) -> str:
    """Single-pass German → English translation with no optimization."""
    result = call_groq(
        system=_translation_system(),
        messages=[{"role": "user", "content": _translation_user(text, emotion)}],
        temperature=0.0,
    )
    time.sleep(delay)
    return result if result else text


# ================================================================
# STRATEGY A — ITERATIVE REGENERATION
# ================================================================

def run_iterative(
    source_text:  str,
    emotion:      str,
    source_dist:  np.ndarray,
    label_order:  list[str],
    max_attempts: int   = 3,
    delay:        float = 0.8,
) -> dict:
    """
    Iterative German → English translation with metric-guided feedback.

    Returns:
      best_translation: str
      best_metrics:     dict
      attempts:         list of per-attempt records
    """
    system      = _translation_system()
    user_prompt = _translation_user(source_text, emotion)
    conv_history: list[dict] = []

    # Initial generation
    current = call_groq(
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.1,
    )
    time.sleep(delay)
    if current is None:
        current = source_text

    best_p       = float("inf")
    best_trans   = current
    best_metrics = None
    attempts     = []

    for attempt in range(max_attempts):
        eval_result = evaluate_translation(
            source_text=source_text,
            translation=current,
            source_dist=source_dist,
            label_order=label_order,
            delay=delay,
        )
        eval_result["attempt"]     = attempt
        eval_result["translation"] = current
        attempts.append(eval_result)

        p = eval_result["penalty"]
        if p < best_p:
            best_p       = p
            best_trans   = current
            best_metrics = {k: v for k, v in eval_result.items()
                            if k not in ("attempt", "translation")}

        if satisfies_constraints(eval_result["semantic"], eval_result):
            break

        if attempt < max_attempts - 1:
            feedback = build_feedback(eval_result["semantic"], eval_result)
            conv_history.append({"role": "assistant", "content": current})
            conv_history.append({"role": "user",      "content": feedback})

            new_trans = call_groq(
                system=system,
                messages=[{"role": "user", "content": user_prompt}] + conv_history,
                temperature=0.0,
            )
            time.sleep(delay)
            if new_trans is None:
                break
            current = new_trans

    return {
        "best_translation": best_trans,
        "best_metrics":     best_metrics,
        "attempts":         attempts,
    }


# ================================================================
# STRATEGY B — CANDIDATE GENERATION
# ================================================================

def run_candidate(
    source_text:  str,
    emotion:      str,
    source_dist:  np.ndarray,
    label_order:  list[str],
    n_candidates: int   = 3,
    delay:        float = 0.8,
) -> dict:
    """
    Generate N independent German → English candidates, select best by penalty.

    Returns:
      best_translation: str
      best_metrics:     dict
      candidates:       list of per-candidate records
      early_exit:       bool
    """
    system      = _translation_system()
    user_prompt = _translation_user(source_text, emotion)

    candidates: list[dict] = []
    best_p      = float("inf")
    best_trans  = source_text
    best_metrics= None
    early_exit  = False

    for i in range(n_candidates):
        cand = call_groq(
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.7,   # diversity needed for meaningful selection
        )
        time.sleep(delay)
        if cand is None:
            cand = source_text

        eval_result = evaluate_translation(
            source_text=source_text,
            translation=cand,
            source_dist=source_dist,
            label_order=label_order,
            delay=delay,
        )
        eval_result["candidate"]   = i
        eval_result["translation"] = cand
        candidates.append(eval_result)

        p = eval_result["penalty"]
        if p < best_p:
            best_p       = p
            best_trans   = cand
            best_metrics = {k: v for k, v in eval_result.items()
                            if k not in ("candidate", "translation")}

        if satisfies_constraints(eval_result["semantic"], eval_result):
            early_exit = True
            break

    return {
        "best_translation": best_trans,
        "best_metrics":     best_metrics,
        "candidates":       candidates,
        "early_exit":       early_exit,
    }