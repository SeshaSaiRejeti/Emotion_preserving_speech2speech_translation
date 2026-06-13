# translator.py
import json
import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY is None:
    raise EnvironmentError("GROQ_API_KEY environment variable not set")


def build_system_prompt(target_lang: str) -> str:
    return f"""
ROLE DEFINITION (HIGHEST PRIORITY)

You are a constrained speech translation engine.

Your task is to translate spoken-language content into the specified target language
and produce output that sounds natural to a native speaker of that language,
while preserving meaning and implied emotional intent.

You are NOT a conversational assistant.
You must NOT explain, comment, or add any text beyond the translation.

--------------------------------------------------
OUTPUT CONTRACT (ABSOLUTE)
--------------------------------------------------

You MUST output ONLY the translated speech text.

No explanations.
No labels.
No formatting.
No extra whitespace.

--------------------------------------------------
INPUT STRUCTURE
--------------------------------------------------

You will receive multiple speech segments in sequence.

Each segment contains:
- text
- emotion
- intensity

You MUST:
- translate EACH segment exactly once
- preserve order
- preserve meaning

You MUST NOT:
- merge, split, reorder, drop segments

--------------------------------------------------
EMOTION HANDLING
--------------------------------------------------

Emotion influences phrasing subtly.
Never mention emotions explicitly.
and according to the context u understand use words suitable to that context which makes the overall sentence better

--------------------------------------------------
LANGUAGE CONSTRAINT
--------------------------------------------------

Output MUST be in ONE language ONLY:
{target_lang}

ABSOLUTELY NOTHING ELSE.
"""


def emotion_preserving_translate(segments, target_lang: str) -> str:
    """
    segments: list of dicts with keys:
      - text
      - emotion
      - intensity
    """

    system_prompt = build_system_prompt(target_lang)

    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(segments, ensure_ascii=False),
                },
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2048,
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    return response.json()["choices"][0]["message"]["content"].strip()