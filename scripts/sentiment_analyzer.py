#!/usr/bin/env python3
"""
Sentiment Analyzer for Ad Transcripts

Uses HuggingFace Inference API with cardiffnlp/twitter-roberta-base-sentiment-latest
to analyze sentiment of transcript segments.

Returns per-segment sentiment (positive/negative/neutral + confidence) and
an overall summary including hook sentiment, CTA sentiment, and emotional arc.
"""

import os
import sys
import json
import re
import urllib.request
import urllib.error

# Load secrets — this script only needs HF token from ~/.cache, no env secrets
sys.path.insert(0, '/data/.openclaw/workspace/lib')
try:
    import secrets
    secrets.load(keys=[])  # load nothing — uses HF token from ~/.cache/huggingface/token
except Exception:
    pass

# ============================================================================
# CONFIG
# ============================================================================

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

# Label mapping from model output to human-readable
LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
}

# Sentiment score mapping: Negative=-1, Neutral=0, Positive=1
SCORE_MAP = {
    "Negative": -1.0,
    "Neutral": 0.0,
    "Positive": 1.0,
}


def _get_hf_token() -> str:
    """Get HuggingFace token from env or cached file."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Try cached token file
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()
        if token:
            return token

    return ""


def _call_hf_inference_single(text: str, token: str) -> list:
    """
    Call HuggingFace Inference API for a single text.
    Returns list of {label, score} dicts sorted by score descending.
    """
    payload = json.dumps({"inputs": text}).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(HF_API_URL, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            # API returns [[{label, score}, ...]] for single input
            if result and isinstance(result[0], list):
                return result[0]
            return result
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HF API error {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"HF API connection error: {e.reason}") from e


def _call_hf_inference_batch(texts: list, token: str) -> list:
    """
    Call HF Inference API for multiple texts (one call per text).
    Returns list of results, one per input text.
    """
    results = []
    for text in texts:
        results.append(_call_hf_inference_single(text, token))
    return results


def split_transcript(transcript: str, max_words: int = 50) -> list:
    """
    Split transcript into segments by sentence boundaries, merging short
    sentences to stay close to max_words per chunk.
    Returns list of non-empty text segments.
    """
    if not transcript or transcript.strip() == "N/A":
        return []

    # Split on sentence boundaries
    raw_sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not raw_sentences:
        return []

    # Merge short sentences into ~max_words chunks
    segments = []
    current = []
    current_len = 0

    for sentence in raw_sentences:
        words = len(sentence.split())
        if current_len + words > max_words and current:
            segments.append(" ".join(current))
            current = [sentence]
            current_len = words
        else:
            current.append(sentence)
            current_len += words

    if current:
        segments.append(" ".join(current))

    return segments


def analyze_sentiment(segments: list) -> dict:
    """
    Analyze sentiment for a list of text segments.

    Args:
        segments: List of text strings to analyze.

    Returns:
        dict with keys:
            - segments: list of {text, label, score, confidence}
            - overall_label: dominant sentiment label
            - overall_score: weighted average score (-1.0 to 1.0)
            - hook_sentiment: sentiment of first segment
            - cta_sentiment: sentiment of last segment
            - emotional_arc: brief description of sentiment progression
    """
    if not segments:
        return {
            "segments": [],
            "overall_label": "Neutral",
            "overall_score": 0.0,
            "hook_sentiment": "N/A",
            "cta_sentiment": "N/A",
            "emotional_arc": "N/A",
        }

    token = _get_hf_token()

    # Call API (one request per segment)
    try:
        raw_results = _call_hf_inference_batch(segments, token)
    except RuntimeError as e:
        print(f"Warning: Sentiment analysis failed — {e}")
        return {
            "segments": [],
            "overall_label": "Unknown",
            "overall_score": 0.0,
            "hook_sentiment": "Error",
            "cta_sentiment": "Error",
            "emotional_arc": "Analysis failed",
        }

    # Parse results
    parsed = []
    for i, segment_text in enumerate(segments):
        if i >= len(raw_results):
            break

        # Each result is a list of label/score pairs sorted by score desc
        segment_scores = raw_results[i]
        if isinstance(segment_scores, list) and segment_scores:
            top = segment_scores[0]
            label_raw = top.get("label", "neutral")
            label = LABEL_MAP.get(label_raw, label_raw.capitalize())
            confidence = round(top.get("score", 0.0), 4)
        else:
            label = "Neutral"
            confidence = 0.0

        numeric_score = SCORE_MAP.get(label, 0.0) * confidence

        parsed.append({
            "text": segment_text[:120] + ("..." if len(segment_text) > 120 else ""),
            "label": label,
            "score": round(numeric_score, 4),
            "confidence": confidence,
        })

    # Overall sentiment
    if parsed:
        avg_score = sum(p["score"] for p in parsed) / len(parsed)
        label_counts = {}
        for p in parsed:
            label_counts[p["label"]] = label_counts.get(p["label"], 0) + 1
        dominant_label = max(label_counts, key=label_counts.get)
    else:
        avg_score = 0.0
        dominant_label = "Neutral"

    # Hook & CTA
    hook = f"{parsed[0]['label']} ({parsed[0]['confidence']:.0%})" if parsed else "N/A"
    cta = f"{parsed[-1]['label']} ({parsed[-1]['confidence']:.0%})" if parsed else "N/A"

    # Emotional arc — compress labels into a progression string
    if len(parsed) >= 2:
        arc_labels = [p["label"].lower() for p in parsed]
        # Deduplicate consecutive labels
        deduped = [arc_labels[0]]
        for lbl in arc_labels[1:]:
            if lbl != deduped[-1]:
                deduped.append(lbl)
        emotional_arc = " → ".join(deduped)
    elif parsed:
        emotional_arc = parsed[0]["label"].lower()
    else:
        emotional_arc = "N/A"

    return {
        "segments": parsed,
        "overall_label": dominant_label,
        "overall_score": round(avg_score, 4),
        "hook_sentiment": hook,
        "cta_sentiment": cta,
        "emotional_arc": emotional_arc,
    }


def analyze_transcript_sentiment(transcript: str, max_words: int = 50) -> dict:
    """
    Convenience wrapper: split transcript and analyze sentiment.

    Args:
        transcript: Full transcript text.
        max_words: Max words per segment.

    Returns:
        Same dict as analyze_sentiment().
    """
    segments = split_transcript(transcript, max_words=max_words)
    return analyze_sentiment(segments)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze sentiment of text segments")
    parser.add_argument("--text", help="Text to analyze")
    parser.add_argument("--file", help="File containing text to analyze")
    parser.add_argument("--max-words", type=int, default=50, help="Max words per segment")
    args = parser.parse_args()

    text = args.text
    if args.file:
        with open(args.file) as f:
            text = f.read()

    if not text:
        print("Provide --text or --file")
        sys.exit(1)

    result = analyze_transcript_sentiment(text, max_words=args.max_words)

    print(json.dumps(result, indent=2))
