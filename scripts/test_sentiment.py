#!/usr/bin/env python3
"""
Test script for sentiment_analyzer module.
Runs a few sample sentences through the HF Inference API and prints results.
"""

import sys
import os
import json

# Auto-load secrets
sys.path.insert(0, '/data/.openclaw/workspace/lib')
try:
    import secrets  # noqa: F401
except Exception:
    pass

from sentiment_analyzer import split_transcript, analyze_sentiment, analyze_transcript_sentiment


def test_split():
    print("=" * 60)
    print("TEST: split_transcript")
    print("=" * 60)

    transcript = (
        "Are you tired of cooking every night? We have the perfect solution. "
        "Fresh meals delivered to your door. Healthy ingredients, amazing flavors. "
        "Our chefs prepare everything so you don't have to. "
        "Try it today and get 50% off your first order. Visit us now!"
    )

    segments = split_transcript(transcript, max_words=50)
    print(f"Input: {len(transcript)} chars")
    print(f"Segments: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"  [{i}] ({len(seg.split())} words) {seg[:80]}...")
    print()


def test_sentiment_analysis():
    print("=" * 60)
    print("TEST: analyze_sentiment (individual segments)")
    print("=" * 60)

    segments = [
        "This is absolutely amazing, I love it so much!",
        "The product was okay, nothing special really.",
        "Terrible experience, worst purchase I've ever made.",
        "Fresh healthy meals delivered right to your door.",
        "Don't miss out! Order now and save 50%!",
    ]

    result = analyze_sentiment(segments)

    print(f"Overall: {result['overall_label']} (score: {result['overall_score']:.2f})")
    print(f"Hook: {result['hook_sentiment']}")
    print(f"CTA: {result['cta_sentiment']}")
    print(f"Arc: {result['emotional_arc']}")
    print()

    for seg in result["segments"]:
        print(f"  [{seg['label']:>8} {seg['confidence']:.0%}] {seg['text']}")
    print()


def test_full_transcript():
    print("=" * 60)
    print("TEST: analyze_transcript_sentiment (full pipeline)")
    print("=" * 60)

    transcript = (
        "Struggling to eat healthy with a busy schedule? You're not alone. "
        "That's why we created Factor. Fresh, chef-prepared meals delivered "
        "right to your door. No shopping, no cooking, no cleanup. "
        "Just heat and enjoy restaurant-quality meals in minutes. "
        "Choose from over 35 dietitian-approved options each week. "
        "Keto, calorie-smart, vegan, you name it. "
        "Join over a million happy customers today. "
        "Head to factormeals.com and use code SAVE50 for 50% off!"
    )

    result = analyze_transcript_sentiment(transcript)

    print(f"Overall: {result['overall_label']} (score: {result['overall_score']:.2f})")
    print(f"Hook: {result['hook_sentiment']}")
    print(f"CTA: {result['cta_sentiment']}")
    print(f"Arc: {result['emotional_arc']}")
    print(f"Segments analyzed: {len(result['segments'])}")
    print()

    for seg in result["segments"]:
        print(f"  [{seg['label']:>8} {seg['confidence']:.0%}] {seg['text']}")
    print()


def test_edge_cases():
    print("=" * 60)
    print("TEST: Edge cases")
    print("=" * 60)

    # Empty transcript
    result = analyze_transcript_sentiment("")
    print(f"Empty transcript: {result['overall_label']} — arc: {result['emotional_arc']}")

    # N/A transcript
    result = analyze_transcript_sentiment("N/A")
    print(f"N/A transcript: {result['overall_label']} — arc: {result['emotional_arc']}")

    # Single word
    result = analyze_transcript_sentiment("Amazing!")
    print(f"Single word 'Amazing!': {result['overall_label']} (score: {result['overall_score']:.2f})")

    print()


if __name__ == "__main__":
    test_split()
    test_sentiment_analysis()
    test_full_transcript()
    test_edge_cases()
    print("All tests completed ✓")
