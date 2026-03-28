---
name: ad-analyzer
description: "Analyze video ads by extracting frames, transcribing audio, identifying visual/audio features, and tagging them for pattern analysis. Use when: (1) analyzing a single video ad to tag features (food, human talking, text, music), (2) building a library of ad analyses correlated with performance metrics, (3) iterating ad creative based on feature patterns."
---

# Ad Analyzer

Analyze video ads to extract actionable insights for creative optimization.

## Quick Start

**What it does:**
1. Extracts key frames from your video ad at regular intervals
2. Transcribes the audio using Google Cloud Speech-to-Text
3. Analyzes frames + transcript for recognizable features (e.g., "shows food", "human talking", "text overlay")
4. Creates a Notion database entry with tags and timestamps

**What you need:**
- A video file (MP4, MOV, WebM, etc.)
- Notion database set up (see Setup below)
- Anthropic API key (required — feature detection via Claude Vision)
- OpenAI API key (optional — narrative/storytelling analysis via GPT-4o)

**Usage:**
```bash
# Full analysis (Claude features + OpenAI narrative)
python ad_analyzer.py --video my_ad.mp4 --notion-db-id <database-id>

# Claude-only mode (skip OpenAI)
python ad_analyzer.py --video my_ad.mp4 --notion-db-id <database-id> --skip-openai
```

## How It Works

### Step 1: Frame Extraction
Extracts frames at regular intervals (default: every 3 seconds). This captures visual content for analysis.

### Step 2: Audio Transcription
Uses Google Cloud Speech-to-Text to transcribe all dialogue and speech in the ad.

### Step 3: Feature Analysis (Claude Vision)
Analyzes frames and transcript to identify features:
- **Visual:** food, people, text, product, motion, color dominant, animation
- **Audio:** human talking, music, voiceover, silence, dialogue
- **Overall:** video length, pacing, color scheme, call-to-action

### Step 3b: Narrative Analysis (OpenAI GPT-4o, optional)
When `OPENAI_API_KEY` is set, sends frames to GPT-4o for storytelling analysis:
- **Narrative arc:** Story structure (problem→solution, testimonial, etc.)
- **Emotional progression:** How emotions evolve through the ad
- **Pacing & transitions:** Rhythm of cuts, scene connections
- **Hook & CTA:** Opening attention-grab and closing effectiveness
- **Storytelling score:** 1-10 rating with improvement suggestions

### Step 4: Notion Upload
Creates a structured entry in your Notion database with:
- Ad metadata (filename, duration)
- Detected features (with confidence scores)
- Creative summary (features + storytelling narrative merged)
- Storytelling analysis section (narrative arc, pacing, emotional progression)
- Transcript
- Ready to correlate with performance metrics

## Feature Tags

See `references/feature-tags.md` for the complete list of recognizable features and what they mean.

## Setup: Notion Database

See `references/notion-setup.md` for step-by-step instructions to create the database schema.

## Example Output

```
Ad: summer_promo.mp4 (15 seconds)

Features Detected:
✓ Shows food (confidence: 0.95)
✓ Human talking (confidence: 0.88)
✓ Music playing (confidence: 0.92)
✓ Fast cuts/motion (confidence: 0.81)
✗ Text overlay

Transcript:
"Come try our new summer menu. Fresh ingredients, bold flavors. Visit us today!"

Notion Entry: Created ✓
```

## Next Steps

After analyzing an ad:
1. Record actual performance metrics (CTR, ROAS, conversions) in Notion
2. Analyze patterns across multiple ads
3. Identify which feature combinations correlate with high performance
4. Iterate creative brief based on insights

---

**Note:** Requires Google Cloud Speech-to-Text API access (you have this from your workspace setup).
