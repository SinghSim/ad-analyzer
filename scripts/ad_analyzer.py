#!/usr/bin/env python3
"""
Ad Analyzer

Analyze video ads by extracting frames, transcribing audio, and identifying features.
Uses dual-API approach: Claude Vision for feature detection + OpenAI GPT-4o for
narrative/storytelling analysis. Uploads results to Notion database.

Usage:
    python ad_analyzer.py --video my_ad.mp4 --notion-db-id <database-id>

Requirements:
    - ffmpeg installed
    - Anthropic API key (ANTHROPIC_API_KEY env var) — feature detection
    - OpenAI API key (OPENAI_API_KEY env var) — narrative analysis (optional)
    - Notion API key (NOTION_API_KEY env var)
"""

import os
import sys
sys.path.insert(0, '/data/.openclaw/workspace/lib')
import secrets as _secrets  # noqa
_secrets.require(['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'NOTION_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS'])
import json
import argparse
import subprocess
import tempfile
import urllib.request
import re
from pathlib import Path
from datetime import timedelta, datetime
from typing import List, Dict, Tuple
import base64

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    try:
        sys.path.insert(0, '/data/.local/lib/python3.14/site-packages')
        import whisper
        WHISPER_AVAILABLE = True
    except (ImportError, Exception):
        WHISPER_AVAILABLE = False

try:
    from anthropic import Anthropic
except ImportError:
    print("Error: anthropic SDK not installed. Run: pip install anthropic")
    sys.exit(1)

# OpenAI is optional — used for narrative/storytelling analysis
_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        _OPENAI_AVAILABLE = True
    else:
        print("Note: OPENAI_API_KEY not set. Narrative analysis will use Claude only.")
except ImportError:
    print("Note: openai SDK not installed. Narrative analysis will use Claude only.")

# ============================================================================
# VIDEO QUALITY ANALYZER
# ============================================================================

def analyze_video_quality(video_path: str) -> Tuple[str, str, Dict]:
    """
    Analyze video quality metrics and return quality level + details + JSON dict.
    Returns: (quality_level, details_string, quality_dict)
    """
    print("Analyzing video quality...")
    
    try:
        import subprocess
        
        # Get video metrics
        video_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream=width,height,r_frame_rate,codec_name,bit_rate',
                     '-of', 'default=noprint_wrappers=1:nokey=1:noprint_wrappers=1',
                     video_path]
        
        audio_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                     '-show_entries', 'stream=codec_name,sample_rate,channels,bit_rate',
                     '-of', 'default=noprint_wrappers=1:nokey=1:noprint_wrappers=1',
                     video_path]
        
        format_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                      '-of', 'default=noprint_wrappers=1:nokey=1',
                      video_path]
        
        v_result = subprocess.run(video_cmd, capture_output=True, text=True).stdout.strip().split('\n')
        a_result = subprocess.run(audio_cmd, capture_output=True, text=True).stdout.strip().split('\n')
        duration = float(subprocess.run(format_cmd, capture_output=True, text=True).stdout.strip())
        
        # Parse video metrics
        v_codec = v_result[0] if len(v_result) > 0 else "Unknown"
        height = int(v_result[1]) if len(v_result) > 1 else 0
        width = int(v_result[2]) if len(v_result) > 2 else 0
        fps_str = v_result[3] if len(v_result) > 3 else "0"
        fps = float(fps_str.split('/')[0]) if '/' in fps_str else float(fps_str) if fps_str else 0
        v_bitrate = int(v_result[4]) if len(v_result) > 4 else 0
        
        # Parse audio metrics
        a_codec = a_result[0] if len(a_result) > 0 else "Unknown"
        sample_rate = int(a_result[1]) if len(a_result) > 1 else 0
        channels = int(a_result[2]) if len(a_result) > 2 else 0
        a_bitrate = int(a_result[3]) if len(a_result) > 3 else 0
        
        # Build quality dict
        quality_dict = {
            "resolution": f"{width}x{height}",
            "fps": round(fps, 1),
            "video_codec": v_codec,
            "video_bitrate_mbps": round(v_bitrate / 1e6, 1),
            "audio_codec": a_codec,
            "sample_rate_khz": round(sample_rate / 1000, 1),
            "audio_channels": channels,
            "audio_bitrate_kbps": round(a_bitrate / 1000, 1),
            "duration_seconds": round(duration, 1)
        }
        
        # Assess quality
        resolution = f"{width}x{height}"
        details = f"Resolution: {resolution}, FPS: {int(fps)}, Video Codec: {v_codec}, Video Bitrate: {v_bitrate/1e6:.1f}Mbps, Audio: {a_codec} {sample_rate/1000:.0f}kHz {channels}ch, Audio Bitrate: {a_bitrate/1000:.0f}Kbps"
        
        # Determine quality level
        quality_scores = 0
        
        # Resolution score
        if height >= 1080:
            quality_scores += 3
        elif height >= 720:
            quality_scores += 2
        elif height >= 480:
            quality_scores += 1
        
        # FPS score
        if fps >= 30:
            quality_scores += 2
        elif fps >= 24:
            quality_scores += 1
        
        # Bitrate score
        if v_bitrate >= 10e6:
            quality_scores += 2
        elif v_bitrate >= 5e6:
            quality_scores += 1
        
        # Audio score
        if sample_rate >= 48000 and channels == 2:
            quality_scores += 2
        elif sample_rate >= 44100:
            quality_scores += 1
        
        # Determine level
        if quality_scores >= 8:
            quality_level = "Professional"
        elif quality_scores >= 6:
            quality_level = "High"
        elif quality_scores >= 4:
            quality_level = "Standard"
        else:
            quality_level = "Low"
        
        print(f"✓ Quality assessed: {quality_level}")
        return quality_level, details, quality_dict
    
    except Exception as e:
        print(f"Warning: Could not analyze video quality ({e})")
        return "Unknown", "", {}

# ============================================================================
# NOTION TAG MANAGEMENT
# ============================================================================

def get_existing_tags(database_id: str) -> List[str]:
    """
    Fetch existing tag options from Notion database's Tags property.
    Returns: list of existing tag names
    """
    print("Checking existing tags in Notion...")
    
    try:
        import subprocess
        
        cmd = f'source ~/.bashrc && notion databases get "{database_id}" 2>&1'
        result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
        
        # Parse JSON response
        db_data = json.loads(result.stdout)
        tags_property = db_data.get('properties', {}).get('Tags', {})
        
        # Extract tag options
        existing_tags = []
        if tags_property.get('type') == 'multi_select':
            options = tags_property.get('multi_select', {}).get('options', [])
            existing_tags = [opt.get('name', '') for opt in options if opt.get('name')]
        
        print(f"✓ Found {len(existing_tags)} existing tags")
        return existing_tags
    
    except Exception as e:
        print(f"Warning: Could not fetch existing tags ({e}). Will create new ones as needed.")
        return []

def map_features_to_tags(detected_features: Dict[str, float], existing_tags: List[str]) -> List[str]:
    """
    Map detected features to existing tags. 
    If feature name matches an existing tag, reuse it.
    Otherwise, create new tag name.
    """
    feature_tags = []
    existing_tags_lower = [t.lower() for t in existing_tags]
    
    for feature, score in sorted(detected_features.items(), key=lambda x: x[1], reverse=True):
        if score <= 0.3:
            continue
        
        feature_name = feature.replace('_', ' ').title()
        feature_name_lower = feature_name.lower()
        
        # Try to find matching existing tag
        matched = False
        for i, existing_lower in enumerate(existing_tags_lower):
            if existing_lower == feature_name_lower or existing_lower in feature_name_lower or feature_name_lower in existing_lower:
                feature_tags.append(existing_tags[i])  # Use exact case from database
                matched = True
                break
        
        # If no match, use the new feature name (will be created in Notion)
        if not matched:
            feature_tags.append(feature_name)
    
    return feature_tags

# ============================================================================
# GOOGLE DRIVE HELPERS
# ============================================================================

def extract_filename_from_drive_link(drive_link: str) -> Tuple[str, str]:
    """
    Extract filename from Google Drive link via Content-Disposition header.
    Returns: (filename, original_drive_link)
    """
    if "drive.google.com" not in drive_link:
        return None, drive_link
    
    # Extract file ID from Drive link
    file_id = None
    if "/d/" in drive_link:
        file_id = drive_link.split("/d/")[1].split("/")[0]
    elif "id=" in drive_link:
        file_id = drive_link.split("id=")[1].split("&")[0]
    
    if not file_id:
        return None, drive_link
    
    # Construct direct download URL
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"Extracting filename from Google Drive link...")
    
    try:
        # Make HEAD request to get Content-Disposition header
        request = urllib.request.Request(download_url, method='HEAD')
        with urllib.request.urlopen(request, timeout=5) as response:
            disposition = response.headers.get('Content-Disposition', '')
            
            # Extract filename from header: filename="name.mp4" or filename*=UTF-8''name.mp4
            match = re.search(r"filename\*?=(?:UTF-8'')?['\"]?([^'\"]+)['\"]?", disposition)
            if match:
                filename = match.group(1).split('/')[-1]  # Get just filename, not path
                print(f"✓ Found filename: {filename}")
                return filename, download_url
    
    except Exception as e:
        print(f"Warning: Could not extract filename from Drive link. {e}")
    
    return None, drive_link

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

VISUAL_FEATURES = {
    # People demographics
    "people_female": "Female/woman/girl visible",
    "people_male": "Male/man/boy visible",
    "people_child": "Child or young person (under 13)",
    "people_teenager": "Teenager or young adult (13-25)",
    "people_young_adult": "Young adult (25-40)",
    "people_middle_aged": "Middle-aged person (40-60)",
    "people_senior": "Senior or older person (60+)",
    # General food
    "shows_food": "Food or drink visible in frame",
    # Specific proteins
    "food_salmon": "Salmon or salmon dish visible",
    "food_chicken": "Chicken or poultry visible",
    "food_beef": "Beef or red meat visible",
    "food_fish": "Fish (not salmon) visible",
    "food_tofu": "Tofu or plant-based protein visible",
    "food_shrimp": "Shrimp or seafood visible",
    # Vegetables
    "food_vegetables": "Vegetables visible (broccoli, spinach, carrots, peppers, etc)",
    "food_leafy_greens": "Leafy greens or salad visible",
    "food_mushrooms": "Mushrooms visible",
    # Fruits
    "food_fruits": "Fruits visible",
    "food_berries": "Berries visible",
    # Carbs
    "food_rice": "Rice or grain dishes visible",
    "food_pasta": "Pasta or noodles visible",
    "food_bread": "Bread or bakery items visible",
    "food_potatoes": "Potatoes or root vegetables visible",
    # Dairy
    "food_cheese": "Cheese visible",
    "food_yogurt": "Yogurt or dairy visible",
    # Prepared/packaged
    "food_prepared_meal": "Prepared or plated meal visible",
    "food_packaged": "Packaged food or ingredients visible",
    # Other features
    "shows_people": "Human faces or bodies visible",
    "text_overlay": "Text/titles overlaid on video",
    "product_visible": "Product being advertised is visible",
    "motion_heavy": "Fast cuts, pans, or dynamic movement",
    "color_dominant_warm": "Warm colors (orange, red, yellow) dominant",
    "color_dominant_cool": "Cool colors (blue, green) dominant",
    "animation": "Animated elements or graphics",
    "outdoor_setting": "Scene is outdoors",
    "indoor_setting": "Scene is indoors",
}

AUDIO_FEATURES = {
    "human_talking": "Human speech/dialogue",
    "music": "Background music playing",
    "voiceover": "Voiceover narration",
    "silence": "Silence or quiet sections",
    "sound_effects": "Sound effects or audio cues",
    "upbeat_audio": "Fast-paced, energetic audio",
    "calm_audio": "Slow, calm audio",
}

OVERALL_FEATURES = {
    "fast_pacing": "Quick cuts and fast editing",
    "emotional_appeal": "Emotional or inspirational messaging",
    "call_to_action": "Clear CTA (call now, visit, buy, etc.)",
    "humor": "Humorous or comedic content",
    "aspirational": "Shows aspirational lifestyle",
    "problem_solution": "Shows problem then solution",
}

# ============================================================================
# FRAME EXTRACTION
# ============================================================================

def extract_frames(video_path: str, interval_seconds: int = 3, max_frames: int = 50) -> List[Dict]:
    """
    Extract frames from video at regular intervals.
    
    Returns list of dicts with keys: timestamp, frame_path, frame_number
    """
    print(f"Extracting frames from {video_path} (every {interval_seconds}s)...")
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Get video duration
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1:noprint_wrappers=1",
        str(video_path)
    ]
    try:
        duration_str = subprocess.check_output(duration_cmd, text=True).strip()
        duration_seconds = float(duration_str)
    except (subprocess.CalledProcessError, ValueError):
        raise RuntimeError("Could not determine video duration. Ensure ffprobe is installed.")
    
    # Create temp dir for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%04d.jpg")
        
        # Extract frames at intervals
        fps = 1.0 / interval_seconds
        extract_cmd = [
            "ffmpeg", "-i", str(video_path), "-vf", f"fps={fps}",
            "-q:v", "2", frame_pattern
        ]
        
        try:
            subprocess.run(extract_cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")
        
        # Collect frames
        frames = []
        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))[:max_frames]
        
        for i, frame_path in enumerate(frame_files):
            timestamp = i * interval_seconds
            if timestamp <= duration_seconds:
                # Read frame data
                with open(frame_path, "rb") as f:
                    frame_data = f.read()
                
                frames.append({
                    "timestamp": timestamp,
                    "frame_number": i,
                    "frame_data": base64.b64encode(frame_data).decode(),  # For Notion
                })
        
        print(f"Extracted {len(frames)} frames")
        return frames, duration_seconds

# ============================================================================
# AUDIO TRANSCRIPTION
# ============================================================================

def extract_subtitles(video_path: str) -> str:
    """
    Extract subtitles/captions from video if they exist.
    Returns subtitle text or empty string if none found.
    """
    try:
        # Try to extract subtitles using ffmpeg
        subtitle_path = "/tmp/subtitles_temp.srt"
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-c", "copy", "-map", "0:s:0", subtitle_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10
        )
        
        if os.path.exists(subtitle_path) and os.path.getsize(subtitle_path) > 100:
            with open(subtitle_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            os.remove(subtitle_path)
            # Parse SRT to extract just the text (skip timestamps)
            lines = content.split("\n")
            text_lines = [line for line in lines if line.strip() and not line.isdigit() and "-->" not in line]
            return " ".join(text_lines).strip()
    except:
        pass
    
    return ""

def detect_voiceover_presence(video_path: str) -> bool:
    """
    Detect if there's significant voiceover/speech in the video.
    Returns True if speech detected, False if only music/ambient sound.
    """
    try:
        # Use Whisper's detection to see if it finds meaningful speech
        if not WHISPER_AVAILABLE:
            return True  # Assume voiceover present if Whisper unavailable
        model = whisper.load_model("base")
        result = model.transcribe(video_path, verbose=False, task="transcribe")
        transcript = result.get("text", "").strip()
        
        # If Whisper found very little text (< 5 chars) or just filler, likely no voiceover
        if len(transcript) < 5:
            return False
        
        # Check if transcript is mostly gibberish (too many unusual chars)
        words = transcript.split()
        if len(words) < 2:
            return False
        
        return True
    except:
        return False

def transcribe_audio(video_path: str, analysis_modules: Dict = None) -> Tuple[str, str]:
    """
    Smart transcription: checks for subtitles first, then voiceover, then returns N/A.
    
    Priority:
    1. Embedded subtitles (if present)
    2. Voiceover transcription (if detected)
    3. "N/A" (music-only or silent videos)
    
    Returns: (transcript, language)
    If analysis_modules dict is provided, populates Whisper and Google Cloud STT entries.
    """
    if analysis_modules is None:
        analysis_modules = {}
    try:
        # Step 1: Check for embedded subtitles
        print("Checking for subtitles...")
        subtitles = extract_subtitles(video_path)
        if subtitles:
            print(f"✓ Found subtitles ({len(subtitles)} chars)")
            return subtitles, "en"
        
        # Step 2: Check if there's actual voiceover
        print("Detecting language and voiceover...")
        has_voiceover = detect_voiceover_presence(video_path)
        
        if not has_voiceover:
            print("ℹ No voiceover detected (music/ambient sound only)")
            return "N/A", "en"
        
        # Step 3: Transcribe the voiceover
        if not WHISPER_AVAILABLE:
            # Skip Whisper detection, go straight to Google STT
            detected_language = "en"
            detect_result = {"text": "", "language": "en"}
            analysis_modules["whisper"] = ("skip", "unavailable")
        else:
            model = whisper.load_model("base")
            detect_result = model.transcribe(video_path, verbose=False, task="transcribe")
            detected_language = detect_result.get("language", "en")
            analysis_modules["whisper"] = ("ok", f"language detected: {detected_language}")
        
        if not WHISPER_AVAILABLE:
            # No Whisper — use Google STT with auto language detection
            detected_language = "auto"

        if detected_language == "en" and WHISPER_AVAILABLE:
            # Use Whisper for English
            print("Transcribing audio with Whisper (English)...")
            transcript = detect_result["text"]
            analysis_modules["google_stt"] = ("skip", "not needed (English + Whisper)")
        else:
            # Use Google Cloud STT for non-English
            print(f"Transcribing audio with Google Cloud STT ({detected_language})...")
            try:
                from google.cloud import speech
                import io
                
                # Extract audio from video (mono for Google Cloud STT)
                audio_path = "/tmp/audio_temp.wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", video_path, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )
                
                # Transcribe with Google
                client = speech.SpeechClient()
                with io.open(audio_path, "rb") as audio_file:
                    content = audio_file.read()
                
                audio = speech.RecognitionAudio(content=content)
                lang_code = "sv-SE" if detected_language in ("sv", "auto") else detected_language
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code=lang_code,
                    alternative_language_codes=["en-US", "nl-NL", "de-DE", "fr-FR"] if detected_language == "auto" else [],
                    enable_automatic_punctuation=True,
                )
                
                response = client.recognize(config=config, audio=audio)
                transcript = " ".join([result.alternatives[0].transcript for result in response.results])
                analysis_modules["google_stt"] = ("ok", f"transcription ({detected_language} → en, {len(transcript)} chars)")
                
                # Cleanup
                os.remove(audio_path)
                
            except Exception as google_error:
                analysis_modules["google_stt"] = ("fail", f"failed ({google_error})")
                if WHISPER_AVAILABLE:
                    print(f"Warning: Google Cloud STT failed ({google_error}). Falling back to Whisper...")
                    transcript = detect_result["text"]
                else:
                    print(f"Warning: Google Cloud STT failed ({google_error}). Whisper unavailable, using Whisper detect result...")
                    transcript = detect_result["text"]
        
        print(f"Transcribed {len(transcript)} characters (Language: {detected_language})")
        return transcript.strip(), detected_language
    
    except Exception as e:
        print(f"Warning: Transcription failed: {e}")
        print("Continuing with visual analysis only...")
        analysis_modules["whisper"] = analysis_modules.get("whisper", ("fail", f"failed ({e})"))
        analysis_modules["google_stt"] = analysis_modules.get("google_stt", ("fail", f"failed ({e})"))
        return "N/A", "unknown"

# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

def has_product_mentions(transcript: str) -> bool:
    """Check if product/service is mentioned in transcript."""
    product_keywords = ["factor", "meal", "product", "service", "our", "brand", "this", "introducing"]
    return any(kw in transcript.lower() for kw in product_keywords)

def describe_video_visually(transcript: str, num_frames: int, duration: float) -> str:
    """
    Create a visual description of video based on transcript and frame count.
    Returns a human-readable description of what happens in the video.
    """
    description = []
    
    # Estimate scene progression based on transcript
    transcript_lower = transcript.lower()
    
    # Opening
    description.append("Video opens with:")
    if any(word in transcript_lower for word in ["start", "begin", "introduce", "presents", "shows"]):
        description.append("- Introduces the topic or product")
    if any(word in transcript_lower for word in ["man", "woman", "person", "character"]):
        description.append("- Features a person/character speaking")
    if any(word in transcript_lower for word in ["meal", "food", "eat", "eating", "unbox", "package"]):
        description.append("- Shows food, meals, or product unboxing")
    
    # Main content based on keywords
    description.append("\nMain content:")
    if any(word in transcript_lower for word in ["problem", "struggle", "difficult", "challenge", "tired"]):
        description.append("- Presents a problem or challenge")
    if any(word in transcript_lower for word in ["solution", "solve", "discover", "introduce", "offer"]):
        description.append("- Introduces a solution or product")
    if any(word in transcript_lower for word in ["work", "busy", "schedule", "time", "quick", "easy"]):
        description.append("- Emphasizes convenience/time-saving")
    if any(word in transcript_lower for word in ["health", "fit", "exercise", "nutrition", "good", "best", "quality"]):
        description.append("- Highlights health, fitness, or quality benefits")
    
    # Scene transitions
    description.append(f"\nVideo structure:")
    description.append(f"- Total duration: {duration:.1f} seconds")
    description.append(f"- Approximately {num_frames} key frames/scenes")
    
    # Call to action
    description.append("\nEnding:")
    if any(word in transcript_lower for word in ["try", "order", "get", "visit", "call", "buy", "subscribe", "click", "download"]):
        description.append("- Includes a clear call-to-action")
    else:
        description.append("- Concludes with product/service message")
    
    return "\n".join(description)

def analyze_visual_features_with_claude(frames: List[Dict], transcript: str = "", override_gender: str = None) -> Dict[str, float]:
    """
    Analyze frames using Claude vision API for accurate feature detection.
    Returns dict of {feature: confidence_score}
    """
    print("Analyzing visual features with Claude vision...")
    
    features = {f: 0.0 for f in VISUAL_FEATURES.keys()}
    
    if not frames:
        return features
    
    # Initialize Anthropic client
    client = Anthropic()
    
    # Sample frames (first, middle, last)
    sample_indices = [0, len(frames)//2, len(frames)-1]
    sample_frames = [frames[i] for i in sample_indices if i < len(frames)]
    
    try:
        # Build messages for Claude vision
        content = [
            {
                "type": "text",
                "text": """Analyze these video frames and identify visual features. For each frame, describe:
1. What objects/products are visible
2. Who appears in the frame (people, faces)
3. Dominant colors and lighting
4. Setting (indoor/outdoor)
5. Any motion or action

Be specific and concise. Then provide a summary of detected features across all frames."""
            }
        ]
        
        # Add frames to content
        for i, frame in enumerate(sample_frames):
            frame_data = frame.get("frame_data", "")
            if frame_data:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame_data
                    }
                })
                content.append({
                    "type": "text",
                    "text": f"Frame {i+1}:"
                })
        
        # Call Claude vision
        response = client.messages.create(
            model="claude-opus-4-1",  # High quality vision model
            max_tokens=1024,
            messages=[
                {"role": "user", "content": content}
            ]
        )
        
        analysis_text = response.content[0].text.lower()
        print(f"✓ Claude vision analysis complete")
        
        # Extract features from Claude's analysis
        # People/faces and demographics
        if any(word in analysis_text for word in ["person", "people", "face", "man", "woman", "guy", "girl", "human"]):
            features["shows_people"] = 0.85
        
        # Gender detection (exclusive - can't be both)
        # Priority: override > explicit keywords > visual cues > pronouns
        analysis_lower = analysis_text.lower()
        
        # Explicit gender keywords
        female_keywords = ["woman", "girl", "female", "women", "girls", "actress", "she's", "her "]
        male_keywords = ["man", "boy", "male", "men", "boys", "actor", "he's", "his ", "him "]
        
        has_explicit_female = any(word in analysis_lower for word in female_keywords)
        has_explicit_male = any(word in analysis_lower for word in male_keywords)
        
        # Pronouns
        has_she_pronouns = " she " in analysis_lower or " her " in analysis_lower or " hers " in analysis_lower
        has_he_pronouns = " he " in analysis_lower or " his " in analysis_lower or " him " in analysis_lower
        
        # Visual appearance cues (feminine/masculine indicators from vision)
        feminine_cues = ["long hair", "makeup", "lipstick", "dress", "skirt", "feminine", "lady", "woman's", "female voice"]
        masculine_cues = ["short hair", "beard", "masculine", "man's", "male voice"]
        
        has_feminine_cues = any(cue in analysis_lower for cue in feminine_cues)
        has_masculine_cues = any(cue in analysis_lower for cue in masculine_cues)
        
        detected_gender = None
        
        # Apply override if provided (highest priority)
        if override_gender:
            detected_gender = override_gender.lower()
        # Otherwise detect from text with priority: explicit keywords > visual cues > pronouns
        elif has_explicit_female:
            detected_gender = "female"
        elif has_explicit_male:
            detected_gender = "male"
        elif has_feminine_cues and not has_masculine_cues:
            detected_gender = "female"
        elif has_masculine_cues and not has_feminine_cues:
            detected_gender = "male"
        elif has_she_pronouns and not has_he_pronouns:
            detected_gender = "female"
        elif has_he_pronouns and not has_she_pronouns:
            detected_gender = "male"
        
        # Set features based on detected gender
        if detected_gender == "female":
            features["people_female"] = 0.85
            features["people_male"] = 0.0
        elif detected_gender == "male":
            features["people_male"] = 0.85
            features["people_female"] = 0.0
        
        # Age range detection
        analysis_lower = analysis_text.lower()
        
        if any(word in analysis_lower for word in ["child", "kid", "toddler", "baby", "young child", "elementary"]):
            features["people_child"] = 0.8
        elif any(word in analysis_lower for word in ["teenager", "teen", "adolescent", "high school", "college", "young adult"]):
            features["people_teenager"] = 0.8
        elif any(word in analysis_lower for word in ["25-40", "30s", "40s early", "professional", "woman in her 30s", "man in his 30s"]):
            features["people_young_adult"] = 0.8
        elif any(word in analysis_lower for word in ["middle-aged", "middle aged", "40-60", "50s", "mature"]):
            features["people_middle_aged"] = 0.8
        elif any(word in analysis_lower for word in ["senior", "older", "elderly", "60+", "retired", "gray hair", "grey hair"]):
            features["people_senior"] = 0.8
        
        # General food
        if any(word in analysis_text for word in ["food", "meal", "dish", "plate", "eat", "eating", "pizza", "salad", "sandwich", "drink"]):
            features["shows_food"] = 0.85
        
        # Specific proteins
        if any(word in analysis_text for word in ["salmon", "fish", "seafood", "sea food"]):
            features["food_salmon"] = 0.85 if "salmon" in analysis_text else 0.7
            features["food_fish"] = 0.8
        
        # Chicken (be specific - don't confuse with beef)
        if any(word in analysis_text for word in ["chicken", "poultry", "fowl", "breast", "thigh", "drumstick"]):
            features["food_chicken"] = 0.85
            features["food_beef"] = 0.0  # Reset beef if chicken is detected
        # Beef (only if explicitly mentioned, not just generic "meat")
        elif any(word in analysis_text for word in ["beef", "steak", "red meat", "burger", "brisket", "ribeye"]):
            features["food_beef"] = 0.85
        
        if any(word in analysis_text for word in ["tofu", "soy", "plant-based", "vegan", "vegetarian"]):
            features["food_tofu"] = 0.8
        
        if any(word in analysis_text for word in ["shrimp", "prawn", "seafood", "shellfish"]):
            features["food_shrimp"] = 0.85
        
        # Vegetables
        if any(word in analysis_text for word in ["vegetables", "vegetable", "veggies", "broccoli", "spinach", "carrot", "carrots", "pepper", "peppers", "asparagus", "zucchini", "eggplant"]):
            features["food_vegetables"] = 0.85
        
        if any(word in analysis_text for word in ["salad", "greens", "leafy", "spinach", "lettuce", "kale", "arugula"]):
            features["food_leafy_greens"] = 0.8
        
        if any(word in analysis_text for word in ["mushroom", "mushrooms", "fungi"]):
            features["food_mushrooms"] = 0.8
        
        # Fruits
        if any(word in analysis_text for word in ["fruit", "fruits", "apple", "banana", "orange", "berry"]):
            features["food_fruits"] = 0.75
        
        if any(word in analysis_text for word in ["berry", "berries", "blueberry", "strawberry", "raspberry"]):
            features["food_berries"] = 0.8
        
        # Carbs
        if any(word in analysis_text for word in ["rice", "grain", "grains", "risotto"]):
            features["food_rice"] = 0.8
        
        if any(word in analysis_text for word in ["pasta", "noodle", "noodles", "spaghetti"]):
            features["food_pasta"] = 0.8
        
        if any(word in analysis_text for word in ["bread", "toast", "bakery", "baked"]):
            features["food_bread"] = 0.75
        
        if any(word in analysis_text for word in ["potato", "potatoes", "root vegetable"]):
            features["food_potatoes"] = 0.75
        
        # Dairy
        if any(word in analysis_text for word in ["cheese", "cheddar", "mozzarella", "parmesan"]):
            features["food_cheese"] = 0.8
        
        # Yogurt vs cream sauce disambiguation
        if any(word in analysis_text for word in ["yogurt", "yoghurt"]):
            features["food_yogurt"] = 0.8
        
        if any(word in analysis_text for word in ["sauce", "cream sauce", "creamy", "gravy", "dressing"]):
            features["food_cream_sauce"] = 0.8
            features["food_yogurt"] = 0.0  # Reset yogurt if cream sauce detected
        elif any(word in analysis_text for word in ["dairy", "milk"]):
            features["food_yogurt"] = 0.7
        
        # Prepared/packaged
        if any(word in analysis_text for word in ["plated", "plate", "prepared", "cooked", "cooking"]):
            features["food_prepared_meal"] = 0.8
        
        if any(word in analysis_text for word in ["package", "packaged", "packaging", "box", "container", "sealed"]):
            features["food_packaged"] = 0.7
        
        # Product visibility
        if any(word in analysis_text for word in ["product", "package", "box", "bottle", "brand", "logo"]) or has_product_mentions(transcript):
            features["product_visible"] = 0.8
        
        # Colors
        if any(word in analysis_text for word in ["warm", "orange", "red", "yellow", "golden", "bright"]):
            features["color_dominant_warm"] = 0.8
        if any(word in analysis_text for word in ["cool", "blue", "green", "purple", "cold", "cyan"]):
            features["color_dominant_cool"] = 0.8
        
        # Settings
        if any(word in analysis_text for word in ["outdoor", "outside", "street", "park", "nature", "sky"]):
            features["outdoor_setting"] = 0.8
        if any(word in analysis_text for word in ["indoor", "inside", "room", "kitchen", "office", "home", "house"]):
            features["indoor_setting"] = 0.8
        
        # Motion/animation
        if any(word in analysis_text for word in ["motion", "moving", "animation", "animated", "action", "dynamic"]):
            features["motion_heavy"] = 0.75
            features["fast_pacing"] = 0.7
        
        # Text overlays
        if any(word in analysis_text for word in ["text", "writing", "title", "subtitle", "label", "caption", "overlay"]):
            features["text_overlay"] = 0.75
        
        # Animation
        if "animation" in analysis_text or "animated" in analysis_text:
            features["animation"] = 0.8
        
        # Defaults for undetected
        if features.get("shows_food", 0) == 0:
            features["shows_food"] = 0.4
        if features.get("shows_people", 0) == 0:
            features["shows_people"] = 0.5
        if features.get("color_dominant_warm", 0) == 0 and features.get("color_dominant_cool", 0) == 0:
            features["color_dominant_warm"] = 0.45
        
        return features
    
    except Exception as e:
        print(f"Warning: Claude vision analysis failed ({e}). Using fallback heuristics.")
        # Fallback to simple heuristics
        features["shows_food"] = 0.4
        features["shows_people"] = 0.5
        features["color_dominant_warm"] = 0.45
        if has_product_mentions(transcript):
            features["product_visible"] = 0.7
        return features

def analyze_visual_features(frames: List[Dict], transcript: str = "", override_gender: str = None) -> Dict[str, float]:
    """Wrapper that uses Claude vision."""
    return analyze_visual_features_with_claude(frames, transcript, override_gender)

# ============================================================================
# OPENAI NARRATIVE ANALYSIS (GPT-4o Vision)
# ============================================================================

def analyze_narrative_with_openai(frames: List[Dict], transcript: str, duration: float) -> Dict:
    """
    Use OpenAI GPT-4o vision to analyze the ad's narrative structure,
    storytelling flow, pacing, transitions, and emotional arc.

    Sends a broader set of frames (up to 15) to capture the full narrative
    progression, unlike Claude which samples 3 for feature detection.

    Returns dict with narrative analysis fields:
      - narrative_arc: description of story structure
      - pacing_analysis: assessment of pacing and rhythm
      - transition_style: how scenes connect
      - emotional_progression: emotional journey
      - storytelling_score: 1-10 rating
      - storytelling_techniques: list of techniques used
      - hook_analysis: opening hook effectiveness
      - cta_effectiveness: closing CTA strength
      - improvement_suggestions: list of suggestions
    """
    if not _OPENAI_AVAILABLE:
        return {}

    print("Analyzing narrative structure with OpenAI GPT-4o...")

    # Sample more frames for narrative flow (up to 15, evenly distributed)
    num_frames = len(frames)
    if num_frames <= 15:
        sample_frames = frames
    else:
        indices = [int(i * (num_frames - 1) / 14) for i in range(15)]
        sample_frames = [frames[i] for i in indices]

    try:
        client = OpenAI()

        # Build content with frames in chronological order
        content = []
        content.append({
            "type": "text",
            "text": f"""You are an expert ad creative analyst specializing in video storytelling and narrative structure.

Analyze these {len(sample_frames)} chronological frames from a {duration:.0f}-second video ad.
{"The ad's transcript: " + transcript[:1500] if transcript and transcript != "N/A" else "This ad has no voiceover (music/visual only)."}

Provide a structured analysis in this EXACT JSON format (no markdown, just JSON):
{{
  "narrative_arc": "Describe the story structure (e.g., problem→solution, hero's journey, day-in-the-life, product showcase, testimonial)",
  "pacing_analysis": "How does the pacing evolve? Fast/slow starts, acceleration, rhythm of cuts",
  "transition_style": "How do scenes connect? Hard cuts, dissolves, matched action, thematic links",
  "emotional_progression": "Map the emotional journey from first to last frame (e.g., curiosity→desire→urgency)",
  "storytelling_score": 7,
  "storytelling_techniques": ["technique1", "technique2"],
  "hook_analysis": "How effective is the opening 3 seconds at grabbing attention?",
  "cta_effectiveness": "How well does the ending drive action?",
  "key_narrative_moments": ["moment1 at ~Xs", "moment2 at ~Xs"],
  "improvement_suggestions": ["suggestion1", "suggestion2"]
}}

Focus on STORYTELLING and NARRATIVE CRAFT, not just listing what's visible. Think like a creative director reviewing this ad."""
        })

        for i, frame in enumerate(sample_frames):
            frame_data = frame.get("frame_data", "")
            if frame_data:
                timestamp = frame.get("timestamp", i * 3)
                content.append({
                    "type": "text",
                    "text": f"[Frame at {timestamp}s]"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_data}",
                        "detail": "low"  # Save tokens, narrative doesn't need pixel detail
                    }
                })

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1200,
            messages=[
                {"role": "user", "content": content}
            ]
        )

        response_text = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        narrative_data = json.loads(response_text)
        print(f"✓ OpenAI narrative analysis complete (storytelling score: {narrative_data.get('storytelling_score', '?')}/10)")
        return narrative_data

    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse OpenAI narrative response as JSON ({e})")
        # Return raw text as fallback
        return {"narrative_arc": response_text[:500] if 'response_text' in dir() else "Parse error"}
    except Exception as e:
        print(f"Warning: OpenAI narrative analysis failed ({e})")
        return {}

def analyze_audio_features(transcript: str) -> Dict[str, float]:
    """
    Analyze transcript for audio features using NLP patterns.
    Returns dict of {feature: confidence_score}
    """
    print("Analyzing audio features...")
    
    features = {f: 0.0 for f in AUDIO_FEATURES.keys()}
    
    if not transcript or len(transcript) < 10:
        return features
    
    transcript_lower = transcript.lower()
    
    # Human talking (presence of transcript = speech)
    features["human_talking"] = min(0.95, 0.5 + len(transcript) / 200)
    
    # Voiceover detection (professional tone, formal words)
    voiceover_keywords = ["we", "you", "discover", "explore", "introducing", "now available", "learn more"]
    voiceover_score = sum(1 for kw in voiceover_keywords if kw in transcript_lower) / len(voiceover_keywords)
    features["voiceover"] = min(0.85, voiceover_score * 1.5)
    
    # Upbeat audio (energetic words)
    upbeat_keywords = ["amazing", "incredible", "fantastic", "awesome", "love", "exciting", "new", "best"]
    upbeat_score = sum(1 for kw in upbeat_keywords if kw in transcript_lower) / max(1, len(upbeat_keywords))
    features["upbeat_audio"] = min(0.8, upbeat_score)
    
    # Calm audio (soothing words)
    calm_keywords = ["relax", "peace", "comfortable", "easy", "simple", "natural", "gentle", "calm"]
    calm_score = sum(1 for kw in calm_keywords if kw in transcript_lower) / max(1, len(calm_keywords))
    features["calm_audio"] = min(0.75, calm_score)
    
    # Music/sound effects (harder without audio analysis, but can detect from context)
    if any(word in transcript_lower for word in ["music", "beat", "sound", "rhythm"]):
        features["music"] = 0.6
    
    return features

def analyze_overall_features(transcript: str) -> Dict[str, float]:
    """
    Analyze overall ad characteristics using pattern matching.
    """
    print("Analyzing overall features...")
    
    features = {f: 0.0 for f in OVERALL_FEATURES.keys()}
    transcript_lower = transcript.lower()
    
    if not transcript:
        return features
    
    # CTA detection (explicit calls to action)
    cta_keywords = ["call", "visit", "buy", "click", "download", "subscribe", "order", "get", "start", "join", "sign up", "try", "explore"]
    cta_score = sum(1 for kw in cta_keywords if kw in transcript_lower) / len(cta_keywords)
    features["call_to_action"] = min(0.95, cta_score * 1.2)
    
    # Emotional appeal (inspirational language)
    emotional_keywords = ["amazing", "incredible", "transform", "life-changing", "discover", "beautiful", "love", "best", "dream", "inspire", "powerful", "good", "great"]
    emotional_score = sum(1 for kw in emotional_keywords if kw in transcript_lower) / max(1, len(emotional_keywords))
    features["emotional_appeal"] = min(0.85, emotional_score + 0.3)  # Boost score
    
    # Humor (comedic tone)
    humor_keywords = ["laugh", "funny", "hilarious", "silly", "crazy", "insane"]
    has_humor = sum(1 for kw in humor_keywords if kw in transcript_lower) > 0
    features["humor"] = 0.75 if has_humor else 0.0
    
    # Problem-solution structure
    problem_keywords = ["problem", "struggle", "tired", "frustrated", "difficult", "challenge", "tired of", "can't", "impossible", "hard", "struggle"]
    solution_keywords = ["solution", "discover", "finally", "now you can", "easy", "simple", "here's", "introducing", "try", "have", "ready-made", "prepared"]
    problem_count = sum(1 for kw in problem_keywords if kw in transcript_lower)
    solution_count = sum(1 for kw in solution_keywords if kw in transcript_lower)
    if problem_count > 0 or solution_count > 0:
        features["problem_solution"] = min(0.9, 0.4 + (problem_count + solution_count) * 0.15)
    
    # Aspirational (lifestyle, dreams, success)
    aspirational_keywords = ["success", "achieve", "dream", "freedom", "lifestyle", "confidence", "strong", "healthy", "fit", "winning"]
    aspirational_score = sum(1 for kw in aspirational_keywords if kw in transcript_lower) / len(aspirational_keywords)
    features["aspirational"] = min(0.8, aspirational_score * 1.3)
    
    # Fast pacing detection (short sentences = fast editing typically)
    sentences = [s.strip() for s in transcript.split('.') if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 10
    features["fast_pacing"] = 0.7 if avg_sentence_length < 10 else 0.3
    
    return features

# ============================================================================
# TRANSLATION
# ============================================================================

def translate_to_english(text: str, source_language: str) -> str:
    """
    Translate text to English if not already in English.
    """
    if source_language == "en" or source_language == "english":
        return text
    
    print(f"Translating from {source_language} to English...")
    
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source_auto=True, target='en').translate(text)
        print(f"✓ Translated {len(text)} chars to {len(translated)} chars")
        return translated
    except ImportError:
        print("Warning: deep_translator not installed. Using original transcript.")
        return text
    except Exception as e:
        print(f"Warning: Translation failed ({e}). Using original transcript.")
        return text

# ============================================================================
# NOTION UPLOAD
# ============================================================================

def generate_narrative_summary(
    visual_features: Dict,
    audio_features: Dict,
    overall_features: Dict,
    openai_narrative: Dict = None,
) -> str:
    """
    Generate a narrative summary combining Claude's feature detection with
    OpenAI's storytelling analysis. Returns a rich, readable summary.

    When OpenAI narrative data is available, the summary includes:
    - Feature-based description (from Claude)
    - Storytelling structure and emotional arc (from OpenAI)
    - Pacing and transition insights (from OpenAI)
    """
    sections = []

    # ── Section 1: Feature-based description (Claude) ──
    feature_parts = []

    # People/talent
    people = []
    if visual_features.get("people_male", 0) > 0.7:
        people.append("male talent")
    if visual_features.get("people_female", 0) > 0.7:
        people.append("female talent")
    if visual_features.get("people_child", 0) > 0.7:
        people.append("children")
    if people:
        feature_parts.append(f"Features {', '.join(people)}")

    # Food/product details
    food_items = []
    food_mapping = {
        "food_vegetables": "fresh vegetables",
        "food_chicken": "chicken",
        "food_rice": "rice",
        "food_potatoes": "potatoes",
        "food_leafy_greens": "leafy greens",
        "food_fruits": "fruits",
        "food_prepared_meal": "prepared meals",
        "food_salmon": "salmon",
        "food_cream_sauce": "creamy sauces",
    }
    for feat, label in food_mapping.items():
        if visual_features.get(feat, 0) > 0.7:
            food_items.append(label)
    if food_items:
        feature_parts.append(f"prominently displays {', '.join(food_items)}")

    # Setting
    if visual_features.get("indoor_setting", 0) > 0.7:
        feature_parts.append("in an indoor kitchen setting")
    elif visual_features.get("outdoor_setting", 0) > 0.7:
        feature_parts.append("in an outdoor setting")

    # Colors
    colors = []
    if visual_features.get("color_dominant_warm", 0) > 0.7:
        colors.append("warm")
    if visual_features.get("color_dominant_cool", 0) > 0.7:
        colors.append("cool")
    if colors:
        feature_parts.append(f"with {'/'.join(colors)} color tones")

    # Motion/pacing
    if overall_features.get("fast_pacing", 0) > 0.7:
        feature_parts.append("using fast-paced cuts and dynamic motion")
    if visual_features.get("motion_heavy", 0) > 0.7:
        feature_parts.append("emphasizing movement and energy")

    # Audio
    audio_desc = []
    if audio_features.get("human_talking", 0) > 0.7:
        audio_desc.append("voiceover narration")
    if audio_features.get("music", 0) > 0.7:
        audio_desc.append("background music")
    if audio_features.get("sound_effects", 0) > 0.7:
        audio_desc.append("sound effects")
    if audio_desc:
        feature_parts.append(f"with {' and '.join(audio_desc)}")

    # Text / product / CTA
    if visual_features.get("text_overlay", 0) > 0.7:
        feature_parts.append("incorporating text overlays")
    if visual_features.get("product_visible", 0) > 0.7:
        feature_parts.append("with clear product visibility")
    if overall_features.get("call_to_action", 0) > 0.7:
        feature_parts.append("and a clear call-to-action")
    if overall_features.get("emotional_appeal", 0) > 0.7:
        feature_parts.append("evoking emotional engagement")
    if overall_features.get("aspirational", 0) > 0.7:
        feature_parts.append("portraying an aspirational lifestyle")

    feature_summary = " ".join(feature_parts)
    if feature_summary:
        feature_summary = feature_summary[0].upper() + feature_summary[1:] + "."
    else:
        feature_summary = "Ad features multiple visual and audio elements working together."

    sections.append(feature_summary)

    # ── Section 2: Storytelling analysis (OpenAI) ──
    if openai_narrative:
        story_parts = []

        # Narrative arc
        arc = openai_narrative.get("narrative_arc", "")
        if arc:
            story_parts.append(f"Narrative structure: {arc}")

        # Emotional progression
        emo = openai_narrative.get("emotional_progression", "")
        if emo:
            story_parts.append(f"Emotional arc: {emo}")

        # Pacing
        pacing = openai_narrative.get("pacing_analysis", "")
        if pacing:
            story_parts.append(f"Pacing: {pacing}")

        # Transitions
        transitions = openai_narrative.get("transition_style", "")
        if transitions:
            story_parts.append(f"Transitions: {transitions}")

        # Hook & CTA
        hook = openai_narrative.get("hook_analysis", "")
        if hook:
            story_parts.append(f"Opening hook: {hook}")

        cta = openai_narrative.get("cta_effectiveness", "")
        if cta:
            story_parts.append(f"Closing CTA: {cta}")

        # Storytelling score
        score = openai_narrative.get("storytelling_score")
        if score is not None:
            story_parts.append(f"Storytelling score: {score}/10")

        if story_parts:
            sections.append("\n".join(story_parts))

        # Techniques
        techniques = openai_narrative.get("storytelling_techniques", [])
        if techniques:
            sections.append("Techniques used: " + ", ".join(techniques))

        # Improvement suggestions
        suggestions = openai_narrative.get("improvement_suggestions", [])
        if suggestions:
            sections.append("Suggestions: " + "; ".join(suggestions))

    return "\n\n".join(sections)

def upload_to_notion(
    database_id: str,
    video_name: str,
    duration: float,
    transcript_english: str,
    transcript_original: str,
    video_path: str,
    source_url: str,
    num_frames: int,
    quality_level: str,
    quality_details: str,
    quality_dict: Dict,
    feature_tags: List[str],  # Pre-mapped feature tags
    visual_features: Dict,
    audio_features: Dict,
    overall_features: Dict,
    all_features: Dict,  # Combined features dict for JSON
    openai_narrative: Dict = None,  # OpenAI storytelling analysis
    sentiment_result: Dict = None,  # Sentiment analysis results
) -> str:
    """
    Upload analysis results to Notion database with full analysis.
    Returns the page URL.
    """
    print("Uploading to Notion...")
    
    # Combine all features (for analysis blocks display)
    all_features = {**visual_features, **audio_features, **overall_features}
    detected_features = {k: v for k, v in all_features.items() if v > 0.3}
    
    try:
        # Generate visual description
        video_description = describe_video_visually(transcript_english, num_frames, duration)
        
        # Create feature scores JSON (all features with their scores)
        feature_scores_json = json.dumps(
            {k: round(v, 2) for k, v in sorted(all_features.items(), key=lambda x: x[1], reverse=True)},
            indent=2
        )
        
        # Create video quality analysis JSON
        quality_analysis_json = json.dumps(quality_dict, indent=2)
        
        # Build properties JSON
        properties = {
            "Video Duration": {"number": duration},
            "Frames analysed": {"number": num_frames},
            "Video Quality": {"select": {"name": quality_level}},  # Single select for quality level
            "Video Quality Analysis": {"rich_text": [{"text": {"content": quality_analysis_json}}]},  # Detailed metrics as JSON
            "Tags": {"multi_select": [{"name": tag} for tag in feature_tags]},
            "Feature_Scores": {"rich_text": [{"text": {"content": feature_scores_json}}]},  # All scores as JSON
            "English transcript": {"rich_text": [{"text": {"content": transcript_english[:2000]}}]},
            "Original transcript": {"rich_text": [{"text": {"content": transcript_original[:2000]}}]},
            "Video script": {"rich_text": [{"text": {"content": video_description}}]},  # Visual description
            "Upload Date": {"date": {"start": datetime.now().isoformat()}},
        }
        
        # Add OpenAI narrative analysis properties if available
        if all_features and openai_narrative:
            # Storytelling Score (number, 0-10)
            storytelling_score = openai_narrative.get("storytelling_score", 0)
            properties["Storytelling Score"] = {"number": storytelling_score}
            
            # Narrative Arc (text)
            narrative_arc = openai_narrative.get("narrative_arc", "N/A")
            properties["Narrative Arc"] = {"rich_text": [{"text": {"content": narrative_arc[:500]}}]}
            
            # Emotional Journey (text)
            emotional_arc = openai_narrative.get("emotional_progression", "N/A")
            properties["Emotional Journey"] = {"rich_text": [{"text": {"content": emotional_arc[:500]}}]}
            
            # Key Techniques (text)
            techniques = openai_narrative.get("storytelling_techniques", [])
            techniques_str = ", ".join(techniques) if techniques else "N/A"
            properties["Key Techniques"] = {"rich_text": [{"text": {"content": techniques_str[:500]}}]}
        
        # Add sentiment analysis properties if available
        if sentiment_result and sentiment_result.get("segments"):
            properties["Sentiment Overall"] = {"select": {"name": sentiment_result.get("overall_label", "Neutral")}}
            properties["Sentiment Score"] = {"number": round(sentiment_result.get("overall_score", 0.0), 4)}
            properties["Hook Sentiment"] = {"rich_text": [{"text": {"content": sentiment_result.get("hook_sentiment", "N/A")}}]}
            properties["CTA Sentiment"] = {"rich_text": [{"text": {"content": sentiment_result.get("cta_sentiment", "N/A")}}]}
            properties["Emotional Arc"] = {"rich_text": [{"text": {"content": sentiment_result.get("emotional_arc", "N/A")}}]}

        # Add source link if provided
        if source_url:
            properties["Source link"] = {"url": source_url}
        
        # Write properties to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(properties, f)
            props_file = f.name
        
        # Create page with --title and --properties
        cmd = f'source ~/.bashrc && notion pages create --title "{video_name}" --parent "database:{database_id}" --properties "$(cat {props_file})"'
        result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True)
        page_data = json.loads(result.stdout)
        page_id = page_data.get("id", "")
        page_url = page_data.get("url", "")
        
        if not page_id:
            print(f"Warning: Could not extract page ID")
            return ""
        
        # Generate narrative summary of features + storytelling insights
        narrative = generate_narrative_summary(visual_features, audio_features, overall_features, openai_narrative)
        
        # Now add full analysis as block content
        blocks_content = [
            {
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Creative Summary"}}]}
            },
            {
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": narrative}}]}
            },
            {
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Detailed Features"}}]}
            },
            {
                "type": "heading_3",
                "heading_3": {"rich_text": [{"type": "text", "text": {"content": "Top Detected Elements"}}]}
            },
        ]
        
        # Add top detected features with scores
        top_features = sorted(detected_features.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, score in top_features:
            feature_name = feature.replace('_', ' ').title()
            blocks_content.append({
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"• {feature_name}: {score:.0%}"}}]}
            })

        # Add storytelling analysis if OpenAI data is available
        if openai_narrative:
            blocks_content.append({
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Storytelling Analysis"}}]}
            })

            story_fields = [
                ("Narrative Arc", "narrative_arc"),
                ("Emotional Progression", "emotional_progression"),
                ("Pacing", "pacing_analysis"),
                ("Transitions", "transition_style"),
                ("Opening Hook", "hook_analysis"),
                ("CTA Effectiveness", "cta_effectiveness"),
            ]
            for label, key in story_fields:
                value = openai_narrative.get(key, "")
                if value:
                    blocks_content.append({
                        "type": "paragraph",
                        "paragraph": {"rich_text": [
                            {"type": "text", "text": {"content": f"{label}: "}, "annotations": {"bold": True}},
                            {"type": "text", "text": {"content": str(value)}}
                        ]}
                    })

            score = openai_narrative.get("storytelling_score")
            if score is not None:
                blocks_content.append({
                    "type": "paragraph",
                    "paragraph": {"rich_text": [
                        {"type": "text", "text": {"content": f"Storytelling Score: "}, "annotations": {"bold": True}},
                        {"type": "text", "text": {"content": f"{score}/10"}}
                    ]}
                })

            techniques = openai_narrative.get("storytelling_techniques", [])
            if techniques:
                blocks_content.append({
                    "type": "paragraph",
                    "paragraph": {"rich_text": [
                        {"type": "text", "text": {"content": "Techniques: "}, "annotations": {"bold": True}},
                        {"type": "text", "text": {"content": ", ".join(techniques)}}
                    ]}
                })

            # Key narrative moments
            moments = openai_narrative.get("key_narrative_moments", [])
            if moments:
                blocks_content.append({
                    "type": "heading_3",
                    "heading_3": {"rich_text": [{"type": "text", "text": {"content": "Key Narrative Moments"}}]}
                })
                for moment in moments:
                    blocks_content.append({
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"• {moment}"}}]}
                    })

            # Improvement suggestions
            suggestions = openai_narrative.get("improvement_suggestions", [])
            if suggestions:
                blocks_content.append({
                    "type": "heading_3",
                    "heading_3": {"rich_text": [{"type": "text", "text": {"content": "Improvement Suggestions"}}]}
                })
                for suggestion in suggestions:
                    blocks_content.append({
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"• {suggestion}"}}]}
                    })
        
        # Add sentiment analysis blocks if available
        if sentiment_result and sentiment_result.get("segments"):
            blocks_content.append({
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Sentiment Analysis"}}]}
            })
            blocks_content.append({
                "type": "paragraph",
                "paragraph": {"rich_text": [
                    {"type": "text", "text": {"content": f"Overall: "}, "annotations": {"bold": True}},
                    {"type": "text", "text": {"content": f"{sentiment_result['overall_label']} (score: {sentiment_result['overall_score']:.2f})"}}
                ]}
            })
            blocks_content.append({
                "type": "paragraph",
                "paragraph": {"rich_text": [
                    {"type": "text", "text": {"content": f"Hook Sentiment: "}, "annotations": {"bold": True}},
                    {"type": "text", "text": {"content": sentiment_result.get("hook_sentiment", "N/A")}}
                ]}
            })
            blocks_content.append({
                "type": "paragraph",
                "paragraph": {"rich_text": [
                    {"type": "text", "text": {"content": f"CTA Sentiment: "}, "annotations": {"bold": True}},
                    {"type": "text", "text": {"content": sentiment_result.get("cta_sentiment", "N/A")}}
                ]}
            })
            blocks_content.append({
                "type": "paragraph",
                "paragraph": {"rich_text": [
                    {"type": "text", "text": {"content": f"Emotional Arc: "}, "annotations": {"bold": True}},
                    {"type": "text", "text": {"content": sentiment_result.get("emotional_arc", "N/A")}}
                ]}
            })

            # Per-segment breakdown
            if len(sentiment_result["segments"]) > 1:
                blocks_content.append({
                    "type": "heading_3",
                    "heading_3": {"rich_text": [{"type": "text", "text": {"content": "Segment Breakdown"}}]}
                })
                for seg in sentiment_result["segments"]:
                    blocks_content.append({
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {
                            "content": f"• [{seg['label']} {seg['confidence']:.0%}] {seg['text']}"
                        }}]}
                    })

        # Write blocks to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(blocks_content, f)
            blocks_file = f.name
        
        # Append blocks to page
        cmd_blocks = f'source ~/.bashrc && notion blocks append "{page_id}" --content "$(cat {blocks_file})"'
        result = subprocess.run(["bash", "-c", cmd_blocks], capture_output=True, text=True, check=True)
        
        print(f"✓ Created Notion page: {page_url}")
        return page_url
    
    except subprocess.CalledProcessError as e:
        print(f"Warning: Notion upload failed. {e.stderr}")
        return ""
    
    finally:
        if 'props_file' in locals() and os.path.exists(props_file):
            os.unlink(props_file)
        if 'blocks_file' in locals() and os.path.exists(blocks_file):
            os.unlink(blocks_file)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze video ads")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--notion-db-id", required=True, help="Notion database ID")
    parser.add_argument("--interval", type=int, default=3, help="Frame extraction interval (seconds)")
    parser.add_argument("--original-name", help="Original video filename (if different from path)")
    parser.add_argument("--source-url", help="Source/download URL of the video")
    parser.add_argument("--language", help="Override detected language (e.g., 'da' for Danish, 'sv' for Swedish)")
    parser.add_argument("--gender", help="Override detected gender (female or male)", choices=["female", "male"])
    parser.add_argument("--skip-openai", action="store_true", help="Skip OpenAI narrative analysis (use Claude only)")
    parser.add_argument("--sentiment", action="store_true", help="Run sentiment analysis on transcript (uses HuggingFace Inference API)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AD ANALYZER")
    print("=" * 70)
    print()
    
    # Track module results: key -> (status, detail)
    # status: "ok", "skip", "fail"
    analysis_modules = {}
    
    try:
        # Fetch existing tags from Notion database
        existing_tags = get_existing_tags(args.notion_db_id)
        
        # Get video filename
        video_filename = args.original_name
        
        # If no original name provided, try to extract from source URL
        if not video_filename and args.source_url:
            extracted_name, _ = extract_filename_from_drive_link(args.source_url)
            video_filename = extracted_name
        
        # Fallback to local filename
        if not video_filename:
            video_filename = Path(args.video).name
        
        # Analyze video quality
        quality_level, quality_details, quality_dict = analyze_video_quality(args.video)
        
        # Extract frames
        frames, duration = extract_frames(args.video, interval_seconds=args.interval)
        analysis_modules["frame_extraction"] = ("ok", f"{len(frames)} frames @ {args.interval}s intervals")
        
        # Transcribe audio (returns transcript + language)
        # Pass analysis_modules so Whisper/Google STT can populate their entries
        transcript_original, detected_language = transcribe_audio(args.video, analysis_modules)
        
        # Use provided language override if given, otherwise use detected language
        language = args.language if args.language else detected_language
        
        # Translate to English if needed
        transcript_english = transcript_original
        if language != "en":
            transcript_english = translate_to_english(transcript_original, language)
        
        # Analyze features (use English transcript)
        visual = analyze_visual_features(frames, transcript_english, args.gender)
        audio = analyze_audio_features(transcript_english)
        overall = analyze_overall_features(transcript_english)
        analysis_modules["claude_vision"] = ("ok", "feature detection")

        # OpenAI narrative/storytelling analysis (optional, runs if API key set)
        openai_narrative = {}
        if _OPENAI_AVAILABLE and not args.skip_openai:
            openai_narrative = analyze_narrative_with_openai(frames, transcript_english, duration)
            if openai_narrative:
                score = openai_narrative.get("storytelling_score", "?")
                analysis_modules["openai_narrative"] = ("ok", f"storytelling score: {score}/10")
            else:
                analysis_modules["openai_narrative"] = ("fail", "no response")
        elif args.skip_openai:
            print("Skipping OpenAI narrative analysis (--skip-openai flag)")
            analysis_modules["openai_narrative"] = ("skip", "skipped (--skip-openai)")
        elif not os.environ.get("OPENAI_API_KEY"):
            analysis_modules["openai_narrative"] = ("skip", "no API key")
        else:
            analysis_modules["openai_narrative"] = ("skip", "SDK not installed")

        # Sentiment analysis (optional)
        sentiment_result = {}
        if args.sentiment:
            print()
            print("Running sentiment analysis on transcript...")
            try:
                from sentiment_analyzer import analyze_transcript_sentiment
                sentiment_result = analyze_transcript_sentiment(transcript_english)
                print(f"✓ Sentiment analysis complete: {sentiment_result.get('overall_label', '?')} "
                      f"(score: {sentiment_result.get('overall_score', 0):.2f})")
                overall_label = sentiment_result.get('overall_label', '?')
                overall_score = sentiment_result.get('overall_score', 0)
                analysis_modules["hf_sentiment"] = ("ok", f"overall: {overall_label} ({overall_score:.2f})")
            except Exception as e:
                print(f"Warning: Sentiment analysis failed ({e})")
                analysis_modules["hf_sentiment"] = ("fail", f"failed ({e})")
        else:
            analysis_modules["hf_sentiment"] = ("skip", "skipped (no --sentiment flag)")

        # Prepare summary
        print()
        print("=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Video: {video_filename}")
        print(f"Duration: {duration:.1f}s")
        print(f"Frames extracted: {len(frames)}")
        print(f"Language: {language}")
        print(f"Transcript length: {len(transcript_english)} chars")
        print()
        
        # Detected features (threshold 0.3 for all features to catch more)
        all_features = {**visual, **audio, **overall}
        detected = {k: v for k, v in all_features.items() if v > 0.3}
        
        # Map detected features to existing tags or create new ones
        feature_tags = map_features_to_tags(detected, existing_tags)
        
        if detected:
            print("Detected Features:")
            for feature, score in sorted(detected.items(), key=lambda x: x[1], reverse=True):
                print(f"  ✓ {feature.replace('_', ' ').title()}: {score:.0%}")
        else:
            print("No high-confidence features detected.")
        
        # Show storytelling insights if available
        if openai_narrative:
            print()
            print("Storytelling Analysis (OpenAI):")
            score = openai_narrative.get("storytelling_score", "?")
            print(f"  Storytelling Score: {score}/10")
            arc = openai_narrative.get("narrative_arc", "")
            if arc:
                print(f"  Narrative Arc: {arc[:120]}...")
            emo = openai_narrative.get("emotional_progression", "")
            if emo:
                print(f"  Emotional Arc: {emo[:120]}...")

        # Show sentiment analysis if available
        if sentiment_result and sentiment_result.get("segments"):
            print()
            print("Sentiment Analysis:")
            print(f"  Overall: {sentiment_result['overall_label']} (score: {sentiment_result['overall_score']:.2f})")
            print(f"  Hook Sentiment: {sentiment_result['hook_sentiment']}")
            print(f"  CTA Sentiment: {sentiment_result['cta_sentiment']}")
            print(f"  Emotional Arc: {sentiment_result['emotional_arc']}")

        print()
        print("Transcript (English):")
        print(f"  {transcript_english[:200]}..." if len(transcript_english) > 200 else f"  {transcript_english}")
        print()
        
        # Upload to Notion (with pre-mapped feature_tags + OpenAI narrative)
        notion_url = upload_to_notion(
            args.notion_db_id,
            video_filename,
            duration,
            transcript_english,
            transcript_original,
            args.video,
            args.source_url or "",
            len(frames),
            quality_level,
            quality_details,
            quality_dict,
            feature_tags,  # Use pre-mapped tags
            visual,
            audio,
            overall,
            all_features,
            openai_narrative=openai_narrative,
            sentiment_result=sentiment_result,
        )
        
        if notion_url:
            print(f"View in Notion: {notion_url}")
            analysis_modules["notion_upload"] = ("ok", notion_url)
        else:
            analysis_modules["notion_upload"] = ("fail", "no URL returned")
        
        # Print Analysis Modules summary
        print()
        print("Analysis Modules:")
        module_display_order = [
            ("frame_extraction", "Frame Extraction"),
            ("whisper", "Whisper"),
            ("google_stt", "Google Cloud STT"),
            ("claude_vision", "Claude Vision"),
            ("openai_narrative", "GPT-4o Narrative"),
            ("hf_sentiment", "HuggingFace Sentiment"),
            ("notion_upload", "Notion Upload"),
        ]
        for key, label in module_display_order:
            status, detail = analysis_modules.get(key, ("skip", "not configured"))
            icon = "✓" if status == "ok" else "✗"
            print(f"  {icon} {label:<24} {detail}")
        
        print()
        print("=" * 70)
        print("✓ Analysis complete")
        print("=" * 70)
        
        # Clean up: Delete the video file after analysis
        try:
            video_path = Path(args.video)
            if video_path.exists():
                video_path.unlink()
                print(f"✓ Cleaned up: Deleted {video_path}")
        except Exception as cleanup_error:
            print(f"Warning: Could not delete video file: {cleanup_error}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
