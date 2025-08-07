import os
import pathlib
import time
import re
import datetime
from datetime import timedelta
import subprocess
import shutil
import argparse
import sys
from faster_whisper import WhisperModel
import srt
import pysrt
from tqdm import tqdm
import torch
import json

# =================================================================
# --- ⚙️ Global Parameters (Configure Everything Here) ---
# =================================================================

# 1. File & Folder Paths
VIDEO_SOURCE_PATH = '/Volumes/Macintosh HD/Downloads/Video/uc'
AUDIO_SOURCE_PATH = '/Volumes/Macintosh HD/Downloads'  # For direct audio input
SRT_OUTPUT_PATH = '/Volumes/Macintosh HD/Downloads/srt'
MODEL_STORAGE_PATH = '/Volumes/Macintosh HD/Downloads/srt/whisper_models'
TEMP_PATH = '/Volumes/Macintosh HD/Downloads/srt/temp'
AUDIO_OUTPUT_PATH = '/Volumes/Macintosh HD/Downloads/srt/audio_cache'  # Permanent audio storage

# 2. Language & Task Settings
SOURCE_LANGUAGE = "ja"              # Source language (ISO code: ja for Japanese)
TARGET_LANGUAGE_CODE = "en-US"      # NOTE: faster-whisper only translates to English. This sets the output language code.
TASK = "translate"                  # "transcribe" or "translate"

# 3. Model & Transcription Configuration
MODEL_SIZE = "large-v3"
# OPTIMIZED: Auto-detect best device with MPS support
DEVICE = "auto"  # Will auto-detect MPS, CUDA, or CPU
COMPUTE_TYPE = "mps" #"auto"  # Will auto-select best compute type for device
CREDIT = "Subbed by Gemini"

# Model caching configuration
FORCE_OFFLINE_MODE = False  # Set to True to prevent any internet downloads
LOCAL_FILES_ONLY = True   # Use only locally cached models

# 4. File Type Configuration
# Supported video formats for conversion
VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
# Supported audio formats for direct transcription
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma']

# 5. Advanced Performance Settings
# MPS-specific optimizations
MPS_HIGH_WATERMARK_RATIO = 0.0  # Use maximum MPS memory
CPU_THREADS = 0  # 0 = auto-detect optimal thread count
NUM_WORKERS = 1  # Number of parallel workers (keep at 1 for MPS stability)

# 6. SRT Cleaning & Word Lists
PUNCT_MATCH = ["。", "、", ",", ".", "〜", "！", "!", "？", "?", "-"]
GARBAGE_LIST = ["a", "hmm", "huh", "oh"]
SUPPRESS_HIGH = ["subscribe", "my channel", "ご視聴ありがとうございました"]

# Language code mapping for output filenames
TO_LANGUAGE_CODE = {
    "ja": "ja-JP",      # Japanese
    "en": "en-US",      # English
    "es": "es-ES",      # Spanish
    "fr": "fr-FR",      # French
    "de": "de-DE",      # German
    "zh": "zh-CN",      # Chinese (Simplified)
    "ko": "ko-KR",      # Korean
    "pt": "pt-BR",      # Portuguese
    "ru": "ru-RU",      # Russian
    "it": "it-IT",      # Italian
    "ar": "ar-SA",      # Arabic
    "hi": "hi-IN",      # Hindi
    "th": "th-TH",      # Thai
    "vi": "vi-VN",      # Vietnamese
    "nl": "nl-NL",      # Dutch
    "sv": "sv-SE",      # Swedish
    "no": "no-NO",      # Norwegian
    "da": "da-DK",      # Danish
    "fi": "fi-FI",      # Finnish
    "pl": "pl-PL",      # Polish
    "cs": "cs-CZ",      # Czech
    "hu": "hu-HU",      # Hungarian
    "tr": "tr-TR",      # Turkish
    "he": "he-IL",      # Hebrew
    "id": "id-ID",      # Indonesian
    "ms": "ms-MY",      # Malay
    "uk": "uk-UA",      # Ukrainian
}

# =================================================================
# --- System Optimization and Environment Setup ---
# =================================================================

def fix_openmp_conflicts():
    """
    Fix OpenMP library conflicts that can cause crashes.
    """
    # Set environment variable to allow duplicate OpenMP libraries
    # This is a workaround for the common OpenMP conflict issue
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads to prevent conflicts

    # Additional MKL optimizations for Intel processors
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    print("🔧 OpenMP conflict fixes applied")

def setup_offline_environment():
    """
    Setup environment variables to force offline mode and use local model cache.
    """
    print("🔒 Setting up offline environment...")

    # Set Hugging Face cache directory
    os.environ['HF_HOME'] = MODEL_STORAGE_PATH
    os.environ['TRANSFORMERS_CACHE'] = MODEL_STORAGE_PATH
    os.environ['HF_HUB_CACHE'] = MODEL_STORAGE_PATH

    if FORCE_OFFLINE_MODE:
        # Force offline mode - no internet downloads
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        print("✅ Offline mode enabled - no internet downloads will occur")

    # Create model storage directory structure
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

    # Check if model exists locally
    model_exists = check_local_model_exists(MODEL_SIZE)
    if model_exists:
        print(f"✅ Model '{MODEL_SIZE}' found locally")
    else:
        print(f"⚠️ Model '{MODEL_SIZE}' not found locally")
        if FORCE_OFFLINE_MODE:
            print("❌ Cannot proceed in offline mode without cached model")
            print(f"💡 To download model, temporarily set FORCE_OFFLINE_MODE = False")
            return False
        else:
            print("📥 Model will be downloaded and cached for future offline use")

    return True

def check_local_model_exists(model_size):
    """
    Check if the whisper model exists locally in the cache.
    """
    # Common paths where faster-whisper stores models
    possible_paths = [
        os.path.join(MODEL_STORAGE_PATH, f"models--Systran--faster-whisper-{model_size}"),
        os.path.join(MODEL_STORAGE_PATH, f"faster-whisper-{model_size}"),
        os.path.join(MODEL_STORAGE_PATH, f"models--openai--whisper-{model_size}"),
        os.path.join(MODEL_STORAGE_PATH, model_size),
    ]

    # Check for key model files
    key_files = ['model.bin', 'config.json', 'tokenizer.json', 'vocabulary.json', 'preprocessor_config.json']

    for path in possible_paths:
        if os.path.exists(path):
            # Check if all required files exist
            files_found = []
            for file in key_files:
                file_path = os.path.join(path, file)
                if os.path.exists(file_path):
                    files_found.append(file)

            if len(files_found) >= 3:  # At least 3 key files should exist
                print(f"📁 Model found at: {path}")
                print(f"📄 Files found: {', '.join(files_found)}")
                return True

    return False

def download_and_cache_model(model_size):
    """
    Download model with proper caching to ensure offline availability.
    """
    print(f"📥 Downloading and caching model '{model_size}'...")
    print("⚠️ This is a one-time download. Future runs will be offline.")

    # Temporarily disable offline mode for download
    old_offline = os.environ.get('HF_HUB_OFFLINE', '0')
    old_transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE', '0')

    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)

    try:
        # Create a temporary model instance to trigger download
        temp_model = WhisperModel(
            model_size,
            device="cpu",  # Use CPU for download to avoid device issues
            compute_type="int8",
            download_root=MODEL_STORAGE_PATH,
            local_files_only=False  # Allow download
        )

        print("✅ Model downloaded and cached successfully")

        # Clean up temporary model
        del temp_model

        # Restore offline settings
        if old_offline == '1':
            os.environ['HF_HUB_OFFLINE'] = '1'
        if old_transformers_offline == '1':
            os.environ['TRANSFORMERS_OFFLINE'] = '1'

        return True

    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def detect_optimal_device_and_compute():
    """
    Detect the best available device and compute type for Apple Silicon M4.
    Returns tuple of (device, compute_type)
    """
    print("🔍 Detecting optimal device configuration...")

    # Fix OpenMP conflicts before device detection
    fix_openmp_conflicts()

    # Check system architecture first
    import platform
    system_info = platform.platform()
    print(f"📱 System: {system_info}")

    # Check for Apple Silicon specifically
    is_apple_silicon = platform.processor() == 'arm' or 'arm64' in platform.machine().lower()

    if is_apple_silicon:
        print("🍎 Apple Silicon detected!")

        # Check for MPS availability with better error handling
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("✅ MPS backend is available and built")

                # Test MPS functionality
                try:
                    test_tensor = torch.tensor([1.0]).to('mps')
                    del test_tensor
                    torch.mps.empty_cache()
                    print("✅ MPS functionality test passed")

                    # Set MPS memory optimization
                    if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                        torch.mps.set_per_process_memory_fraction(0.8)

                    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = str(MPS_HIGH_WATERMARK_RATIO)
                    return "mps", "float16"

                except Exception as mps_error:
                    print(f"⚠️ MPS test failed: {mps_error}")
                    print("🔄 Falling back to optimized CPU")

            else:
                print("⚠️ MPS backend not available or not built")

        except Exception as e:
            print(f"⚠️ MPS detection error: {e}")

    # Check for CUDA (unlikely on Mac but just in case)
    elif torch.cuda.is_available():
        print("✅ CUDA detected!")
        return "cuda", "float16"

    # Fallback to optimized CPU with better compute type for Apple Silicon
    print("🔧 Using CPU with optimizations")
    if is_apple_silicon:
        return "cpu", "float32"  # Better for Apple Silicon
    else:
        return "cpu", "int8"

def optimize_for_apple_silicon():
    """Apply Apple Silicon specific optimizations."""
    import platform

    # Fix OpenMP conflicts first
    fix_openmp_conflicts()

    is_apple_silicon = platform.processor() == 'arm' or 'arm64' in platform.machine().lower()

    if is_apple_silicon:
        print("🚀 Applying Apple Silicon optimizations...")

        # Set optimal thread count for Apple Silicon
        if CPU_THREADS == 0:
            # M4 typically has 10 cores (4P + 6E), but limit for stability
            optimal_threads = min(torch.get_num_threads(), 4)  # Conservative threading
            torch.set_num_threads(optimal_threads)
            print(f"📊 Using {optimal_threads} CPU threads")
        else:
            torch.set_num_threads(CPU_THREADS)

        # Enable MPS fallback for unsupported operations only if MPS is available
        if torch.backends.mps.is_available():
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

            # Clear any existing MPS cache
            try:
                torch.mps.empty_cache()
            except:
                pass  # Ignore if MPS cache operations fail

        print("✅ Apple Silicon optimizations applied")
    else:
        print("🔧 Applying Intel/x86 optimizations...")
        # Conservative settings for Intel processors
        if CPU_THREADS == 0:
            optimal_threads = min(torch.get_num_threads(), 4)
            torch.set_num_threads(optimal_threads)
            print(f"📊 Using {optimal_threads} CPU threads")
        print("✅ Intel optimizations applied")

def setup_model_with_fallback(model_size, device, compute_type):
    """
    Setup WhisperModel with intelligent fallback and proper offline caching.
    """
    print(f"📦 Loading Whisper model '{model_size}' from cache...")
    print(f"🎯 Target device: {device}, compute_type: {compute_type}")

    # Apply additional fixes for stability
    fix_openmp_conflicts()

    # First, try with the detected optimal settings and local files only
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=MODEL_STORAGE_PATH,
            local_files_only=LOCAL_FILES_ONLY,  # Force local files only
            num_workers=1  # Force single worker to prevent multiprocessing issues
        )
        print(f"✅ Model loaded successfully on {device} with {compute_type} (offline)")
        return model, device, compute_type

    except Exception as e:
        print(f"⚠️ Failed to load from cache on {device}: {e}")

        # If offline mode is disabled and model not found, try to download
        if not FORCE_OFFLINE_MODE and ("No such file or directory" in str(e) or "not found" in str(e).lower()):
            print("📥 Attempting to download model for first-time setup...")
            if download_and_cache_model(model_size):
                # Try again after download
                try:
                    model = WhisperModel(
                        model_size,
                        device=device,
                        compute_type=compute_type,
                        download_root=MODEL_STORAGE_PATH,
                        local_files_only=True,  # Now use cached version
                        num_workers=1
                    )
                    print(f"✅ Model loaded after download on {device} with {compute_type}")
                    return model, device, compute_type
                except Exception as download_error:
                    print(f"❌ Still failed after download: {download_error}")

        # Try fallback combinations with more conservative settings
        fallback_configs = [
            ("cpu", "float32"),  # Better for Apple Silicon
            ("cpu", "int8"),     # Most compatible
            ("auto", "auto")     # Let the library decide
        ]

        for fallback_device, fallback_compute in fallback_configs:
            try:
                print(f"🔄 Trying fallback: {fallback_device} with {fallback_compute}")
                model = WhisperModel(
                    model_size,
                    device=fallback_device,
                    compute_type=fallback_compute,
                    download_root=MODEL_STORAGE_PATH,
                    local_files_only=LOCAL_FILES_ONLY,
                    num_workers=1  # Single worker for stability
                )
                print(f"✅ Model loaded on fallback: {fallback_device} with {fallback_compute} (offline)")
                return model, fallback_device, fallback_compute

            except Exception as fallback_error:
                print(f"❌ Fallback {fallback_device} failed: {fallback_error}")
                continue

        # If all fallbacks fail
        error_msg = "Unable to load model. "
        if LOCAL_FILES_ONLY:
            error_msg += "Try downloading the model first by setting FORCE_OFFLINE_MODE = False"
        raise Exception(error_msg)

# =================================================================
# --- Language Support and Validation ---
# =================================================================

# Supported language codes for faster-whisper (ISO 639-1 codes)
SUPPORTED_LANGUAGES = [
    'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs',
    'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi',
    'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy',
    'id', 'is', 'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb',
    'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt',
    'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru',
    'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw',
    'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi',
    'yi', 'yo', 'zh', 'yue'
]

# Common language name to ISO code mapping for user convenience
LANGUAGE_NAME_TO_CODE = {
    'japanese': 'ja', 'english': 'en', 'spanish': 'es', 'french': 'fr',
    'german': 'de', 'chinese': 'zh', 'korean': 'ko', 'portuguese': 'pt',
    'russian': 'ru', 'italian': 'it', 'arabic': 'ar', 'hindi': 'hi',
    'thai': 'th', 'vietnamese': 'vi', 'dutch': 'nl', 'swedish': 'sv',
    'norwegian': 'no', 'danish': 'da', 'finnish': 'fi', 'polish': 'pl',
    'czech': 'cs', 'hungarian': 'hu', 'turkish': 'tr', 'hebrew': 'he',
    'indonesian': 'id', 'malay': 'ms', 'ukrainian': 'uk'
}

def validate_and_convert_language(language_input):
    """
    Validate language input and convert to proper ISO code if needed.
    Returns: (valid_iso_code, is_valid)
    """
    if not language_input:
        return None, False

    lang_lower = language_input.lower().strip()

    # Check if it's already a valid ISO code
    if lang_lower in SUPPORTED_LANGUAGES:
        return lang_lower, True

    # Check if it's a language name that can be converted
    if lang_lower in LANGUAGE_NAME_TO_CODE:
        iso_code = LANGUAGE_NAME_TO_CODE[lang_lower]
        print(f"📝 Converting '{language_input}' to ISO code: '{iso_code}'")
        return iso_code, True

    return language_input, False

def print_language_help():
    """Print helpful language information."""
    print("\n📚 Language Code Reference:")
    print("Common languages and their codes:")
    popular_langs = {
        'ja': 'Japanese', 'en': 'English', 'es': 'Spanish', 'fr': 'French',
        'de': 'German', 'zh': 'Chinese', 'ko': 'Korean', 'pt': 'Portuguese',
        'ru': 'Russian', 'it': 'Italian', 'ar': 'Arabic', 'hi': 'Hindi'
    }

    for code, name in popular_langs.items():
        print(f"  {code:<3} = {name}")

    print(f"\n💡 You can also use full names like 'japanese' (will convert to 'ja')")
    print(f"📋 Total supported languages: {len(SUPPORTED_LANGUAGES)}")

# =================================================================
# --- File Type Detection and Path Resolution ---
# =================================================================

def detect_file_type(filename):
    """
    Detect if the input file is a video or audio file.
    Returns: 'video', 'audio', or 'unknown'
    """
    _, ext = os.path.splitext(filename.lower())

    if ext in VIDEO_EXTENSIONS:
        return 'video'
    elif ext in AUDIO_EXTENSIONS:
        return 'audio'
    else:
        return 'unknown'

def find_input_file(filename):
    """
    Find the input file in either video or audio source paths.
    Returns: (full_path, file_type) or (None, None) if not found
    """
    # Try video path first
    video_path = os.path.join(VIDEO_SOURCE_PATH, filename)
    if os.path.exists(video_path):
        file_type = detect_file_type(filename)
        return video_path, file_type

    # Try audio path
    audio_path = os.path.join(AUDIO_SOURCE_PATH, filename)
    if os.path.exists(audio_path):
        file_type = detect_file_type(filename)
        return audio_path, file_type

    # Try both paths regardless of detected type (in case user has misnamed files)
    for path in [VIDEO_SOURCE_PATH, AUDIO_SOURCE_PATH]:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            file_type = detect_file_type(filename)
            return full_path, file_type

    return None, None

def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe for progress bar calculation."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        return duration
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError):
        return None

def convert_video_to_audio_with_progress(video_path, audio_output_path):
    """
    Converts a video file to MP3 using ffmpeg with real-time progress bar.
    """
    print(f"\n▶️ Converting video to audio: {os.path.basename(video_path)}")

    # Get video duration for progress calculation
    duration = get_video_duration(video_path)
    if duration:
        print(f"📹 Video duration: {duration:.1f} seconds")

    try:
        # Optimized ffmpeg command with progress output
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",  # No video
            "-acodec", "libmp3lame",  # Use LAME MP3 encoder
            "-ab", "192k",  # Good quality bitrate
            "-ar", "44100",  # Standard sample rate
            "-ac", "2",  # Stereo
            "-progress", "pipe:1",  # Output progress to stdout
            audio_output_path
        ]

        # Start ffmpeg process
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        # Initialize progress bar
        progress_bar = None
        if duration:
            progress_bar = tqdm(
                total=duration,
                desc="🎵 Converting",
                unit="sec",
                bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}] {rate_fmt}"
            )

        # Parse ffmpeg progress output
        current_time = 0
        for line in process.stdout:
            line = line.strip()
            if line.startswith('out_time_ms='):
                try:
                    # Extract time in microseconds and convert to seconds
                    time_ms = int(line.split('=')[1])
                    current_time = time_ms / 1000000.0  # Convert to seconds

                    if progress_bar and current_time <= duration:
                        progress_bar.n = current_time
                        progress_bar.refresh()
                except (ValueError, IndexError):
                    continue
            elif line.startswith('progress=end'):
                if progress_bar:
                    progress_bar.n = duration if duration else progress_bar.total
                    progress_bar.refresh()
                break

        # Wait for process to complete
        process.wait()

        if progress_bar:
            progress_bar.close()

        if process.returncode == 0:
            print(f"✅ Conversion successful: {os.path.basename(audio_output_path)}")
            return audio_output_path
        else:
            stderr_output = process.stderr.read() if process.stderr else "No error details"
            print(f"❌ FFmpeg conversion failed with return code {process.returncode}")
            print(f"   Error details: {stderr_output}")
            return None

    except FileNotFoundError:
        print("❌ Error: 'ffmpeg' not found. Install with: brew install ffmpeg")
        return None
    except Exception as e:
        print(f"❌ Unexpected error during conversion: {e}")
        return None

def clean_srt_file_english(source_file, target_dir, final_filename, hallucination_lists):
    print("-> Running English SRT cleaning process...")
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            subs = list(srt.parse(f.read()))
    except Exception as e:
        print(f"Error reading SRT file: {e}"); return None
    patterns = [r'\(.*?\)', r'\[.*?\]', r'★.*?★']
    subs = [sub for sub in subs if not any(re.search(pattern, sub.content) for pattern in patterns)]
    hallucination_sentences = [sentence for sublist in hallucination_lists for sentence in sublist]
    subs = [sub for sub in subs if re.sub(r'\W+', '', sub.content).strip() not in map(lambda s: re.sub(r'\W+', '', s).strip(), hallucination_sentences)]
    subs = [sub for sub in subs if sub.content.strip() != '']
    subs.sort(key=lambda sub: sub.start)
    for i, sub in enumerate(subs): sub.index = i + 1
    i = 0
    while i < len(subs):
        if i < len(subs) - 1 and subs[i].content == subs[i+1].content:
            subs[i].start = subs[i+1].start; del subs[i+1]; continue
        i += 1
    for sub in subs:
        words = sub.content.split()
        unique_words = [words[0]] if words else []
        for word in words[1:]:
            if word != unique_words[-1]: unique_words.append(word)
        sub.content = ' '.join(unique_words)
    subs = [s for s in subs if s.content.strip() != '']
    for i, sub in enumerate(subs): sub.index = i + 1
    target_file = os.path.join(target_dir, final_filename)
    with open(target_file, 'w', encoding='utf-8') as f: f.write(srt.compose(subs))
    return target_file

def clean_srt_file_japanese(source_file, target_dir, final_filename, hallucination_lists):
    print("-> Running Japanese SRT cleaning process...")
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            subs = list(srt.parse(f.read()))
    except Exception as e:
        print(f"Error reading SRT file: {e}"); return None
    patterns = [r'★.*?★', r'「.*?」', r'【.*?】', '^「', '^★']
    subs = [sub for sub in subs if not any(re.search(pattern, sub.content) for pattern in patterns)]
    hallucination_sentences = [sentence for sublist in hallucination_lists for sentence in sublist]
    subs = [sub for sub in subs if re.sub(r'\W+', '', sub.content).strip() not in map(lambda s: re.sub(r'\W+', '', s).strip(), hallucination_sentences)]
    subs_cleaned = [s for s in subs if s.content.strip() != '']
    for i, sub in enumerate(subs_cleaned): sub.index = i + 1
    target_file = os.path.join(target_dir, final_filename)
    with open(target_file, 'w', encoding='utf-8') as f: f.write(srt.compose(subs_cleaned))
    return target_file

def stamp_srt_file(file_path, marker_text):
    print(f"-> Stamping file with credit: '{marker_text}'")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            subs = list(srt.parse(file.read()))
    except Exception as e:
        print(f"Could not stamp file, error reading: {e}"); return
    if marker_text:
        first_sub = srt.Subtitle(index=1, start=timedelta(0), end=timedelta(milliseconds=1000), content=marker_text)
        subs.insert(0, first_sub)
        for i, sub in enumerate(subs, start=1): sub.index = i
    with open(file_path, 'w', encoding='utf-8') as file: file.write(srt.compose(subs))

# =================================================================
# --- Main Application Logic ---
# =================================================================

def run_transcription_and_cleaning(input_filename):
    """Main function to run the full transcription and cleaning pipeline with MPS optimization."""
    # 1. Setup Environment and Paths
    print("🚀 --- Initializing Apple Silicon Optimized Transcription --- 🚀")

    # Validate source language before proceeding
    validated_lang, is_valid = validate_and_convert_language(SOURCE_LANGUAGE)
    if not is_valid:
        print(f"❌ Invalid language code: '{SOURCE_LANGUAGE}'")
        print(f"🔍 Supported codes: {', '.join(SUPPORTED_LANGUAGES[:20])}...")
        print_language_help()
        return

    # Use validated language code
    source_lang = validated_lang
    print(f"🌐 Source language: {source_lang.upper()}")

    os.makedirs(SRT_OUTPUT_PATH, exist_ok=True)
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
    os.makedirs(TEMP_PATH, exist_ok=True)
    os.makedirs(AUDIO_OUTPUT_PATH, exist_ok=True)  # Create audio cache directory
    os.makedirs(AUDIO_SOURCE_PATH, exist_ok=True)  # Create audio source directory

    # Setup offline environment and model caching
    if not setup_offline_environment():
        return

    # Apply Apple Silicon optimizations
    optimize_for_apple_silicon()

    # 2. Find and identify input file
    input_path, file_type = find_input_file(input_filename)
    if not input_path:
        print(f"❌ Error: Input file '{input_filename}' not found in:")
        print(f"   📁 Video path: {VIDEO_SOURCE_PATH}")
        print(f"   🎵 Audio path: {AUDIO_SOURCE_PATH}")
        return

    file_basename = os.path.splitext(input_filename)[0]

    print(f"\n📁 Found file: {input_path}")
    print(f"📋 File type: {file_type.upper()}")

    try:
        # 3. Handle Audio Preparation Based on File Type
        audio_for_transcription = None

        if file_type == 'video':
            print(f"\n🎬 Processing video file...")

            # Check if converted audio already exists in cache
            cached_audio_path = os.path.join(AUDIO_OUTPUT_PATH, f"{file_basename}.mp3")

            if os.path.exists(cached_audio_path):
                print(f"🎵 Found cached audio file: {os.path.basename(cached_audio_path)}")
                user_choice = input("Use cached audio? (y/n, default=y): ").lower().strip()

                if user_choice in ['', 'y', 'yes']:
                    audio_for_transcription = cached_audio_path
                    print("✅ Using cached audio file")
                else:
                    print("🔄 Re-converting video to audio...")
                    audio_for_transcription = convert_video_to_audio_with_progress(input_path, cached_audio_path)
            else:
                print("🎵 Converting video to audio (will be cached for future use)...")
                audio_for_transcription = convert_video_to_audio_with_progress(input_path, cached_audio_path)

            if not audio_for_transcription:
                raise Exception("Video to audio conversion failed.")

        elif file_type == 'audio':
            print(f"\n🎵 Processing audio file directly...")
            audio_for_transcription = input_path
            print(f"✅ Using audio file: {os.path.basename(audio_for_transcription)}")

        else:
            supported_formats = VIDEO_EXTENSIONS + AUDIO_EXTENSIONS
            print(f"❌ Unsupported file format. Supported formats:")
            print(f"   📹 Video: {', '.join(VIDEO_EXTENSIONS)}")
            print(f"   🎵 Audio: {', '.join(AUDIO_EXTENSIONS)}")
            return

        # 4. Setup Model with Optimal Device Detection
        print("\n--- Model Setup ---")
        optimal_device, optimal_compute = detect_optimal_device_and_compute()
        model, actual_device, actual_compute = setup_model_with_fallback(
            MODEL_SIZE, optimal_device, optimal_compute
        )

        # 5. Run Transcription with Progress Monitoring
        print("\n--- Transcription ---")
        print(f"🎬 Processing: {input_filename}")
        print(f"🔧 Device: {actual_device} | Compute: {actual_compute}")

        start_time = time.time()

        # Enhanced transcription parameters for better quality and stability
        segments, info = model.transcribe(
            audio=audio_for_transcription,
            task=TASK,
            language=source_lang,  # Use validated language code
            beam_size=5,  # Good balance of quality vs speed
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500),  # Reduce false positive segments
            word_timestamps=False,  # Disable for better performance unless needed
            temperature=0.0,  # Deterministic output
            compression_ratio_threshold=2.4,  # Filter out repetitive segments
            log_prob_threshold=-1.0,  # Filter out low-confidence segments
            no_speech_threshold=0.6,  # Sensitivity for detecting speech
            condition_on_previous_text=True,  # Better context understanding
            prompt_reset_on_temperature=0.5,  # Reset context if needed
            initial_prompt=None,  # Could add context-specific prompts here
        )

        # Convert segments to list for processing (with progress)
        segment_list = list(segments)
        transcription_time = time.time() - start_time

        print(f"⏱️ Transcription completed in {transcription_time:.2f} seconds")
        print(f"📊 Detected language: {info.language} (confidence: {info.language_probability:.2f})")

        # 6. Save Raw SRT (in Temp Folder)
        raw_srt_path = os.path.join(TEMP_PATH, f"{file_basename}.raw.srt")
        subs = [
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=s.start),
                end=timedelta(seconds=s.end),
                content=s.text.strip()
            ) for i, s in enumerate(segment_list, 1)
        ]

        with open(raw_srt_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(subs))
        print(f"📝 Raw transcription saved ({len(subs)} segments)")

        # 7. Clean and Stamp Final SRT
        print("\n--- Post-Processing ---")
        lang_code = TARGET_LANGUAGE_CODE if TASK == "translate" else TO_LANGUAGE_CODE.get(source_lang, source_lang)
        final_srt_filename = f"{file_basename}.{lang_code}.srt"
        hallucination_lists = [SUPPRESS_HIGH, GARBAGE_LIST]

        cleaned_srt_path = None
        if TASK == "translate":
            cleaned_srt_path = clean_srt_file_english(raw_srt_path, SRT_OUTPUT_PATH, final_srt_filename, hallucination_lists)
        else: # transcribe
            if source_lang == "ja":  # Use validated language code
                cleaned_srt_path = clean_srt_file_japanese(raw_srt_path, SRT_OUTPUT_PATH, final_srt_filename, hallucination_lists)
            else:
                cleaned_srt_path = clean_srt_file_english(raw_srt_path, SRT_OUTPUT_PATH, final_srt_filename, hallucination_lists)

        if cleaned_srt_path and os.path.exists(cleaned_srt_path):
            stamp_srt_file(cleaned_srt_path, CREDIT)

            # Performance summary
            total_time = time.time() - start_time
            print(f"\n✨ --- Transcription Complete! --- ✨")
            print(f"✅ Final SRT: {cleaned_srt_path}")
            print(f"⏱️ Total processing time: {total_time:.2f} seconds")
            print(f"🚀 Device used: {actual_device} ({actual_compute})")

            if file_type == 'video':
                print(f"🎵 Audio cached: {os.path.join(AUDIO_OUTPUT_PATH, f'{file_basename}.mp3')}")

            if actual_device == "mps":
                print("🎉 Successfully utilized Apple Silicon MPS acceleration!")
        else:
            print("\n⚠️ Warning: SRT cleaning failed. No final file was created.")

    except Exception as e:
        print(f"\n💥 An error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 8. Cleanup with better error handling
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()  # Clear MPS memory
        except:
            pass  # Ignore cleanup errors

        try:
            if 'model' in locals():
                del model  # Explicitly delete model to free memory
        except:
            pass

        if os.path.exists(TEMP_PATH):
            try:
                print(f"\n🧹 Cleaning up temporary folder: {TEMP_PATH}")
                shutil.rmtree(TEMP_PATH)
            except:
                print("⚠️ Could not clean up temp folder (may be in use)")

if __name__ == "__main__":
    # Apply early fixes
    fix_openmp_conflicts()

    # Display system info
    print("🍎 Apple Silicon MPS-Optimized Whisper Transcription")
    print("=" * 60)

    if torch.backends.mps.is_available():
        print("✅ MPS Support: Available")
    else:
        print("⚠️ MPS Support: Not Available")

    print(f"🔧 PyTorch Version: {torch.__version__}")
    print(f"🧠 CPU Threads: {torch.get_num_threads()}")
    print(f"🔒 Offline Mode: {'Enabled' if FORCE_OFFLINE_MODE else 'Disabled'}")
    print(f"📁 Model Cache: {MODEL_STORAGE_PATH}")
    print("=" * 60)

    parser = argparse.ArgumentParser(
        description="Apple Silicon optimized transcription with automatic file type detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
File Type Support:
  📹 Video: .mp4, .mkv, .avi, .mov, .wmv, .flv, .webm, .m4v
  🎵 Audio: .mp3, .wav, .flac, .m4a, .aac, .ogg, .wma

Search Paths:
  📁 Video files: VIDEO_SOURCE_PATH
  🎵 Audio files: AUDIO_SOURCE_PATH
  
Audio Caching:
  - Converted audio files are stored in AUDIO_OUTPUT_PATH
  - Cached files are reused for faster processing on repeat runs
  
Examples:
  python transcribe.py video.mp4           # Convert video to audio, then transcribe
  python transcribe.py audio.mp3           # Directly transcribe audio file
  python transcribe.py "my file.mkv"       # Handle files with spaces
  
Performance Tips:
  - Audio files process faster (no conversion step)
  - Cached audio files from previous video conversions are reused
  - Close memory-intensive applications for best MPS performance
        """
    )

    parser.add_argument(
        "filename",
        type=str,
        help="Video or audio file to process (searches in VIDEO_SOURCE_PATH and AUDIO_SOURCE_PATH)"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    run_transcription_and_cleaning(args.filename)
