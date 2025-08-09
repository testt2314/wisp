#!/usr/bin/env python3
"""
Whisper Transcription Tool for Mac mini
Converts video/audio files to English subtitles using Whisper or faster-whisper
Usage: python transcribe.py [file]

Requirements:
- For regular Whisper: pip install transformers torch librosa
- For faster-whisper: pip install faster-whisper librosa
"""

import os
import sys
import json
import datetime
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import re
import srt
from datetime import timedelta
import torch
import librosa
import numpy as np
import time
import threading

# Try to import psutil for CPU detection
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Note: Install psutil for automatic CPU thread detection: pip install psutil")

# Try to import both whisper implementations
try:
    from transformers import pipeline

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not available. Install with: pip install faster-whisper")

# Master Configuration - Embedded in script
CONFIG = {
    "srt_location": "/Volumes/Macintosh HD/Downloads/srt",
    "temp_location": "/Volumes/Macintosh HD/Downloads/srt/temp",
    "audio_source": "/Volumes/Macintosh HD/Downloads",  # Source audio files location
    "video_source": "/Volumes/Macintosh HD/Downloads/video",  # Source video files location
    "audio_export": "/Volumes/Macintosh HD/Downloads/audio/exported",  # Exported audio from video conversion
    "whisper_models_location": "/Volumes/Macintosh HD/Downloads/srt/whisper_models",
    "ffmpeg_path": "/Volumes/Macintosh HD/Downloads/srt/whisper_models/ffmpeg",
    "ffprobe_path": "/Volumes/Macintosh HD/Downloads/srt/whisper_models/ffprobe",
    "model_size": "openai/whisper-large-v3",
    "chunk_length_s": 30,
    "vad_threshold": 0.05,  # Much lower for soft voices (was 0.15)
    "chunk_duration": 15.0,
    "credit": "Created using Whisper Transcription Tool",
    "use_mps": True,  # Enable MPS acceleration on Apple Silicon
    "save_audio_to_export_location": True,  # Save converted audio to audio_export instead of temp
    "use_faster_whisper": True,  # Set to True to use faster-whisper, False for regular whisper
    "faster_whisper_model_size": "large-v3",  # Model size for faster-whisper (different naming)
    "faster_whisper_local_model_path": "/Volumes/Macintosh HD/Downloads/srt/whisper_models/faster-whisper-large-v3",
    # Local model folder name
    "faster_whisper_compute_type": "int8",  # Changed from float16 for M4 CPU compatibility
    "faster_whisper_device": "auto",  # Device: auto, cpu, cuda, or specific device
    "faster_whisper_cpu_threads": "auto",  # Number of CPU threads: auto, or specific number (e.g., 4, 8)
    "faster_whisper_num_workers": 1,  # Number of parallel workers for faster-whisper
    "faster_whisper_beam_size": 5,  # Reduced from 10 to prevent repetition loops
    "faster_whisper_best_of": 5,  # Reduced from 10 to prevent repetition loops
    # New settings for better soft voice handling and accuracy
    "faster_whisper_temperature": 0.2,  # Slightly higher to add variation (was 0.0)
    "faster_whisper_patience": 1.0,  # Patience for beam search
    "faster_whisper_length_penalty": 1.2,  # Encourage longer segments (was 1.0)
    "faster_whisper_repetition_penalty": 1.1,  # Stronger penalty for repetitions (was 1.01)
    "faster_whisper_no_repeat_ngram_size": 3,  # Prevent 3-gram repetitions (was 0)
    "faster_whisper_suppress_blank": True,  # Suppress blank outputs
    "faster_whisper_suppress_tokens": [-1],  # Suppress specific tokens
    "faster_whisper_without_timestamps": False,  # Keep timestamps
    "faster_whisper_max_initial_timestamp": 1.0,  # Allow some initial timestamp flexibility
    "faster_whisper_word_timestamps": False,  # Disable word timestamps to reduce segmentation issues
    "faster_whisper_prepend_punctuations": "\"'([{-",  # Punctuation handling
    "faster_whisper_append_punctuations": "\"'.,:!?)]}",  # Punctuation handling
    # Enhanced VAD settings for soft voices but less aggressive
    "faster_whisper_vad_filter": True,  # Enable VAD
    "faster_whisper_vad_threshold": 0.15,  # Less aggressive than 0.05 to prevent over-segmentation
    "faster_whisper_min_silence_duration_ms": 500,  # Longer silence detection to group words better
    "faster_whisper_max_speech_duration_s": 60,  # Allow longer segments (was 30)
    "faster_whisper_min_speech_duration_ms": 300,  # Longer minimum speech to avoid tiny segments
    # Audio preprocessing for better soft voice detection
    "audio_normalize": False,  # Disable normalization to prevent math errors
    "audio_gain_db": 0.0,  # Disable gain boost to prevent overflow
    "audio_noise_reduction": False  # Disable noise reduction as it may cause artifacts
}

# Global constants
TQDM_FORMAT = "{desc}: {percentage:3.1f}% |{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate:.2f} audio s / real time s]"

# Garbage patterns to remove from transcriptions
GARBAGE_PATTERNS = [
    "Thank you.",
    "Thanks for watching.",
    "Please subscribe.",
    "Don't forget to like and subscribe.",
    "See you next time.",
    "Bye bye.",
]

REMOVE_QUOTES = dict.fromkeys(map(ord, '"„"‟"＂「」'), None)


class WhisperTranscriber:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipe = None  # For HF Transformers
        self.model = None  # For faster-whisper
        self.device = None
        self.use_faster_whisper = config.get("use_faster_whisper", False)
        self.transcription_complete = False
        self.current_progress = 0.0

        # Validate that the required library is available
        if self.use_faster_whisper and not FASTER_WHISPER_AVAILABLE:
            print("Error: faster-whisper requested but not installed.")
            print("Install with: pip install faster-whisper")
            sys.exit(1)
        elif not self.use_faster_whisper and not HF_AVAILABLE:
            print("Error: transformers requested but not installed.")
            print("Install with: pip install transformers")
            sys.exit(1)

        self._ensure_directories()
        self._setup_device()

    def _update_progress(self, progress: float):
        """Update transcription progress."""
        self.current_progress = progress
        # Print progress update
        print(f"\r⏳ Transcription Progress: {progress:.1f}%", end="", flush=True)

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.config["srt_location"],
            self.config["temp_location"],
            self.config["audio_source"],
            self.config["video_source"],
            self.config["audio_export"],
            self.config["whisper_models_location"]
        ]
        for path in directories:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _setup_device(self):
        """Setup the best available device (MPS, CUDA, or CPU)."""
        if self.use_faster_whisper:
            # faster-whisper device setup
            device_config = self.config.get("faster_whisper_device", "auto")
            if device_config == "auto":
                if torch.backends.mps.is_available() and self.config.get("use_mps", True):
                    self.device = "cpu"  # faster-whisper doesn't support MPS directly, use CPU
                    print("Using CPU for faster-whisper (MPS not supported by faster-whisper)")
                elif torch.cuda.is_available():
                    self.device = "cuda"
                    print("Using CUDA for faster-whisper")
                else:
                    self.device = "cpu"
                    print("Using CPU for faster-whisper")
            else:
                self.device = device_config
                print(f"Using configured device for faster-whisper: {self.device}")
        else:
            # Regular whisper device setup
            if self.config.get("use_mps", True) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple Silicon MPS acceleration for Transformers Whisper")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA acceleration for Transformers Whisper")
            else:
                self.device = torch.device("cpu")
                print("Using CPU for Transformers Whisper")

    def _check_model_exists(self) -> bool:
        """Check if the Whisper model is already downloaded."""
        cache_dir = self.config["whisper_models_location"]
        model_name = self.config["model_size"].replace("/", "--")

        # Check for common cached model directory patterns
        potential_paths = [
            os.path.join(cache_dir, "models--" + model_name),
            os.path.join(cache_dir, model_name),
            os.path.join(cache_dir, self.config["model_size"]),
        ]

        for path in potential_paths:
            if os.path.exists(path) and os.listdir(path):
                print(f"Found cached model at: {path}")
                return True

        return False

    def _load_model(self):
        """Load the appropriate Whisper model based on configuration."""
        if self.use_faster_whisper:
            self._load_faster_whisper_model()
        else:
            self._load_hf_model()

    def _get_optimal_cpu_threads(self) -> int:
        """Determine optimal number of CPU threads for faster-whisper on M4."""
        cpu_threads_config = self.config.get("faster_whisper_cpu_threads", "auto")

        if cpu_threads_config == "auto":
            if PSUTIL_AVAILABLE:
                # Get system info using psutil
                total_cores = psutil.cpu_count(logical=False)  # Physical cores
                total_threads = psutil.cpu_count(logical=True)  # Logical cores (with hyperthreading)

                # M4 specific optimization
                # M4 has 4 performance cores + 6 efficiency cores = 10 total cores
                if total_cores >= 10:  # Likely M4
                    # Use performance cores primarily (4) + some efficiency cores
                    optimal_threads = 6  # Conservative for stability
                elif total_cores >= 8:  # Likely M3 or similar
                    optimal_threads = min(6, total_cores - 1)
                elif total_cores >= 4:
                    optimal_threads = total_cores - 1
                else:
                    optimal_threads = total_cores

                print(f"Auto-detected CPU: {total_cores} cores, {total_threads} threads")
                print(f"Using {optimal_threads} threads for faster-whisper (M4 optimized)")
                return optimal_threads
            else:
                # Fallback without psutil - conservative for M4
                import os
                total_threads = os.cpu_count() or 4
                optimal_threads = min(6, max(1, total_threads - 2))  # Conservative approach
                print(f"Detected {total_threads} CPU threads (install psutil for better detection)")
                print(f"Using {optimal_threads} threads for faster-whisper")
                return optimal_threads
        else:
            # Use configured value
            threads = int(cpu_threads_config)
            print(f"Using configured {threads} threads for faster-whisper")
            return threads

    def _download_faster_whisper_model(self, model_size: str, local_path: str) -> str:
        """Download faster-whisper model and return the path."""
        print(f"🔄 Downloading faster-whisper model: {model_size}")
        print(f"   This may take several minutes...")

        try:
            # Create a temporary WhisperModel to trigger download
            temp_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                download_root=self.config["whisper_models_location"]
            )

            # The model should now be downloaded to the cache
            # Find where it was actually downloaded
            cache_dir = self.config["whisper_models_location"]

            # Check multiple possible cache locations
            possible_locations = [
                os.path.join(cache_dir, f"models--Systran--faster-whisper-{model_size}"),
                os.path.join(cache_dir, f"faster-whisper-{model_size}"),
                os.path.join(cache_dir, model_size),
                # Also check for the exact downloaded path format
                os.path.join(cache_dir, f"models--Systran--faster-whisper-{model_size}", "snapshots")
            ]

            downloaded_path = None
            for location in possible_locations:
                if os.path.exists(location):
                    # For models with snapshots, find the actual model files
                    if location.endswith("snapshots"):
                        snapshots_dir = location
                        if os.path.exists(snapshots_dir):
                            # Find the latest snapshot
                            snapshots = [d for d in os.listdir(snapshots_dir) if
                                         os.path.isdir(os.path.join(snapshots_dir, d))]
                            if snapshots:
                                # Use the first (usually only) snapshot
                                snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                                if any(f.endswith('.bin') for f in os.listdir(snapshot_path)):
                                    downloaded_path = snapshot_path
                                    break
                    else:
                        # Check if this location has model files
                        if self._check_model_files_exist(location):
                            downloaded_path = location
                            break
                        # Also check snapshots subdirectory
                        snapshots_path = os.path.join(location, "snapshots")
                        if os.path.exists(snapshots_path):
                            snapshots = [d for d in os.listdir(snapshots_path) if
                                         os.path.isdir(os.path.join(snapshots_path, d))]
                            if snapshots:
                                snapshot_path = os.path.join(snapshots_path, snapshots[0])
                                if self._check_model_files_exist(snapshot_path):
                                    downloaded_path = snapshot_path
                                    break

            if downloaded_path:
                print(f"✅ Model downloaded to: {downloaded_path}")
                return downloaded_path
            else:
                print("⚠️  Model downloaded but location not found, using model name directly")
                return model_size

        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("🔄 Will attempt to use model name directly...")
            return model_size

    def _check_model_files_exist(self, path: str) -> bool:
        """Check if model files exist in the given path."""
        if not os.path.exists(path):
            return False

        files = os.listdir(path)
        # Check for typical faster-whisper model files
        required_files = ['config.json']
        model_files = [f for f in files if f.endswith('.bin')]

        has_config = any(f in files for f in required_files)
        has_model = len(model_files) > 0

        return has_config and has_model

    def _load_faster_whisper_model(self):
        """Load the faster-whisper model with CPU thread optimization."""
        if self.model is None:
            model_size = self.config.get("faster_whisper_model_size", "large-v3")
            compute_type = self.config.get("faster_whisper_compute_type", "float16")
            num_workers = self.config.get("faster_whisper_num_workers", 1)
            local_model_path = self.config.get("faster_whisper_local_model_path",
                                               "/Volumes/Macintosh HD/Downloads/srt/whisper_models/faster-whisper-large-v3")

            # Get optimal CPU threads
            cpu_threads = self._get_optimal_cpu_threads()

            print(f"Compute type: {compute_type}")
            print(f"Device: {self.device}")
            print(f"CPU threads: {cpu_threads}")
            print(f"Workers: {num_workers}")

            # Check for the actual downloaded model first
            cache_dir = self.config["whisper_models_location"]
            actual_model_path = os.path.join(cache_dir, f"models--Systran--faster-whisper-{model_size}")

            model_path_to_use = model_size  # Default to downloading

            # Check if the actual downloaded model exists
            if os.path.exists(actual_model_path):
                # Check snapshots subdirectory
                snapshots_path = os.path.join(actual_model_path, "snapshots")
                if os.path.exists(snapshots_path):
                    snapshots = [d for d in os.listdir(snapshots_path) if
                                 os.path.isdir(os.path.join(snapshots_path, d))]
                    if snapshots:
                        snapshot_path = os.path.join(snapshots_path, snapshots[0])
                        if self._check_model_files_exist(snapshot_path):
                            print(f"✅ Found actual downloaded model at: {snapshot_path}")
                            model_path_to_use = snapshot_path
                        else:
                            print(f"⚠️  Snapshot exists but incomplete: {snapshot_path}")
                elif self._check_model_files_exist(actual_model_path):
                    print(f"✅ Found actual downloaded model at: {actual_model_path}")
                    model_path_to_use = actual_model_path
                else:
                    print(f"⚠️  Model directory exists but incomplete: {actual_model_path}")

            # If actual model not found, check user-specified local path
            if model_path_to_use == model_size and os.path.exists(local_model_path):
                if self._check_model_files_exist(local_model_path):
                    print(f"✅ Found local faster-whisper model at: {local_model_path}")
                    model_path_to_use = local_model_path
                else:
                    print(f"⚠️  Local model path exists but appears incomplete: {local_model_path}")
                    print(f"🔄 Will download model instead")

            # If we still need to download, do it
            if model_path_to_use == model_size:
                print(f"🔍 No valid local model found, downloading: {model_size}")
                model_path_to_use = self._download_faster_whisper_model(model_size, local_model_path)

            try:
                print("🔄 Loading faster-whisper model...")
                self.model = WhisperModel(
                    model_path_to_use,
                    device=self.device,
                    compute_type=compute_type,
                    download_root=self.config["whisper_models_location"],
                    cpu_threads=cpu_threads,
                    num_workers=num_workers
                )
                print("✅ faster-whisper model loaded successfully!")

                # Verify model is actually loaded
                if self.model is None:
                    raise RuntimeError("Model loaded but is still None")

            except Exception as e:
                print(f"❌ Error loading faster-whisper model with optimizations: {e}")
                print("🔄 Trying with basic settings...")

                try:
                    # Try with minimal settings using the same path
                    self.model = WhisperModel(
                        model_path_to_use,
                        device="cpu",
                        compute_type="int8"
                    )
                    print("✅ faster-whisper model loaded with basic settings!")

                except Exception as basic_e:
                    print(f"❌ Basic model loading also failed: {basic_e}")
                    print("🔄 Trying fallback model...")

                    try:
                        # Last resort: try downloading base model
                        self.model = WhisperModel(
                            "base",
                            device="cpu",
                            compute_type="int8",
                            download_root=self.config["whisper_models_location"]
                        )
                        print("✅ Fallback base model loaded!")
                    except Exception as fallback_e:
                        print(f"❌ All model loading attempts failed: {fallback_e}")
                        raise RuntimeError(f"Could not load any faster-whisper model. Last error: {fallback_e}")

        # Final verification
        if self.model is None:
            raise RuntimeError("Model is None after loading attempts")

    def _load_hf_model(self):
        """Load the Hugging Face Transformers Whisper model."""
        if self.pipe is None:
            # Check if model is already downloaded
            model_exists = self._check_model_exists()
            if model_exists:
                print(f"Using cached HF Whisper model: {self.config['model_size']}")
            else:
                print(f"Downloading HF Whisper model: {self.config['model_size']}")

            # Set cache directory to our models location
            cache_dir = self.config["whisper_models_location"]
            os.makedirs(cache_dir, exist_ok=True)

            try:
                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model=self.config["model_size"],
                    chunk_length_s=self.config["chunk_length_s"],
                    device=self.device,
                    model_kwargs={"cache_dir": cache_dir},
                    return_timestamps=True
                )
                print("HF Transformers Whisper model loaded successfully!")
            except Exception as e:
                print(f"Error loading HF model: {e}")
                print("Falling back to smaller model...")
                # Fallback to a smaller model if the large one fails
                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-base",
                    chunk_length_s=self.config["chunk_length_s"],
                    device=self.device,
                    model_kwargs={"cache_dir": cache_dir},
                    return_timestamps=True
                )
                print("Fallback HF Transformers model loaded successfully!")

    def _is_video_file_by_extension(self, extension: str) -> bool:
        """Check if the file extension indicates a video file."""
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
        return extension.lower() in video_extensions

    def _is_audio_file_by_extension(self, extension: str) -> bool:
        """Check if the file extension indicates an audio file."""
        audio_extensions = {'.mp3', '.wav', '.aac', '.m4a', '.ogg', '.opus', '.flac'}
        return extension.lower() in audio_extensions

    def _is_video_file(self, file_path: str) -> bool:
        """Check if the file is a video file."""
        return self._is_video_file_by_extension(Path(file_path).suffix)

    def _is_audio_file(self, file_path: str) -> bool:
        """Check if the file is an audio file."""
        return self._is_audio_file_by_extension(Path(file_path).suffix)

    def _find_input_file(self, filename: str) -> str:
        """Find the input file in the configured source directories.

        Args:
            filename: The filename to search for (can be with or without full path)

        Returns:
            str: Full path to the found file

        Raises:
            FileNotFoundError: If file is not found in any source location
        """
        print(f"Searching for file: {filename}")

        # If it's already a full path that exists, use it
        if os.path.exists(filename):
            print(f"Found file at provided path: {filename}")
            return filename

        # Extract just the filename if a path was provided
        base_filename = os.path.basename(filename)
        print(f"Base filename: {base_filename}")

        # Determine file type and search in appropriate directories
        file_extension = Path(base_filename).suffix.lower()

        # Define search locations based on file type
        if self._is_audio_file_by_extension(file_extension):
            primary_locations = [self.config["audio_source"], self.config["audio_export"]]
            print(f"Audio file detected, searching in audio directories first")
        elif self._is_video_file_by_extension(file_extension):
            primary_locations = [self.config["video_source"]]
            print(f"Video file detected, searching in video directories first")
        else:
            primary_locations = []
            print(f"Unknown file type, searching in all directories")

        # Build complete search list (primary locations first, then fallbacks)
        search_locations = primary_locations + [
            self.config["audio_source"],
            self.config["video_source"],
            self.config["audio_export"],
            os.getcwd(),  # Current working directory
        ]

        # Remove duplicates while preserving order
        seen = set()
        search_locations = [x for x in search_locations if not (x in seen or seen.add(x))]

        print(f"Searching in these locations:")
        for i, location in enumerate(search_locations, 1):
            potential_path = os.path.join(location, base_filename)
            print(f"  {i}. {potential_path}")

            # Check if directory exists first
            if os.path.exists(location):
                print(f"     Directory exists: ✓")
                if os.path.exists(potential_path):
                    print(f"     File found: ✓")
                    return potential_path
                else:
                    print(f"     File not found: ✗")
            else:
                print(f"     Directory does not exist: ✗")

        # If still not found, provide helpful error message
        error_msg = f"File '{base_filename}' not found in any location.\n"
        error_msg += f"Searched in {len(search_locations)} directories:\n"
        for i, location in enumerate(search_locations, 1):
            potential_path = os.path.join(location, base_filename)
            exists = "✓" if os.path.exists(location) else "✗"
            error_msg += f"  {i}. {potential_path} (dir exists: {exists})\n"
        error_msg += f"\nPlease check:\n"
        error_msg += f"1. File '{base_filename}' exists\n"
        error_msg += f"2. For audio files: place in {self.config['audio_source']}\n"
        error_msg += f"3. For video files: place in {self.config['video_source']}\n"
        error_msg += f"4. File permissions allow reading\n"

        raise FileNotFoundError(error_msg)

    def _load_audio(self, audio_path: str) -> Dict[str, Any]:
        """Load audio file using librosa with safe preprocessing."""
        print(f"Loading audio file: {audio_path}")

        try:
            # Load audio with librosa (automatically handles various formats)
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz

            # Check for audio issues that cause mathematical errors
            if len(audio_array) == 0:
                raise ValueError("Audio file is empty or corrupted")

            # Check for NaN or infinite values
            if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
                print("⚠️  Audio contains NaN or infinite values, cleaning...")
                audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Check audio dynamic range
            audio_max = np.max(np.abs(audio_array))
            audio_rms = np.sqrt(np.mean(audio_array ** 2))

            print(f"📊 Audio stats - Max: {audio_max:.4f}, RMS: {audio_rms:.4f}")

            # Only apply minimal processing if audio is extremely quiet
            if audio_rms < 0.001:  # Very quiet audio
                print("🔧 Very quiet audio detected, applying minimal boost...")
                # Safe, minimal gain boost
                audio_array = audio_array * 2.0
                # Ensure no clipping
                audio_array = np.clip(audio_array, -0.95, 0.95)
                print(f"📊 After boost - RMS: {np.sqrt(np.mean(audio_array ** 2)):.4f}")

            # Final safety check
            if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
                print("❌ Audio still contains invalid values after processing")
                raise ValueError("Audio processing failed - contains NaN or infinite values")

        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            raise

        return {
            "array": audio_array,
            "sampling_rate": sample_rate,
            "path": audio_path
        }

    def _convert_timestamps_to_srt(self, chunks: List[Dict], audio_duration: float) -> List[srt.Subtitle]:
        """Convert Hugging Face pipeline timestamps to SRT format."""
        subs = []

        print(f"Processing {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks, start=1):
            # Extract timestamp and text from chunk
            timestamp = chunk.get("timestamp", [0.0, audio_duration])
            text = chunk.get("text", "").strip()

            if not text:
                continue

            # Handle timestamp format
            if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                start_time = float(timestamp[0]) if timestamp[0] is not None else 0.0
                end_time = float(timestamp[1]) if timestamp[1] is not None else start_time + 1.0
            else:
                # If timestamp format is unexpected, create reasonable defaults
                start_time = i * 2.0  # 2 seconds per segment as fallback
                end_time = start_time + 2.0

            # Ensure end_time is after start_time
            if end_time <= start_time:
                end_time = start_time + 1.0

            sub = srt.Subtitle(
                index=i,
                start=timedelta(seconds=start_time),
                end=timedelta(seconds=end_time),
                content=text
            )
            subs.append(sub)

        return subs

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg and ffprobe are available."""
        ffmpeg_path = self.config["ffmpeg_path"]
        ffprobe_path = self.config["ffprobe_path"]

        if not (os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path)):
            print(f"Warning: ffmpeg/ffprobe not found at specified location.")
            print(f"Checking system PATH...")

            # Check if they're in system PATH
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
                subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
                self.config["ffmpeg_path"] = "ffmpeg"
                self.config["ffprobe_path"] = "ffprobe"
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Error: ffmpeg/ffprobe not found. Please install ffmpeg.")
                return False
        return True

    def _convert_to_audio(self, video_path: str) -> tuple[str, bool]:
        """Convert video file to audio format.

        Returns:
            tuple: (audio_path, is_temporary) - path to audio file and whether it should be cleaned up
        """
        if not self._check_ffmpeg():
            raise RuntimeError("ffmpeg is required for video conversion")

        video_name = Path(video_path).stem

        # Choose output location based on config
        if self.config.get("save_audio_to_export_location", True):
            audio_path = os.path.join(self.config["audio_export"], f"{video_name}.mp3")
            is_temporary = False  # Don't clean up if saving to audio_export
            print(f"Converting video to audio (permanent): {video_path}")
        else:
            audio_path = os.path.join(self.config["temp_location"], f"{video_name}.mp3")
            is_temporary = True  # Clean up temp files
            print(f"Converting video to audio (temporary): {video_path}")

        print(f"Output: {audio_path}")

        cmd = [
            self.config["ffmpeg_path"],
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-ab", "192k",  # Audio bitrate
            "-ar", "22050",  # Audio sample rate
            "-y",  # Overwrite output file
            audio_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Video conversion completed!")
            return audio_path, is_temporary
        except subprocess.CalledProcessError as e:
            print(f"Error converting video: {e}")
            print(f"FFmpeg output: {e.stderr}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean up transcribed text."""
        # Remove common garbage phrases
        for garbage in GARBAGE_PATTERNS:
            text = text.replace(garbage, "")

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove quotes
        text = text.translate(REMOVE_QUOTES)

        return text

    def _clean_srt_segments(self, segments: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Clean and filter SRT segments."""
        cleaned_segments = []

        for segment in segments:
            # Clean the text
            cleaned_text = self._clean_text(segment.content)

            # Skip if text is too short or empty
            if len(cleaned_text.strip()) < 3:
                continue

            # Skip if it's likely garbage
            if any(garbage.lower() in cleaned_text.lower() for garbage in GARBAGE_PATTERNS):
                continue

            # Update the segment with cleaned text
            segment.content = cleaned_text
            cleaned_segments.append(segment)

        # Renumber segments
        for i, segment in enumerate(cleaned_segments, 1):
            segment.index = i

        return cleaned_segments

    def _add_credit_to_srt(self, srt_path: str, credit: str):
        """Add credit line to the end of SRT file."""
        if not credit:
            return

        with open(srt_path, 'r', encoding='utf-8') as f:
            subs = list(srt.parse(f.read()))

        if subs:
            # Add credit at the end
            last_sub_end_time = subs[-1].end
            credit_sub = srt.Subtitle(
                index=len(subs) + 1,
                start=last_sub_end_time,
                end=last_sub_end_time + timedelta(seconds=2),
                content=credit
            )
            subs.append(credit_sub)

        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subs))

    def _transcribe_with_faster_whisper(self, audio_data: Dict[str, Any], audio_duration: float) -> Dict[str, Any]:
        """Transcribe using faster-whisper with enhanced settings for soft voices and accuracy."""
        # Verify model is loaded
        if self.model is None:
            raise RuntimeError("faster-whisper model is not loaded")

        try:
            # Get enhanced faster-whisper parameters for accuracy
            beam_size = self.config.get("faster_whisper_beam_size", 10)
            best_of = self.config.get("faster_whisper_best_of", 10)
            temperature = self.config.get("faster_whisper_temperature", 0.0)
            patience = self.config.get("faster_whisper_patience", 1.0)
            length_penalty = self.config.get("faster_whisper_length_penalty", 1.0)
            repetition_penalty = self.config.get("faster_whisper_repetition_penalty", 1.01)
            word_timestamps = self.config.get("faster_whisper_word_timestamps", True)

            # Enhanced VAD parameters for balanced performance
            vad_threshold = self.config.get("faster_whisper_vad_threshold", 0.15)
            min_silence_duration_ms = self.config.get("faster_whisper_min_silence_duration_ms", 500)
            max_speech_duration_s = self.config.get("faster_whisper_max_speech_duration_s", 60)
            min_speech_duration_ms = self.config.get("faster_whisper_min_speech_duration_ms", 300)

            print(f"🎛️  Balanced transcription settings (anti-repetition):")
            print(f"   Beam size: {beam_size}, Best of: {best_of}")
            print(f"   Temperature: {temperature} (adds variation)")
            print(f"   Repetition penalty: {repetition_penalty}")
            print(f"   No-repeat n-gram size: {self.config.get('faster_whisper_no_repeat_ngram_size', 3)}")
            print(f"   VAD threshold: {vad_threshold} (balanced sensitivity)")
            print(f"   Min silence: {min_silence_duration_ms}ms (groups words better)")
            print(f"   Word timestamps: {word_timestamps}")
            print(f"   Model verification: {type(self.model)}")

            print("🔄 Starting enhanced faster-whisper transcription...")

            # Enhanced VAD parameters
            vad_parameters = {
                "threshold": vad_threshold,
                "min_silence_duration_ms": min_silence_duration_ms,
                "max_speech_duration_s": max_speech_duration_s,
                "min_speech_duration_ms": min_speech_duration_ms
            }

            segments_generator, info = self.model.transcribe(
                audio_data["array"],
                task="translate",  # Always translate to English
                language=None,  # Auto-detect
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                patience=patience,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=self.config.get("faster_whisper_no_repeat_ngram_size", 0),
                suppress_blank=self.config.get("faster_whisper_suppress_blank", True),
                suppress_tokens=self.config.get("faster_whisper_suppress_tokens", [-1]),
                without_timestamps=self.config.get("faster_whisper_without_timestamps", False),
                max_initial_timestamp=self.config.get("faster_whisper_max_initial_timestamp", 0.0),
                word_timestamps=word_timestamps,
                prepend_punctuations=self.config.get("faster_whisper_prepend_punctuations", "\"'([{-"),
                append_punctuations=self.config.get("faster_whisper_append_punctuations", "\"'.,:!?)]}"),
                vad_filter=self.config.get("faster_whisper_vad_filter", True),
                vad_parameters=vad_parameters
            )

            print(f"📊 Detected language: {info.language} (probability: {info.language_probability:.2f})")
            if info.language_probability < 0.8:
                print(f"⚠️  Language detection confidence is low ({info.language_probability:.2f})")
                print("   Consider manually specifying the language for better accuracy")

            print(f"🔄 Processing segments with enhanced accuracy...")

            # Convert generator to list while tracking progress
            chunks = []
            last_end_time = 0.0
            segment_count = 0
            soft_voice_segments = 0

            for segment in segments_generator:
                segment_count += 1

                # Track soft voice detection (segments with lower confidence or volume)
                if hasattr(segment, 'avg_logprob') and segment.avg_logprob < -0.8:
                    soft_voice_segments += 1

                chunk_data = {
                    "text": segment.text,
                    "timestamp": [segment.start, segment.end]
                }

                # Add word-level timestamps if available
                if word_timestamps and hasattr(segment, 'words') and segment.words:
                    chunk_data["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": getattr(word, 'probability', 1.0)
                        }
                        for word in segment.words
                    ]

                chunks.append(chunk_data)
                last_end_time = segment.end

                # Update progress based on audio processed
                progress = min(100.0, (last_end_time / audio_duration) * 100) if audio_duration > 0 else 0
                self._update_progress(progress)

                # Print progress every 10 segments
                if segment_count % 10 == 0:
                    print(f"\n   📝 Processed {segment_count} segments, {progress:.1f}% complete")
                    if soft_voice_segments > 0:
                        print(f"   🔍 Detected {soft_voice_segments} potential soft voice segments")

            # Ensure we reach 100%
            self._update_progress(100.0)

            print(f"\n✅ Processed {segment_count} segments total")
            if soft_voice_segments > 0:
                print(f"🎙️  Successfully captured {soft_voice_segments} soft voice segments")

            # Analyze transcription quality
            avg_segment_length = np.mean([len(chunk["text"]) for chunk in chunks]) if chunks else 0
            print(f"📊 Average segment length: {avg_segment_length:.1f} characters")

            if avg_segment_length < 20:
                print("⚠️  Short average segments detected - audio might be very quiet or fragmented")

            # Create result in HF pipeline format
            full_text = " ".join([chunk["text"] for chunk in chunks])
            return {
                "text": full_text,
                "chunks": chunks
            }

        except Exception as e:
            print(f"\n❌ Enhanced faster-whisper transcription error: {e}")
            print(f"   Model type: {type(self.model)}")
            print(f"   Audio shape: {audio_data['array'].shape}")
            print(f"   Audio duration: {audio_duration}")
            print(f"   Audio RMS level: {np.sqrt(np.mean(audio_data['array'] ** 2)):.4f}")
            raise

    def _transcribe_with_hf(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe using Hugging Face Transformers."""

        # HF Transformers doesn't provide progress callbacks, so we simulate progress
        def simulate_hf_progress():
            # Simulate progress over expected duration (rough estimate)
            estimated_duration = len(audio_data["array"]) / audio_data["sampling_rate"] * 0.3  # ~30% of audio duration
            steps = 100
            for i in range(steps + 1):
                if hasattr(self, '_transcription_complete') and self._transcription_complete:
                    break
                progress = (i / steps) * 100
                self._update_progress(progress)
                time.sleep(estimated_duration / steps)

        # Start progress simulation
        progress_thread = threading.Thread(target=simulate_hf_progress)
        progress_thread.daemon = True
        progress_thread.start()

        try:
            # Force translation to English regardless of source language
            result = self.pipe(
                audio_data["array"].copy(),
                return_timestamps=True,
                generate_kwargs={
                    "task": "translate",  # Always translate to English
                    "language": None  # Auto-detect source language
                }
            )
            self._transcription_complete = True
            self._update_progress(100.0)
            return result
        except Exception as e:
            print(f"\nHF Whisper transcription error: {e}")
            print("Trying fallback without explicit translation task...")
            # Fallback without explicit translation task
            result = self.pipe(
                audio_data["array"].copy(),
                return_timestamps=True
            )
            self._transcription_complete = True
            self._update_progress(100.0)
            return result

    def transcribe_file(self, file_path: str) -> str:
        """Main transcription function supporting both faster-whisper and HF Transformers."""
        # Find the actual file path using our search logic
        actual_file_path = self._find_input_file(file_path)

        if not os.path.exists(actual_file_path):
            raise FileNotFoundError(f"File not found: {actual_file_path}")

        self._load_model()

        # Determine file type and prepare audio file
        audio_path = actual_file_path
        temp_audio = False

        if self._is_video_file(actual_file_path):
            print("Video file detected, converting to audio...")
            audio_path, temp_audio = self._convert_to_audio(actual_file_path)
        elif not self._is_audio_file(actual_file_path):
            raise ValueError(f"Unsupported file type: {Path(actual_file_path).suffix}")

        try:
            # Generate output filename based on original file
            base_name = Path(actual_file_path).stem
            srt_path = os.path.join(self.config["srt_location"], f"{base_name}.srt")

            print(f"Transcribing: {audio_path}")
            print(f"Output SRT: {srt_path}")
            print(f"Using device: {self.device}")
            print(f"Using engine: {'faster-whisper' if self.use_faster_whisper else 'HF Transformers'}")

            # Load audio data using the _load_audio method
            audio_data = self._load_audio(audio_path)
            audio_duration = len(audio_data["array"]) / audio_data["sampling_rate"]

            print(f"Audio duration: {audio_duration:.2f} seconds")

            # Run transcription with progress tracking
            print("Starting transcription...")
            print("Note: This may take several minutes depending on audio length...")

            start_time = time.time()
            self.transcription_complete = False

            try:
                if self.use_faster_whisper:
                    result = self._transcribe_with_faster_whisper(audio_data, audio_duration)
                else:
                    result = self._transcribe_with_hf(audio_data)
            finally:
                # Mark transcription as complete
                self.transcription_complete = True
                elapsed = time.time() - start_time
                mins, secs = divmod(elapsed, 60)
                print(f"\n✅ Transcription completed in {int(mins):02d}:{int(secs):02d}")

            # Extract chunks from result
            chunks = result.get("chunks", [])
            if not chunks:
                print("Warning: No transcription chunks found")
                # Create a single chunk with the full text if no chunks
                chunks = [{
                    "text": result.get("text", ""),
                    "timestamp": [0.0, audio_duration]
                }]

            print(f"Generated {len(chunks)} transcription chunks")

            # Convert to SRT format
            subs = self._convert_timestamps_to_srt(chunks, audio_duration)

            if not subs:
                raise ValueError("No valid subtitles generated from transcription")

            # Clean the segments
            cleaned_subs = self._clean_srt_segments(subs)

            if not cleaned_subs:
                print("Warning: All segments were filtered out during cleaning")
                cleaned_subs = subs  # Use original if cleaning removes everything

            # Write SRT file
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt.compose(cleaned_subs))

            # Add credit
            self._add_credit_to_srt(srt_path, self.config["credit"])

            print(f"Transcription completed successfully!")
            print(f"SRT file saved: {srt_path}")
            print(f"Total segments: {len(cleaned_subs)}")

            if not temp_audio:
                print(f"Audio file saved permanently: {audio_path}")

            return srt_path

        finally:
            # Clean up temporary audio file if created
            if temp_audio and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"Cleaned up temporary file: {audio_path}")
                except OSError:
                    print(f"Warning: Could not remove temporary file: {audio_path}")


def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file if it exists, otherwise use embedded defaults."""
    # First, use embedded configuration as base
    config = CONFIG.copy()

    # Try to load from external file if it exists (optional override)
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            # Merge with embedded config (file overrides embedded)
            config.update(file_config)
            print(f"Configuration loaded from: {config_file}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using embedded configuration.")
    else:
        print("Using embedded configuration (no config.json found).")

    return config


def save_config(config: Dict[str, Any], config_file: str = "config.json"):
    """Save configuration to file."""
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {config_file}")
    except IOError as e:
        print(f"Warning: Could not save config file: {e}")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py [file]")
        print("  file: Audio or video file to transcribe")
        print("\nFile locations:")
        print("  Audio files: place in audio_source directory")
        print("  Video files: place in video_source directory")
        print("  Converted audio: saved to audio_export directory")
        print("\nConfiguration:")
        print("  Set 'use_faster_whisper': true in config.json for faster processing")
        print("  Set 'use_faster_whisper': false for HF Transformers (better MPS support)")
        print("\nfaster-whisper CPU optimization:")
        print("  'faster_whisper_cpu_threads': 'auto' (recommended for M4)")
        print("  'faster_whisper_cpu_threads': 4 (use specific number of cores)")
        print("  'faster_whisper_cpu_threads': 8 (use 8 cores - good for M4)")
        print("\nRequirements:")
        print("  For faster-whisper: pip install faster-whisper")
        print("  For HF Transformers: pip install transformers torch")
        print("  For CPU optimization: pip install psutil (optional)")
        print("  For SRT processing: pip install srt")
        sys.exit(1)

    input_file = sys.argv[1]

    # Load configuration
    config = load_config()

    # Show which engine will be used
    engine = "faster-whisper" if config.get("use_faster_whisper", False) else "HF Transformers Whisper"
    print(f"🚀 Using {engine} for transcription")

    if config.get("use_faster_whisper", False):
        cpu_threads = config.get("faster_whisper_cpu_threads", "auto")
        print(f"💻 CPU threads setting: {cpu_threads}")

    # Create transcriber and run
    transcriber = WhisperTranscriber(config)

    try:
        result = transcriber.transcribe_file(input_file)
        print(f"\n✅ Success! SRT file created: {result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()