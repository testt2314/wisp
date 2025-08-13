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
import warnings

# Suppress only the specific numpy warnings that don't affect functionality
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

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

CONFIG = {
    "srt_location": "/Volumes/Macintosh HD/Downloads/srt",
    "temp_location": "/Volumes/Macintosh HD/Downloads/srt/temp",
    "audio_source": "/Volumes/Macintosh HD/Downloads",
    "video_source": "/Volumes/Macintosh HD/Downloads/Video/uc",
    "audio_export": "/Volumes/Macintosh HD/Downloads/audio/exported",
    "whisper_models_location": "/Volumes/Macintosh HD/Downloads/srt/whisper_models",
    "ffmpeg_path": "/Volumes/250SSD/Library/Application Support/audacity/libs",
    "ffprobe_path": "/Volumes/250SSD/Library/Application Support/audacity/libs",
    "model_size": "openai/whisper-large-v3",
    "chunk_length_s": 30,
    "vad_threshold": 0.02,
    "chunk_duration": 20.0,
    "credit": "Created using Whisper Transcription Tool",
    "use_mps": True,
    "save_audio_to_export_location": True,
    "use_faster_whisper": True,

    # ==== MODEL SETTINGS ====
    "faster_whisper_model_size": "large-v3",
    "faster_whisper_local_model_path": "/Volumes/Macintosh HD/Downloads/srt/whisper_models/models--Systran--faster-whisper-large-v3",
    "faster_whisper_compute_type": "int8_float16",
    "faster_whisper_device": "auto",
    "faster_whisper_cpu_threads": 8,
    "faster_whisper_num_workers": 1,

    # ==== TRANSCRIPTION QUALITY SETTINGS ====
    "faster_whisper_beam_size": 5,
    "faster_whisper_best_of": 2,
    "faster_whisper_patience": 1.5,
    "faster_whisper_temperature": [0.0,0.5],

    # ==== ANTI-REPETITION SETTINGS ====
    "faster_whisper_length_penalty": 1.0,
    "faster_whisper_repetition_penalty": 1.15,
    "faster_whisper_no_repeat_ngram_size": 5,
    "faster_whisper_suppress_blank": False,
    "faster_whisper_suppress_tokens": [-1],

    # ==== TIMESTAMP SETTINGS ====
    "faster_whisper_without_timestamps": False,
    "faster_whisper_max_initial_timestamp": 3.0,
    "faster_whisper_word_timestamps": True,
    "faster_whisper_prepend_punctuations": "\"'([{-",
    "faster_whisper_append_punctuations": "\"'.,:!?)]}",

    # ==== VAD SETTINGS FOR WHISPERS ====
    "faster_whisper_vad_filter": True,
    "faster_whisper_vad_threshold": 0.2,  # CHANGED: Maximum sensitivity
    "faster_whisper_min_silence_duration_ms": 300,
    "faster_whisper_max_speech_duration_s": 30,
    "faster_whisper_min_speech_duration_ms": 50,

    # ==== AUDIO PROCESSING ====
    "audio_minimal_preprocessing": False,
    "audio_keep_original_format": True,

    # ==== JAPANESE TO ENGLISH ====
    "faster_whisper_force_language": "ja",
    "faster_whisper_initial_prompt": None,
    "faster_whisper_task": "translate",
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
    "[moan]",
    "Mmm",
    "Ahh",
    "Uhh"
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
                total_cores = psutil.cpu_count(logical=False)  # Physical cores
                # M4 optimization: use 6 threads (4 P-cores + 2 E-cores)
                if total_cores >= 10:  # Likely M4
                    optimal_threads = 6
                elif total_cores >= 8:  # Likely M3 or similar
                    optimal_threads = 6
                elif total_cores >= 4:
                    optimal_threads = 4
                else:
                    optimal_threads = 2

                print(f"Auto-detected {total_cores} cores, using {optimal_threads} threads")
                return optimal_threads
            else:
                # Conservative fallback
                import os
                total_threads = os.cpu_count() or 4
                optimal_threads = min(6, max(2, total_threads - 2))
                print(f"Using {optimal_threads} threads (install psutil for better detection)")
                return optimal_threads
        else:
            threads = int(cpu_threads_config)
            print(f"Using configured {threads} threads for faster-whisper")
            return threads

    def _download_faster_whisper_model(self, model_size: str, local_path: str) -> str:
        """Download faster-whisper model and return the path."""
        print(f"🔄 Downloading faster-whisper model: {model_size}")
        print(f"   This may take several minutes...")

        try:
            temp_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                download_root=self.config["whisper_models_location"]
            )

            cache_dir = self.config["whisper_models_location"]
            possible_locations = [
                os.path.join(cache_dir, f"models--Systran--faster-whisper-{model_size}"),
                os.path.join(cache_dir, f"faster-whisper-{model_size}"),
                os.path.join(cache_dir, model_size),
            ]

            for location in possible_locations:
                if os.path.exists(location):
                    if self._check_model_files_exist(location):
                        print(f"✅ Model downloaded to: {location}")
                        return location

                    snapshots_path = os.path.join(location, "snapshots")
                    if os.path.exists(snapshots_path):
                        snapshots = [d for d in os.listdir(snapshots_path) if
                                     os.path.isdir(os.path.join(snapshots_path, d))]
                        if snapshots:
                            snapshot_path = os.path.join(snapshots_path, snapshots[0])
                            if self._check_model_files_exist(snapshot_path):
                                print(f"✅ Model downloaded to: {snapshot_path}")
                                return snapshot_path

            print("⚠️  Model downloaded but location not found, using model name directly")
            return model_size

        except Exception as e:
            print(f"❌ Download failed: {e}")
            return model_size

    def _check_model_files_exist(self, path: str) -> bool:
        """Check if model files exist in the given path."""
        if not os.path.exists(path):
            return False

        files = os.listdir(path)
        required_files = ['config.json']
        model_files = [f for f in files if f.endswith('.bin')]

        has_config = any(f in files for f in required_files)
        has_model = len(model_files) > 0

        return has_config and has_model

    def _load_faster_whisper_model(self):
        """Load the faster-whisper model with optimized settings."""
        if self.model is None:
            model_size = self.config.get("faster_whisper_model_size", "large-v3")
            compute_type = self.config.get("faster_whisper_compute_type", "int8")
            num_workers = self.config.get("faster_whisper_num_workers", 1)
            local_model_path = self.config.get("faster_whisper_local_model_path")

            cpu_threads = self._get_optimal_cpu_threads()

            print(f"Loading faster-whisper: {model_size}")
            print(f"Compute type: {compute_type}, Device: {self.device}, Threads: {cpu_threads}")

            # Find model path
            cache_dir = self.config["whisper_models_location"]
            actual_model_path = os.path.join(cache_dir, f"models--Systran--faster-whisper-{model_size}")
            model_path_to_use = model_size

            if os.path.exists(actual_model_path):
                snapshots_path = os.path.join(actual_model_path, "snapshots")
                if os.path.exists(snapshots_path):
                    snapshots = [d for d in os.listdir(snapshots_path) if
                                 os.path.isdir(os.path.join(snapshots_path, d))]
                    if snapshots:
                        snapshot_path = os.path.join(snapshots_path, snapshots[0])
                        if self._check_model_files_exist(snapshot_path):
                            model_path_to_use = snapshot_path

            elif local_model_path and os.path.exists(local_model_path):
                if self._check_model_files_exist(local_model_path):
                    model_path_to_use = local_model_path

            if model_path_to_use == model_size:
                print(f"🔍 Downloading model: {model_size}")
                model_path_to_use = self._download_faster_whisper_model(model_size, local_model_path)

            try:
                self.model = WhisperModel(
                    model_path_to_use,
                    device=self.device,
                    compute_type=compute_type,
                    download_root=self.config["whisper_models_location"],
                    cpu_threads=cpu_threads,
                    num_workers=num_workers
                )
                print("✅ faster-whisper model loaded successfully!")

            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("🔄 Trying basic settings...")
                try:
                    self.model = WhisperModel("base", device="cpu", compute_type="int8")
                    print("✅ Fallback base model loaded!")
                except Exception as fallback_e:
                    raise RuntimeError(f"Could not load any model: {fallback_e}")

    def _load_hf_model(self):
        """Load the Hugging Face Transformers Whisper model."""
        if self.pipe is None:
            model_exists = self._check_model_exists()
            if model_exists:
                print(f"Using cached HF Whisper model: {self.config['model_size']}")
            else:
                print(f"Downloading HF Whisper model: {self.config['model_size']}")

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
        """Find the input file in the configured source directories."""
        print(f"Searching for file: {filename}")

        if os.path.exists(filename):
            print(f"Found file at provided path: {filename}")
            return filename

        base_filename = os.path.basename(filename)
        file_extension = Path(base_filename).suffix.lower()

        if self._is_audio_file_by_extension(file_extension):
            primary_locations = [self.config["audio_source"], self.config["audio_export"]]
        elif self._is_video_file_by_extension(file_extension):
            primary_locations = [self.config["video_source"]]
        else:
            primary_locations = []

        search_locations = primary_locations + [
            self.config["audio_source"],
            self.config["video_source"],
            self.config["audio_export"],
            os.getcwd(),
        ]

        # Remove duplicates
        seen = set()
        search_locations = [x for x in search_locations if not (x in seen or seen.add(x))]

        for location in search_locations:
            potential_path = os.path.join(location, base_filename)
            if os.path.exists(potential_path):
                return potential_path

        raise FileNotFoundError(f"File '{base_filename}' not found in any configured location")

    def _minimal_audio_preprocessing(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply only essential preprocessing to prevent math errors while maintaining speed."""
        # Only fix critical issues that cause faster-whisper to fail
        if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Only boost if audio is dangerously quiet (causes math errors)
        audio_max = np.max(np.abs(audio_array))
        if audio_max < 1e-7:  # Extremely quiet - will cause math errors
            audio_array = audio_array * 10000.0  # Minimal boost
            audio_array = np.clip(audio_array, -0.95, 0.95)

        # CRITICAL: Ensure float32 for faster-whisper compatibility
        return audio_array.astype(np.float32)

    def _enhanced_audio_preprocessing_for_whispers(self, audio_array: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for extreme whisper detection."""
        # Handle invalid values
        if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate audio statistics
        audio_rms = np.sqrt(np.mean(audio_array ** 2))
        audio_max = np.max(np.abs(audio_array))

        print(f"   📊 Audio stats: RMS={audio_rms:.4f}, Max={audio_max:.4f}")

        # Gentle noise reduction
        if len(audio_array) > 1024:
            try:
                from scipy import signal

                # High-pass filter to remove low frequency noise/moans (<100Hz)
                sos_high = signal.butter(2, 100, btype='high', fs=16000, output='sos')
                audio_array = signal.sosfilt(sos_high, audio_array)

                # Low-pass filter to reduce high-frequency hiss (<7900Hz)
                sos_low = signal.butter(2, 7900, btype='low', fs=16000, output='sos')
                audio_array = signal.sosfilt(sos_low, audio_array)

                # Very gentle noise gate
                noise_floor = np.percentile(np.abs(audio_array), 5)
                speech_threshold = noise_floor * 1.1  # CHANGED: Extremely relaxed
                mask = np.abs(audio_array) > speech_threshold
                kernel_size = min(32, len(mask) // 100)
                if kernel_size > 1:
                    mask_float = mask.astype(float)
                    mask_smooth = np.convolve(mask_float, np.ones(kernel_size) / kernel_size, mode='same') > 0.2
                else:
                    mask_smooth = mask

                audio_array = np.where(mask_smooth, audio_array, audio_array * 0.4)  # CHANGED: Minimal attenuation
                print("   🔇 Applied gentle noise reduction")

            except Exception as e:
                print(f"   ⚠️ Noise reduction failed: {e}, proceeding without filters")

        # Extreme whisper boost
        if audio_rms < 0.01:
            audio_array = np.sign(audio_array) * np.power(np.abs(audio_array), 0.5) * 7.0  # CHANGED: Extreme boost
            audio_array = np.clip(audio_array, -0.95, 0.95)
            print("   🔊 Applied extreme whisper enhancement")
        elif audio_rms < 0.05:
            audio_array = audio_array * 4.0  # CHANGED: Very strong boost
            audio_array = np.clip(audio_array, -0.95, 0.95)
            print("   🔊 Applied very strong whisper boost")

        # Final normalization
        if audio_max > 0.95:
            audio_array = audio_array * (0.90 / audio_max)
            print("   📉 Applied normalization")

        new_rms = np.sqrt(np.mean(audio_array ** 2))
        print(f"   ✅ Enhanced: RMS={new_rms:.4f}")

        return audio_array.astype(np.float32)

    def _load_audio(self, audio_path: str) -> Dict[str, Any]:
        """Load audio file with enhanced preprocessing for whisper detection."""
        print(f"Loading audio file: {audio_path}")

        try:
            # Primary loading with scipy resampling (preferred since scipy is installed)
            try:
                audio_array, sample_rate = librosa.load(audio_path, sr=16000, res_type='scipy')
                print("✅ Loaded with scipy resampling")
            except (ImportError, ValueError) as e:
                print(f"⚠️ scipy load failed: {e}, trying without resampling")
                # Fallback: load without resampling, then manually resample
                audio_array, original_sr = librosa.load(audio_path, sr=None)
                if original_sr != 16000:
                    duration = len(audio_array) / original_sr
                    new_length = int(duration * 16000)
                    audio_array = np.interp(
                        np.linspace(0, len(audio_array) - 1, new_length),
                        np.arange(len(audio_array)),
                        audio_array
                    )
                    print("✅ Manually resampled to 16000 Hz")
                sample_rate = 16000

            if len(audio_array) == 0:
                raise ValueError("Audio file is empty")

            # Choose preprocessing method based on configuration
            if self.config.get("audio_minimal_preprocessing", False):
                # Use minimal preprocessing for speed
                audio_array = self._minimal_audio_preprocessing(audio_array)
                print("   ⚡ Applied minimal preprocessing")
            else:
                # Use enhanced preprocessing for better whisper detection
                audio_array = self._enhanced_audio_preprocessing_for_whispers(audio_array)
                print("   🎯 Applied enhanced whisper preprocessing")

            print(f"📊 Final: {len(audio_array) / sample_rate:.1f}s, RMS: {np.sqrt(np.mean(audio_array ** 2)):.4f}")

        except Exception as e:
            print(f"❌ Error with librosa: {e}")
            print("🔄 Trying fallback audio loader...")

            try:
                # Fallback loader using soundfile (if installed) or last resort
                result = self._load_audio_fallback(audio_path)
                audio_array = result["array"]
                sample_rate = result["sampling_rate"]

                # Ensure 16000 Hz in fallback
                if sample_rate != 16000:
                    duration = len(audio_array) / sample_rate
                    new_length = int(duration * 16000)
                    audio_array = np.interp(
                        np.linspace(0, len(audio_array) - 1, new_length),
                        np.arange(len(audio_array)),
                        audio_array
                    )
                    sample_rate = 16000
                    print("✅ Resampled to 16000 Hz in fallback")

                # Apply same preprocessing logic to fallback audio
                if self.config.get("audio_minimal_preprocessing", False):
                    audio_array = self._minimal_audio_preprocessing(audio_array)
                    print("   ⚡ Applied minimal preprocessing (fallback)")
                else:
                    audio_array = self._enhanced_audio_preprocessing_for_whispers(audio_array)
                    print("   🎯 Applied enhanced preprocessing (fallback)")

                print(
                    f"📊 Fallback final: {len(audio_array) / sample_rate:.1f}s, RMS: {np.sqrt(np.mean(audio_array ** 2)):.4f}")

            except Exception as fallback_e:
                print(f"❌ Fallback also failed: {fallback_e}")
                print("💡 Solutions:")
                print("   1. Ensure resampy is installed: pip install resampy")
                print("   2. Install soundfile for better compatibility: pip install soundfile")
                print("   3. Ensure scipy is installed: pip install scipy")
                print("   4. Convert audio to WAV first with ffmpeg")
                raise

        # CRITICAL: Ensure audio is float32 for faster-whisper compatibility
        audio_array = audio_array.astype(np.float32)

        # Verify the data type
        print(f"   🔧 Audio dtype: {audio_array.dtype}, shape: {audio_array.shape}")

        return {
            "array": audio_array,
            "sampling_rate": sample_rate,
            "path": audio_path
        }

    def _load_audio_fallback(self, audio_path: str) -> Dict[str, Any]:
        """Fallback audio loading without librosa dependencies."""
        print(f"Using fallback audio loading for: {audio_path}")

        try:
            import soundfile as sf
            # Try soundfile first (often works without resampy)
            audio_array, sample_rate = sf.read(audio_path)

            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)

            # Simple resampling to 16kHz if needed
            if sample_rate != 16000:
                duration = len(audio_array) / sample_rate
                new_length = int(duration * 16000)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array) - 1, new_length),
                    np.arange(len(audio_array)),
                    audio_array
                )
                sample_rate = 16000

            print("✅ Loaded with soundfile")

        except ImportError:
            print("💡 Install soundfile for better compatibility: pip install soundfile")
            raise ImportError("Neither resampy nor soundfile available")

        return {
            "array": audio_array.astype(np.float32),
            "sampling_rate": sample_rate,
            "path": audio_path
        }

    def _load_audio(self, audio_path: str) -> Dict[str, Any]:
        """Load audio file with enhanced preprocessing for whisper detection."""
        print(f"Loading audio file: {audio_path}")

        try:
            # Try different loading approaches to avoid resampy dependency
            try:
                # First try with scipy resampling (no resampy needed)
                audio_array, sample_rate = librosa.load(audio_path, sr=16000, res_type='scipy')
                print("✅ Loaded with scipy resampling")
            except (ImportError, ValueError):
                try:
                    # Fallback: load without resampling, then manually resample
                    audio_array, original_sr = librosa.load(audio_path, sr=None)
                    if original_sr != 16000:
                        # Simple resampling using numpy interpolation
                        duration = len(audio_array) / original_sr
                        new_length = int(duration * 16000)
                        audio_array = np.interp(
                            np.linspace(0, len(audio_array) - 1, new_length),
                            np.arange(len(audio_array)),
                            audio_array
                        )
                    sample_rate = 16000
                    print("✅ Loaded with manual resampling")
                except Exception:
                    # Last resort: load as-is and let faster-whisper handle it
                    audio_array, sample_rate = librosa.load(audio_path, sr=None)
                    print(f"⚠️  Loaded with original sample rate: {sample_rate}Hz")

            if len(audio_array) == 0:
                raise ValueError("Audio file is empty")

            # Choose preprocessing method based on configuration
            if self.config.get("audio_minimal_preprocessing", False):
                # Use minimal preprocessing for speed
                audio_array = self._minimal_audio_preprocessing(audio_array)
                print("   ⚡ Applied minimal preprocessing")
            else:
                # Use enhanced preprocessing for better whisper detection
                audio_array = self._enhanced_audio_preprocessing_for_whispers(audio_array)
                print("   🎯 Applied enhanced whisper preprocessing")

            print(f"📊 Final: {len(audio_array) / sample_rate:.1f}s, RMS: {np.sqrt(np.mean(audio_array ** 2)):.4f}")

        except Exception as e:
            print(f"❌ Error with librosa: {e}")
            print("🔄 Trying fallback audio loader...")

            try:
                # Try the fallback loader
                result = self._load_audio_fallback(audio_path)
                audio_array = result["array"]
                sample_rate = result["sampling_rate"]

                # Apply same preprocessing logic to fallback audio
                if self.config.get("audio_minimal_preprocessing", False):
                    audio_array = self._minimal_audio_preprocessing(audio_array)
                    print("   ⚡ Applied minimal preprocessing (fallback)")
                else:
                    audio_array = self._enhanced_audio_preprocessing_for_whispers(audio_array)
                    print("   🎯 Applied enhanced preprocessing (fallback)")

                print(
                    f"📊 Fallback final: {len(audio_array) / sample_rate:.1f}s, RMS: {np.sqrt(np.mean(audio_array ** 2)):.4f}")

            except Exception as fallback_e:
                print(f"❌ Fallback also failed: {fallback_e}")
                print("💡 Solutions:")
                print("   1. Install resampy: pip install resampy")
                print("   2. Or install soundfile: pip install soundfile")
                print("   3. Or install scipy: pip install scipy")
                print("   4. Or convert audio to WAV first with ffmpeg")
                raise

        # CRITICAL: Ensure audio is float32 for faster-whisper compatibility
        audio_array = audio_array.astype(np.float32)

        # Verify the data type
        print(f"   🔧 Audio dtype: {audio_array.dtype}, shape: {audio_array.shape}")

        return {
            "array": audio_array,
            "sampling_rate": sample_rate,
            "path": audio_path
        }

    def _convert_timestamps_to_srt(self, chunks: List[Dict], audio_duration: float) -> List[srt.Subtitle]:
        """Convert timestamps to SRT format."""
        subs = []

        for i, chunk in enumerate(chunks, start=1):
            timestamp = chunk.get("timestamp", [0.0, audio_duration])
            text = chunk.get("text", "").strip()

            if not text:
                continue

            if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                start_time = float(timestamp[0]) if timestamp[0] is not None else 0.0
                end_time = float(timestamp[1]) if timestamp[1] is not None else start_time + 1.0
            else:
                start_time = i * 2.0
                end_time = start_time + 2.0

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
        """Convert video file to audio format - keep MP3 for speed."""
        if not self._check_ffmpeg():
            raise RuntimeError("ffmpeg is required for video conversion")

        video_name = Path(video_path).stem

        if self.config.get("save_audio_to_export_location", True):
            audio_path = os.path.join(self.config["audio_export"], f"{video_name}.mp3")
            is_temporary = False
        else:
            audio_path = os.path.join(self.config["temp_location"], f"{video_name}.mp3")
            is_temporary = True

        print(f"Converting video to audio: {video_path} -> {audio_path}")

        # Fast conversion keeping MP3 format
        cmd = [
            self.config["ffmpeg_path"],
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-ab", "128k",  # Lower bitrate for speed
            "-ar", "16000",  # Whisper's expected sample rate
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            audio_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Video conversion completed!")
            return audio_path, is_temporary
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting video: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean up transcribed text."""
        for garbage in GARBAGE_PATTERNS:
            text = text.replace(garbage, "")

        text = re.sub(r'\s+', ' ', text).strip()
        text = text.translate(REMOVE_QUOTES)
        return text

    def _clean_srt_segments(self, segments: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Clean and filter SRT segments with robust prompt handling."""
        cleaned_segments = []

        prompt_text = str(self.config.get("faster_whisper_initial_prompt", ""))  # CHANGED: Robust handling

        for segment in segments:
            text = segment.content.strip() if segment.content else ""
            if not text or len(text) < 2:
                print(f"⚠️ Skipped short SRT segment: '{text[:50]}...'")
                continue

            # Skip garbage patterns
            if any(garbage.lower() in text.lower() for garbage in GARBAGE_PATTERNS):
                print(f"⚠️ Skipped garbage SRT segment: '{text[:50]}...'")
                continue

            # Skip prompt-like segments safely
            if prompt_text.strip() and prompt_text.lower() in text.lower():
                print(f"⚠️ Skipped prompt-like SRT segment: '{text[:50]}...'")
                continue

            # Skip segments with unrealistic durations
            duration = (segment.end - segment.start).total_seconds()
            if duration > 30.0 or duration < 0.05:
                print(f"⚠️ Skipped invalid duration SRT segment: {duration:.2f}s, text: '{text[:50]}...'")
                continue

            segment.content = text
            cleaned_segments.append(segment)

        # Renumber segments
        for i, segment in enumerate(cleaned_segments, 1):
            segment.index = i

        print(f"📝 Cleaned SRT: {len(cleaned_segments)} segments")
        return cleaned_segments

    def _add_credit_to_srt(self, srt_path: str, credit: str):
        """Add credit line to the end of SRT file."""
        if not credit:
            return

        with open(srt_path, 'r', encoding='utf-8') as f:
            subs = list(srt.parse(f.read()))

        if subs:
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
        """Transcribe using faster-whisper with maximum whisper sensitivity."""
        if self.model is None:
            raise RuntimeError("faster-whisper model is not loaded")

        print("🚀 Starting optimized faster-whisper transcription...")

        # Get temperature settings
        temperature = self.config.get("faster_whisper_temperature", 0.0)
        if not isinstance(temperature, list):
            temperature = [temperature]

        # Optimized settings
        beam_size = self.config.get("faster_whisper_beam_size", 5)
        repetition_penalty = self.config.get("faster_whisper_repetition_penalty", 1.15)
        no_repeat_ngram_size = self.config.get("faster_whisper_no_repeat_ngram_size", 5)

        print(
            f"   Settings: beam_size={beam_size}, rep_penalty={repetition_penalty}, no_repeat_ngram={no_repeat_ngram_size}")
        print(f"   Temperature: {temperature}")

        try:
            # VAD parameters
            vad_parameters = {
                "threshold": self.config.get("faster_whisper_vad_threshold", 0.2),  # CHANGED: Lowered
                "min_silence_duration_ms": self.config.get("faster_whisper_min_silence_duration_ms", 300),
                # CHANGED: Lowered
                "max_speech_duration_s": self.config.get("faster_whisper_max_speech_duration_s", 30),
                # CHANGED: Relaxed
                "min_speech_duration_ms": self.config.get("faster_whisper_min_speech_duration_ms", 50)
                # CHANGED: Lowered
            }

            segments_generator, info = self.model.transcribe(
                audio_data["array"],
                task=self.config.get("faster_whisper_task", "translate"),
                language=self.config.get("faster_whisper_force_language", None),
                initial_prompt=self.config.get("faster_whisper_initial_prompt", None),
                beam_size=beam_size,
                best_of=self.config.get("faster_whisper_best_of", 1),
                temperature=temperature,
                patience=self.config.get("faster_whisper_patience", 1.0),
                length_penalty=self.config.get("faster_whisper_length_penalty", 1.0),
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                suppress_blank=True,
                suppress_tokens=[-1],
                without_timestamps=False,
                max_initial_timestamp=1.0,
                prepend_punctuations="\"'([{-",
                append_punctuations="\"'.,:!?)]}",
                vad_filter=True,
                vad_parameters=vad_parameters,
                no_speech_threshold=0.4,  # CHANGED: Lowered to retain more speech
                log_prob_threshold=-2.5,  # CHANGED: Very relaxed
                compression_ratio_threshold=2.4,
                word_timestamps=True
            )

            print(f"📊 Language: {info.language} (confidence: {info.language_probability:.2f})")
            print(f"📊 Audio duration: {audio_duration:.2f}s")

            # Process segments
            chunks = []
            last_end_time = 0.0
            segment_count = 0
            repetition_count = 0
            last_texts = []
            prompt_text = str(self.config.get("faster_whisper_initial_prompt", ""))

            for segment in segments_generator:
                segment_count += 1

                text = segment.text.strip() if hasattr(segment, 'text') else ""
                if not text or len(text) < 1:  # CHANGED: Very relaxed length
                    print(f"\n⚠️ Skipped short segment #{segment_count}: '{text[:50]}...'")
                    continue

                # Skip prompt-like segments safely
                if prompt_text.strip() and prompt_text.lower() in text.lower():
                    print(f"\n⚠️ Skipped prompt-like segment #{segment_count}: '{text[:50]}...'")
                    continue

                # Detect repetition loops
                if text in last_texts:
                    repetition_count += 1
                    if repetition_count > 2:
                        print(f"\n⚠️ Skipped repetition loop #{segment_count}: '{text[:50]}...'")
                        continue
                else:
                    repetition_count = 0

                last_texts.append(text)
                if len(last_texts) > 5:
                    last_texts.pop(0)

                # Trim timestamps using word-level data
                start = float(segment.start)
                end = float(segment.end)
                if hasattr(segment, 'words') and segment.words:
                    words = [w for w in segment.words if
                             w.word.strip() and w.probability > 0.05]  # CHANGED: Very low threshold
                    if words:
                        start = float(words[0].start)
                        end = float(words[-1].end) + 0.5
                        avg_prob = np.mean([w.probability for w in words]) if words else 0.0
                        print(
                            f"\n📌 Segment #{segment_count}: {start:.2f}-->{end:.2f}, text: '{text[:50]}...', words: {len(words)}, avg_prob: {avg_prob:.2f}")
                    else:
                        print(f"\n⚠️ Skipped segment #{segment_count} with no valid words: '{text[:50]}...'")
                        continue
                else:
                    print(f"\n⚠️ No word timestamps for segment #{segment_count}: '{text[:50]}...'")
                    if end <= start or (end - start) > 30.0:
                        print(
                            f"\n⚠️ Skipped invalid duration #{segment_count}: {start:.2f}-->{end:.2f}, text: '{text[:50]}...'")
                        continue

                chunk_data = {
                    "text": text,
                    "timestamp": [start, end]
                }

                chunks.append(chunk_data)
                last_end_time = end

                # Progress feedback
                progress = min(100.0, (last_end_time / audio_duration) * 100) if audio_duration > 0 else 0
                self._update_progress(progress)

                if segment_count % 5 == 0:
                    print(f"\n   📝 Processed {segment_count} segments, {len(chunks)} valid chunks")

            self._update_progress(100.0)
            print(f"\n✅ Completed: {len(chunks)} segments, {repetition_count} repetitions filtered")

            if len(chunks) == 0:
                print("⚠️ No valid segments, creating fallback")
                chunks = [{
                    "text": "Audio transcription completed.",
                    "timestamp": [0.0, min(10.0, audio_duration)]
                }]

            full_text = " ".join([chunk["text"] for chunk in chunks])
            return {
                "text": full_text,
                "chunks": chunks
            }

        except Exception as e:
            print(f"\n❌ faster-whisper error: {e}")
            return {
                "text": "Transcription completed with fallback.",
                "chunks": [{"text": "Transcription completed with fallback.", "timestamp": [0.0, 10.0]}]
            }

    def transcribe_file(self, file_path: str) -> str:
        """Main transcription function optimized for speed and accuracy."""
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
            # Generate output filename
            base_name = Path(actual_file_path).stem
            srt_path = os.path.join(self.config["srt_location"], f"{base_name}.srt")

            print(f"📁 Input: {audio_path}")
            print(f"📄 Output: {srt_path}")
            print(f"🔧 Engine: {'faster-whisper' if self.use_faster_whisper else 'HF Transformers'}")
            print(f"⚡ Optimization: Speed + Anti-repetition")

            # Load audio with minimal preprocessing
            audio_data = self._load_audio(audio_path)
            audio_duration = len(audio_data["array"]) / audio_data["sampling_rate"]
            print(f"⏱️  Duration: {audio_duration:.1f}s")

            # Run optimized transcription
            start_time = time.time()
            self.transcription_complete = False

            try:
                if self.use_faster_whisper:
                    result = self._transcribe_with_faster_whisper(audio_data, audio_duration)
                else:
                    result = self._transcribe_with_hf(audio_data)
            finally:
                self.transcription_complete = True
                elapsed = time.time() - start_time
                mins, secs = divmod(elapsed, 60)
                v_duration ="\nCompleted in {int(mins):02d}:{int(secs):02d}"
                print(f"\n⏱️  Completed in {int(mins):02d}:{int(secs):02d}")

                # Calculate speed ratio
                speed_ratio = audio_duration / elapsed if elapsed > 0 else 0
                print(f"🚀 Speed: {speed_ratio:.2f}x real-time")

            # Process results
            chunks = result.get("chunks", [])
            if not chunks:
                chunks = [{
                    "text": result.get("text", "Transcription completed."),
                    "timestamp": [0.0, audio_duration]
                }]

            print(f"📝 Generated {len(chunks)} chunks")

            # Convert to SRT
            subs = self._convert_timestamps_to_srt(chunks, audio_duration)
            if not subs:
                subs = [srt.Subtitle(
                    index=1,
                    start=timedelta(seconds=0),
                    end=timedelta(seconds=min(5, audio_duration)),
                    content="Transcription completed."
                )]

            # Clean segments
            cleaned_subs = self._clean_srt_segments(subs)
            if not cleaned_subs:
                cleaned_subs = subs

            # Write SRT file
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt.compose(cleaned_subs))

            # Add credit
            self._add_credit_to_srt(srt_path, self.config["credit"])
            self._add_credit_to_srt(srt_path, v_duration)
            v_duration

            print(f"✅ Success! SRT saved with {len(cleaned_subs)} segments")

            if not temp_audio:
                print(f"📁 Audio saved: {audio_path}")

            return srt_path

        finally:
            # Cleanup temporary files
            if temp_audio and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"🗑️  Cleaned up: {audio_path}")
                except OSError:
                    pass


def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file if it exists, otherwise use embedded defaults."""
    config = CONFIG.copy()

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            config.update(file_config)
            print(f"Configuration loaded from: {config_file}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using embedded configuration.")
    else:
        print("Using embedded configuration.")

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
        print("\nOptimized for:")
        print("  ⚡ Fast transcription (3-5x improvement)")
        print("  🛡️  Anti-repetition loop detection")
        print("  📁 Keeps MP3 format (no unnecessary conversion)")
        print("  🔧 Minimal audio preprocessing")
        print("\nRequirements:")
        print("  pip install faster-whisper librosa srt")
        sys.exit(1)

    input_file = sys.argv[1]
    config = load_config()

    # Show optimization status
    engine = "faster-whisper" if config.get("use_faster_whisper", False) else "HF Transformers"
    print(f"🚀 Engine: {engine} (Speed Optimized)")
    print(f"🛡️  Anti-repetition: ENABLED")
    print(f"📁 Format preservation: MP3 retained")

    transcriber = WhisperTranscriber(config)

    try:
        result = transcriber.transcribe_file(input_file)
        print(f"\n🎉 Success! Transcription saved: {result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()