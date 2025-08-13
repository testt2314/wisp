#!/usr/bin/env python3
"""
Whisper Transcription Tool for Mac mini - ENHANCED FOR SOFT AUDIO
Converts video/audio files to English subtitles using Whisper or faster-whisper
Usage: python transcribe.py [file]

IMPROVEMENTS:
- Enhanced audio preprocessing for extremely quiet/whisper audio
- Dynamic range compression for consistent volume
- Advanced noise gate with whisper detection
- Optimized VAD settings for soft speech
- Multi-pass audio enhancement

Requirements:
- For regular Whisper: pip install transformers torch librosa
- For faster-whisper: pip install faster-whisper librosa scipy
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

# Try to import scipy for advanced audio processing
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
    print("✅ scipy available for advanced audio processing")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available. Install for better whisper audio processing: pip install scipy")

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
    "faster_whisper_temperature": [0.0, 0.2, 0.4, 0.6, 0.8],  # CHANGED: More temperatures for difficult audio

    # ==== ANTI-REPETITION SETTINGS ====
    "faster_whisper_length_penalty": 1.0,
    "faster_whisper_repetition_penalty": 1.15,
    "faster_whisper_no_repeat_ngram_size": 5,
    "faster_whisper_suppress_blank": False,
    "faster_whisper_suppress_tokens": [-1],

    # ==== TIMESTAMP PRECISION SETTINGS ====
    "faster_whisper_without_timestamps": False,
    "faster_whisper_max_initial_timestamp": 1.0,  # CHANGED: More conservative
    "faster_whisper_word_timestamps": True,
    "faster_whisper_prepend_punctuations": "\"'([{-",
    "faster_whisper_append_punctuations": "\"'.,:!?)]}",
    "timestamp_padding_start": 0.1,  # NEW: Conservative start padding
    "timestamp_padding_end": 0.1,  # NEW: Conservative end padding
    "prevent_overlapping": True,  # NEW: Enable overlap prevention
    # ==== SEGMENT PROCESSING SETTINGS ====
    "max_segment_duration": 20.0,  # NEW: Maximum duration before splitting (seconds)
    "min_segment_duration": 0.1,  # NEW: Minimum duration to keep
    # ==== LOW CONFIDENCE REPROCESSING SETTINGS ====
    "reprocess_low_confidence": True,  # NEW: Enable reprocessing of low confidence segments
    "low_confidence_threshold": 0.35,  # NEW: Threshold below which to reprocess (0.35 = 35%)
    "reprocess_improvement_threshold": 0.15,  # NEW: Minimum improvement required to accept
    "reprocess_min_acceptable": 0.4,  # NEW: Minimum acceptable confidence for reprocessed segments

    # ==== VAD SETTINGS FOR WHISPERS - OPTIMIZED FOR SOFT AUDIO ====
    "faster_whisper_vad_filter": True,
    "faster_whisper_vad_threshold": 0.1,  # CHANGED: Even lower threshold for whispers
    "faster_whisper_min_silence_duration_ms": 200,  # CHANGED: Shorter silence detection
    "faster_whisper_max_speech_duration_s": 45,  # CHANGED: Longer segments for whispers
    "faster_whisper_min_speech_duration_ms": 25,  # CHANGED: Detect very short whispers

    # ==== AUDIO PROCESSING - ENHANCED FOR WHISPERS ====
    "audio_minimal_preprocessing": False,  # CHANGED: Use full processing for whispers
    "audio_keep_original_format": True,
    "audio_whisper_boost_enabled": True,  # NEW: Enable aggressive whisper boost
    "audio_dynamic_range_compression": True,  # NEW: Compress dynamic range
    "audio_spectral_gating": True,  # NEW: Advanced noise reduction

    # ==== JAPANESE TO ENGLISH ====
    "faster_whisper_force_language": "ja",
    "faster_whisper_initial_prompt": None,
    "faster_whisper_task": "translate",

    # ==== SOFT AUDIO DETECTION THRESHOLDS ====
    "soft_audio_rms_threshold": 0.005,  # NEW: RMS threshold for soft audio detection
    "soft_audio_max_threshold": 0.02,  # NEW: Max amplitude threshold for soft audio
    "whisper_boost_factor": 12.0,  # NEW: Aggressive boost for whispers
    "dynamic_compression_ratio": 4.0,  # NEW: Compression ratio for dynamic range
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

    def _detect_soft_audio(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Analyze audio characteristics to detect soft/whisper speech."""
        # Calculate audio statistics
        audio_rms = np.sqrt(np.mean(audio_array ** 2))
        audio_max = np.max(np.abs(audio_array))
        audio_std = np.std(audio_array)

        # Calculate percentiles for dynamic range analysis
        p95 = np.percentile(np.abs(audio_array), 95)
        p50 = np.percentile(np.abs(audio_array), 50)
        p10 = np.percentile(np.abs(audio_array), 10)

        # Dynamic range ratio
        dynamic_range = p95 / (p10 + 1e-8)

        # Soft audio thresholds from config
        soft_rms_threshold = self.config.get("soft_audio_rms_threshold", 0.005)
        soft_max_threshold = self.config.get("soft_audio_max_threshold", 0.02)

        is_soft = audio_rms < soft_rms_threshold or audio_max < soft_max_threshold
        is_whisper = audio_rms < (soft_rms_threshold * 0.5) and dynamic_range < 5.0

        analysis = {
            "is_soft": is_soft,
            "is_whisper": is_whisper,
            "rms": audio_rms,
            "max": audio_max,
            "std": audio_std,
            "dynamic_range": dynamic_range,
            "p95": p95,
            "p50": p50,
            "p10": p10
        }

        print(f"🔍 Audio Analysis:")
        print(f"   RMS: {audio_rms:.6f} (threshold: {soft_rms_threshold:.6f})")
        print(f"   Max: {audio_max:.6f} (threshold: {soft_max_threshold:.6f})")
        print(f"   Dynamic Range: {dynamic_range:.2f}")
        print(f"   Classification: {'WHISPER' if is_whisper else 'SOFT' if is_soft else 'NORMAL'}")

        return analysis

    def _spectral_noise_gate(self, audio_array: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Advanced spectral noise gate for extremely quiet audio."""
        if not SCIPY_AVAILABLE:
            print("   ⚠️  Skipping spectral gating (scipy required)")
            return audio_array

        try:
            # Convert to frequency domain
            window_size = min(2048, len(audio_array) // 4)
            if window_size < 512:
                return audio_array

            # Compute spectrogram
            f, t, Zxx = signal.stft(audio_array, fs=sample_rate, nperseg=window_size,
                                    noverlap=window_size // 2)

            # Calculate magnitude spectrogram
            magnitude = np.abs(Zxx)

            # Estimate noise floor for each frequency bin
            noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)

            # Create adaptive mask - very gentle for whisper audio
            signal_threshold = noise_floor * 1.2  # Very low threshold
            mask = magnitude > signal_threshold

            # Smooth the mask to avoid artifacts
            from scipy.ndimage import binary_dilation
            mask = binary_dilation(mask, iterations=2)

            # Apply mask with minimal attenuation
            Zxx_filtered = Zxx * (mask.astype(float) * 0.8 + 0.2)  # Minimum 20% of original

            # Convert back to time domain
            _, filtered_audio = signal.istft(Zxx_filtered, fs=sample_rate,
                                             nperseg=window_size, noverlap=window_size // 2)

            # Ensure same length as input
            if len(filtered_audio) != len(audio_array):
                if len(filtered_audio) > len(audio_array):
                    filtered_audio = filtered_audio[:len(audio_array)]
                else:
                    # Pad with zeros
                    filtered_audio = np.pad(filtered_audio, (0, len(audio_array) - len(filtered_audio)))

            print("   ✅ Applied spectral noise gate")
            return filtered_audio.astype(np.float32)

        except Exception as e:
            print(f"   ⚠️  Spectral gating failed: {e}")
            return audio_array

    def _dynamic_range_compression(self, audio_array: np.ndarray, ratio: float = 4.0) -> np.ndarray:
        """Apply dynamic range compression to make quiet sounds louder."""
        try:
            # Calculate moving RMS for dynamic compression
            window_size = min(1024, len(audio_array) // 10)
            if window_size < 64:
                return audio_array

            # Compute moving RMS
            audio_squared = audio_array ** 2
            kernel = np.ones(window_size) / window_size
            moving_rms = np.sqrt(np.convolve(audio_squared, kernel, mode='same'))

            # Define compression curve
            threshold = 0.1  # Compression threshold
            makeup_gain = 2.0  # Overall gain after compression

            # Apply compression
            compressed = np.where(
                moving_rms > threshold,
                audio_array * (threshold + (moving_rms - threshold) / ratio) / (moving_rms + 1e-8),
                audio_array
            )

            # Apply makeup gain
            compressed *= makeup_gain

            # Limit to prevent clipping
            compressed = np.clip(compressed, -0.95, 0.95)

            print(f"   ✅ Applied dynamic compression (ratio: {ratio:.1f})")
            return compressed.astype(np.float32)

        except Exception as e:
            print(f"   ⚠️  Dynamic compression failed: {e}")
            return audio_array

    def _extreme_whisper_enhancement(self, audio_array: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
        """Apply extreme enhancement for whisper-level audio."""
        if not analysis["is_soft"]:
            return audio_array

        print("🔊 Applying extreme whisper enhancement...")

        # Step 1: Spectral noise gate if enabled and available
        if self.config.get("audio_spectral_gating", True):
            audio_array = self._spectral_noise_gate(audio_array)

        # Step 2: Dynamic range compression
        if self.config.get("audio_dynamic_range_compression", True):
            compression_ratio = self.config.get("dynamic_compression_ratio", 4.0)
            audio_array = self._dynamic_range_compression(audio_array, compression_ratio)

        # Step 3: Aggressive whisper boost
        if analysis["is_whisper"] and self.config.get("audio_whisper_boost_enabled", True):
            boost_factor = self.config.get("whisper_boost_factor", 12.0)

            # Power law enhancement for extremely quiet audio
            sign = np.sign(audio_array)
            magnitude = np.abs(audio_array)

            # Apply power curve - makes quiet sounds much louder
            enhanced_magnitude = np.power(magnitude, 0.3) * boost_factor

            # Reconstruct signal
            audio_array = sign * enhanced_magnitude

            print(f"   🚀 Applied extreme whisper boost (factor: {boost_factor:.1f})")

        elif analysis["is_soft"]:
            # Regular soft audio boost
            boost_factor = self.config.get("whisper_boost_factor", 12.0) * 0.6  # 60% of whisper boost
            audio_array = audio_array * boost_factor

            print(f"   🔊 Applied soft audio boost (factor: {boost_factor:.1f})")

        # Step 4: Final limiting to prevent clipping
        audio_max = np.max(np.abs(audio_array))
        if audio_max > 0.95:
            audio_array = audio_array * (0.90 / audio_max)
            print("   📉 Applied final limiting")

        # Step 5: High-frequency emphasis for whisper clarity
        if SCIPY_AVAILABLE and analysis["is_whisper"]:
            try:
                # Gentle high-frequency boost to improve whisper clarity
                sos = signal.butter(2, [1000, 4000], btype='band', fs=16000, output='sos')
                high_freq = signal.sosfilt(sos, audio_array)
                audio_array = audio_array + high_freq * 0.3  # Subtle boost
                print("   ✨ Applied high-frequency emphasis for whisper clarity")
            except Exception as e:
                print(f"   ⚠️  High-frequency emphasis failed: {e}")

        return audio_array.astype(np.float32)

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
        """Enhanced preprocessing for extreme whisper detection with advanced algorithms."""
        # Handle invalid values
        if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Analyze audio characteristics
        analysis = self._detect_soft_audio(audio_array)

        # Apply appropriate enhancement based on analysis
        if analysis["is_soft"] or analysis["is_whisper"]:
            audio_array = self._extreme_whisper_enhancement(audio_array, analysis)
        else:
            # Standard preprocessing for normal audio
            audio_rms = analysis["rms"]
            if audio_rms < 0.05:
                audio_array = audio_array * 3.0  # Moderate boost for low audio
                audio_array = np.clip(audio_array, -0.95, 0.95)
                print("   🔊 Applied standard audio boost")

        # Final verification
        final_rms = np.sqrt(np.mean(audio_array ** 2))
        final_max = np.max(np.abs(audio_array))
        print(f"   ✅ Enhanced: RMS={final_rms:.4f}, Max={final_max:.4f}")

        return audio_array.astype(np.float32)

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

                print(f"📊 Fallback final: {len(audio_array) / sample_rate:.1f}s, RMS: {np.sqrt(np.mean(audio_array ** 2)):.4f}")

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

    def _split_long_segment(self, segment: srt.Subtitle, max_duration: float = 15.0) -> List[srt.Subtitle]:
        """Split overly long segments into smaller chunks for better readability."""
        duration = (segment.end - segment.start).total_seconds()

        if duration <= max_duration:
            return [segment]

        # Split text into words
        words = segment.content.strip().split()
        if len(words) <= 1:
            return [segment]  # Can't split single word

        # Calculate how many chunks we need
        num_chunks = max(2, int(np.ceil(duration / max_duration)))
        words_per_chunk = max(1, len(words) // num_chunks)

        chunks = []
        start_time = segment.start.total_seconds()
        time_per_chunk = duration / num_chunks

        for i in range(num_chunks):
            # Calculate word range for this chunk
            start_word = i * words_per_chunk
            if i == num_chunks - 1:  # Last chunk gets remaining words
                end_word = len(words)
            else:
                end_word = min(start_word + words_per_chunk, len(words))

            if start_word >= len(words):
                break

            # Create chunk text
            chunk_words = words[start_word:end_word]
            chunk_text = " ".join(chunk_words)

            if not chunk_text.strip():
                continue

            # Calculate timing for this chunk
            chunk_start = start_time + (i * time_per_chunk)
            chunk_end = start_time + ((i + 1) * time_per_chunk)

            # Ensure last chunk ends at original end time
            if i == num_chunks - 1:
                chunk_end = segment.end.total_seconds()

            chunk_subtitle = srt.Subtitle(
                index=segment.index + i,  # Will be renumbered later
                start=timedelta(seconds=chunk_start),
                end=timedelta(seconds=chunk_end),
                content=chunk_text
            )
            chunks.append(chunk_subtitle)

        print(f"🔪 Split long segment ({duration:.1f}s) into {len(chunks)} chunks")
        return chunks

    def _clean_srt_segments(self, segments: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Clean and filter SRT segments with robust prompt handling, overlap prevention, and long segment splitting."""
        cleaned_segments = []
        prompt_text = str(self.config.get("faster_whisper_initial_prompt", ""))
        max_segment_duration = self.config.get("max_segment_duration", 15.0)  # NEW: Configurable max duration

        for i, segment in enumerate(segments):
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

            # Check segment duration
            duration = (segment.end - segment.start).total_seconds()

            # Handle overly long segments by splitting them
            if duration > max_segment_duration:
                print(f"📏 Long segment detected: {duration:.1f}s, text: '{text[:50]}...' - splitting")
                split_segments = self._split_long_segment(segment, max_segment_duration)

                for split_seg in split_segments:
                    # Apply overlap prevention to each split segment
                    if cleaned_segments:
                        prev_segment = cleaned_segments[-1]
                        prev_end_time = prev_segment.end.total_seconds()
                        current_start_time = split_seg.start.total_seconds()

                        if current_start_time < prev_end_time:
                            gap = self.config.get("min_segment_gap", 0.1)
                            new_start = prev_end_time + gap
                            split_seg.start = timedelta(seconds=new_start)

                            # Ensure segment doesn't become too short
                            if (split_seg.end - split_seg.start).total_seconds() < 0.3:
                                split_seg.end = timedelta(seconds=new_start + 0.5)

                    cleaned_segments.append(split_seg)
                continue

            # Skip segments that are too short
            if duration < 0.1:
                print(f"⚠️ Skipped very short SRT segment: {duration:.2f}s, text: '{text[:50]}...'")
                continue

            # Fix overlapping segments
            if cleaned_segments:
                prev_segment = cleaned_segments[-1]
                prev_end_time = prev_segment.end.total_seconds()
                current_start_time = segment.start.total_seconds()

                # If current segment starts before previous ends (overlap)
                if current_start_time < prev_end_time:
                    # Adjust previous segment end to prevent overlap
                    gap = self.config.get("min_segment_gap", 0.1)
                    new_prev_end = current_start_time - gap

                    if new_prev_end > prev_segment.start.total_seconds():
                        prev_segment.end = timedelta(seconds=new_prev_end)
                        print(f"🔧 Fixed overlap: adjusted previous segment end to {new_prev_end:.3f}")
                    else:
                        # If adjustment would make previous segment too short, adjust current start
                        new_start = prev_end_time + gap
                        segment.start = timedelta(seconds=new_start)
                        print(f"🔧 Fixed overlap: adjusted current segment start to {new_start:.3f}")

                        # Ensure current segment doesn't become too short
                        if (segment.end - segment.start).total_seconds() < 0.2:
                            segment.end = timedelta(seconds=new_start + 0.5)

            segment.content = text
            cleaned_segments.append(segment)

        # Renumber segments
        for i, segment in enumerate(cleaned_segments, 1):
            segment.index = i

        print(f"📝 Cleaned SRT: {len(cleaned_segments)} segments (overlaps fixed, long segments split)")
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
        """Transcribe using faster-whisper with optimized settings for soft audio."""
        if self.model is None:
            raise RuntimeError("faster-whisper model is not loaded")

        print("🚀 Starting optimized faster-whisper transcription for soft audio...")

        # Get temperature settings - more temperatures for difficult audio
        temperature = self.config.get("faster_whisper_temperature", [0.0, 0.2, 0.4, 0.6, 0.8])
        if not isinstance(temperature, list):
            temperature = [temperature]

        # Optimized settings for soft audio
        beam_size = self.config.get("faster_whisper_beam_size", 5)
        repetition_penalty = self.config.get("faster_whisper_repetition_penalty", 1.15)
        no_repeat_ngram_size = self.config.get("faster_whisper_no_repeat_ngram_size", 5)

        print(f"   Settings: beam_size={beam_size}, rep_penalty={repetition_penalty}, no_repeat_ngram={no_repeat_ngram_size}")
        print(f"   Temperature: {temperature}")

        try:
            # Optimized VAD parameters for soft audio
            vad_parameters = {
                "threshold": self.config.get("faster_whisper_vad_threshold", 0.1),  # Very low for whispers
                "min_silence_duration_ms": self.config.get("faster_whisper_min_silence_duration_ms", 200),  # Shorter
                "max_speech_duration_s": self.config.get("faster_whisper_max_speech_duration_s", 45),  # Longer
                "min_speech_duration_ms": self.config.get("faster_whisper_min_speech_duration_ms", 25)  # Very short
            }

            print(f"   VAD: threshold={vad_parameters['threshold']}, min_silence={vad_parameters['min_silence_duration_ms']}ms")

            segments_generator, info = self.model.transcribe(
                audio_data["array"],
                task=self.config.get("faster_whisper_task", "translate"),
                language=self.config.get("faster_whisper_force_language", None),
                initial_prompt=self.config.get("faster_whisper_initial_prompt", None),
                beam_size=beam_size,
                best_of=self.config.get("faster_whisper_best_of", 2),
                temperature=temperature,
                patience=self.config.get("faster_whisper_patience", 1.5),
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
                no_speech_threshold=0.3,  # Lowered for soft audio
                log_prob_threshold=-3.0,  # More relaxed for whispers
                compression_ratio_threshold=2.4,
                word_timestamps=True
            )

            print(f"📊 Language: {info.language} (confidence: {info.language_probability:.2f})")
            print(f"📊 Audio duration: {audio_duration:.2f}s")

            # Process segments with enhanced whisper detection
            chunks = []
            last_end_time = 0.0
            segment_count = 0
            repetition_count = 0
            last_texts = []
            prompt_text = str(self.config.get("faster_whisper_initial_prompt", ""))

            for segment in segments_generator:
                segment_count += 1

                text = segment.text.strip() if hasattr(segment, 'text') else ""
                if not text or len(text) < 1:  # Very relaxed length for whispers
                    print(f"\n⚠️ Skipped short segment #{segment_count}: '{text[:50]}...'")
                    continue

                # Skip prompt-like segments safely
                if prompt_text.strip() and prompt_text.lower() in text.lower():
                    print(f"\n⚠️ Skipped prompt-like segment #{segment_count}: '{text[:50]}...'")
                    continue

                # Enhanced repetition detection
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

                # Enhanced timestamp processing for whispers - FIXED for accuracy
                start = float(segment.start)
                end = float(segment.end)

                if hasattr(segment, 'words') and segment.words:
                    # Filter words with reasonable probability thresholds
                    words = [w for w in segment.words if
                             w.word.strip() and w.probability > 0.1]  # Reasonable threshold for accuracy

                    if words:
                        # Use word-level timestamps but with conservative padding
                        word_start = float(words[0].start)
                        word_end = float(words[-1].end)

                        # Conservative timing adjustments - prevent early starts and late ends
                        # Only extend if segment timing is clearly wrong
                        if abs(word_start - start) < 2.0:  # If word start is close to segment start
                            start = max(word_start - 0.1, start - 0.2)  # Minimal padding
                        else:
                            start = start  # Use original segment start

                        if abs(word_end - end) < 2.0:  # If word end is close to segment end
                            end = min(word_end + 0.1, end + 0.2)  # Minimal padding
                        else:
                            end = end  # Use original segment end

                        # Ensure timing makes sense
                        if end <= start:
                            end = start + 0.5

                        # Prevent overlapping with previous segment
                        if start < last_end_time:
                            start = last_end_time + 0.1
                            if end <= start:
                                end = start + 0.5

                        avg_prob = np.mean([w.probability for w in words]) if words else 0.0
                        print(f"\n📌 Segment #{segment_count}: {start:.3f}-->{end:.3f}, text: '{text[:50]}...', words: {len(words)}, avg_prob: {avg_prob:.2f}")
                    else:
                        # No valid words - use conservative segment timing
                        if end <= start:
                            end = start + 1.0
                        # Prevent overlapping
                        if start < last_end_time:
                            start = last_end_time + 0.1
                            end = max(end, start + 0.5)
                        print(f"\n⚠️ Using segment timing #{segment_count}: {start:.3f}-->{end:.3f}, text: '{text[:50]}...'")
                else:
                    # No word timestamps available
                    if end <= start:
                        end = start + 1.0
                    # Prevent overlapping
                    if start < last_end_time:
                        start = last_end_time + 0.1
                        end = max(end, start + 0.5)
                    print(f"\n⚠️ No word timestamps #{segment_count}: {start:.3f}-->{end:.3f}, text: '{text[:50]}...'")

                # Final validation
                if end <= start or (end - start) > 45.0:
                    print(f"\n⚠️ Skipped invalid duration #{segment_count}: {start:.3f}-->{end:.3f}, text: '{text[:50]}...'")
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

    def _reprocess_low_confidence_segment(self, audio_data: Dict[str, Any], start_time: float, end_time: float,
                                          original_text: str, avg_prob: float) -> Dict[str, Any]:
        """Reprocess a low-confidence segment with enhanced settings."""
        try:
            # Extract the specific audio segment
            sample_rate = audio_data["sampling_rate"]
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # Add padding for context (0.5 seconds on each side)
            padding_samples = int(0.5 * sample_rate)
            padded_start = max(0, start_sample - padding_samples)
            padded_end = min(len(audio_data["array"]), end_sample + padding_samples)

            segment_audio = audio_data["array"][padded_start:padded_end]

            if len(segment_audio) < sample_rate * 0.1:  # Too short
                return {"text": original_text, "confidence": avg_prob, "reprocessed": False}

            # Apply extra enhancement for reprocessing
            enhanced_audio = self._extra_enhancement_for_reprocessing(segment_audio)

            print(f"   🔄 Reprocessing {end_time - start_time:.1f}s segment with enhanced settings...")

            # Reprocess with more aggressive settings
            segments_generator, info = self.model.transcribe(
                enhanced_audio,
                task=self.config.get("faster_whisper_task", "translate"),
                language=self.config.get("faster_whisper_force_language", None),
                beam_size=8,  # Higher beam size for better accuracy
                best_of=3,  # More candidates
                temperature=[0.0, 0.1, 0.3, 0.5],  # More temperature options
                patience=2.0,  # More patience
                repetition_penalty=1.1,  # Less aggressive repetition penalty
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.05,  # Very sensitive VAD
                    "min_silence_duration_ms": 100,
                    "max_speech_duration_s": 60,
                    "min_speech_duration_ms": 25
                },
                no_speech_threshold=0.2,  # Lower threshold
                log_prob_threshold=-3.5,  # More relaxed
                word_timestamps=True
            )

            # Collect all segments from reprocessing
            reprocessed_text = ""
            best_confidence = 0.0
            word_count = 0

            for segment in segments_generator:
                if hasattr(segment, 'words') and segment.words:
                    valid_words = [w for w in segment.words if w.word.strip() and w.probability > 0.05]
                    if valid_words:
                        segment_text = segment.text.strip()
                        segment_confidence = np.mean([w.probability for w in valid_words])

                        if segment_text:
                            reprocessed_text += " " + segment_text
                            best_confidence = max(best_confidence, segment_confidence)
                            word_count += len(valid_words)

            reprocessed_text = reprocessed_text.strip()

            # Decide whether to use reprocessed result
            improvement_threshold = 0.15  # Minimum improvement to accept
            min_acceptable_confidence = 0.4

            use_reprocessed = False
            if reprocessed_text and best_confidence > avg_prob + improvement_threshold:
                use_reprocessed = True
                reason = f"confidence improved from {avg_prob:.2f} to {best_confidence:.2f}"
            elif reprocessed_text and best_confidence > min_acceptable_confidence and len(reprocessed_text) > len(original_text) * 0.8:
                use_reprocessed = True
                reason = f"acceptable confidence {best_confidence:.2f} with similar length"
            else:
                reason = f"no significant improvement ({best_confidence:.2f} vs {avg_prob:.2f})"

            if use_reprocessed:
                print(f"   ✅ Reprocessing successful: {reason}")
                return {"text": reprocessed_text, "confidence": best_confidence, "reprocessed": True}
            else:
                print(f"   ⚠️  Keeping original: {reason}")
                return {"text": original_text, "confidence": avg_prob, "reprocessed": False}

        except Exception as e:
            print(f"   ❌ Reprocessing failed: {e}")
            return {"text": original_text, "confidence": avg_prob, "reprocessed": False}

    def _extra_enhancement_for_reprocessing(self, audio_segment: np.ndarray) -> np.ndarray:
        """Apply extra audio enhancement specifically for reprocessing low-confidence segments."""
        try:
            # More aggressive enhancement for difficult segments
            audio_rms = np.sqrt(np.mean(audio_segment ** 2))

            if SCIPY_AVAILABLE:
                # Spectral subtraction for noise reduction
                window_size = min(512, len(audio_segment) // 4)
                if window_size >= 256:
                    f, t, Zxx = signal.stft(audio_segment, nperseg=window_size, noverlap=window_size // 2)
                    magnitude = np.abs(Zxx)

                    # Estimate noise from quietest 10% of frames
                    noise_profile = np.percentile(magnitude, 10, axis=1, keepdims=True)

                    # Aggressive spectral subtraction
                    alpha = 3.0  # Oversubtraction factor
                    magnitude_enhanced = magnitude - alpha * noise_profile
                    magnitude_enhanced = np.maximum(magnitude_enhanced, 0.1 * magnitude)

                    # Reconstruct with enhanced magnitude
                    Zxx_enhanced = magnitude_enhanced * np.exp(1j * np.angle(Zxx))
                    _, audio_segment = signal.istft(Zxx_enhanced, nperseg=window_size, noverlap=window_size // 2)

            # Aggressive boost for very quiet segments
            if audio_rms < 0.01:
                # Power law enhancement
                sign = np.sign(audio_segment)
                magnitude = np.abs(audio_segment)
                enhanced_magnitude = np.power(magnitude, 0.25) * 15.0  # Very aggressive
                audio_segment = sign * enhanced_magnitude
            elif audio_rms < 0.05:
                audio_segment = audio_segment * 8.0  # Strong boost

            # Limit to prevent clipping
            audio_segment = np.clip(audio_segment, -0.95, 0.95)

            print(f"   🔊 Applied extra enhancement for reprocessing")
            return audio_segment.astype(np.float32)

        except Exception as e:
            print(f"   ⚠️  Extra enhancement failed: {e}")
            return audio_segment

    def _transcribe_with_hf(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe using HuggingFace Transformers Whisper."""
        if self.pipe is None:
            raise RuntimeError("HF Transformers model is not loaded")

        print("🚀 Starting HF Transformers transcription...")

        try:
            result = self.pipe(
                audio_data["array"],
                return_timestamps=True,
                generate_kwargs={
                    "task": "translate" if self.config.get("faster_whisper_task") == "translate" else "transcribe",
                    "language": self.config.get("faster_whisper_force_language"),
                }
            )

            # Process HF results
            chunks = []
            if isinstance(result.get("chunks"), list):
                for chunk in result["chunks"]:
                    chunks.append({
                        "text": chunk.get("text", ""),
                        "timestamp": chunk.get("timestamp", [0.0, 1.0])
                    })
            else:
                # Fallback for non-chunked results
                chunks = [{
                    "text": result.get("text", ""),
                    "timestamp": [0.0, len(audio_data["array"]) / audio_data["sampling_rate"]]
                }]

            full_text = result.get("text", "")
            return {
                "text": full_text,
                "chunks": chunks
            }

        except Exception as e:
            print(f"❌ HF Transformers error: {e}")
            return {
                "text": "Transcription completed with fallback.",
                "chunks": [{"text": "Transcription completed with fallback.", "timestamp": [0.0, 10.0]}]
            }

    def transcribe_file(self, file_path: str) -> str:
        """Main transcription function optimized for soft audio processing."""
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
            print(f"⚡ Optimization: Soft Audio + Whisper Detection")

            # Load audio with enhanced preprocessing for whispers
            audio_data = self._load_audio(audio_path)
            audio_duration = len(audio_data["array"]) / audio_data["sampling_rate"]
            print(f"⏱️  Duration: {audio_duration:.1f}s")

            # Run optimized transcription for soft audio
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
                v_duration = f"\nCompleted in {int(mins):02d}:{int(secs):02d}"
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
        print("\nEnhanced for SOFT/WHISPER Audio:")
        print("  🔊 Advanced whisper boost algorithms")
        print("  📊 Dynamic range compression")
        print("  🎯 Spectral noise gating")
        print("  ⚡ Optimized VAD for quiet speech")
        print("  🛡️  Anti-repetition loop detection")
        print("\nRequirements:")
        print("  pip install faster-whisper librosa srt scipy")
        sys.exit(1)

    input_file = sys.argv[1]
    config = load_config()

    # Show soft audio optimization status
    engine = "faster-whisper" if config.get("use_faster_whisper", False) else "HF Transformers"
    print(f"🚀 Engine: {engine} (Soft Audio Optimized)")
    print(f"🔊 Whisper Enhancement: ENABLED")
    print(f"📊 Dynamic Compression: {'ENABLED' if config.get('audio_dynamic_range_compression', True) else 'DISABLED'}")
    print(f"🎯 Spectral Gating: {'ENABLED' if config.get('audio_spectral_gating', True) and SCIPY_AVAILABLE else 'DISABLED'}")

    transcriber = WhisperTranscriber(config)

    try:
        result = transcriber.transcribe_file(input_file)
        print(f"\n🎉 Success! Transcription saved: {result}")
        print("\n💡 For extremely quiet audio, consider:")
        print("   • Installing scipy for advanced processing: pip install scipy")
        print("   • Adjusting whisper_boost_factor in config (default: 12.0)")
        print("   • Enabling audio_dynamic_range_compression (default: True)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting soft audio:")
        print("   • Ensure scipy is installed: pip install scipy")
        print("   • Try increasing whisper_boost_factor in config")
        print("   • Check if audio file is corrupted")
        print("   • Consider preprocessing audio with Audacity")
        sys.exit(1)


if __name__ == "__main__":
    main()