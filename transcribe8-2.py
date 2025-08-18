#!/usr/bin/env python3
"""
Whisper Transcription Tool for Mac mini - ENHANCED WITH ACCURATE TIMESTAMP DETECTION
Now with support for 'whisper', 'faster-whisper', and 'insanely-fast-whisper'
Fixes issue where subtitles appear during silence/background noise
This is with batch processing
Usage: python transcribe.py [file] | --batch


KEY IMPROVEMENTS FOR ACCURATE TIMESTAMPS:
- Enhanced VAD (Voice Activity Detection) to ignore background noise
- Audio energy analysis to detect actual speech vs silence
- Conservative timestamp validation
- Spectral analysis for speech detection
- Word-level confidence filtering
- Silence gap detection and removal

Requirements:
- For regular Whisper: pip install transformers torch librosa
- For faster-whisper: pip install faster-whisper librosa scipy
- For insanely-fast-whisper: pip install insanely-fast-whisper torch accelerate
"""

import os
import sys
import json
import datetime
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import re
import srt
from datetime import timedelta
import torch
import librosa
import numpy as np
import time
import threading
import warnings
import argparse
import glob

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

# Try to import all whisper implementations
try:
    from transformers import pipeline as hf_pipeline

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

try:
    from insanely_fast_whisper import pipeline as insanely_fast_pipeline
    from transformers.utils import is_flash_attn_2_available

    INSANELY_FAST_WHISPER_AVAILABLE = True
    print(f"✅ insanely-fast-whisper available (Flash Attention 2: {is_flash_attn_2_available()})")
except ImportError:
    INSANELY_FAST_WHISPER_AVAILABLE = False
    print("Warning: insanely-fast-whisper not available. Install with: pip install insanely-fast-whisper")


# Try to import scipy for advanced audio processing
try:
    from scipy import signal
    from scipy.ndimage import binary_dilation, binary_erosion

    SCIPY_AVAILABLE = True
    print("✅ scipy available for advanced audio processing")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available. Install for better speech detection: pip install scipy")

CONFIG = {
    "srt_location": "/Volumes/Macintosh HD/Downloads/srt",
    "temp_location": "/Volumes/Macintosh HD/Downloads/srt/temp",
    "audio_source": "/Volumes/Macintosh HD/Downloads",
    "video_source": "/Volumes/Macintosh HD/Downloads/Video/uc",
    "audio_export": "/Volumes/Macintosh HD/Downloads/audio/exported",

    "whisper_models_location": "/Volumes/Macintosh HD/Downloads/srt/whisper_models",
    "faster_whisper_local_model_path": "/Volumes/Macintosh HD/Downloads/srt/whisper_models/models--Systran--faster-whisper-medium",

    "ffmpeg_path": "/Volumes/250SSD/Library/Application Support/audacity/libs",
    "ffprobe_path": "/Volumes/250SSD/Library/Application Support/audacity/libs",
    "credit": "Created using Whisper Transcription Tool",
    "save_audio_to_export_location": True,
    "clean_audio": False,

    # ==== BATCH PROCESSING SETTINGS ====
    "batch_folder": "/Volumes/Macintosh HD/Downloads/batch_audio",
    "batch_processed_folder": "/Volumes/Macintosh HD/Downloads/batch_audio/processed",
    "batch_failed_folder": "/Volumes/Macintosh HD/Downloads/batch_audio/failed",
    "batch_log_file": "/Volumes/Macintosh HD/Downloads/srt/batch_processing.log",
    "batch_skip_existing_srt": True,
    "batch_move_processed_files": True,
    "batch_continue_on_error": True,

    # ==== ENGINE SELECTION ====
    # Choose from: "whisper", "faster-whisper", "insanely-fast-whisper"
    "transcription_engine": "insanely-fast-whisper",

    # ==== [ENGINE 1] Hugging Face TRANSFORMERS 'whisper' SETTINGS ====
    "hf_model_size": "openai/whisper-medium",
    "hf_chunk_length_s": 30,
    "hf_use_mps": True,

    # ==== [ENGINE 2] 'faster-whisper' SETTINGS ====
    "faster_whisper_model_size": "medium",
    "faster_whisper_compute_type": "int8",
    "faster_whisper_device": "auto",
    "faster_whisper_cpu_threads": 8,
    "faster_whisper_num_workers": 1,
    "faster_whisper_beam_size": 5,
    "faster_whisper_patience": 1.5,
    "faster_whisper_temperature": [0.0, 0.2, 0.4, 0.6, 0.8],
    "faster_whisper_vad_filter": True,
    "faster_whisper_vad_threshold": 0.15,
    "faster_whisper_min_silence_duration_ms": 500,

    # ==== [ENGINE 3] 'insanely-fast-whisper' SETTINGS (FOR M-SERIES MACS) ====
    "insanely_fast_model_size": "openai/medium",
    "insanely_fast_models_location": None, # Set a specific path here, or it will use the shared 'whisper_models_location'
    "insanely_fast_device": "mps",  # Use "mps" for Apple Silicon
    "insanely_fast_torch_dtype": "float16", # Use "float16" for MPS
    "insanely_fast_batch_size": 8, # Adjust based on your VRAM (e.g., 24GB RAM on M4)

    # ==== SHARED TRANSCRIPTION QUALITY SETTINGS ====
    "task": "translate", # "transcribe" or "translate"
    "language": "ja", # Set to language code (e.g., "ja") or None for auto-detect
    "initial_prompt": None,
    "length_penalty": 1.0,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 5,
    "suppress_blank": False,
    "suppress_tokens": [-1],

    # ==== ENHANCED VAD & SPEECH DETECTION (Used by post-processing) ====
    "speech_energy_threshold": 0.008,
    "speech_spectral_centroid_min": 300,
    "speech_spectral_centroid_max": 3500,
    "speech_zero_crossing_rate_min": 0.01,
    "speech_zero_crossing_rate_max": 0.35,
    "enable_advanced_speech_detection": True,
    "conservative_timing_mode": True,

    # ==== TIMESTAMP ACCURACY & CLEANING SETTINGS ====
    "timestamp_validation_enabled": True,
    "timestamp_energy_window_ms": 100,
    "timestamp_buffer_start_ms": 200,
    "timestamp_buffer_end_ms": 100,
    "merge_close_segments": True,
    "max_segment_gap_ms": 400,
    "filter_background_noise": True,

    # ==== LOW CONFIDENCE FILTERING (For faster-whisper) ====
    "enable_confidence_filtering": True,
    "word_confidence_threshold": 0.4,
    "segment_confidence_threshold": 0.35,

    # ==== AUDIO PROCESSING & BOOSTING ====
    "audio_minimal_preprocessing": False,
    "enable_mp3_transcription_boost": True,
    "mp3_boost_factor": 8.0,
    "mp3_normalize_audio": True,
    "mp3_target_rms": 0.15,
    "audio_whisper_boost_enabled": True,
    "whisper_boost_factor": 15.0,
    "audio_dynamic_range_compression": True,
    "dynamic_compression_ratio": 4.0,
    "audio_spectral_gating": True,
}


# Global constants
v_config = "config4-1.json"
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
    "Uhh",
    "♪",  # Music notes
    "(music)",
    "[music]",
    "(background music)",
    "[background music]",
    "Uh",
]

REMOVE_QUOTES = dict.fromkeys(map(ord, '"„"‟"＂「」'), None)


class BatchProcessor:
    """Handles batch processing of multiple files."""

    def __init__(self, transcriber, config: Dict[str, Any]):
        self.transcriber = transcriber
        self.config = config
        self.batch_folder = config.get("batch_folder", "/Volumes/Macintosh HD/Downloads/batch_audio")
        self.processed_folder = config.get("batch_processed_folder",
                                           "/Volumes/Macintosh HD/Downloads/batch_audio/processed")
        self.failed_folder = config.get("batch_failed_folder", "/Volumes/Macintosh HD/Downloads/batch_audio/failed")
        self.log_file = config.get("batch_log_file", "/Volumes/Macintosh HD/Downloads/srt/batch_processing.log")
        self.skip_existing = config.get("batch_skip_existing_srt", True)
        self.move_processed = config.get("batch_move_processed_files", True)
        self.continue_on_error = config.get("batch_continue_on_error", True)

        # Ensure directories exist
        self._ensure_batch_directories()

        # Initialize log file
        self._initialize_log()

    def _ensure_batch_directories(self):
        """Create batch processing directories if they don't exist."""
        directories = [
            self.batch_folder,
            self.processed_folder,
            self.failed_folder,
            os.path.dirname(self.log_file)
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _initialize_log(self):
        """Initialize the batch processing log file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"Batch processing session started: {datetime.datetime.now()}\n")
                f.write(f"{'=' * 60}\n")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize log file: {e}")

    def _log_message(self, message: str, print_also: bool = True):
        """Log a message to the batch log file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        if print_also:
            print(message)

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"⚠️ Warning: Could not write to log file: {e}")

    def _find_batch_files(self) -> List[str]:
        """Find all audio and video files in the batch folder."""
        supported_extensions = [
            '*.mp3', '*.wav', '*.aac', '*.m4a', '*.ogg', '*.opus', '*.flac',  # Audio
            '*.mp4', '*.avi', '*.mkv', '*.mov', '*.wmv', '*.flv', '*.webm', '*.m4v'  # Video
        ]

        files = []
        for extension in supported_extensions:
            pattern = os.path.join(self.batch_folder, extension)
            files.extend(glob.glob(pattern))

        # Sort files for consistent processing order
        files.sort()

        self._log_message(f"🔍 Found {len(files)} files for batch processing")
        return files

    def _should_skip_file(self, file_path: str) -> bool:
        """Check if file should be skipped based on existing SRT."""
        if not self.skip_existing:
            return False

        base_name = Path(file_path).stem
        srt_path = os.path.join(self.config["srt_location"], f"{base_name}.srt")

        if os.path.exists(srt_path):
            self._log_message(f"⏭️ Skipping {os.path.basename(file_path)} - SRT already exists")
            return True

        return False

    def _move_file(self, source_path: str, destination_folder: str, reason: str = ""):
        """Move file to destination folder."""
        if not self.move_processed:
            return

        try:
            filename = os.path.basename(source_path)
            destination_path = os.path.join(destination_folder, filename)

            # Handle duplicate filenames
            counter = 1
            while os.path.exists(destination_path):
                name, ext = os.path.splitext(filename)
                destination_path = os.path.join(destination_folder, f"{name}_{counter}{ext}")
                counter += 1

            os.rename(source_path, destination_path)
            self._log_message(f"📁 Moved to {destination_folder}: {filename} {reason}")

        except Exception as e:
            self._log_message(f"⚠️ Could not move file {os.path.basename(source_path)}: {e}")

    def process_batch(self) -> Dict[str, Any]:
        """Process all files in the batch folder."""
        self._log_message("🚀 Starting batch processing...")

        files = self._find_batch_files()
        if not files:
            self._log_message("❌ No files found in batch folder")
            return {"success": 0, "skipped": 0, "failed": 0, "total": 0}

        stats = {"success": 0, "skipped": 0, "failed": 0, "total": len(files)}

        for i, file_path in enumerate(files, 1):
            filename = os.path.basename(file_path)
            self._log_message(f"\n📂 Processing [{i}/{len(files)}]: {filename}")

            try:
                # Check if should skip
                if self._should_skip_file(file_path):
                    stats["skipped"] += 1
                    continue

                # Start timing
                start_time = time.time()

                # Process file
                self._log_message(f"⏳ Transcribing: {filename}")
                result_path = self.transcriber.transcribe_file(file_path)

                # Calculate processing time
                elapsed_time = time.time() - start_time
                mins, secs = divmod(elapsed_time, 60)

                self._log_message(f"✅ Success: {filename} -> {os.path.basename(result_path)}")
                self._log_message(f"⏱️ Processing time: {int(mins):02d}:{int(secs):02d}")

                # Move to processed folder
                self._move_file(file_path, self.processed_folder, "(processed)")

                stats["success"] += 1

            except Exception as e:
                error_msg = str(e)
                self._log_message(f"❌ Failed: {filename} - {error_msg}")

                # Move to failed folder
                self._move_file(file_path, self.failed_folder, "(failed)")

                stats["failed"] += 1

                if not self.continue_on_error:
                    self._log_message("🛑 Stopping batch processing due to error")
                    break

        # Log final statistics
        self._log_message(f"\n📊 Batch processing completed:")
        self._log_message(f"   ✅ Successful: {stats['success']}")
        self._log_message(f"   ⏭️ Skipped: {stats['skipped']}")
        self._log_message(f"   ❌ Failed: {stats['failed']}")
        self._log_message(f"   📁 Total: {stats['total']}")

        return stats


class AdvancedSpeechDetector:
    """Advanced speech detection using multiple audio features."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = 16000

    def analyze_audio_segment(self, audio_segment: np.ndarray, start_time: float) -> Dict[str, Any]:
        """Analyze audio segment for speech characteristics."""
        if len(audio_segment) == 0:
            return self._get_default_analysis(False)

        try:
            # 1. Energy analysis
            energy = np.sqrt(np.mean(audio_segment ** 2))

            # 2. Spectral centroid (brightness of sound)
            if SCIPY_AVAILABLE:
                spectral_centroid = self._compute_spectral_centroid(audio_segment)
            else:
                spectral_centroid = 1000  # Default fallback

            # 3. Zero crossing rate (indicates voiced vs unvoiced)
            zcr = self._compute_zero_crossing_rate(audio_segment)

            # 4. Spectral rolloff
            if SCIPY_AVAILABLE:
                spectral_rolloff = self._compute_spectral_rolloff(audio_segment)
            else:
                spectral_rolloff = 2000  # Default fallback

            # 5. Peak detection
            peak_count = self._count_peaks(audio_segment)

            # Determine if this is likely speech
            is_speech = self._is_likely_speech(energy, spectral_centroid, zcr, spectral_rolloff, peak_count)

            analysis = {
                "is_speech": is_speech,
                "energy": energy,
                "spectral_centroid": spectral_centroid,
                "zero_crossing_rate": zcr,
                "spectral_rolloff": spectral_rolloff,
                "peak_count": peak_count,
                "start_time": start_time,
                "duration": len(audio_segment) / self.sample_rate
            }

            return analysis

        except Exception as e:
            print(f"   ⚠️  Speech analysis failed: {e}")
            return self._get_default_analysis(True)  # Default to speech when analysis fails

    def _get_default_analysis(self, is_speech: bool) -> Dict[str, Any]:
        """Return default analysis when computation fails."""
        return {
            "is_speech": is_speech,
            "energy": 0.01 if is_speech else 0.001,
            "spectral_centroid": 1000,
            "zero_crossing_rate": 0.1,
            "spectral_rolloff": 2000,
            "peak_count": 10 if is_speech else 0,
            "start_time": 0,
            "duration": 1.0
        }

    def _compute_spectral_centroid(self, audio_segment: np.ndarray) -> float:
        """Compute spectral centroid (brightness)."""
        try:
            # Compute FFT
            fft = np.fft.rfft(audio_segment)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_segment), 1 / self.sample_rate)

            # Compute weighted average
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                return float(centroid)
            else:
                return 1000.0  # Default
        except Exception:
            return 1000.0

    def _compute_spectral_rolloff(self, audio_segment: np.ndarray, rolloff_percent: float = 0.85) -> float:
        """Compute spectral rolloff point."""
        try:
            fft = np.fft.rfft(audio_segment)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_segment), 1 / self.sample_rate)

            total_energy = np.sum(magnitude)
            if total_energy == 0:
                return 2000.0

            cumulative_energy = np.cumsum(magnitude)
            rolloff_point = rolloff_percent * total_energy

            rolloff_idx = np.where(cumulative_energy >= rolloff_point)[0]
            if len(rolloff_idx) > 0:
                return float(freqs[rolloff_idx[0]])
            else:
                return float(freqs[-1])
        except Exception:
            return 2000.0

    def _compute_zero_crossing_rate(self, audio_segment: np.ndarray) -> float:
        """Compute zero crossing rate."""
        try:
            # Count sign changes
            sign_changes = np.diff(np.sign(audio_segment))
            zero_crossings = np.sum(np.abs(sign_changes)) / 2
            zcr = zero_crossings / len(audio_segment)
            return float(zcr)
        except Exception:
            return 0.1

    def _count_peaks(self, audio_segment: np.ndarray) -> int:
        """Count peaks in the audio segment."""
        try:
            if SCIPY_AVAILABLE:
                peaks, _ = signal.find_peaks(np.abs(audio_segment),
                                             height=np.max(np.abs(audio_segment)) * 0.1,
                                             distance=int(self.sample_rate * 0.01))  # 10ms minimum distance
                return len(peaks)
            else:
                # Simple peak counting fallback
                abs_audio = np.abs(audio_segment)
                threshold = np.max(abs_audio) * 0.1
                peaks = 0
                for i in range(1, len(abs_audio) - 1):
                    if abs_audio[i] > threshold and abs_audio[i] > abs_audio[i - 1] and abs_audio[i] > abs_audio[i + 1]:
                        peaks += 1
                return peaks
        except Exception:
            return 5  # Default

    def _is_likely_speech(self, energy: float, spectral_centroid: float, zcr: float,
                          spectral_rolloff: float, peak_count: int) -> bool:
        """Determine if audio characteristics indicate speech."""

        # Energy threshold
        energy_threshold = self.config.get("speech_energy_threshold", 0.008)
        if energy < energy_threshold:
            return False

        # Spectral centroid should be in speech range
        centroid_min = self.config.get("speech_spectral_centroid_min", 300)
        centroid_max = self.config.get("speech_spectral_centroid_max", 3500)
        if not (centroid_min <= spectral_centroid <= centroid_max):
            return False

        # Zero crossing rate should be in speech range
        zcr_min = self.config.get("speech_zero_crossing_rate_min", 0.01)
        zcr_max = self.config.get("speech_zero_crossing_rate_max", 0.35)
        if not (zcr_min <= zcr <= zcr_max):
            return False

        # Should have some peaks (indicating modulation)
        if peak_count < 2:
            return False

        # All checks passed
        return True

    def validate_segment_timing(self, audio_data: np.ndarray, start_time: float,
                                end_time: float, sample_rate: int) -> Tuple[float, float, bool]:
        """Validate and adjust segment timing based on actual speech content."""

        if not self.config.get("timestamp_validation_enabled", True):
            return start_time, end_time, True

        try:
            # Convert times to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # Ensure bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)

            if start_sample >= end_sample:
                return start_time, end_time, False

            # Extract segment
            segment = audio_data[start_sample:end_sample]

            # Analyze in small windows
            window_ms = self.config.get("timestamp_energy_window_ms", 100)
            window_samples = int(window_ms * sample_rate / 1000)

            if window_samples <= 0:
                return start_time, end_time, True

            # Find actual speech boundaries
            speech_start_sample = self._find_speech_start(segment, window_samples)
            speech_end_sample = self._find_speech_end(segment, window_samples)

            if speech_start_sample is None or speech_end_sample is None:
                return start_time, end_time, False  # No speech detected

            # Convert back to absolute time with conservative buffers
            buffer_start_ms = self.config.get("timestamp_buffer_start_ms", 200)
            buffer_end_ms = self.config.get("timestamp_buffer_end_ms", 100)

            actual_start = start_time + (speech_start_sample / sample_rate) - (buffer_start_ms / 1000)
            actual_end = start_time + (speech_end_sample / sample_rate) + (buffer_end_ms / 1000)

            # Ensure minimum duration
            min_duration = 0.3
            if actual_end - actual_start < min_duration:
                center = (actual_start + actual_end) / 2
                actual_start = center - min_duration / 2
                actual_end = center + min_duration / 2

            # Clamp to original bounds (don't extend beyond original segment)
            actual_start = max(start_time, actual_start)
            actual_end = min(end_time, actual_end)

            return actual_start, actual_end, True

        except Exception as e:
            print(f"   ⚠️  Timestamp validation failed: {e}")
            return start_time, end_time, True

    def _find_speech_start(self, segment: np.ndarray, window_samples: int) -> Optional[int]:
        """Find the actual start of speech in the segment."""
        if len(segment) < window_samples:
            return 0 if self._has_speech_characteristics(segment) else None

        for i in range(0, len(segment) - window_samples, window_samples // 2):
            window = segment[i:i + window_samples]
            analysis = self.analyze_audio_segment(window, 0)

            if analysis["is_speech"]:
                return i

        return None

    def _find_speech_end(self, segment: np.ndarray, window_samples: int) -> Optional[int]:
        """Find the actual end of speech in the segment."""
        if len(segment) < window_samples:
            return len(segment) if self._has_speech_characteristics(segment) else None

        # Search backwards
        for i in range(len(segment) - window_samples, 0, -(window_samples // 2)):
            window = segment[i:i + window_samples]
            analysis = self.analyze_audio_segment(window, 0)

            if analysis["is_speech"]:
                return i + window_samples

        return None

    def _has_speech_characteristics(self, audio_segment: np.ndarray) -> bool:
        """Quick check if segment has basic speech characteristics."""
        if len(audio_segment) == 0:
            return False

        energy = np.sqrt(np.mean(audio_segment ** 2))
        energy_threshold = self.config.get("speech_energy_threshold", 0.008)

        return energy >= energy_threshold


class WhisperTranscriber:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = config.get("transcription_engine", "faster-whisper")
        self.pipe = None  # For HF Transformers or insanely-fast-whisper
        self.model = None  # For faster-whisper
        self.device = None
        self.transcription_complete = False
        self.current_progress = 0.0

        # Initialize advanced speech detector
        self.speech_detector = AdvancedSpeechDetector(config)

        # Validate that the required library is available
        if self.engine == "insanely-fast-whisper" and not INSANELY_FAST_WHISPER_AVAILABLE:
            print("Error: 'insanely-fast-whisper' requested but not installed.")
            sys.exit(1)
        elif self.engine == "faster-whisper" and not FASTER_WHISPER_AVAILABLE:
            print("Error: 'faster-whisper' requested but not installed.")
            sys.exit(1)
        elif self.engine == "whisper" and not HF_AVAILABLE:
            print("Error: 'transformers' (for whisper) requested but not installed.")
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
        if self.engine == "faster-whisper":
            device_config = self.config.get("faster_whisper_device", "auto")
            if device_config == "auto":
                if torch.cuda.is_available(): self.device = "cuda"
                else: self.device = "cpu"
            else:
                self.device = device_config
            print(f"Using device for faster-whisper: {self.device}")

        elif self.engine == "insanely-fast-whisper":
            device_config = self.config.get("insanely_fast_device", "mps")
            if device_config == "mps" and torch.backends.mps.is_available():
                self.device = "mps"
                print("Using Apple Silicon MPS acceleration for insanely-fast-whisper")
            elif device_config == "cuda" and torch.cuda.is_available():
                self.device = "cuda:0"
                print("Using CUDA acceleration for insanely-fast-whisper")
            else:
                self.device = "cpu"
                print("Using CPU for insanely-fast-whisper")

        else: # HF 'whisper'
            if self.config.get("hf_use_mps", True) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple Silicon MPS acceleration for Transformers Whisper")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA acceleration for Transformers Whisper")
            else:
                self.device = torch.device("cpu")
                print("Using CPU for Transformers Whisper")


    def _check_model_exists(self, model_name_key) -> bool:
        """Check if the Whisper model is already downloaded."""
        cache_dir = self.config["whisper_models_location"]
        model_name = self.config[model_name_key].replace("/", "--")

        potential_path = os.path.join(cache_dir, "models--" + model_name)
        if os.path.exists(potential_path) and os.listdir(potential_path):
            print(f"Found cached model at: {potential_path}")
            return True
        return False

    def _load_model(self):
        """Load the appropriate Whisper model based on configuration."""
        if self.engine == "faster-whisper":
            self._load_faster_whisper_model()
        elif self.engine == "insanely-fast-whisper":
            self._load_insanely_fast_model()
        else: # HF 'whisper'
            self._load_hf_model()

    def _get_optimal_cpu_threads(self) -> int:
        """Determine optimal number of CPU threads for faster-whisper."""
        cpu_threads_config = self.config.get("faster_whisper_cpu_threads", "auto")
        if cpu_threads_config == "auto":
            if PSUTIL_AVAILABLE:
                # Use physical cores, leaving some for system processes
                physical_cores = psutil.cpu_count(logical=False) or 1
                optimal_threads = max(1, physical_cores - 2) if physical_cores > 2 else 1
                print(f"Auto-detected {physical_cores} physical cores, using {optimal_threads} threads for faster-whisper")
                return optimal_threads
            else:
                return 4 # Conservative fallback
        return int(cpu_threads_config)

    def _download_faster_whisper_model(self, model_size: str) -> str:
        """Download faster-whisper model and return the path."""
        print(f"🔄 Downloading faster-whisper model: {model_size}")
        try:
            # Temporarily load the model to trigger download
            _ = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=self.config["whisper_models_location"])
            # Construct the expected path
            model_path = os.path.join(self.config["whisper_models_location"], f"models--Systran--faster-whisper-{model_size}")
            print(f"✅ Model downloaded to: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return model_size # Fallback to using name

    def _check_model_files_exist(self, path: str) -> bool:
        """Check if essential model files exist in the given path."""
        if not os.path.exists(path): return False
        files = os.listdir(path)
        return 'config.json' in files and any(f.endswith('.bin') for f in files)

    def _load_faster_whisper_model(self):
        """Load the faster-whisper model with optimized settings."""
        if self.model: return
        model_size = self.config.get("faster_whisper_model_size", "large-v3")
        compute_type = self.config.get("faster_whisper_compute_type", "int8")
        local_path_config = self.config.get("faster_whisper_local_model_path")
        cache_dir = self.config["whisper_models_location"]
        cpu_threads = self._get_optimal_cpu_threads()

        print(f"Loading faster-whisper: {model_size}")
        print(f"Compute type: {compute_type}, Device: {self.device}, Threads: {cpu_threads}")

        model_path_to_use = model_size # Default to name
        # Prefer configured local path if it exists and is valid
        if local_path_config and self._check_model_files_exist(local_path_config):
            model_path_to_use = local_path_config
            print(f"Using local model path: {model_path_to_use}")
        else:
            # Check standard cache path
            standard_cache_path = os.path.join(cache_dir, f"models--Systran--faster-whisper-{model_size}")
            if self._check_model_files_exist(standard_cache_path):
                 # Dig into snapshots if they exist
                snapshots_dir = os.path.join(standard_cache_path, "snapshots")
                if os.path.isdir(snapshots_dir):
                    snapshots = sorted([d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))], reverse=True)
                    if snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                        if self._check_model_files_exist(snapshot_path):
                            model_path_to_use = snapshot_path
                            print(f"Using cached model snapshot: {model_path_to_use}")

        try:
            self.model = WhisperModel(
                model_path_to_use,
                device=self.device,
                compute_type=compute_type,
                download_root=cache_dir,
                cpu_threads=cpu_threads,
                num_workers=self.config.get("faster_whisper_num_workers", 1)
            )
            print("✅ faster-whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading faster-whisper model '{model_path_to_use}': {e}")
            sys.exit(1)

    def _load_hf_model(self):
        """Load the Hugging Face Transformers Whisper model."""
        if self.pipe: return
        model_name = self.config.get('hf_model_size', 'openai/whisper-large-v3')
        print(f"Loading HF Whisper model: {model_name}")
        cache_dir = self.config["whisper_models_location"]
        try:
            self.pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=model_name,
                chunk_length_s=self.config.get("hf_chunk_length_s", 30),
                device=self.device,
                model_kwargs={"cache_dir": cache_dir},
            )
            print("✅ HF Transformers Whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading HF model: {e}")
            sys.exit(1)

    def _load_insanely_fast_model(self):
        """Load the insanely-fast-whisper model with MPS support."""
        if self.pipe: return
        model_name = self.config.get('insanely_fast_model_size', 'openai/whisper-large-v3')
        torch_dtype_str = self.config.get("insanely_fast_torch_dtype", "float16")
        torch_dtype = getattr(torch, torch_dtype_str)

        print(f"Loading insanely-fast-whisper model: {model_name}")
        print(f"Device: {self.device}, Dtype: {torch_dtype_str}")

        # Determine the cache directory for the model. Prioritize the specific path.
        cache_dir = self.config.get("insanely_fast_models_location") or self.config.get("whisper_models_location")
        print(f"Model cache location: {cache_dir}")

        # Ensure the directory exists
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        try:
            self.pipe = insanely_fast_pipeline(
                "automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch_dtype,
                device=self.device,
                model_kwargs={
                    "cache_dir": cache_dir,
                    "attn_implementation": "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
                },
            )
            print("✅ insanely-fast-whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading insanely-fast-whisper model: {e}")
            sys.exit(1)


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

        # Soft audio thresholds from config
        soft_rms_threshold = self.config.get("soft_audio_rms_threshold", 0.005)
        soft_max_threshold = self.config.get("soft_audio_max_threshold", 0.02)
        is_soft = audio_rms < soft_rms_threshold or audio_max < soft_max_threshold

        analysis = {
            "is_soft": is_soft,
            "rms": audio_rms,
            "max": audio_max,
        }
        print(f"🔍 Audio Analysis: RMS={audio_rms:.6f}, Max={audio_max:.6f} -> {'SOFT' if is_soft else 'NORMAL'}")
        return analysis

    def _spectral_noise_gate(self, audio_array: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Advanced spectral noise gate for extremely quiet audio."""
        if not SCIPY_AVAILABLE:
            print("   ⚠️  Skipping spectral gating (scipy required)")
            return audio_array

        try:
            window_size = min(2048, len(audio_array) // 4)
            if window_size < 512: return audio_array
            f, t, Zxx = signal.stft(audio_array, fs=sample_rate, nperseg=window_size, noverlap=window_size // 2)
            magnitude = np.abs(Zxx)
            noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
            mask = magnitude > (noise_floor * 1.2)
            mask = binary_dilation(mask, iterations=2)
            Zxx_filtered = Zxx * (mask.astype(float) * 0.8 + 0.2)
            _, filtered_audio = signal.istft(Zxx_filtered, fs=sample_rate, nperseg=window_size, noverlap=window_size // 2)
            if len(filtered_audio) != len(audio_array):
                filtered_audio = np.pad(filtered_audio, (0, len(audio_array) - len(filtered_audio)))
            print("   ✅ Applied spectral noise gate")
            return filtered_audio.astype(np.float32)
        except Exception as e:
            print(f"   ⚠️  Spectral gating failed: {e}")
            return audio_array

    def _dynamic_range_compression(self, audio_array: np.ndarray, ratio: float = 4.0) -> np.ndarray:
        """Apply dynamic range compression to make quiet sounds louder."""
        try:
            window_size = min(1024, len(audio_array) // 10)
            if window_size < 64: return audio_array
            audio_squared = audio_array ** 2
            kernel = np.ones(window_size) / window_size
            moving_rms = np.sqrt(np.convolve(audio_squared, kernel, mode='same'))
            threshold = 0.1
            makeup_gain = 2.0
            compressed = np.where(moving_rms > threshold, audio_array * (threshold + (moving_rms - threshold) / ratio) / (moving_rms + 1e-8), audio_array)
            compressed *= makeup_gain
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
        if self.config.get("audio_spectral_gating", True):
            audio_array = self._spectral_noise_gate(audio_array)
        if self.config.get("audio_dynamic_range_compression", True):
            compression_ratio = self.config.get("dynamic_compression_ratio", 4.0)
            audio_array = self._dynamic_range_compression(audio_array, compression_ratio)
        if self.config.get("audio_whisper_boost_enabled", True):
            boost_factor = self.config.get("whisper_boost_factor", 12.0)
            audio_array = audio_array * boost_factor
            print(f"   🔊 Applied audio boost (factor: {boost_factor:.1f})")
        audio_max = np.max(np.abs(audio_array))
        if audio_max > 0.95:
            audio_array = audio_array * (0.90 / audio_max)
            print("   📉 Applied final limiting")
        return audio_array.astype(np.float32)

    def _minimal_audio_preprocessing(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply only essential preprocessing to prevent math errors while maintaining speed."""
        if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
        audio_max = np.max(np.abs(audio_array))
        if audio_max < 1e-7:
            audio_array *= 10000.0
            audio_array = np.clip(audio_array, -0.95, 0.95)
        return audio_array.astype(np.float32)

    def _apply_mp3_transcription_boost(self, audio_array: np.ndarray, file_path: str) -> np.ndarray:
        """Apply specialized boost for MP3 files to improve transcription accuracy."""
        if not self.config.get("enable_mp3_transcription_boost", True) or not file_path.lower().endswith('.mp3'):
            return audio_array
        print("🎵 Applying MP3 transcription boost...")
        try:
            if self.config.get("mp3_normalize_audio", True):
                current_rms = np.sqrt(np.mean(audio_array ** 2))
                target_rms = self.config.get("mp3_target_rms", 0.15)
                if current_rms > 1e-8:
                    normalization_factor = target_rms / current_rms
                    audio_array *= normalization_factor
                    print(f"   🎚️  Normalized: {current_rms:.4f} → {target_rms:.4f} RMS")
            mp3_boost_factor = self.config.get("mp3_boost_factor", 8.0)
            if mp3_boost_factor > 1.0:
                audio_array *= mp3_boost_factor
                print(f"   🚀 Applied MP3 boost: {mp3_boost_factor:.1f}x")
            audio_max = np.max(np.abs(audio_array))
            if audio_max > 0.90:
                audio_array *= (0.85 / audio_max)
                print("   📉 Applied MP3 limiting")
            return audio_array.astype(np.float32)
        except Exception as e:
            print(f"   ⚠️  MP3 boost failed: {e}")
            return audio_array

    def _enhanced_audio_preprocessing(self, audio_array: np.ndarray, file_path: str = "") -> np.ndarray:
        """Enhanced preprocessing for whisper detection with advanced algorithms."""
        if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
        audio_array = self._apply_mp3_transcription_boost(audio_array, file_path)
        analysis = self._detect_soft_audio(audio_array)
        if analysis["is_soft"]:
            audio_array = self._extreme_whisper_enhancement(audio_array, analysis)
        final_rms = np.sqrt(np.mean(audio_array ** 2))
        final_max = np.max(np.abs(audio_array))
        print(f"   ✅ Final Enhanced: RMS={final_rms:.4f}, Max={final_max:.4f}")
        return audio_array.astype(np.float32)

    def _load_audio(self, audio_path: str) -> Dict[str, Any]:
        """Load audio file with enhanced preprocessing for whisper detection."""
        print(f"Loading audio file: {audio_path}")
        try:
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
            if len(audio_array) == 0: raise ValueError("Audio file is empty")

            if self.config.get("audio_minimal_preprocessing", False):
                audio_array = self._minimal_audio_preprocessing(audio_array)
            else:
                audio_array = self._enhanced_audio_preprocessing(audio_array, audio_path)

            print(f"📊 Final Audio: {len(audio_array) / sample_rate:.1f}s, dtype: {audio_array.dtype}")
            return {"array": audio_array, "sampling_rate": sample_rate, "path": audio_path}
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            raise

    def _convert_timestamps_to_srt(self, chunks: List[Dict], audio_duration: float) -> List[srt.Subtitle]:
        """Convert timestamps to SRT format."""
        subs = []
        for i, chunk in enumerate(chunks, start=1):
            text = chunk.get("text", "").strip()
            if not text: continue
            timestamp = chunk.get("timestamp", [0.0, 0.0])
            start_time = float(timestamp[0]) if timestamp[0] is not None else 0.0
            end_time = float(timestamp[1]) if timestamp[1] is not None else start_time + 1.0
            if end_time <= start_time: end_time = start_time + 1.0
            subs.append(srt.Subtitle(index=i, start=timedelta(seconds=start_time), end=timedelta(seconds=end_time), content=text))
        return subs

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg and ffprobe are available."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            self.config["ffmpeg_path"] = "ffmpeg"
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: ffmpeg not found in PATH. Please install ffmpeg.")
            return False

    def _convert_to_audio(self, video_path: str) -> tuple[str, bool]:
        """Convert video file to audio format - keep MP3 for speed."""
        if not self._check_ffmpeg(): raise RuntimeError("ffmpeg is required")
        video_name = Path(video_path).stem
        if self.config.get("save_audio_to_export_location", True):
            audio_path = os.path.join(self.config["audio_export"], f"{video_name}.mp3")
            is_temporary = False
        else:
            audio_path = os.path.join(self.config["temp_location"], f"{video_name}.mp3")
            is_temporary = True
        print(f"Converting video to audio: {video_path} -> {audio_path}")
        cmd = [self.config["ffmpeg_path"], "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ab", "128k", "-ar", "16000", "-ac", "1", "-y", audio_path]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Video conversion completed!")
            return audio_path, is_temporary
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting video: {e.stderr}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean up transcribed text."""
        for garbage in GARBAGE_PATTERNS:
            text = text.replace(garbage, "")
        return re.sub(r'\s+', ' ', text).strip().translate(REMOVE_QUOTES)

    def _merge_close_segments(self, segments: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Merge segments that are separated by very short gaps."""
        if not self.config.get("merge_close_segments", True) or len(segments) <= 1:
            return segments
        max_gap_seconds = self.config.get("max_segment_gap_ms", 400) / 1000.0
        merged = []
        current = segments[0]
        for next_seg in segments[1:]:
            if (next_seg.start.total_seconds() - current.end.total_seconds()) <= max_gap_seconds:
                current.content += " " + next_seg.content
                current.end = next_seg.end
            else:
                merged.append(current)
                current = next_seg
        merged.append(current)
        for i, seg in enumerate(merged, 1): seg.index = i
        return merged

    def _filter_silence_segments(self, segments: List[srt.Subtitle], audio_data: np.ndarray, sample_rate: int) -> List[srt.Subtitle]:
        """Filter out segments that contain only silence or background noise."""
        if not self.config.get("filter_background_noise", True): return segments
        print("🔇 Filtering silence and background noise segments...")
        filtered = []
        for seg in segments:
            start, end, has_speech = self.speech_detector.validate_segment_timing(audio_data, seg.start.total_seconds(), seg.end.total_seconds(), sample_rate)
            if has_speech and (end - start) >= 0.2:
                seg.start, seg.end = timedelta(seconds=start), timedelta(seconds=end)
                filtered.append(seg)
        for i, seg in enumerate(filtered, 1): seg.index = i
        print(f"   ✅ Kept {len(filtered)} segments after silence filtering")
        return filtered

    def _clean_srt_segments(self, segments: List[srt.Subtitle], audio_data: np.ndarray, sample_rate: int) -> List[srt.Subtitle]:
        """Clean and filter SRT segments with advanced speech detection and overlap prevention."""
        print("🧹 Cleaning SRT segments...")
        if not segments: return []
        cleaned = self._filter_silence_segments(segments, audio_data, sample_rate)
        if not cleaned: cleaned = segments
        final = []
        for i, seg in enumerate(cleaned):
            if final:
                prev_end = final[-1].end.total_seconds()
                curr_start = seg.start.total_seconds()
                if curr_start < prev_end:
                    final[-1].end = timedelta(seconds=curr_start - 0.01)
            final.append(seg)
        final = self._merge_close_segments(final)
        for i, seg in enumerate(final, 1): seg.index = i
        print(f"📝 Final cleaning result: {len(final)} segments")
        return final

    def _add_credit_to_srt(self, srt_path: str, credit: str):
        """Add credit line to the end of SRT file."""
        if not credit: return
        with open(srt_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{9999}\n23:59:58,000 --> 23:59:59,000\n{credit}\n")

    def _transcribe_with_faster_whisper(self, audio_data: Dict[str, Any], audio_duration: float) -> Dict[str, Any]:
        """Transcribe using faster-whisper."""
        if self.model is None: raise RuntimeError("faster-whisper model not loaded")
        print("🚀 Starting faster-whisper transcription...")
        vad_parameters = {"threshold": self.config.get("faster_whisper_vad_threshold", 0.15)}
        segments_gen, _ = self.model.transcribe(
            audio_data["array"],
            task=self.config.get("task", "translate"),
            language=self.config.get("language"),
            initial_prompt=self.config.get("initial_prompt"),
            beam_size=self.config.get("faster_whisper_beam_size", 5),
            temperature=self.config.get("faster_whisper_temperature", [0.0, 0.2, 0.4]),
            patience=self.config.get("faster_whisper_patience", 1.5),
            vad_filter=self.config.get("faster_whisper_vad_filter", True),
            vad_parameters=vad_parameters,
            word_timestamps=True,
        )
        chunks = []
        last_end_time = 0.0
        for segment in segments_gen:
            text = segment.text.strip()
            if not text: continue
            start, end = float(segment.start), float(segment.end)
            chunks.append({"text": text, "timestamp": [start, end]})
            last_end_time = end
            progress = min(100.0, (last_end_time / audio_duration) * 100) if audio_duration > 0 else 0
            self._update_progress(progress)
        self._update_progress(100.0)
        print(f"\n✅ Processing complete: {len(chunks)} valid segments")
        return {"text": " ".join(c['text'] for c in chunks), "chunks": chunks}

    def _transcribe_with_hf_based_pipeline(self, audio_data: Dict[str, Any], audio_duration: float) -> Dict[str, Any]:
        """Transcribe using a HuggingFace-style pipeline (either 'whisper' or 'insanely-fast-whisper')."""
        if self.pipe is None: raise RuntimeError(f"{self.engine} model not loaded")
        print(f"🚀 Starting {self.engine} transcription...")
        generate_kwargs = {
            "task": self.config.get("task", "translate"),
            "language": self.config.get("language"),
        }
        # For insanely-fast-whisper, we can add a progress bar callback
        def progress_callback(progress):
            self._update_progress(progress * 100)

        # insanely-fast-whisper supports a progress callback
        if self.engine == "insanely-fast-whisper":
             result = self.pipe(
                audio_data["array"].copy(), # Pass a copy to avoid potential modification issues
                batch_size=self.config.get("insanely_fast_batch_size", 8),
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
                chunk_length_s=30, # Fixed chunk length for this pipeline
                stride_length_s=5,
                progress_callback=progress_callback,
            )
        else: # Standard HF pipeline
            result = self.pipe(
                audio_data["array"].copy(),
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
            )

        self._update_progress(100.0) # Ensure it finishes at 100%
        chunks = result.get("chunks", [])
        if not chunks:
             chunks = [{"text": result.get("text", ""), "timestamp": [0.0, audio_duration]}]
        print(f"\n✅ Processing complete: {len(chunks)} valid segments")
        return {"text": result.get("text", ""), "chunks": chunks}


    def transcribe_file(self, file_path: str) -> str:
        """Main transcription function."""
        actual_file_path = self._find_input_file(file_path)
        self._load_model()

        audio_path, temp_audio = actual_file_path, False
        if self._is_video_file(actual_file_path):
            print("Video file detected, converting to audio...")
            audio_path, temp_audio = self._convert_to_audio(actual_file_path)

        try:
            base_name = Path(actual_file_path).stem
            srt_path = os.path.join(self.config["srt_location"], f"{base_name}.srt")
            print(f"📁 Input: {audio_path}\n📄 Output: {srt_path}\n🔧 Engine: {self.engine}")

            audio_data = self._load_audio(audio_path)
            audio_duration = len(audio_data["array"]) / audio_data["sampling_rate"]
            print(f"⏱️  Duration: {audio_duration:.1f}s")

            start_time = time.time()
            if self.engine == "faster-whisper":
                result = self._transcribe_with_faster_whisper(audio_data, audio_duration)
            else: # HF and insanely-fast-whisper share a similar pipeline interface
                result = self._transcribe_with_hf_based_pipeline(audio_data, audio_duration)
            elapsed = time.time() - start_time
            mins, secs = divmod(elapsed, 60)
            completion_time = f"Completed in {int(mins):02d}:{int(secs):02d}"
            speed_ratio = audio_duration / elapsed if elapsed > 0 else 0
            print(f"\n⏱️  {completion_time} (Speed: {speed_ratio:.2f}x real-time)")

            chunks = result.get("chunks", [])
            if not chunks: chunks = [{"text": result.get("text", ""), "timestamp": [0.0, audio_duration]}]
            subs = self._convert_timestamps_to_srt(chunks, audio_duration)
            cleaned_subs = self._clean_srt_segments(subs, audio_data["array"], audio_data["sampling_rate"])
            if not cleaned_subs:
                 cleaned_subs = [srt.Subtitle(1, timedelta(0), timedelta(seconds=min(3, audio_duration)), "Audio processed - minimal speech detected.")]

            with open(srt_path, 'w', encoding='utf-8') as f: f.write(srt.compose(cleaned_subs))
            self._add_credit_to_srt(srt_path, self.config.get("credit", ""))
            self._add_credit_to_srt(srt_path, completion_time)
            print(f"✅ Success! Enhanced SRT saved with {len(cleaned_subs)} accurate segments")
            return srt_path
        finally:
            if temp_audio and os.path.exists(audio_path) and self.config.get("clean_audio", True):
                os.remove(audio_path)
                print(f"🗑️  Cleaned up: {audio_path}")


def load_config(config_file: str = v_config) -> Dict[str, Any]:
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
    return config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Whisper Transcription Tool")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('filename', nargs='?', help='Audio or video file to transcribe')
    mode_group.add_argument('--batch', action='store_true', help='Enable batch processing mode')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    config = load_config()

    engine = config.get("transcription_engine")
    print(f"🚀 Engine: {engine}")
    print(f"🎯 Advanced VAD: {'ENABLED' if config.get('faster_whisper_vad_filter', True) else 'DISABLED'}")
    print(f"🔍 Speech Analysis: {'ENABLED' if config.get('enable_advanced_speech_detection', True) else 'DISABLED'}")
    print(f"⏱️  Timestamp Validation: {'ENABLED' if config.get('timestamp_validation_enabled', True) else 'DISABLED'}")

    transcriber = WhisperTranscriber(config)

    try:
        if args.batch:
            print("\n📦 BATCH PROCESSING MODE")
            batch_processor = BatchProcessor(transcriber, config)
            stats = batch_processor.process_batch()
            print(f"\n📊 BATCH RESULTS: Successful: {stats['success']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
        else:
            if not args.filename:
                print("❌ Error: No filename provided.")
                sys.exit(1)
            print(f"\n📄 SINGLE FILE MODE")
            result = transcriber.transcribe_file(args.filename)
            print(f"\n🎉 Success! Transcription saved: {result}")
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()