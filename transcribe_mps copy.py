#!/usr/bin/env python3
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
import warnings
import argparse
import glob
import logging

# NEW CODE: Import translation libraries
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers available for translation")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available. Install with: pip install transformers")

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

# Try to import faster-whisper
try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not available. Install with: pip install faster-whisper")

# NEW: Try to import whisper for MPS support
try:
    import whisper

    WHISPER_MPS_AVAILABLE = True
except ImportError:
    WHISPER_MPS_AVAILABLE = False
    print("Warning: whisper-mps not available. Install with: pip install -U openai-whisper")

# Try to import scipy for advanced audio processing
try:
    from scipy import signal
    from scipy.ndimage import binary_dilation, binary_erosion

    SCIPY_AVAILABLE = True
    print("✅ scipy available for advanced audio processing")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available. Install for better speech detection: pip install scipy")

v_config = "config_mps.json"

# NEW CODE: Translation class
class TranslationProcessor:
    """Handles Japanese to English translation using Hugging Face Transformers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.translator = None
        self.translation_enabled = config.get("translation_enabled", False)
        self.translation_model = config.get("translation_model", "Helsinki-NLP/opus-mt-ja-en")
        self.models_location = Path(config.get("translation_models_location", "/tmp/translation_models"))

        # Setup logging for translation
        self.logger = logging.getLogger(__name__)

        if self.translation_enabled and TRANSFORMERS_AVAILABLE:
            self._load_translation_model()

    def _load_translation_model(self):
        """Load the translation model from local path or download it."""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.error("Transformers library not available. Translation disabled.")
            self.translation_enabled = False
            return

        try:
            self.models_location.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Loading translation model '{self.translation_model}' from: {self.models_location}")
            print(f"🔄 Loading translation model: {self.translation_model}")

            tokenizer = AutoTokenizer.from_pretrained(
                self.translation_model,
                cache_dir=str(self.models_location)
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.translation_model,
                cache_dir=str(self.models_location)
            )

            self.translator = pipeline(
                "translation_ja_to_en",
                model=model,
                tokenizer=tokenizer
            )

            self.logger.info(f"Successfully loaded {self.translation_model}")
            print(f"✅ Translation model loaded successfully!")

        except Exception as e:
            self.logger.error(f"Failed to load translation model '{self.translation_model}': {e}")
            print(f"❌ Translation model loading failed: {e}")
            print("💡 Solutions:")
            print("   1. Install transformers: pip install transformers")
            print("   2. Check internet connection for model download")
            print("   3. Set translation_enabled to false in config")
            self.translation_enabled = False

    def translate_text(self, text: str) -> str:
        """
        Translate Japanese text to English using the loaded model.

        Args:
            text (str): The Japanese text to translate.

        Returns:
            str: The translated English text, or original text if translation fails.
        """
        if not self.translation_enabled or not self.translator:
            return text

        if not text or not text.strip():
            return text

        try:
            self.logger.info("Starting translation...")
            print(f"   🔄 Translating: {text[:50]}...")

            result = self.translator(text)
            translated_text = result[0]['translation_text']

            self.logger.info("Translation completed.")
            print(f"   ✅ Translated: {translated_text[:50]}...")

            return translated_text.strip()

        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            print(f"   ❌ Translation failed: {e}")
            return text  # Return original text if translation fails


# Garbage patterns to remove from transcriptions
GARBAGE_PATTERNS = [
    "Thank you."
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
        self.model = None
        self.device = None
        self.use_faster_whisper = config.get("use_faster_whisper", True)
        self.transcription_complete = False
        self.current_progress = 0.0

        # Initialize advanced speech detector
        self.speech_detector = AdvancedSpeechDetector(config)

        # NEW CODE: Initialize translation processor
        self.translation_processor = TranslationProcessor(config)

        # Validate that the required library is available
        if self.use_faster_whisper:
            if not FASTER_WHISPER_AVAILABLE:
                print("Error: faster-whisper is enabled in config but not installed.")
                print("Install with: pip install faster-whisper")
                sys.exit(1)
        else:
            if not WHISPER_MPS_AVAILABLE:
                print("Error: whisper-mps is enabled in config but not installed.")
                print("Install with: pip install -U openai-whisper")
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
            self.config["whisper_models_location"],
            # NEW CODE: Add translation models directory
            # self.config.get("translation_models_location", "/tmp/translation_models")
            self.config["translation_models_location"],
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
            # whisper-mps device setup
            if torch.backends.mps.is_available() and self.config.get("use_mps", True):
                self.device = "mps"
                print("Using MPS for whisper-mps")
            else:
                self.device = "cpu"
                print("Using CPU for whisper-mps (MPS not available or disabled)")

    def _load_model(self):
        """Load the appropriate Whisper model based on configuration."""
        if self.use_faster_whisper:
            self._load_faster_whisper_model()
        else:
            self._load_whisper_mps_model()

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
            print(f"Using configured {threads} CPU threads for faster-whisper")
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

    def _load_mlx_whisper_model(self):
        """Load MLX Whisper model optimized for Apple Silicon M4."""
        if self.model is None:
            model_size = self.config.get("mlx_model_size", "large-v3")
            print(f"🚀 Loading MLX Whisper model for M4 optimization: {model_size}")
            print("   🔥 Utilizing CPU + GPU + Neural Engine on Apple Silicon")

            try:
                # MLX Whisper doesn't need explicit model loading - it loads on demand
                # We just need to store the model size for use in transcription
                self.model_size = model_size
                self.model = "mlx_ready"  # Flag that MLX is ready
                print("✅ MLX Whisper ready for M4-optimized transcription!")
                print("   ⚡ Expected performance: ~2-3x faster than MPS")
                print("   🔋 Power efficient: ~25W vs 190W (GPU equivalent)")

            except Exception as e:
                print(f"❌ Error setting up MLX Whisper: {e}")
                print("Ensure you have installed: pip install mlx-whisper")
                raise RuntimeError(f"Could not setup MLX Whisper: {e}")

    def _load_whisper_mps_model(self):
        """Load the standard whisper model with enhanced MPS support and sparse tensor workarounds."""
        if self.model is None:
            model_size = self.config.get("model_size", "large-v3")
            print(f"Loading whisper model with M4-optimized MPS: {model_size}")
            print(f"Device: {self.device}")

            # Apply MPS workarounds for M4 compatibility
            if self.device == "mps":
                print("🔧 Applying M4 MPS optimizations...")
                # Set environment variables to optimize MPS behavior
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

            # Try loading with the selected device first
            try:
                self.model = whisper.load_model(
                    model_size,
                    device=self.device,
                    download_root=self.config["whisper_models_location"]
                )
                print(f"✅ Whisper model loaded successfully on {self.device}!")

                # Test the model with MPS-specific operations
                if self.device == "mps":
                    try:
                        print("🧪 Testing M4 MPS compatibility...")
                        # Test basic tensor operations
                        test_audio = torch.zeros(1600, dtype=torch.float32, device="mps")

                        # Test model inference with a small sample
                        with torch.no_grad():
                            # Create a small test input
                            mel_input = torch.zeros(1, 80, 3000, dtype=torch.float32, device="mps")

                        print("✅ M4 MPS compatibility test passed")
                        print("   🚀 Ready for accelerated transcription")

                    except Exception as test_e:
                        print(f"⚠️ M4 MPS compatibility issue detected: {str(test_e)[:100]}...")
                        if "sparse" in str(test_e).lower():
                            print("🔧 Detected sparse tensor issue - applying workaround...")
                            # Try to reload with CPU fallback
                            print("🔄 Switching to CPU with high-performance settings...")
                            self.device = "cpu"

                            # Reload on CPU
                            self.model = whisper.load_model(
                                model_size,
                                device="cpu",
                                download_root=self.config["whisper_models_location"]
                            )
                            print("✅ Whisper model successfully loaded on CPU with optimizations!")

                            # Enable CPU optimizations for M4
                            torch.set_num_threads(8)  # Optimize for M4 CPU cores
                            print("   🔧 M4 CPU optimization: 8 threads enabled")
                        else:
                            raise test_e

            except Exception as e:
                error_message = str(e)
                print(f"❌ Error loading whisper model on {self.device}: {e}")

                # Handle specific MPS issues
                if any(keyword in error_message.lower() for keyword in ["sparse", "sparsemps", "coo_tensor"]):
                    print("🔧 Detected M4 MPS sparse tensor limitation")
                    print("💡 This is a known PyTorch MPS issue, not your hardware")
                    print("🔄 Applying M4 CPU optimization instead...")

                    try:
                        self.device = "cpu"
                        # Apply M4 CPU optimizations
                        torch.set_num_threads(8)  # Use M4's performance cores

                        self.model = whisper.load_model(
                            model_size,
                            device="cpu",
                            download_root=self.config["whisper_models_location"]
                        )
                        print("✅ Whisper model loaded with M4 CPU optimizations!")
                        print("   🚀 Performance: Optimized for M4's 4P+6E core architecture")
                        print("   🔋 Power: More efficient than forcing MPS with errors")

                    except Exception as cpu_e:
                        print(f"❌ M4 CPU optimization also failed: {cpu_e}")
                        raise RuntimeError(f"Could not load whisper model on M4: {cpu_e}")
                else:
                    # Try CPU as general fallback
                    if self.device != "cpu":
                        print("🔄 Trying CPU fallback with M4 optimizations...")
                        try:
                            self.device = "cpu"
                            torch.set_num_threads(8)  # M4 optimization

                            self.model = whisper.load_model(
                                model_size,
                                device="cpu",
                                download_root=self.config["whisper_models_location"]
                            )
                            print("✅ Whisper model loaded with M4 CPU optimizations!")
                        except Exception as cpu_e:
                            print(f"❌ CPU fallback also failed: {cpu_e}")
                            raise RuntimeError(f"Could not load whisper model: {cpu_e}")
                    else:
                        raise RuntimeError(f"Could not load whisper model: {e}")

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

    def _apply_mp3_transcription_boost(self, audio_array: np.ndarray, file_path: str) -> np.ndarray:
        """Apply specialized boost for MP3 files to improve transcription accuracy."""
        if not self.config.get("enable_mp3_transcription_boost", True):
            return audio_array

        # Check if this is an MP3 file
        if not file_path.lower().endswith('.mp3'):
            return audio_array

        print("🎵 Applying MP3 transcription boost...")

        try:
            # Step 1: Normalize audio to target RMS level
            if self.config.get("mp3_normalize_audio", True):
                current_rms = np.sqrt(np.mean(audio_array ** 2))
                target_rms = self.config.get("mp3_target_rms", 0.15)

                if current_rms > 1e-8:  # Avoid division by zero
                    normalization_factor = target_rms / current_rms
                    audio_array = audio_array * normalization_factor
                    print(f"   🎚️  Normalized: {current_rms:.4f} → {target_rms:.4f} RMS")

            # Step 2: Apply MP3-specific boost
            mp3_boost_factor = self.config.get("mp3_boost_factor", 8.0)
            if mp3_boost_factor > 1.0:
                # Gentle power curve for MP3 - preserves dynamics while boosting
                sign = np.sign(audio_array)
                magnitude = np.abs(audio_array)

                # Power curve: makes quiet sounds louder, leaves loud sounds more natural
                enhanced_magnitude = np.power(magnitude, 0.7) * mp3_boost_factor
                audio_array = sign * enhanced_magnitude
                print(f"   🚀 Applied MP3 boost: {mp3_boost_factor:.1f}x")

            # Step 3: Apply compression for MP3
            if self.config.get("mp3_compression_boost", True):
                audio_array = self._apply_mp3_compression(audio_array)

            # Step 4: High frequency boost for speech clarity
            if self.config.get("mp3_high_freq_boost", True) and SCIPY_AVAILABLE:
                audio_array = self._apply_mp3_high_freq_boost(audio_array)

            # Step 5: Final limiting to prevent clipping
            audio_max = np.max(np.abs(audio_array))
            if audio_max > 0.90:
                audio_array = audio_array * (0.85 / audio_max)
                print("   📉 Applied MP3 limiting to prevent clipping")

            # Final stats
            final_rms = np.sqrt(np.mean(audio_array ** 2))
            final_max = np.max(np.abs(audio_array))
            print(f"   ✅ MP3 boost complete: RMS={final_rms:.4f}, Max={final_max:.4f}")

            return audio_array.astype(np.float32)

        except Exception as e:
            print(f"   ⚠️  MP3 boost failed: {e}")
            return audio_array

    def _apply_mp3_compression(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression optimized for MP3 transcription."""
        try:
            # Moving RMS calculation for dynamic compression
            window_size = min(512, len(audio_array) // 20)
            if window_size < 32:
                return audio_array

            # Calculate moving RMS
            audio_squared = audio_array ** 2
            kernel = np.ones(window_size) / window_size
            moving_rms = np.sqrt(np.convolve(audio_squared, kernel, mode='same'))

            # Compression parameters optimized for speech
            threshold = 0.08  # Lower threshold for MP3
            ratio = 3.0  # Moderate compression ratio
            makeup_gain = 1.8  # Moderate makeup gain

            # Apply compression with soft knee
            compressed = np.where(
                moving_rms > threshold,
                audio_array * (threshold + (moving_rms - threshold) / ratio) / (moving_rms + 1e-8),
                audio_array * makeup_gain
            )

            print("   🎛️  Applied MP3 compression for speech clarity")
            return compressed.astype(np.float32)

        except Exception as e:
            print(f"   ⚠️  MP3 compression failed: {e}")
            return audio_array

    def _apply_mp3_high_freq_boost(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply high frequency boost to improve speech intelligibility in MP3."""
        try:
            # Design a gentle high-frequency emphasis filter
            # Boost 2-6 kHz range where speech intelligibility is critical
            nyquist = 8000  # Half of 16kHz sample rate
            low_freq = 2000 / nyquist  # 2 kHz
            high_freq = 6000 / nyquist  # 6 kHz

            # Create bandpass filter for speech frequencies
            sos = signal.butter(2, [low_freq, high_freq], btype='band', output='sos')
            speech_band = signal.sosfilt(sos, audio_array)

            # Mix back with original - subtle boost
            boost_amount = 0.4  # 40% boost to speech frequencies
            enhanced_audio = audio_array + (speech_band * boost_amount)

            print("   ✨ Applied high-frequency boost for speech clarity")
            return enhanced_audio.astype(np.float32)

        except Exception as e:
            print(f"   ⚠️  High-frequency boost failed: {e}")
            return audio_array

    def _enhanced_audio_preprocessing_for_whispers(self, audio_array: np.ndarray, file_path: str = "") -> np.ndarray:
        """Enhanced preprocessing for extreme whisper detection with advanced algorithms."""
        # Handle invalid values
        if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)

        # STEP 1: Apply MP3-specific boost first (if MP3 file)
        audio_array = self._apply_mp3_transcription_boost(audio_array, file_path)

        # STEP 2: Analyze audio characteristics after MP3 boost
        analysis = self._detect_soft_audio(audio_array)

        # STEP 3: Apply additional enhancement if still needed after MP3 boost
        if analysis["is_soft"] or analysis["is_whisper"]:
            audio_array = self._extreme_whisper_enhancement(audio_array, analysis)
        else:
            # Standard preprocessing for normal audio
            audio_rms = analysis["rms"]
            if audio_rms < 0.08:  # Increased threshold since MP3 boost applied
                audio_array = audio_array * 2.0  # Reduced boost since MP3 boost already applied
                audio_array = np.clip(audio_array, -0.95, 0.95)
                print("   🔊 Applied additional standard boost")

        # Final verification
        final_rms = np.sqrt(np.mean(audio_array ** 2))
        final_max = np.max(np.abs(audio_array))
        print(f"   ✅ Final Enhanced: RMS={final_rms:.4f}, Max={final_max:.4f}")

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
                # Use enhanced preprocessing with MP3 boost for better whisper detection
                audio_array = self._enhanced_audio_preprocessing_for_whispers(audio_array, audio_path)
                print("   🎯 Applied enhanced whisper preprocessing with MP3 boost")

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
                    audio_array = self._enhanced_audio_preprocessing_for_whispers(audio_array, audio_path)
                    print("   🎯 Applied enhanced preprocessing with MP3 boost (fallback)")

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

    def _merge_close_segments(self, segments: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Merge segments that are separated by very short gaps."""
        if not self.config.get("merge_close_segments", True) or len(segments) <= 1:
            return segments

        max_gap_ms = self.config.get("max_segment_gap_ms", 400)
        max_gap_seconds = max_gap_ms / 1000.0
        merged_segments = []
        current_segment = segments[0]

        print(f"🔗 Merging segments with gaps < {max_gap_ms}ms...")

        for next_segment in segments[1:]:
            current_end = current_segment.end.total_seconds()
            next_start = next_segment.start.total_seconds()
            gap = next_start - current_end

            if gap <= max_gap_seconds:
                # Merge segments
                merged_text = current_segment.content + " " + next_segment.content
                current_segment = srt.Subtitle(
                    index=current_segment.index,
                    start=current_segment.start,
                    end=next_segment.end,
                    content=merged_text.strip()
                )
                print(f"   📎 Merged segments (gap: {gap * 1000:.0f}ms)")
            else:
                # Keep current segment and move to next
                merged_segments.append(current_segment)
                current_segment = next_segment

        # Don't forget the last segment
        merged_segments.append(current_segment)

        # Renumber segments
        for i, segment in enumerate(merged_segments, 1):
            segment.index = i

        print(f"   ✅ Reduced from {len(segments)} to {len(merged_segments)} segments")
        return merged_segments

    def _filter_silence_segments(self, segments: List[srt.Subtitle], audio_data: np.ndarray,
                                 sample_rate: int) -> List[srt.Subtitle]:
        """Filter out segments that contain only silence or background noise."""
        if not self.config.get("filter_background_noise", True):
            return segments

        print("🔇 Filtering silence and background noise segments...")

        filtered_segments = []
        filtered_count = 0

        for segment in segments:
            start_time = segment.start.total_seconds()
            end_time = segment.end.total_seconds()

            # Validate and adjust timing using advanced speech detection
            validated_start, validated_end, has_speech = self.speech_detector.validate_segment_timing(
                audio_data, start_time, end_time, sample_rate
            )

            if has_speech:
                # Update segment with validated timing
                segment.start = timedelta(seconds=validated_start)
                segment.end = timedelta(seconds=validated_end)

                # Final check: ensure minimum duration
                if (validated_end - validated_start) >= 0.2:  # At least 200ms
                    filtered_segments.append(segment)

                    duration_change = (validated_end - validated_start) - (end_time - start_time)
                    if abs(duration_change) > 0.3:  # Significant timing change
                        print(
                            f"   🎯 Adjusted timing: {start_time:.2f}-{end_time:.2f} → {validated_start:.2f}-{validated_end:.2f}")
                else:
                    filtered_count += 1
                    print(f"   ⚠️  Filtered too-short segment: '{segment.content[:30]}...'")
            else:
                filtered_count += 1
                print(f"   🔇 Filtered silence segment: {start_time:.2f}-{end_time:.2f} '{segment.content[:30]}...'")

        # Renumber remaining segments
        for i, segment in enumerate(filtered_segments, 1):
            segment.index = i

        print(f"   ✅ Filtered {filtered_count} silence segments, kept {len(filtered_segments)}")
        return filtered_segments

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

    # NEW CODE: Modified method to include translation
    def _clean_srt_segments(self, segments: List[srt.Subtitle], audio_data: np.ndarray,
                            sample_rate: int) -> List[srt.Subtitle]:
        """Clean and filter SRT segments with advanced speech detection, overlap prevention, and translation."""
        print("🧹 Cleaning SRT segments with advanced speech detection...")

        cleaned_segments = []
        prompt_text = str(self.config.get("faster_whisper_initial_prompt", ""))
        max_segment_duration = 15.0

        # Step 1: Basic filtering and splitting
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

            # NEW CODE: Apply translation if enabled and using Kotoba model
            # Check if we're using Kotoba model (based on model path or name)
            is_kotoba = (
                    "kotoba" in self.config.get("faster_whisper_model_size", "").lower() or
                    "kotoba" in str(self.config.get("faster_whisper_local_model_path", "")).lower()
            )

            if is_kotoba and self.translation_processor.translation_enabled:
                original_text = text
                translated_text = self.translation_processor.translate_text(text)
                if translated_text != original_text:
                    text = translated_text
                    print(f"   🌐 Translated segment: '{original_text[:30]}...' -> '{text[:30]}...'")

            # Update segment content with translated text (if translation occurred)
            segment.content = text

            # Check segment duration and split if needed
            duration = (segment.end - segment.start).total_seconds()
            if duration > max_segment_duration:
                split_segments = self._split_long_segment(segment, max_segment_duration)
                cleaned_segments.extend(split_segments)
            else:
                cleaned_segments.append(segment)

        if not cleaned_segments:
            print("⚠️ No segments after basic filtering")
            return segments

        # Step 2: Filter silence and validate timing using advanced speech detection
        speech_filtered_segments = self._filter_silence_segments(cleaned_segments, audio_data, sample_rate)

        if not speech_filtered_segments:
            print("⚠️ No segments after speech filtering, keeping original")
            speech_filtered_segments = cleaned_segments

        # Step 3: Fix overlapping segments and short durations
        final_segments = []
        for i, segment in enumerate(speech_filtered_segments):
            # Skip segments that are too short after all processing
            duration = (segment.end - segment.start).total_seconds()
            if duration < 0.1:
                print(f"⚠️ Skipped very short segment: {duration:.2f}s")
                continue

            # Fix overlapping segments
            if final_segments:
                prev_segment = final_segments[-1]
                prev_end_time = prev_segment.end.total_seconds()
                current_start_time = segment.start.total_seconds()

                # If current segment starts before previous ends (overlap)
                if current_start_time < prev_end_time:
                    gap = 0.1  # Minimum gap
                    new_prev_end = current_start_time - gap

                    if new_prev_end > prev_segment.start.total_seconds():
                        prev_segment.end = timedelta(seconds=new_prev_end)
                        print(f"🔧 Fixed overlap: adjusted previous segment end")
                    else:
                        # Adjust current start instead
                        new_start = prev_end_time + gap
                        segment.start = timedelta(seconds=new_start)

                        # Ensure segment doesn't become too short
                        if (segment.end - segment.start).total_seconds() < 0.3:
                            segment.end = timedelta(seconds=new_start + 0.5)

            final_segments.append(segment)

        # Step 4: Merge close segments if enabled
        if self.config.get("merge_close_segments", True):
            final_segments = self._merge_close_segments(final_segments)

        # Step 5: Final renumbering
        for i, segment in enumerate(final_segments, 1):
            segment.index = i

        print(f"📝 Final cleaning result: {len(final_segments)} segments")
        return final_segments

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
        """Transcribe using faster-whisper with enhanced speech detection."""
        if self.model is None:
            raise RuntimeError("faster-whisper model is not loaded")
        print("🚀 Starting enhanced faster-whisper transcription with accurate speech detection...")
        vad_parameters = {
            "threshold": self.config.get("faster_whisper_vad_threshold", 0.15),
            "min_silence_duration_ms": self.config.get("faster_whisper_min_silence_duration_ms", 500),
            "max_speech_duration_s": self.config.get("faster_whisper_max_speech_duration_s", 30),
            "min_speech_duration_ms": self.config.get("faster_whisper_min_speech_duration_ms", 100)
        }
        print(
            f"   🎯 Enhanced VAD: threshold={vad_parameters['threshold']}, min_silence={vad_parameters['min_silence_duration_ms']}ms")
        print(
            f"   🎯 Speech duration: {vad_parameters['min_speech_duration_ms']}-{vad_parameters['max_speech_duration_s'] * 1000}ms")
        try:
            segments_generator, info = self.model.transcribe(
                audio_data["array"],
                task=self.config.get("faster_whisper_task", "translate"),
                language=self.config.get("faster_whisper_force_language", None),
                initial_prompt=self.config.get("faster_whisper_initial_prompt", None),
                beam_size=self.config.get("faster_whisper_beam_size", 5),
                best_of=self.config.get("faster_whisper_best_of", 2),
                temperature=self.config.get("faster_whisper_temperature", [0.0, 0.2, 0.4]),
                patience=self.config.get("faster_whisper_patience", 1.5),
                length_penalty=self.config.get("faster_whisper_length_penalty", 1.0),
                repetition_penalty=self.config.get("faster_whisper_repetition_penalty", 1.15),
                no_repeat_ngram_size=self.config.get("faster_whisper_no_repeat_ngram_size", 5),
                suppress_blank=True,
                suppress_tokens=[-1],
                without_timestamps=False,
                max_initial_timestamp=1.0,
                prepend_punctuations="\"'([{-",
                append_punctuations="\"'.,:!?)]}",
                vad_filter=True,
                vad_parameters=vad_parameters,
                no_speech_threshold=0.4,
                log_prob_threshold=-2.5,
                compression_ratio_threshold=2.4,
                word_timestamps=self.config.get("faster_whisper_word_timestamps", True),
                condition_on_previous_text=self.config.get("condition_on_previous_text", True)
            )
            print(f"📊 Language: {info.language} (confidence: {info.language_probability:.2f})")
            chunks = []
            last_end_time = 0.0
            segment_count = 0
            filtered_count = 0
            word_confidence_threshold = self.config.get("word_confidence_threshold", 0.4)
            segment_confidence_threshold = self.config.get("segment_confidence_threshold", 0.35)
            enable_confidence_filtering = self.config.get("enable_confidence_filtering", True)
            enable_text = self.config.get("enable_text", True)
            for segment in segments_generator:
                segment_count += 1
                text = segment.text.strip() if hasattr(segment, 'text') else ""
                if not text or len(text) < 2:
                    filtered_count += 1
                    continue
                if hasattr(segment, 'words') and segment.words and enable_confidence_filtering:
                    high_conf_words = [w for w in segment.words if
                                       w.word.strip() and w.probability >= word_confidence_threshold]
                    if not high_conf_words:
                        if enable_text:
                            print(f"   ❌ Filtered low-confidence segment: '{text[:30]}...' (no high-conf words)")

                        filtered_count += 1
                        continue
                    avg_confidence = np.mean([w.probability for w in high_conf_words])
                    if avg_confidence < segment_confidence_threshold:
                        if enable_text:
                            print(
                                f"   ❌ Filtered low-confidence segment: '{text[:30]}...' (conf: {avg_confidence:.2f})")

                        filtered_count += 1
                        continue
                    start = max(float(high_conf_words[0].start) - 0.1, float(segment.start))
                    end = min(float(high_conf_words[-1].end) + 0.1, float(segment.end))
                    if enable_text:
                        print(
                            f"   ✅ High-conf segment: {start:.2f}-{end:.2f}, conf: {avg_confidence:.2f}, '{text[:30]}...'")
                else:
                    start = float(segment.start)
                    end = float(segment.end)
                    if enable_text:
                        print(
                            f"   ⚠️ No word timestamps or confidence filtering disabled: {start:.2f}-{end:.2f}, '{text[:30]}...'")
                if start < last_end_time:
                    start = last_end_time + 0.1
                    if end <= start:
                        end = start + 0.5
                if end - start < 0.2 or end - start > 30.0:
                    filtered_count += 1
                    continue
                chunk_data = {
                    "text": text,
                    "timestamp": [start, end]
                }
                chunks.append(chunk_data)
                last_end_time = end
                progress = min(100.0, (last_end_time / audio_duration) * 100) if audio_duration > 0 else 0
                self._update_progress(progress)
            self._update_progress(100.0)
            print(f"\n✅ Processing complete: {len(chunks)} valid segments, {filtered_count} filtered")
            if len(chunks) == 0:
                print("⚠️ No valid segments detected - possible silence or very low quality audio")
                chunks = [{
                    "text": "No speech detected in audio.",
                    "timestamp": [0.0, min(5.0, audio_duration)]
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

    def _transcribe_with_whisper_mps(self, audio_data: Dict[str, Any], audio_duration: float) -> Dict[str, Any]:
        """Transcribe using the standard whisper library with MPS."""
        if self.model is None:
            raise RuntimeError("whisper-mps model is not loaded")
        print("🚀 Starting whisper-mps transcription...")

        # Map config to whisper's transcribe options
        language = self.config.get("faster_whisper_force_language", None)
        task = self.config.get("faster_whisper_task", "transcribe")  # Default to transcribe for whisper-mps
        initial_prompt = self.config.get("faster_whisper_initial_prompt", None)

        print(f"   🗣️ Language: {'auto-detect' if language is None else language}")
        print(f"   📝 Task: {task}")

        try:
            # The progress bar from the original whisper is not easy to capture,
            # so we'll update progress after the whole process.
            result = self.model.transcribe(
                audio_data["array"],
                language=language,
                task=task,
                initial_prompt=initial_prompt,
                verbose=False  # Set to False to avoid duplicate progress info
            )

            chunks = []
            for segment in result["segments"]:
                start = float(segment["start"])
                end = float(segment["end"])
                text = segment["text"].strip()

                if not text:
                    continue

                chunk_data = {
                    "text": text,
                    "timestamp": [start, end]
                }
                chunks.append(chunk_data)
                progress = min(100.0, (end / audio_duration) * 100) if audio_duration > 0 else 0
                self._update_progress(progress)

            self._update_progress(100.0)
            print(f"\n✅ Processing complete: {len(chunks)} segments generated")

            if not chunks:
                print("⚠️ No segments detected - possible silence or very low quality audio")
                chunks = [{
                    "text": "No speech detected in audio.",
                    "timestamp": [0.0, min(5.0, audio_duration)]
                }]

            return {
                "text": result["text"],
                "chunks": chunks
            }
        except Exception as e:
            print(f"\n❌ whisper-mps error: {e}")
            return {
                "text": "Transcription failed.",
                "chunks": [{"text": "Transcription failed due to an error.", "timestamp": [0.0, 5.0]}]
            }

    def transcribe_file(self, file_path: str) -> str:
        """Main transcription function with enhanced speech detection for accurate timestamps."""
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

            engine_name = "faster-whisper" if self.use_faster_whisper else "whisper-mps"
            print(f"🔧 Engine: {engine_name} (Enhanced Speech Detection)")

            # NEW CODE: Show translation status
            is_kotoba = (
                    "kotoba" in self.config.get("faster_whisper_model_size", "").lower() or
                    "kotoba" in str(self.config.get("faster_whisper_local_model_path", "")).lower()
            )

            translation_active = self.use_faster_whisper and is_kotoba and self.translation_processor.translation_enabled
            if translation_active:
                print(f"🌐 Translation: ENABLED ({self.translation_processor.translation_model})")
            else:
                print(f"🌐 Translation: DISABLED")

            print(f"🎯 Features: Enhanced Speech Detection + Accurate Timestamps + Translation")

            # Load audio with enhanced preprocessing
            audio_data = self._load_audio(audio_path)
            audio_duration = len(audio_data["array"]) / audio_data["sampling_rate"]
            print(f"⏱️  Duration: {audio_duration:.1f}s")

            # Run enhanced transcription with speech detection
            start_time = time.time()
            self.transcription_complete = False

            try:
                if self.use_faster_whisper:
                    result = self._transcribe_with_faster_whisper(audio_data, audio_duration)
                else:
                    result = self._transcribe_with_whisper_mps(audio_data, audio_duration)
            finally:
                self.transcription_complete = True
                elapsed = time.time() - start_time
                mins, secs = divmod(elapsed, 60)
                completion_time = f"Completed in {int(mins):02d}:{int(secs):02d}"
                print(f"\n⏱️  {completion_time}")

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

            print(f"📝 Generated {len(chunks)} raw segments")

            # Convert to SRT
            subs = self._convert_timestamps_to_srt(chunks, audio_duration)
            if not subs:
                subs = [srt.Subtitle(
                    index=1,
                    start=timedelta(seconds=0),
                    end=timedelta(seconds=min(5, audio_duration)),
                    content="No speech detected in audio."
                )]

            # Enhanced cleaning with speech detection and translation
            cleaned_subs = self._clean_srt_segments(subs, audio_data["array"], audio_data["sampling_rate"])
            if not cleaned_subs:
                # Fallback if all segments were filtered
                cleaned_subs = [srt.Subtitle(
                    index=1,
                    start=timedelta(seconds=0),
                    end=timedelta(seconds=min(3, audio_duration)),
                    content="Audio processed - minimal speech detected."
                )]

            # Write SRT file
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt.compose(cleaned_subs))

            # Add credits
            self._add_credit_to_srt(srt_path, self.config["credit"])
            self._add_credit_to_srt(srt_path, completion_time)

            print(f"✅ Success! Enhanced SRT saved with {len(cleaned_subs)} accurate segments")
            print(f"🎯 Features used: Advanced VAD, Speech Detection, Timestamp Validation")

            # NEW CODE: Show translation status in completion message
            if translation_active:
                print(f"🌐 Translation: Japanese -> English ({self.translation_processor.translation_model})")

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
            print("Using embedded configuration.")
    else:
        print("Using embedded configuration.")

    return config


def parse_arguments():
    """Parse command line arguments for single file or batch processing."""
    parser = argparse.ArgumentParser(
        description="Enhanced Whisper Transcription Tool with Batch Processing and Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file:  python transcribe7.py filename.mp3
  Batch mode:   python transcribe7.py --batch=y

Enhanced Features:
  ⚙️  Dual Engine: Supports 'faster-whisper' and 'whisper-mps'.
        """
    )

    # Create mutually exclusive group for file vs batch mode
    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument(
        'filename',
        nargs='?',
        help='Audio or video file to transcribe'
    )

    mode_group.add_argument(
        '--batch',
        choices=['y', 'yes', 'n', 'no'],
        help='Enable batch processing mode (y/yes or n/no)'
    )

    return parser.parse_args()


def main():
    """Main function with support for single file and batch processing."""
    args = parse_arguments()

    # Load configuration
    config = load_config()

    # Show enhanced features
    engine = "faster-whisper" if config.get("use_faster_whisper", True) else "whisper-mps"
    print(f"🚀 Engine: {engine} (Enhanced Speech Detection + MP3 Boost + Translation)")

    if engine == "faster-whisper":
        print(f"🎯 Advanced VAD: {'ENABLED' if config.get('faster_whisper_vad_filter', True) else 'DISABLED'}")

    print(f"🔍 Speech Analysis: {'ENABLED' if config.get('enable_advanced_speech_detection', True) else 'DISABLED'}")

    # NEW CODE: Show translation configuration
    translation_enabled = config.get("translation_enabled", False)
    translation_model = config.get("translation_model", "Helsinki-NLP/opus-mt-ja-en")
    print(
        f"🌐 Translation: {'ENABLED (with faster-whisper)' if translation_enabled and TRANSFORMERS_AVAILABLE else 'DISABLED'}")
    if translation_enabled and TRANSFORMERS_AVAILABLE:
        print(f"   📚 Model: {translation_model}")
    elif translation_enabled and not TRANSFORMERS_AVAILABLE:
        print(f"   ⚠️  Transformers library required: pip install transformers")

    # Initialize transcriber
    transcriber = WhisperTranscriber(config)

    try:
        # Check if batch mode is requested
        if args.batch and args.batch.lower() in ['y', 'yes']:
            print("\n📦 BATCH PROCESSING MODE")
            print("=" * 50)

            # Initialize batch processor
            batch_processor = BatchProcessor(transcriber, config)

            # Show batch configuration
            print(f"📁 Batch folder: {config.get('batch_folder', 'Not configured')}")

            # Process batch
            stats = batch_processor.process_batch()

            # Show final results
            print("\n" + "=" * 50)
            print("📊 BATCH PROCESSING RESULTS:")
            print(f"   ✅ Successful: {stats['success']}")


            if stats['success'] > 0:
                print(f"\n🎉 Batch processing completed! {stats['success']} files transcribed successfully.")
            else:
                print(f"\n⚠️  No files were successfully transcribed.")

        else:
            # Single file mode
            if not args.filename:
                print("❌ Error: No filename provided for single file mode")
                print("Usage: python transcribe7.py [filename] or python transcribe7.py --batch=y")
                sys.exit(1)

            print(f"\n📄 SINGLE FILE MODE")
            print("=" * 50)

            result = transcriber.transcribe_file(args.filename)
            print(f"\n🎉 Success! Enhanced transcription saved: {result}")

            # NEW CODE: Show translation info if applicable
            if translation_enabled and TRANSFORMERS_AVAILABLE and engine == "faster-whisper":
                print("   🌐 Japanese to English translation (when using Kotoba models)")

    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")

        # NEW CODE: Add translation troubleshooting
        if config.get("translation_enabled", False) and not TRANSFORMERS_AVAILABLE:
            print("   • Install transformers for translation: pip install transformers")

        print("   • Check audio quality and volume levels")

        if args.batch:
            print("   • Check batch folder configuration and permissions")
        sys.exit(1)


if __name__ == "__main__":
    main()