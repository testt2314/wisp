#!/usr/bin/env python3
"""
Whisper Transcription Tool for Mac mini
Converts video/audio files to English subtitles using faster-whisper
Usage: python transcribe.py [file]
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
from tqdm import tqdm
import torch
from transformers import pipeline
import librosa
import numpy as np

# Master Configuration - Embedded in script
CONFIG = {
    "srt_location": "/Volumes/Macintosh HD/Downloads/srt",
    "temp_location": "/Volumes/Macintosh HD/Downloads/srt/temp",
    "audio_location": "/Volumes/Macintosh HD/Downloads",
    "whisper_models_location": "/Volumes/Macintosh HD/Downloads/srt/whisper_models",
    "ffmpeg_path": "/Volumes/Macintosh HD/Downloads/srt/whisper_models/ffmpeg",
    "ffprobe_path": "/Volumes/Macintosh HD/Downloads/srt/whisper_models/ffprobe",
    "model_size": "openai/whisper-large-v3",
    "chunk_length_s": 30,
    "vad_threshold": 0.15,
    "chunk_duration": 15.0,
    "credit": "Created using Whisper Transcription Tool",
    "use_mps": True  # Enable MPS acceleration on Apple Silicon
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
        self.pipe = None
        self.device = None
        self._ensure_directories()
        self._setup_device()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.config["srt_location"], self.config["temp_location"]]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _setup_device(self):
        """Setup the best available device (MPS, CUDA, or CPU)."""
        if self.config.get("use_mps", True) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU (no acceleration available)")

    def _load_model(self):
        """Load the Whisper model using Hugging Face pipeline."""
        if self.pipe is None:
            print(f"Loading Whisper model: {self.config['model_size']}")

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
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
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
                print("Fallback model loaded successfully!")

    def _load_audio(self, audio_path: str) -> Dict[str, Any]:
        """Load audio file using librosa."""
        print(f"Loading audio file: {audio_path}")

        # Load audio with librosa (automatically handles various formats)
        audio_array, sample_rate = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz

        return {
            "array": audio_array,
            "sampling_rate": sample_rate,
            "path": audio_path
        }

    def _check_ffmpeg(self) -> bool:
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

    def _is_video_file(self, file_path: str) -> bool:
        """Check if the file is a video file."""
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
        return Path(file_path).suffix.lower() in video_extensions

    def _is_audio_file(self, file_path: str) -> bool:
        """Check if the file is an audio file."""
        audio_extensions = {'.mp3', '.wav', '.aac', '.m4a', '.ogg', '.opus', '.flac'}
        return Path(file_path).suffix.lower() in audio_extensions

    def _convert_to_audio(self, video_path: str) -> str:
        """Convert video file to MP3 audio."""
        if not self._check_ffmpeg():
            raise RuntimeError("ffmpeg is required for video conversion")

        video_name = Path(video_path).stem
        audio_path = os.path.join(self.config["temp_location"], f"{video_name}.mp3")

        print(f"Converting video to audio: {video_path}")
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
            return audio_path
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

    def transcribe_file(self, file_path: str) -> str:
        """Main transcription function using Hugging Face pipeline."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self._load_model()

        # Determine file type and prepare audio file
        audio_path = file_path
        temp_audio = False

        if self._is_video_file(file_path):
            print("Video file detected, converting to audio...")
            audio_path = self._convert_to_audio(file_path)
            temp_audio = True
        elif not self._is_audio_file(file_path):
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

        try:
            # Generate output filename
            base_name = Path(file_path).stem
            srt_path = os.path.join(self.config["srt_location"], f"{base_name}.srt")

            print(f"Transcribing: {audio_path}")
            print(f"Output SRT: {srt_path}")
            print(f"Using device: {self.device}")

            # Load audio data
            audio_data = self._load_audio(audio_path)
            audio_duration = len(audio_data["array"]) / audio_data["sampling_rate"]

            print(f"Audio duration: {audio_duration:.2f} seconds")

            # Run transcription with progress tracking
            print("Starting transcription...")
            with tqdm(total=100, desc="Transcribing") as pbar:
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
                    pbar.update(100)
                except Exception as e:
                    print(f"Transcription error: {e}")
                    # Fallback without explicit translation task
                    result = self.pipe(
                        audio_data["array"].copy(),
                        return_timestamps=True
                    )
                    pbar.update(100)

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
        sys.exit(1)

    input_file = sys.argv[1]

    # Load configuration
    config = load_config()

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