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

# =================================================================
# --- ⚙️ Global Parameters (Configure Everything Here) ---
# =================================================================

# 1. File & Folder Paths
VIDEO_SOURCE_PATH = '/Volumes/Macintosh HD/Downloads/Video/uc'
SRT_OUTPUT_PATH = '/Volumes/Macintosh HD/Downloads/srt'
MODEL_STORAGE_PATH = '/Volumes/Macintosh HD/Downloads/srt/whisper_models'
TEMP_PATH = '/Volumes/Macintosh HD/Downloads/srt/temp'

# 2. Language & Task Settings
SOURCE_LANGUAGE = "japanese"       # Source language of the video
TARGET_LANGUAGE_CODE = "en-US"     # NOTE: faster-whisper only translates to English. This sets the output language code.
TASK = "translate"               # "transcribe" or "translate"

# 3. Model & Transcription Configuration
MODEL_SIZE = "large-v3"
DEVICE = "mps"
COMPUTE_TYPE = "float16"
CREDIT = "Subbed by Gemini"

# 4. SRT Cleaning & Word Lists
PUNCT_MATCH = ["。", "、", ",", ".", "〜", "！", "!", "？", "?", "-"]
GARBAGE_LIST = ["a", "hmm", "huh", "oh"]
SUPPRESS_HIGH = ["subscribe", "my channel", "ご視聴ありがとうございました"]

# =================================================================
# --- Helper Functions (All Unchanged) ---
# =================================================================
def convert_video_to_audio(video_path, audio_output_path):
    """Converts a video file to MP3 using ffmpeg."""
    print(f"\n▶️ Converting video to temporary audio file...")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_output_path],
            check=True, capture_output=True, text=True
        )
        print(f"✅ Conversion successful: {os.path.basename(audio_output_path)}")
        return audio_output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during conversion: {e}\n   FFmpeg stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("❌ Error: 'ffmpeg' not found. Please ensure it's installed and in your system's PATH.")
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

def run_transcription_and_cleaning(video_filename):
    """Main function to run the full transcription and cleaning pipeline."""
    # 1. Setup Environment and Paths
    print("--- Initializing ---")
    os.makedirs(SRT_OUTPUT_PATH, exist_ok=True)
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
    os.makedirs(TEMP_PATH, exist_ok=True)
    os.environ['HF_HOME'] = MODEL_STORAGE_PATH

    input_video_path = os.path.join(VIDEO_SOURCE_PATH, video_filename)
    file_basename = os.path.splitext(video_filename)[0]

    if not os.path.exists(input_video_path):
        print(f"❌ Error: Input file not found at {input_video_path}")
        return

    try:
        # 2. Convert Video to Audio (in Temp Folder)
        temp_audio_path = os.path.join(TEMP_PATH, f"{file_basename}.mp3")
        audio_for_transcription = convert_video_to_audio(input_video_path, temp_audio_path)
        if not audio_for_transcription:
            raise Exception("Audio conversion failed.")

        # 3. Transcribe Audio
        print("\n--- Transcription ---")
        print(f"📦 Loading Whisper model '{MODEL_SIZE}' from: {MODEL_STORAGE_PATH}")
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print(f"✅ Model loaded. Starting transcription...")
        segments, info = model.transcribe(
            audio=audio_for_transcription, task=TASK, language=SOURCE_LANGUAGE,
            beam_size=5, vad_filter=True
        )

        # 4. Save Raw SRT (in Temp Folder)
        raw_srt_path = os.path.join(TEMP_PATH, f"{file_basename}.raw.srt")
        subs = [srt.Subtitle(index=i, start=timedelta(seconds=s.start), end=timedelta(seconds=s.end), content=s.text.strip()) for i, s in enumerate(segments, 1)]
        with open(raw_srt_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(subs))
        print(f"📝 Raw transcription saved to temporary folder.")

        # 5. Clean and Stamp Final SRT
        print("\n--- Post-Processing ---")
        # ---- 👇 LOGIC UPDATED TO USE NEW PARAMETER 👇 ----
        lang_code = TARGET_LANGUAGE_CODE if TASK == "translate" else TO_LANGUAGE_CODE.get(SOURCE_LANGUAGE, SOURCE_LANGUAGE)
        final_srt_filename = f"{file_basename}.{lang_code}.srt"
        hallucination_lists = [SUPPRESS_HIGH, GARBAGE_LIST]

        cleaned_srt_path = None
        if TASK == "translate":
            cleaned_srt_path = clean_srt_file_english(raw_srt_path, SRT_OUTPUT_PATH, final_srt_filename, hallucination_lists)
        else: # transcribe
            if SOURCE_LANGUAGE == "japanese":
                cleaned_srt_path = clean_srt_file_japanese(raw_srt_path, SRT_OUTPUT_PATH, final_srt_filename, hallucination_lists)
            else:
                cleaned_srt_path = clean_srt_file_english(raw_srt_path, SRT_OUTPUT_PATH, final_srt_filename, hallucination_lists)

        if cleaned_srt_path and os.path.exists(cleaned_srt_path):
            stamp_srt_file(cleaned_srt_path, CREDIT)
            print("\n✨ --- All Done! --- ✨")
            print(f"✅ Final SRT file saved to: {cleaned_srt_path}")
        else:
            print("\n⚠️ Warning: SRT cleaning failed. No final file was created.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # 6. Cleanup Temp Folder
        if os.path.exists(TEMP_PATH):
            print(f"\n🧹 Cleaning up temporary folder: {TEMP_PATH}")
            shutil.rmtree(TEMP_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a video file and generate cleaned SRT subtitles.")
    parser.add_argument("filename", type=str, help="The name of the video file to process (e.g., 'my_video.mp4'). Must be located in the VIDEO_SOURCE_PATH.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    run_transcription_and_cleaning(args.filename)
