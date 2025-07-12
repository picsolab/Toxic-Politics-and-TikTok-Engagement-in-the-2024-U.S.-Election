# TikTok Political Video Feature Extraction Script
# This script extracts transcripts, RGB stats, and visual features from videos

from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import librosa

def extract_audio_from_video(video_path, audio_path):
    """Extract audio track from video and save as a file."""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def convert_audio_to_wav(audio_path, wav_path):
    """Convert extracted audio to WAV format."""
    audio = AudioSegment.from_file(audio_path)
    audio.export(wav_path, format="wav")

def load_and_process_audio(wav_path):
    """Load and process WAV audio for further tasks."""
    audio, rate = librosa.load(wav_path, sr=None)
    return audio, rate


import whisper

class VoiceAnalyzer:
    def __init__(self, model_name='base'):
        self.model = whisper.load_model(model_name)

    def transcribe(self, path):
        result = self.model.transcribe(path)
        print(f"Transcript: {result['text'][:200]}...")
        return result


import cv2
import numpy as np
from deepface import DeepFace

def extract_keyframes(video_path, frame_rate=1):
    """Extract keyframes from video at specified rate (1 frame per second)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(1)) % int(fps * frame_rate) == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames

def analyze_frame_deepface(frame):
    """Extract demographics and emotion from a video frame using DeepFace."""
    try:
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
        return result[0]
    except Exception as e:
        print("DeepFace error:", e)
        return None

def extract_rgb_stats(frame):
    """Calculate mean RGB values from a frame."""
    means = cv2.mean(frame)[:3]
    return dict(mean_r=means[2], mean_g=means[1], mean_b=means[0])


# Example: Apply feature extraction on a video dataset
import os

def process_video(video_path):
    print(f"Processing: {video_path}")

    # --- Audio Transcript ---
    va = VoiceAnalyzer("base")
    transcript_data = va.transcribe(video_path)

    # --- Keyframe Extraction ---
    keyframes = extract_keyframes(video_path)
    rgb_stats_list = []
    demographics_list = []

    for frame in keyframes:
        rgb_stats_list.append(extract_rgb_stats(frame))
        demo_result = analyze_frame_deepface(frame)
        if demo_result:
            demographics_list.append(demo_result)

    return {
        "video_path": video_path,
        "transcript": transcript_data['text'],
        "rgb_stats": rgb_stats_list,
        "demographics": demographics_list,
    }

# Example usage:
# dataset_dir = "path_to_video_dataset"
# all_results = [process_video(os.path.join(dataset_dir, f)) for f in os.listdir(dataset_dir) if f.endswith(".mp4")]
