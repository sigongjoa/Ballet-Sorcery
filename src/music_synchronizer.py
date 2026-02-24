import librosa
import librosa.display
import numpy as np
import subprocess
import os
import matplotlib
import matplotlib.pyplot as plt
import json

# Set the matplotlib backend to 'Agg' for non-interactive plotting
matplotlib.use('Agg')

# Configure font for Korean characters
import matplotlib.font_manager as fm
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' # Path to NanumGothic font
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False # To prevent breaking minus sign
else:
    print(f"Warning: Korean font not found at {font_path}. Falling back to default font.")


def extract_audio_from_video(video_path, audio_output_path):
    """
    Extracts audio from a video file using ffmpeg.
    """
    if os.path.exists(audio_output_path):
        os.remove(audio_output_path)
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # no video
        '-acodec', 'pcm_s16le',  # audio codec
        '-ar', '44100',  # audio sample rate
        '-ac', '1',  # mono channel
        audio_output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio extracted to {audio_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"FFmpeg stdout: {e.stdout.decode()}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        return None
    return audio_output_path

def analyze_music_beats(audio_path, sr=22050):
    """
    Analyzes music beats using librosa.
    Returns BPM, beat frames, and beat timestamps.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_timestamps = librosa.frames_to_time(beat_frames, sr=sr)
        return tempo[0], beat_frames, beat_timestamps
    except Exception as e:
        print(f"Error analyzing music beats: {e}")
        return None, None, None

def map_beats_to_video_frames(beat_timestamps, video_fps):
    """
    Maps beat timestamps to video frame indices.
    """
    frame_indices = (beat_timestamps * video_fps).astype(int)
    return frame_indices

def plot_beat_detection(y, sr, beat_timestamps, output_path):
    """
    Plots the waveform with detected beats.
    """
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.vlines(beat_timestamps, -1, 1, color='r', linestyle='--', label='Beats')
    plt.title('Waveform and Beat Detection')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    video_path = "data/sample/reference_grand_battement.mp4"
    audio_output_dir = "reports/2026-02-16_phase2_3d_comparison/assets/"
    os.makedirs(audio_output_dir, exist_ok=True)
    audio_output_path = os.path.join(audio_output_dir, "reference_grand_battement.wav")
    video_fps = 24  # Target FPS as per POC_SPEC.md

    print(f"Starting music synchronization for: {video_path}")

    # 1. Extract audio
    extracted_audio_path = extract_audio_from_video(video_path, audio_output_path)
    if not extracted_audio_path:
        return

    # 2. Analyze music beats
    tempo, beat_frames, beat_timestamps = analyze_music_beats(extracted_audio_path)
    if tempo is None:
        return
    
    print(f"Detected Tempo (BPM): {tempo:.2f}")
    print(f"Number of beats detected: {len(beat_timestamps)}")

    # 3. Map beats to video frames
    beat_video_frames = map_beats_to_video_frames(beat_timestamps, video_fps)
    print(f"Mapped beat frames (first 10): {beat_video_frames[:10]}")

    # Plot beat detection
    y, sr = librosa.load(extracted_audio_path, sr=22050)
    plot_output_path = os.path.join(audio_output_dir, "beat_detection.png")
    plot_beat_detection(y, sr, beat_timestamps, plot_output_path)
    print(f"Beat detection plot saved to {plot_output_path}")

    # Save beat information to a JSON file
    beat_info = {
        "video_path": video_path,
        "audio_path": extracted_audio_path,
        "tempo_bpm": tempo,
        "beat_timestamps_sec": beat_timestamps.tolist(),
        "beat_video_frames_24fps": beat_video_frames.tolist(),
        "video_fps_for_mapping": video_fps
    }
    beat_info_output_path = os.path.join(audio_output_dir, "beat_info.json")
    with open(beat_info_output_path, 'w', encoding='utf-8') as f:
        json.dump(beat_info, f, ensure_ascii=False, indent=4)
    print(f"Beat information saved to {beat_info_output_path}")

if __name__ == "__main__":
    main()
