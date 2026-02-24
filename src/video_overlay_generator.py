import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt # For font configuration only

# Configure font for Korean characters (matplotlib is used for font config but not plotting here)
import matplotlib.font_manager as fm
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' # Path to NanumGothic font
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False # To prevent breaking minus sign
else:
    print(f"Warning: Korean font not found at {font_path}. Falling back to default font.")


OUTPUT_DIR = os.environ.get(
    "JUBEOP_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "reports", "2026-02-16_phase2_3d_comparison", "assets")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pose landmark connections
POSE_CONNECTIONS = [
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
    ('LEFT_SHOULDER', 'LEFT_ELBOW'), ('LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_SHOULDER', 'LEFT_HIP'), ('RIGHT_SHOULDER', 'RIGHT_HIP'),
    ('LEFT_HIP', 'RIGHT_HIP'),
    ('LEFT_HIP', 'LEFT_KNEE'), ('LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_ANKLE'),
    ('LEFT_ANKLE', 'LEFT_HEEL'), ('RIGHT_ANKLE', 'RIGHT_HEEL'),
    ('LEFT_HEEL', 'LEFT_FOOT_INDEX'), ('RIGHT_HEEL', 'RIGHT_FOOT_INDEX'),
]

def load_pose_data(filepath):
    """Loads pose data from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)["frames"]

def get_landmark_coords_2d(frame_data, landmark_type, landmark_name, img_width, img_height):
    """
    Extracts x, y coordinates for a given landmark type and name,
    and scales them to image dimensions.
    """
    landmarks = frame_data.get(landmark_type, {})
    coords = landmarks.get(landmark_name)
    if coords and 'x' in coords and 'y' in coords:
        # MediaPipe landmarks are normalized [0, 1]. Scale to image dimensions.
        return int(coords['x'] * img_width), int(coords['y'] * img_height)
    return None

def draw_skeleton(image, pose_data_frame, landmark_type, connections, color, img_width, img_height):
    """
    Draws a 2D skeleton on the given image using OpenCV drawing functions.
    """
    # Draw connections (limbs)
    for connection in connections:
        p1_name, p2_name = connection
        p1 = get_landmark_coords_2d(pose_data_frame, landmark_type, p1_name, img_width, img_height)
        p2 = get_landmark_coords_2d(pose_data_frame, landmark_type, p2_name, img_width, img_height)

        if p1 is not None and p2 is not None:
            cv2.line(image, p1, p2, color, 2)

    # Draw individual joint points
    for landmark_name in pose_data_frame.get(landmark_type, {}):
        coords = get_landmark_coords_2d(pose_data_frame, landmark_type, landmark_name, img_width, img_height)
        if coords is not None:
            cv2.circle(image, coords, 5, color, -1) # -1 fills the circle

def main():
    ref_video_path = "data/sample/reference_grand_battement.mp4"
    output_video_path = os.path.join(OUTPUT_DIR, "reference_video_skeletons.mp4")

    ref_poses_data = load_pose_data("data/sample/reference_poses_3d.json")
    cmp_poses_data = load_pose_data("data/sample/compare_poses_3d.json")

    cap = cv2.VideoCapture(ref_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {ref_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use XVID codec for wider compatibility, or 'mp4v' or 'MJPG'
    # Check available codecs: https://www.fourcc.org/codecs.php
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not create video writer for {output_video_path}")
        cap.release()
        return

    num_poses_frames = min(len(ref_poses_data), len(cmp_poses_data))

    for frame_idx in range(min(total_frames, num_poses_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Get pose data for the current frame
        ref_frame_data = ref_poses_data[frame_idx]
        cmp_frame_data = cmp_poses_data[frame_idx]

        # Draw skeletons on the frame
        # We use "landmarks" for 2D coordinates normalized to image space
        draw_skeleton(frame, ref_frame_data, "landmarks", POSE_CONNECTIONS, (255, 0, 0), width, height) # Blue for reference
        draw_skeleton(frame, cmp_frame_data, "landmarks", POSE_CONNECTIONS, (0, 0, 255), width, height) # Red for comparison

        out.write(frame)

    cap.release()
    out.release()
    print(f"Overlay video saved to {output_video_path}")

if __name__ == "__main__":
    main()
