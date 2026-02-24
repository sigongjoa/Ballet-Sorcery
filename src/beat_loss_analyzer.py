import json
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Tuple

# Set the matplotlib backend to 'Agg' for non-interactive plotting
matplotlib.use('Agg')

# Configure font for Korean characters
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

# Define joint weights (higher weight for lower body as per POC_SPEC.md)
JOINT_WEIGHTS = {
    "LEFT_HIP": 2.0, "RIGHT_HIP": 2.0,
    "LEFT_KNEE": 2.0, "RIGHT_KNEE": 2.0,
    "LEFT_ANKLE": 2.0, "RIGHT_ANKLE": 2.0,
    "LEFT_HEEL": 2.0, "RIGHT_HEEL": 2.0,
    "LEFT_FOOT_INDEX": 2.0, "RIGHT_FOOT_INDEX": 2.0,
    # Other joints will implicitly have a weight of 1.0
}

def load_pose_data(filepath):
    """Loads pose data from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)["frames"]

def load_beat_info(filepath):
    """Loads beat information from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_world_landmark_coords(frame_data, landmark_name):
    """Extracts x, y, z coordinates for a given world landmark."""
    coords = frame_data.get("world_landmarks", {}).get(landmark_name)
    if coords:
        return np.array([coords['x'], coords['y'], coords['z']])
    return None

def calculate_frame_loss(ref_frame, cmp_frame, landmark_names, joint_weights):
    """
    Calculates the weighted L2 distance (Euclidean distance) between corresponding
    world landmarks of two frames.
    """
    weighted_total_loss_sq = 0.0
    total_weight = 0.0
    for name in landmark_names:
        weight = joint_weights.get(name, 1.0) # Default weight is 1.0
        ref_coords = get_world_landmark_coords(ref_frame, name)
        cmp_coords = get_world_landmark_coords(cmp_frame, name) # Corrected typo

        if ref_coords is not None and cmp_coords is not None:
            diff_sq = np.linalg.norm(ref_coords - cmp_coords) ** 2
            weighted_total_loss_sq += weight * diff_sq
            total_weight += weight
    
    if total_weight > 0:
        return np.sqrt(weighted_total_loss_sq / total_weight) # Weighted Root Mean Square Error
    return 0.0

def calculate_individual_joint_losses(ref_poses, cmp_poses, landmark_names):
    """
    Calculates the average L2 distance (Euclidean distance) for each joint
    across all frames.
    """
    joint_losses_sum = {name: 0.0 for name in landmark_names}
    joint_counts = {name: 0 for name in landmark_names}

    num_frames = min(len(ref_poses), len(cmp_poses))

    for i in range(num_frames):
        ref_frame = ref_poses[i]
        cmp_frame = cmp_poses[i]
        for name in landmark_names:
            ref_coords = get_world_landmark_coords(ref_frame, name)
            cmp_coords = get_world_landmark_coords(cmp_frame, name)

            if ref_coords is not None and cmp_coords is not None:
                loss = np.linalg.norm(ref_coords - cmp_coords) # L2 distance
                joint_losses_sum[name] += loss
                joint_counts[name] += 1
    
    average_joint_losses = {}
    for name in landmark_names:
        if joint_counts[name] > 0:
            average_joint_losses[name] = joint_losses_sum[name] / joint_counts[name]
        else:
            average_joint_losses[name] = 0.0 # No data for this joint
            
    return average_joint_losses

def calculate_delta_t_sync_per_beat(beat_video_frames, frame_losses, beat_info, video_fps):
    """
    Calculates a conceptual delta_t_sync for each beat.
    Assumes "movement completion" occurs at the minimum loss frame within a beat interval.
    delta_t_sync is the time difference between the beat timestamp and the timestamp
    of this minimum loss frame.
    """
    delta_t_syncs = []
    beat_timestamps_sec = np.array(beat_info["beat_timestamps_sec"])

    # Ensure beat_video_frames has at least two elements for intervals
    # The first beat_video_frame is usually 0, and the last is num_frames
    # We are interested in the actual beat events within these intervals
    actual_beat_events_timestamps = beat_timestamps_sec # These are the actual times of the beats

    # Each interval starts at beat_video_frames[i] and ends at beat_video_frames[i+1]
    # The beat that defines this interval is considered actual_beat_events_timestamps[i]
    # provided i is a valid index for actual_beat_events_timestamps
    
    for i in range(len(beat_video_frames) - 1):
        start_frame_idx = beat_video_frames[i]
        end_frame_idx = beat_video_frames[i+1] # This is exclusive
        
        # Adjust end_frame_idx if it goes beyond frame_losses array
        end_frame_idx = min(end_frame_idx, len(frame_losses))

        interval_frames_abs = np.arange(start_frame_idx, end_frame_idx)
        interval_losses = frame_losses[start_frame_idx:end_frame_idx]

        if len(interval_losses) > 0:
            # Find the frame index within the original full sequence that corresponds to min loss in this interval
            min_loss_relative_idx = np.argmin(interval_losses)
            min_loss_abs_frame_idx = interval_frames_abs[min_loss_relative_idx]
            
            # Get timestamp of this min loss frame
            min_loss_timestamp = min_loss_abs_frame_idx / video_fps

            # Get the actual beat timestamp that *should* align with this interval's completion
            # Assuming beat_timestamps_sec has one entry per beat interval start
            # The beat_video_frames are beat_video_frames_24fps, which are 0-indexed.
            # beat_info["beat_timestamps_sec"] is also 0-indexed corresponding to beat events.
            # The number of beat events is len(beat_timestamps_sec).
            # The beat_video_frames array has len(beat_timestamps_sec) + 1 (for initial 0 frame) + 1 (for end of video) elements
            
            # We need to find the closest actual beat timestamp to the start_frame_idx or within the interval
            # For simplicity, we can use the beat timestamp that "starts" this interval, if available
            
            # Let's map beat_video_frames to beat_timestamps_sec directly.
            # beat_video_frames: [0, beat1_frame, beat2_frame, ..., last_beat_frame, num_frames]
            # beat_timestamps_sec: [beat1_time, beat2_time, ..., last_beat_time]
            # So, for interval i, the relevant beat timestamp is beat_timestamps_sec[i]
            
            # Need to handle the edge case where the first interval starts at frame 0, but no beat_timestamp_sec[0] exists
            # Also handle the last interval that goes to num_frames, potentially no corresponding beat timestamp
            
            # A more robust mapping:
            # We iterate len(beat_timestamps_sec) times for actual beats
            if i < len(beat_timestamps_sec): # Only for intervals defined by actual beats
                actual_beat_timestamp = beat_timestamps_sec[i]
                delta_t_sync = abs(min_loss_timestamp - actual_beat_timestamp)
                delta_t_syncs.append(delta_t_sync)
            else:
                delta_t_syncs.append(0.0) # No beat event for this interval, assume 0 timing error
        else:
            delta_t_syncs.append(0.0) # No frames in interval
            
    return delta_t_syncs


def calibrate_lambdas(beat_losses: list[float], delta_t_syncs: list[float]) -> Tuple[float, float]:
    """
    Min-Max Normalization 기반으로 포즈 손실(beat_losses)과 타이밍 오차(delta_t_syncs)의
    가중치(λ1, λ2)를 자동 보정합니다. 각 시그널의 최대값을 기준으로 정규화하여
    두 항이 동등한 기여도를 갖도록 설정합니다.
    """
    max_loss = max(beat_losses) if beat_losses else 0
    max_sync = max(delta_t_syncs) if delta_t_syncs else 0
    
    lambda1 = 1.0 / max_loss if max_loss > 0 else 1.0
    lambda2 = 1.0 / max_sync if max_sync > 0 else 1.0
    
    return lambda1, lambda2


def main():
    ref_poses = load_pose_data("data/sample/reference_poses_3d.json")
    cmp_poses = load_pose_data("data/sample/compare_poses_3d.json")
    beat_info = load_beat_info(os.path.join(OUTPUT_DIR, "beat_info.json"))

    # Ensure consistent data length
    num_frames = min(len(ref_poses), len(cmp_poses))
    ref_poses = ref_poses[:num_frames]
    cmp_poses = cmp_poses[:num_frames]

    # Get a list of all landmark names from the first reference frame
    landmark_names = list(ref_poses[0]["world_landmarks"].keys())

    # Calculate frame-by-frame loss
    frame_losses = []
    for i in range(num_frames):
        loss = calculate_frame_loss(ref_poses[i], cmp_poses[i], landmark_names, JOINT_WEIGHTS)
        frame_losses.append(loss)
    
    # Convert to numpy array for easier slicing and aggregation
    frame_losses = np.array(frame_losses)

    # Aggregate loss per beat
    beat_video_frames = np.array(beat_info["beat_video_frames_24fps"])
    beat_losses = []
    beat_intervals = []

    # Add the start of the video as the first "beat" if not present
    if beat_video_frames[0] != 0:
        beat_video_frames = np.insert(beat_video_frames, 0, 0)

    # Ensure the last beat goes until the end of the video
    beat_video_frames = np.append(beat_video_frames, num_frames)

    for i in range(len(beat_video_frames) - 1):
        start_frame = beat_video_frames[i]
        end_frame = beat_video_frames[i+1]
        
        # Ensure end_frame does not exceed num_frames
        end_frame = min(end_frame, num_frames)

        # Slice frames within the current beat interval
        interval_losses = frame_losses[start_frame:end_frame]
        
        if len(interval_losses) > 0:
            beat_loss = np.mean(interval_losses) # Average loss within the beat interval
            beat_losses.append(beat_loss)
            beat_intervals.append(f"[{start_frame}-{end_frame-1}]")
        else:
            # Handle cases where an interval might be empty (e.g., last beat interval)
            beat_losses.append(0.0) # Or NaN, depending on desired behavior
            beat_intervals.append(f"[{start_frame}-{end_frame-1}]")


    # Plot beat-wise loss
    plt.figure(figsize=(12, 6), dpi=150)
    plt.bar(range(len(beat_losses)), beat_losses, tick_label=beat_intervals, color='skyblue')
    plt.title('비트별 포즈 손실 (평균 L2 거리)')
    plt.xlabel('비트 구간 (프레임)')
    plt.ylabel('평균 L2 손실')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "beat_wise_loss.png"))
    plt.close()

    # Save beat-wise loss information to JSON
    beat_wise_loss_info = {
        "beat_intervals_frames": beat_intervals,
        "beat_average_losses": beat_losses
    }
    with open(os.path.join(OUTPUT_DIR, "beat_wise_loss.json"), 'w', encoding='utf-8') as f:
        json.dump(beat_wise_loss_info, f, ensure_ascii=False, indent=4)

    print(f"Beat-wise loss plot saved to {os.path.join(OUTPUT_DIR, 'beat_wise_loss.png')}")
    print(f"Beat-wise loss info saved to {os.path.join(OUTPUT_DIR, 'beat_wise_loss.json')}")

    # --- Plot frame-by-frame loss ---
    plt.figure(figsize=(15, 6), dpi=150)
    plt.plot(frame_losses, color='darkorange')
    plt.title('프레임별 포즈 손실 (가중 평균 L2 거리)')
    plt.xlabel('프레임 인덱스')
    plt.ylabel('가중 평균 L2 손실')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "frame_by_frame_loss.png"))
    plt.close()
    print(f"Frame-by-frame loss plot saved to {os.path.join(OUTPUT_DIR, 'frame_by_frame_loss.png')}")

    # --- Plot frame-by-frame loss with high error sections highlighted ---
    plt.figure(figsize=(15, 6), dpi=150)
    plt.plot(frame_losses, color='darkorange', label='프레임별 손실')

    # Identify high error sections
    mean_loss = np.mean(frame_losses)
    std_loss = np.std(frame_losses)
    # Define threshold as mean + 1 standard deviation
    threshold = mean_loss + std_loss

    # Highlight sections above threshold
    highlight_frames = np.where(frame_losses > threshold)[0]
    
    # Plot vertical spans for high error regions
    for i in range(len(highlight_frames)):
        # Check if this is the start of a new contiguous high-error segment
        if i == 0 or highlight_frames[i] > highlight_frames[i-1] + 1:
            # Determine the end of the current contiguous segment
            j = i
            while j + 1 < len(highlight_frames) and highlight_frames[j+1] == highlight_frames[j] + 1:
                j += 1
            plt.axvspan(highlight_frames[i], highlight_frames[j] + 1, color='red', alpha=0.3, label='높은 에러 구간' if i == 0 else "")
            i = j # Move index to the end of the current segment

    plt.axhline(y=threshold, color='red', linestyle='--', label='에러 임계값 (평균 + 1 표준편차)')
    
    plt.title('프레임별 포즈 손실 및 높은 에러 구간')
    plt.xlabel('프레임 인덱스')
    plt.ylabel('가중 평균 L2 손실')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "highlighted_error_loss.png"))
    plt.close()
    print(f"Highlighted error loss plot saved to {os.path.join(OUTPUT_DIR, 'highlighted_error_loss.png')}")

    # --- Calculate and plot average joint losses ---
    average_joint_losses = calculate_individual_joint_losses(ref_poses, cmp_poses, landmark_names)
    
    # Sort joints by loss for better visualization
    sorted_joints = sorted(average_joint_losses.items(), key=lambda item: item[1], reverse=True)
    joint_names = [item[0] for item in sorted_joints]
    losses = [item[1] for item in sorted_joints]

    plt.figure(figsize=(15, 8), dpi=150)
    plt.bar(joint_names, losses, color='lightgreen')
    plt.title('관절별 평균 포즈 손실 (L2 거리)')
    plt.xlabel('관절')
    plt.ylabel('평균 L2 손실')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "average_joint_losses.png"))
    plt.close()
    print(f"Average joint losses plot saved to {os.path.join(OUTPUT_DIR, 'average_joint_losses.png')}")

    # Save average joint losses to JSON
    with open(os.path.join(OUTPUT_DIR, "average_joint_losses.json"), 'w', encoding='utf-8') as f:
        json.dump(average_joint_losses, f, ensure_ascii=False, indent=4)
    print(f"Average joint losses info saved to {os.path.join(OUTPUT_DIR, 'average_joint_losses.json')}")

    # --- Calculate conceptual delta_t_sync per beat ---
    # Need video_fps from beat_info or metadata, for now use a fixed value.
    # The beat_info file has "video_fps_for_mapping" key
    video_fps_for_timing = beat_info["video_fps_for_mapping"]
    conceptual_delta_t_syncs = calculate_delta_t_sync_per_beat(
        beat_video_frames, frame_losses, beat_info, video_fps_for_timing
    )

    # --- Calculate conceptual Loss_jubeop per beat ---
    # Loss_jubeop = λ1 * ||P_act - P_tar||^2 + λ2 * Δt_sync
    # Using beat_losses for ||P_act - P_tar||^2
    lambda1, lambda2 = calibrate_lambdas(beat_losses, conceptual_delta_t_syncs)
    print(f'보정된 λ값: λ1={lambda1:.4f}, λ2={lambda2:.4f}')
    
    loss_jubeop_per_beat = []
    for i in range(len(beat_losses)):
        # Ensure conceptual_delta_t_syncs has corresponding entry
        if i < len(conceptual_delta_t_syncs):
            loss_jubeop = (lambda1 * beat_losses[i]) + (lambda2 * conceptual_delta_t_syncs[i])
            loss_jubeop_per_beat.append(loss_jubeop)
        else:
            # Handle cases where beat_losses might have more entries than conceptual_delta_t_syncs
            # This can happen if the last interval in beat_losses doesn't correspond to an actual beat event
            loss_jubeop_per_beat.append(lambda1 * beat_losses[i]) 

    # --- Plot conceptual Loss_jubeop per beat ---
    plt.figure(figsize=(12, 6), dpi=150)
    plt.bar(range(len(loss_jubeop_per_beat)), loss_jubeop_per_beat, tick_label=beat_intervals, color='purple')
    plt.title('비트별 컨셉적 파뢰주법 손실')
    plt.xlabel('비트 구간 (프레임)')
    plt.ylabel('컨셉적 파뢰주법 손실')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "conceptual_loss_jubeop.png"))
    plt.close()
    print(f"Conceptual Loss_jubeop plot saved to {os.path.join(OUTPUT_DIR, 'conceptual_loss_jubeop.png')}")

    # Save conceptual Loss_jubeop information to JSON
    conceptual_loss_jubeop_info = {
        "beat_intervals_frames": beat_intervals,
        "conceptual_delta_t_syncs": conceptual_delta_t_syncs,
        "loss_jubeop_per_beat": loss_jubeop_per_beat,
        "lambda1": lambda1,
        "lambda2": lambda2
    }
    with open(os.path.join(OUTPUT_DIR, "conceptual_loss_jubeop.json"), 'w', encoding='utf-8') as f:
        json.dump(conceptual_loss_jubeop_info, f, ensure_ascii=False, indent=4)
    print(f"Conceptual Loss_jubeop info saved to {os.path.join(OUTPUT_DIR, 'conceptual_loss_jubeop.json')}")



if __name__ == "__main__":
    main()
