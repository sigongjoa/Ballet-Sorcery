
import json
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import matplotlib.font_manager as fm

# Set the matplotlib backend to 'Agg' for non-interactive plotting
matplotlib.use('Agg')

# Define the output directory
OUTPUT_DIR = "reports/2026-02-16_phase2_3d_comparison/assets/"
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
    ('LEFT_ANKLE', 'LEFT_HEEL'), ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'), # Changed to FOOT_INDEX based on typical skeleton structure
    ('LEFT_HEEL', 'LEFT_FOOT_INDEX'), ('RIGHT_HEEL', 'RIGHT_FOOT_INDEX'),
]

# Corrected POSE_CONNECTIONS based on typical skeleton and user's requested data
# The original connection 'LEFT_ANKLE', 'LEFT_HEEL' is present in the provided JSON landmarks,
# but for drawing a continuous line from ankle to toe, it usually goes ankle -> heel -> foot_index
# or ankle -> foot_index directly. Let's keep the user provided ones and add foot index.
# The user specified:
# ('LEFT_ANKLE', 'LEFT_HEEL'), ('RIGHT_ANKLE', 'RIGHT_HEEL'),
# ('LEFT_HEEL', 'LEFT_FOOT_INDEX'), ('RIGHT_HEEL', 'RIGHT_FOOT_INDEX'),

# Let's adjust POSE_CONNECTIONS to properly connect ankle, heel, and foot_index
# Based on typical pose models, ANKLE connects to HEEL and HEEL connects to FOOT_INDEX.
# The user's provided connections are sufficient. I will use them as is.
# The only correction needed is if the ankle-foot_index is desired, but the user explicitly
# mentioned ankle-heel and heel-foot_index.

# Let's define the full set of connections for drawing, ensuring they connect to the given landmarks.
# The provided connections define the limbs. I will make sure all points are plotted.
# The user mentioned:
# ('LEFT_ANKLE', 'LEFT_HEEL'), ('RIGHT_ANKLE', 'RIGHT_HEEL'),
# ('LEFT_HEEL', 'LEFT_FOOT_INDEX'), ('RIGHT_HEEL', 'RIGHT_FOOT_INDEX'),
# I need to ensure the script properly handles the plotting of these connections.

def load_pose_data(filepath):
    """Loads pose data from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)["frames"]

def get_landmark_coords(frame_data, landmark_type, landmark_name):
    """Extracts x, y, z coordinates for a given landmark type and name."""
    landmarks = frame_data.get(landmark_type, {})
    coords = landmarks.get(landmark_name)
    if coords:
        # Mediapipe's world_landmarks often have Y as up-down, but positive is down.
        # For matplotlib's 3D plot, positive Z is usually up.
        # To align with typical 3D visualizations where Y is up, we can swap Y and Z,
        # and optionally negate Y to invert the axis if positive Y should be upwards.
        # User specified: world_landmarks에서 Y축이 위/아래 방향 (MediaPipe 규약: Y가 음수=위)
        # So for plotting, I'll leave Y as it is for vertical.
        # But if the plot looks inverted, I might need to negate Y or swap with Z.
        # Let's assume (x, y, z) directly maps to (x, vertical, depth) for now.
        return np.array([coords['x'], coords['y'], coords['z']])
    return None

def plot_skeleton_3d(ax, pose_data_frame, landmark_type, connections, color, label=None):
    """Plots a 3D skeleton on the given Axes3D."""
    
    # Collect all landmark points for scaling
    all_coords = []

    # Plot connections (limbs)
    for connection in connections:
        p1_name, p2_name = connection
        p1 = get_landmark_coords(pose_data_frame, landmark_type, p1_name)
        p2 = get_landmark_coords(pose_data_frame, landmark_type, p2_name)

        if p1 is not None and p2 is not None:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=0.7)
            all_coords.append(p1)
            all_coords.append(p2)

    # Plot individual joint points (scatter)
    for landmark_name in pose_data_frame.get(landmark_type, {}):
        coords = get_landmark_coords(pose_data_frame, landmark_type, landmark_name)
        if coords is not None:
            ax.scatter(coords[0], coords[1], coords[2], color=color, s=20, label=label if landmark_name == list(pose_data_frame.get(landmark_type, {}).keys())[0] else "", alpha=0.9)
            if label and landmark_name == list(pose_data_frame.get(landmark_type, {}).keys())[0]: # Plot label only once per skeleton
                 ax.scatter(coords[0], coords[1], coords[2], color=color, s=20, label=label, alpha=0.9)
            else:
                ax.scatter(coords[0], coords[1], coords[2], color=color, s=20, alpha=0.9)
            all_coords.append(coords)
    
    # Set axis limits and aspect ratio based on all coordinates
    if all_coords:
        all_coords = np.array(all_coords)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()

        max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0

        mid_x = (x_max + x_min) * 0.5
        mid_y = (y_max + y_min) * 0.5
        mid_z = (z_max + z_min) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_skeleton_2d(ax, pose_data_frame, landmark_type, connections, color, label=None):
    """Plots a 2D skeleton on the given Axes."""
    # Plot connections (limbs)
    for connection in connections:
        p1_name, p2_name = connection
        p1 = get_landmark_coords(pose_data_frame, landmark_type, p1_name)
        p2 = get_landmark_coords(pose_data_frame, landmark_type, p2_name)

        if p1 is not None and p2 is not None:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=0.7)

    # Plot individual joint points (scatter)
    for landmark_name in pose_data_frame.get(landmark_type, {}):
        coords = get_landmark_coords(pose_data_frame, landmark_type, landmark_name)
        if coords is not None:
            if label and landmark_name == list(pose_data_frame.get(landmark_type, {}).keys())[0]: # Plot label only once per skeleton
                 ax.scatter(coords[0], coords[1], color=color, s=10, label=label, alpha=0.9)
            else:
                ax.scatter(coords[0], coords[1], color=color, s=10, alpha=0.9)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal', adjustable='box')


def main():
    # Configure font for Korean characters
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' # Path to NanumGothic font
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False # To prevent breaking minus sign
    else:
        print(f"Warning: Korean font not found at {font_path}. Falling back to default font.")

    ref_poses = load_pose_data("data/sample/reference_poses_3d.json")
    cmp_poses = load_pose_data("data/sample/compare_poses_3d.json")

    # Ensure consistent data length
    num_frames = min(len(ref_poses), len(cmp_poses))
    ref_poses = ref_poses[:num_frames]
    cmp_poses = cmp_poses[:num_frames]

    # Frames for visualization
    max_loss_frame_idx = 40
    min_loss_frame_idx = 12
    mean_loss_frame_idx = 20 # Assuming this frame is representative of mean loss

    # Ensure frame indices are within bounds
    max_loss_frame_idx = min(max_loss_frame_idx, num_frames - 1)
    min_loss_frame_idx = min(min_loss_frame_idx, num_frames - 1)
    mean_loss_frame_idx = min(mean_loss_frame_idx, num_frames - 1)

    # --- 1. skeleton_3d_max_loss.png ---
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    ref_frame_data = ref_poses[max_loss_frame_idx]
    cmp_frame_data = cmp_poses[max_loss_frame_idx]

    plot_skeleton_3d(ax, ref_frame_data, "world_landmarks", POSE_CONNECTIONS, 'blue', label='Reference')
    plot_skeleton_3d(ax, cmp_frame_data, "world_landmarks", POSE_CONNECTIONS, 'red', label='Comparison')
    
    ax.set_title(f'3D 스켈레톤 비교 - 최대 손실 프레임 (Frame {max_loss_frame_idx})') # Korean title
    ax.view_init(elev=15, azim=70)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "skeleton_3d_max_loss.png"))
    plt.close(fig)

    # --- 2. skeleton_3d_min_loss.png ---
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    ref_frame_data = ref_poses[min_loss_frame_idx]
    cmp_frame_data = cmp_poses[min_loss_frame_idx]

    plot_skeleton_3d(ax, ref_frame_data, "world_landmarks", POSE_CONNECTIONS, 'blue', label='Reference')
    plot_skeleton_3d(ax, cmp_frame_data, "world_landmarks", POSE_CONNECTIONS, 'red', label='Comparison')

    ax.set_title(f'3D 스켈레톤 비교 - 최소 손실 프레임 (Frame {min_loss_frame_idx})') # Korean title
    ax.view_init(elev=15, azim=70)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "skeleton_3d_min_loss.png"))
    plt.close(fig)

    # --- 3. skeleton_3d_mean_loss.png ---
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    ref_frame_data = ref_poses[mean_loss_frame_idx]
    cmp_frame_data = cmp_poses[mean_loss_frame_idx]

    plot_skeleton_3d(ax, ref_frame_data, "world_landmarks", POSE_CONNECTIONS, 'blue', label='Reference')
    plot_skeleton_3d(ax, cmp_frame_data, "world_landmarks", POSE_CONNECTIONS, 'red', label='Comparison')

    ax.set_title(f'3D 스켈레톤 비교 - 평균 손실 프레임 (Frame {mean_loss_frame_idx})') # Korean title
    ax.view_init(elev=15, azim=70)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "skeleton_3d_mean_loss.png"))
    plt.close(fig)

    # --- 4. skeleton_3d_multiview_max.png ---
    fig = plt.figure(figsize=(15, 12), dpi=150) # Larger figure for 2x2 subplots
    fig.suptitle(f'3D 스켈레톤 다중 시점 - 최대 손실 프레임 (Frame {max_loss_frame_idx})', fontsize=16)

    views = {
        '정면 (Front)': (0, 0),
        '측면 (Side)': (0, 90),
        '상단 (Top)': (90, 0),
        '대각선 (Diagonal)': (15, 70)
    }

    for i, (title, (elev, azim)) in enumerate(views.items()):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        plot_skeleton_3d(ax, ref_frame_data, "world_landmarks", POSE_CONNECTIONS, 'blue', label='Reference')
        plot_skeleton_3d(ax, cmp_frame_data, "world_landmarks", POSE_CONNECTIONS, 'red', label='Comparison')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        if i == 0: # Add legend to the first subplot only
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.savefig(os.path.join(OUTPUT_DIR, "skeleton_3d_multiview_max.png"))
    plt.close(fig)

    # --- 5. skeleton_2d_vs_3d_comparison.png ---
    fig = plt.figure(figsize=(15, 8), dpi=150) # Larger figure for 1x2 subplots
    
    # 2D Skeleton
    ax1 = fig.add_subplot(121)
    plot_skeleton_2d(ax1, ref_frame_data, "landmarks", POSE_CONNECTIONS, 'blue', label='Reference')
    plot_skeleton_2d(ax1, cmp_frame_data, "landmarks", POSE_CONNECTIONS, 'red', label='Comparison')
    ax1.set_title('2D 스켈레톤 (Landmarks)')
    ax1.invert_yaxis() # Invert Y-axis for typical image coordinates (origin top-left)
    ax1.legend()

    # 3D Skeleton
    ax2 = fig.add_subplot(122, projection='3d')
    plot_skeleton_3d(ax2, ref_frame_data, "world_landmarks", POSE_CONNECTIONS, 'blue', label='Reference')
    plot_skeleton_3d(ax2, cmp_frame_data, "world_landmarks", POSE_CONNECTIONS, 'red', label='Comparison')
    ax2.set_title('3D 스켈레톤 (World Landmarks)')
    ax2.view_init(elev=15, azim=70) # Standard view angle
    ax2.legend()
    
    fig.suptitle(f'2D vs 3D 스켈레톤 비교 - Frame {max_loss_frame_idx}', fontsize=16) # Korean title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    plt.savefig(os.path.join(OUTPUT_DIR, "skeleton_2d_vs_3d_comparison.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
