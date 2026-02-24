import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# Set Matplotlib backend to Agg (non-interactive)
plt.switch_backend('Agg')

# Configure font for Korean characters
# Try to find a suitable font on the system
font_name = None
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    if 'NanumGothic' in Path(font).name:
        font_name = Path(font).name.replace('.ttf', '')
        break
    elif 'AppleGothic' in Path(font).name: # For macOS
        font_name = Path(font).name.replace('.ttf', '')
        break

if font_name:
    plt.rcParams['font.family'] = font_name
else:
    print("Warning: NanumGothic or AppleGothic font not found. Falling back to generic sans-serif.")
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False # Prevent minus sign from breaking with Korean font

OUTPUT_DIR = Path('reports/2026-02-16_phase2_3d_comparison/assets/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File paths
FILE_2D = 'data/sample/comparison_result.json'
FILE_3D = 'data/sample/comparison_result_3d.json'

def load_data(filepath):
    """Loads JSON data from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_data(data):
    """Processes loaded data into a DataFrame."""
    per_frame_loss = data['per_frame_loss']
    df = pd.DataFrame(per_frame_loss)
    # Extract total_loss and joint_losses
    df['total_loss'] = df['total_loss']
    # Normalize joint_losses into a separate DataFrame
    joint_losses_df = pd.json_normalize(df['joint_losses'])
    return df, joint_losses_df

def plot_loss_comparison_2d_vs_3d(df_2d, df_3d, output_path):
    """
    1. loss_comparison_2d_vs_3d.png
       - 2개의 subplot (상: 2D loss, 하: 3D loss)
       - X축: 프레임 인덱스, Y축: total_loss
       - 각각 mean loss 수평선 표시
       - 제목은 한국어
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=150, sharex=True)

    # 2D Loss Plot
    axes[0].plot(df_2d['frame_idx'], df_2d['total_loss'], label='2D Loss', color='blue')
    mean_loss_2d = df_2d['total_loss'].mean()
    axes[0].axhline(mean_loss_2d, color='red', linestyle='--', label=f'2D 평균 손실: {mean_loss_2d:.2f}')
    axes[0].set_title('2D 포즈 비교 - 프레임별 총 손실')
    axes[0].set_ylabel('총 손실 (total_loss)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 3D Loss Plot
    axes[1].plot(df_3d['frame_idx'], df_3d['total_loss'], label='3D Loss', color='green')
    mean_loss_3d = df_3d['total_loss'].mean()
    axes[1].axhline(mean_loss_3d, color='red', linestyle='--', label=f'3D 평균 손실: {mean_loss_3d:.2f}')
    axes[1].set_title('3D 포즈 비교 - 프레임별 총 손실')
    axes[1].set_xlabel('프레임 인덱스')
    axes[1].set_ylabel('총 손실 (total_loss)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path / 'loss_comparison_2d_vs_3d.png')
    plt.close()
    print(f"Generated {output_path / 'loss_comparison_2d_vs_3d.png'}")


def plot_joint_heatmap_3d(joint_losses_df_3d, output_path):
    """
    2. joint_heatmap_3d.png
       - 3D 비교의 관절별 평균 손실 상위 10개 bar chart
       - X축: 관절명, Y축: 평균 손실
       - 제목은 한국어
    """
    # Calculate mean joint losses, excluding potential non-numeric columns
    mean_joint_losses_3d = joint_losses_df_3d.mean(numeric_only=True).sort_values(ascending=False)
    
    # Get top 10 joints
    top_10_joints = mean_joint_losses_3d.head(10)

    plt.figure(figsize=(12, 7), dpi=150)
    top_10_joints.plot(kind='bar', color='skyblue')
    plt.title('3D 포즈 비교 - 관절별 평균 손실 상위 10개')
    plt.xlabel('관절명')
    plt.ylabel('평균 손실')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path / 'joint_heatmap_3d.png')
    plt.close()
    print(f"Generated {output_path / 'joint_heatmap_3d.png'}")


def plot_loss_overlay_2d_3d(df_2d, df_3d, output_path):
    """
    3. loss_overlay_2d_3d.png
       - 2D와 3D의 프레임별 loss를 하나의 차트에 중첩 (dual y-axis 사용)
       - 왼쪽 Y축: 2D loss (파란색), 오른쪽 Y축: 3D loss (빨간색)
       - X축: 프레임 인덱스
       - 범례 표시
    """
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

    ax1.set_xlabel('프레임 인덱스')
    ax1.set_ylabel('2D 총 손실 (total_loss)', color='blue')
    ax1.plot(df_2d['frame_idx'], df_2d['total_loss'], color='blue', label='2D 총 손실')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('3D 총 손실 (total_loss)', color='red')
    ax2.plot(df_3d['frame_idx'], df_3d['total_loss'], color='red', label='3D 총 손실')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.suptitle('2D 및 3D 포즈 비교 - 프레임별 총 손실 중첩')
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path / 'loss_overlay_2d_3d.png')
    plt.close()
    print(f"Generated {output_path / 'loss_overlay_2d_3d.png'}")


def plot_std_comparison(df_2d, df_3d, output_path):
    """
    4. std_comparison.png
       - 2D와 3D의 통계 비교 bar chart (mean, max, std)
       - 2D와 3D를 나란히 (grouped bar chart)
    """
    metrics_2d = {
        '평균 손실': df_2d['total_loss'].mean(),
        '최대 손실': df_2d['total_loss'].max(),
        '표준 편차': df_2d['total_loss'].std()
    }
    metrics_3d = {
        '평균 손실': df_3d['total_loss'].mean(),
        '최대 손실': df_3d['total_loss'].max(),
        '표준 편차': df_3d['total_loss'].std()
    }

    metrics_df = pd.DataFrame({
        '2D': metrics_2d,
        '3D': metrics_3d
    })

    plt.figure(figsize=(10, 6), dpi=150)
    metrics_df.plot(kind='bar', rot=0, figsize=(10, 6), width=0.8)
    plt.title('2D 및 3D 포즈 비교 - 통계 요약')
    plt.xlabel('측정 항목')
    plt.ylabel('값')
    plt.legend(title='모델')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path / 'std_comparison.png')
    plt.close()
    print(f"Generated {output_path / 'std_comparison.png'}")


def main():
    # Load and process data
    data_2d = load_data(FILE_2D)
    data_3d = load_data(FILE_3D)

    df_2d, joint_losses_df_2d = process_data(data_2d)
    df_3d, joint_losses_df_3d = process_data(data_3d)

    # Generate plots
    plot_loss_comparison_2d_vs_3d(df_2d, df_3d, OUTPUT_DIR)
    plot_joint_heatmap_3d(joint_losses_df_3d, OUTPUT_DIR)
    plot_loss_overlay_2d_3d(df_2d, df_3d, OUTPUT_DIR)
    plot_std_comparison(df_2d, df_3d, OUTPUT_DIR)

if __name__ == '__main__':
    main()
