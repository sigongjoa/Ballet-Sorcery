# -*- coding: utf-8 -*-
"""
비교 결과를 시각화하는 모듈.
"""

import matplotlib
matplotlib.use('Agg') # Agg 백엔드 사용 (GUI 없이 이미지 생성)
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import sys

# src 디렉토리를 sys.path에 추가하여 normalizer를 임포트할 수 있도록 함
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normalizer

# 관절 이름 목록 (인덱스 순서)
LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
    "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
    "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

# MediaPipe의 33개 관절 중 주요 연결선 (이름 기반)
POSE_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"), ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("LEFT_HEEL", "LEFT_FOOT_INDEX"), ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
]

def generate_loss_graph(comparison_result: dict, output_path: str) -> str:
    """
    프레임별 total_loss를 선 그래프로 시각화합니다.

    Args:
        comparison_result (dict): comparator.py의 compare_poses()가 반환하는 결과 딕셔너리.
        output_path (str): 생성될 PNG 파일의 저장 경로.

    Returns:
        str: 생성된 PNG 파일의 경로.
    """
    plt.figure(figsize=(12, 6), dpi=150)

    total_losses = [frame['total_loss'] for frame in comparison_result['per_frame_loss']]
    frame_indices = list(range(len(total_losses)))

    plt.plot(frame_indices, total_losses, label='Total Loss per Frame')

    mean_loss = np.mean(total_losses)
    plt.axhline(y=mean_loss, color='g', linestyle='--', label=f'Mean Loss: {mean_loss:.2f}')

    max_loss_idx = np.argmax(total_losses)
    max_loss_value = total_losses[max_loss_idx]
    plt.plot(max_loss_idx, max_loss_value, 'ro', markersize=8, label=f'Max Loss: {max_loss_value:.2f} at Frame {max_loss_idx}')
    plt.annotate(f'{max_loss_value:.2f}', (max_loss_idx, max_loss_value), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('Total Loss per Frame')
    plt.xlabel('Frame Index')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path

def generate_skeleton_comparison(ref_json_path: str, comp_json_path: str, frame_idx: int, output_path: str, normalize: bool = True) -> str:
    """
    특정 프레임의 두 스켈레톤을 하나의 이미지에 오버레이하여 시각화합니다.

    Args:
        ref_json_path (str): 참조 포즈 데이터가 담긴 JSON 파일 경로.
        comp_json_path (str): 비교 포즈 데이터가 담긴 JSON 파일 경로.
        frame_idx (int): 시각화할 프레임 인덱스.
        output_path (str): 생성될 PNG 파일의 저장 경로.
        normalize (bool): 포즈 데이터를 정규화할지 여부. 기본값은 True.

    Returns:
        str: 생성된 PNG 파일의 경로.
    """
    def _draw_skeleton(ax, landmarks_dict, connections, color, label=None):
        """스켈레톤을 그리는 헬퍼 함수. landmarks_dict는 {"NOSE": {"x":..,"y":..}, ...} 형태."""
        # 관절 점 그리기
        xs = [v['x'] for v in landmarks_dict.values() if 'x' in v]
        ys = [v['y'] for v in landmarks_dict.values() if 'y' in v]
        ax.scatter(xs, ys, c=color, s=20, zorder=5, label=label)

        # 연결선 그리기
        for name_a, name_b in connections:
            if name_a in landmarks_dict and name_b in landmarks_dict:
                a = landmarks_dict[name_a]
                b = landmarks_dict[name_b]
                ax.plot([a['x'], b['x']], [a['y'], b['y']], color=color, linewidth=1.5)

    with open(ref_json_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
    with open(comp_json_path, 'r', encoding='utf-8') as f:
        comp_data = json.load(f)

    if frame_idx >= len(ref_data['frames']) or frame_idx >= len(comp_data['frames']):
        raise IndexError(f"Frame index {frame_idx} out of bounds for pose data.")

    ref_landmarks = ref_data['frames'][frame_idx]['landmarks']
    comp_landmarks = comp_data['frames'][frame_idx]['landmarks']

    # 정규화 적용
    if normalize:
        ref_single, *_ = normalizer.normalize_single_frame(ref_landmarks)
        comp_single, *_ = normalizer.normalize_single_frame(comp_landmarks)
        ref_landmarks = ref_single
        comp_landmarks = comp_single

    plt.figure(figsize=(8, 8), dpi=150)
    ax = plt.gca()

    # Y축 반전 (이미지 좌표계와 유사하게)
    ax.invert_yaxis()

    _draw_skeleton(ax, ref_landmarks, POSE_CONNECTIONS, 'blue', 'Reference')
    _draw_skeleton(ax, comp_landmarks, POSE_CONNECTIONS, 'red', 'Comparison')

    ax.set_title(f'Skeleton Comparison at Frame {frame_idx}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.set_aspect('equal', adjustable='box') # 비율 유지
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path

def generate_joint_heatmap(comparison_result: dict, output_path: str) -> str:
    """
    관절별 평균 Loss를 히트맵(바 차트)으로 시각화합니다.
    Loss가 높은 상위 10개 관절만 표시합니다.

    Args:
        comparison_result (dict): comparator.py의 compare_poses()가 반환하는 결과 딕셔너리.
        output_path (str): 생성될 PNG 파일의 저장 경로.

    Returns:
        str: 생성된 PNG 파일의 경로.
    """
    joint_losses = {}
    total_frames = len(comparison_result['per_frame_loss'])

    for frame_result in comparison_result['per_frame_loss']:
        if 'joint_losses' in frame_result:
            for joint_name, loss in frame_result['joint_losses'].items():
                if joint_name not in joint_losses:
                    joint_losses[joint_name] = []
                joint_losses[joint_name].append(loss)

    # 관절별 평균 Loss 계산
    avg_joint_losses = {name: np.mean(losses) for name, losses in joint_losses.items()}

    # Loss가 높은 순서대로 정렬하고 상위 10개 추출
    sorted_joint_losses = sorted(avg_joint_losses.items(), key=lambda item: item[1], reverse=True)[:10]

    if not sorted_joint_losses:
        print("No joint losses found in the comparison result to generate heatmap.")
        # 빈 그래프를 저장하거나, 예외를 발생시키거나, 사용자에게 알림
        plt.figure(figsize=(10, 6), dpi=150)
        plt.title('Joint-wise Average Loss (No Data)')
        plt.text(0.5, 0.5, 'No joint loss data available.', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.savefig(output_path)
        plt.close()
        return output_path


    # 관절 이름으로 라벨 생성
    joint_labels = [name for name, _ in sorted_joint_losses]
    losses = [loss for _, loss in sorted_joint_losses]

    plt.figure(figsize=(10, 6), dpi=150)
    plt.barh(joint_labels, losses, color='skyblue')
    plt.xlabel('Average Loss Value')
    plt.ylabel('Joint Index')
    plt.title('Top 10 Joint-wise Average Losses')
    plt.gca().invert_yaxis() # Loss가 높은 관절이 위로 오도록 정렬
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path
