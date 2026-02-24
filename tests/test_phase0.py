# -*- coding: utf-8 -*-
# pytest 테스트 파일을 생성합니다.

import pytest
from unittest import mock
import sys
import os
import json

# src 디렉토리를 sys.path에 추가하여 모듈을 임포트할 수 있도록 합니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# src 모듈 임포트
from pose_extractor import extract_poses
from normalizer import normalize_poses, normalize_single_frame

# 테스트용 더미 비디오 파일 경로
DUMMY_VIDEO_PATH = "non_existent_video.mp4"
DUMMY_OUTPUT_JSON_PATH = "/tmp/output_pose_data.json" # 실제 파일 생성 방지를 위해 /tmp 사용

# Fixture for sample pose data for normalizer tests
@pytest.fixture
def sample_landmarks_basic():
    """
    기본 정규화 테스트를 위한 샘플 랜드마크 데이터를 제공합니다.
    골반 중심이 (0,0,0) 근처로 이동해야 합니다.
    """
    # MediaPipe PoseLandmark 이름을 직접 정의 (environment issue, or for explicit mapping)
    POSE_LANDMARKS_TEST = [
        "NOSE", "MOUTH_LEFT", "RIGHT_SHOULDER", "LEFT_SHOULDER", "RIGHT_HIP", "LEFT_HIP"
    ]
    landmarks_list = [
        [0.5, 0.5, 0.5],  # NOSE
        [0.5, 0.6, 0.5],  # MOUTH_LEFT (acting as neck for simplicity)
        [0.4, 0.6, 0.4],  # RIGHT_SHOULDER
        [0.6, 0.6, 0.6],  # LEFT_SHOULDER
        [0.45, 0.8, 0.45], # RIGHT_HIP
        [0.55, 0.8, 0.55]  # LEFT_HIP
    ]
    landmarks_dict = {}
    for i, name in enumerate(POSE_LANDMARKS_TEST):
        landmarks_dict[name] = {
            "x": landmarks_list[i][0],
            "y": landmarks_list[i][1],
            "z": landmarks_list[i][2],
            "visibility": 0.9
        }
    return landmarks_dict

@pytest.fixture
def sample_landmarks_no_shoulder():
    """
    어깨 랜드마크가 없는 경우를 위한 샘플 데이터를 제공합니다.
    """
    # MediaPipe PoseLandmark 이름을 직접 정의 (environment issue, or for explicit mapping)
    POSE_LANDMARKS_TEST = [
        "NOSE", "MOUTH_LEFT", "RIGHT_SHOULDER", "LEFT_SHOULDER", "RIGHT_HIP", "LEFT_HIP"
    ]
    landmarks_list = [
        [0.5, 0.5, 0.5],  # NOSE
        [0.5, 0.6, 0.5],  # MOUTH_LEFT (acting as neck for simplicity)
        [0.0, 0.0, 0.0],  # RIGHT_SHOULDER (missing or 0)
        [0.0, 0.0, 0.0],  # LEFT_SHOULDER (missing or 0)
        [0.45, 0.8, 0.45], # RIGHT_HIP
        [0.55, 0.8, 0.55]  # LEFT_HIP
    ]
    landmarks_dict = {}
    for i, name in enumerate(POSE_LANDMARKS_TEST):
        visibility = 0.0 if name in ["RIGHT_SHOULDER", "LEFT_SHOULDER"] else 0.9
        landmarks_dict[name] = {
            "x": landmarks_list[i][0],
            "y": landmarks_list[i][1],
            "z": landmarks_list[i][2],
            "visibility": visibility
        }
    return landmarks_dict

@pytest.fixture
def sample_pose_data_full(sample_landmarks_basic):
    """
    여러 프레임으로 구성된 전체 포즈 데이터를 제공합니다.
    """
    frame1 = {
        "frame_id": 1,
        "landmarks": sample_landmarks_basic,
        "metadata": {"timestamp": 0.04}
    }
    # 두 번째 프레임은 첫 번째 프레임에서 약간 이동한 것으로 가정
    frame2_landmarks = {}
    for name, coords in sample_landmarks_basic.items():
        frame2_landmarks[name] = {
            "x": coords["x"] + 0.01,
            "y": coords["y"] + 0.01,
            "z": coords["z"] + 0.01,
            "visibility": coords["visibility"]
        }
    frame2 = {
        "frame_id": 2,
        "landmarks": frame2_landmarks,
        "metadata": {"timestamp": 0.08}
    }
    return {
        "metadata": {"video_path": "dummy.mp4", "fps": 25},
        "frames": [frame1, frame2]
    }

# 1. test_extract_poses_file_not_found: 존재하지 않는 파일 → FileNotFoundError 발생 확인
@pytest.mark.reflexion  # Learn from file handling errors
@mock.patch('os.path.exists', return_value=False)
@mock.patch('cv2.VideoCapture')
def test_extract_poses_file_not_found(mock_video_capture, mock_exists):
    """
    존재하지 않는 비디오 파일 경로에 대해 extract_poses가 FileNotFoundError를 발생시키는지 확인합니다.
    
    Uses reflexion pattern to learn from file handling errors.
    """
    with pytest.raises(FileNotFoundError):
        extract_poses(DUMMY_VIDEO_PATH, DUMMY_OUTPUT_JSON_PATH)
    mock_exists.assert_called_with(DUMMY_VIDEO_PATH)
    mock_video_capture.assert_not_called() # 비디오 캡처가 시도되지 않아야 합니다.

# 2. test_normalize_single_frame_basic: 샘플 랜드마크로 정규화 후 골반 중심이 (0,0,0)에 가까운지 확인
@pytest.mark.self_check  # Validate normalization with evidence
@pytest.mark.complexity("medium")  # Medium complexity: custom algorithm
def test_normalize_single_frame_basic(sample_landmarks_basic):
    """
    기본 랜드마크 데이터를 사용하여 normalize_single_frame을 테스트하고,
    정규화된 골반 중심이 원점 (0,0,0)에 가까운지 확인합니다.
    
    Uses self-check protocol to validate normalization with evidence.
    """
    normalized_landmarks, scale, _ = normalize_single_frame( # _ for actual_pelvis_center
        sample_landmarks_basic, prev_scale=1.0 # 초기 prev_scale은 중요하지 않음
    )

    # 정규화된 랜드마크에서 골반 중심 계산
    right_hip_norm = normalized_landmarks["RIGHT_HIP"]
    left_hip_norm = normalized_landmarks["LEFT_HIP"]
    
    norm_pelvis_center_x = (right_hip_norm["x"] + left_hip_norm["x"]) / 2
    norm_pelvis_center_y = (right_hip_norm["y"] + left_hip_norm["y"]) / 2
    norm_pelvis_center_z = (right_hip_norm["z"] + left_hip_norm["z"]) / 2

    # 골반 중심이 0에 가까운지 확인 (부동 소수점 오차 고려)
    assert abs(norm_pelvis_center_x) < 1e-6
    assert abs(norm_pelvis_center_y) < 1e-6
    assert abs(norm_pelvis_center_z) < 1e-6
    
    # 스케일이 0이 아닌 유효한 값인지 확인
    assert scale > 0

# 3. test_normalize_single_frame_no_shoulder: 어깨 없는 데이터에서 prev_scale이 사용되는지 확인
@pytest.mark.reflexion  # Learn from edge case handling
@pytest.mark.complexity("medium")  # Medium complexity: edge case handling
def test_normalize_single_frame_no_shoulder(sample_landmarks_no_shoulder):
    """
    어깨 랜드마크가 없는 경우 normalize_single_frame이 prev_scale을 사용하여 정규화를 수행하는지 확인합니다.
    
    Uses reflexion pattern to learn from edge case handling.
    """
    initial_prev_scale = 10.0 # 의도적으로 큰 prev_scale 값 설정

    normalized_landmarks, scale, pelvis_center = normalize_single_frame(
        sample_landmarks_no_shoulder, prev_scale=initial_prev_scale
    )

    # 어깨 랜드마크가 0이므로, 새로운 스케일 계산 대신 prev_scale이 사용되었는지 확인
    # 스케일은 prev_scale과 동일하거나 매우 유사해야 합니다.
    assert abs(scale - initial_prev_scale) < 1e-6

    # 정규화된 랜드마크가 유효한지 (모두 0은 아닌지) 확인
    # 랜드마크 딕셔너리를 순회하도록 수정
    assert any(any(lm_data[coord] != 0 for coord in ['x', 'y', 'z']) for lm_data in normalized_landmarks.values())

# 4. test_normalize_poses_full: 여러 프레임의 정규화가 순차적으로 올바르게 적용되는지 확인
@pytest.mark.self_check  # Validate multi-frame normalization with evidence
@pytest.mark.complexity("complex")  # Complex: multi-frame sequential processing
def test_normalize_poses_full(sample_pose_data_full):
    """
    여러 프레임으로 구성된 전체 포즈 데이터에 대해 normalize_poses를 테스트하고,
    각 프레임이 순차적으로 올바르게 정규화되는지 확인합니다.
    
    Uses self-check protocol to validate multi-frame normalization with evidence.
    """
    initial_pose_data = json.loads(json.dumps(sample_pose_data_full)) # 원본 데이터 보존
    normalized_data = normalize_poses(sample_pose_data_full)

    # 첫 번째 프레임의 스케일과 골반 중심 추출 (normalize_single_frame 동작 확인용)
    _, scale_frame1, pelvis_center_frame1 = normalize_single_frame(
        initial_pose_data["frames"][0]["landmarks"], prev_scale=1.0
    )

    # 두 번째 프레임의 스케일과 골반 중심 추출 (이때는 prev_scale로 첫 번째 프레임의 스케일 사용)
    _, scale_frame2, pelvis_center_frame2 = normalize_single_frame(
        initial_pose_data["frames"][1]["landmarks"], prev_scale=scale_frame1
    )

    # normalize_poses 결과와 개별 프레임 정규화 결과 비교
    assert len(normalized_data["frames"]) == len(initial_pose_data["frames"])

    # 첫 번째 프레임 결과 확인
    assert normalized_data["frames"][0]["normalized_scale"] == pytest.approx(scale_frame1)
    assert normalized_data["frames"][0]["normalized_pelvis_center"][0] == pytest.approx(pelvis_center_frame1[0])
    assert normalized_data["frames"][0]["normalized_pelvis_center"][1] == pytest.approx(pelvis_center_frame1[1])
    assert normalized_data["frames"][0]["normalized_pelvis_center"][2] == pytest.approx(pelvis_center_frame1[2])

    # 두 번째 프레임 결과 확인
    assert normalized_data["frames"][1]["normalized_scale"] == pytest.approx(scale_frame2)
    assert normalized_data["frames"][1]["normalized_pelvis_center"][0] == pytest.approx(pelvis_center_frame2[0])
    assert normalized_data["frames"][1]["normalized_pelvis_center"][1] == pytest.approx(pelvis_center_frame2[1])
    assert normalized_data["frames"][1]["normalized_pelvis_center"][2] == pytest.approx(pelvis_center_frame2[2])


# 5. test_normalize_poses_metadata: 정규화 후 metadata에 normalized=True가 있는지 확인
@pytest.mark.complexity("simple")  # Simple: metadata validation
def test_normalize_poses_metadata(sample_pose_data_full):
    """
    normalize_poses 함수 호출 후 반환된 데이터의 메타데이터에 'normalized': True가 설정되었는지 확인합니다.
    
    Simple test with minimal token budget.
    """
    normalized_data = normalize_poses(sample_pose_data_full)
    assert "normalized" in normalized_data["metadata"]
    assert normalized_data["metadata"]["normalized"] is True

    # 각 프레임 메타데이터에도 normalized_scale, normalized_pelvis_center 가 추가되었는지 확인
    for frame in normalized_data["frames"]:
        assert "normalized_scale" in frame
        assert "normalized_pelvis_center" in frame
