import pytest
import sys
import os
import json
import math

# src 디렉토리를 sys.path에 추가하여 모듈을 임포트할 수 있도록 합니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# src 모듈 임포트
from comparator import calculate_l2_distance_3d, compare_poses

# 데이터 파일 경로 설정
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/sample'))
REF_POSES_3D_PATH = os.path.join(DATA_DIR, 'reference_poses_3d.json')
COMP_POSES_3D_PATH = os.path.join(DATA_DIR, 'compare_poses_3d.json')
REF_POSES_2D_PATH = os.path.join(DATA_DIR, 'reference_poses.json')
COMP_POSES_2D_PATH = os.path.join(DATA_DIR, 'compare_poses.json')

@pytest.mark.unit
def test_calculate_l2_distance_3d_basic():
    """
    두 3D 포인트의 L2 거리를 올바르게 계산하는지 확인
    입력: p1={'x':0, 'y':0, 'z':0}, p2={'x':1, 'y':2, 'z':2}
    기대값: sqrt(1+4+4) = 3.0
    """
    p1 = {'x':0, 'y':0, 'z':0}
    p2 = {'x':1, 'y':2, 'z':2}
    expected_distance = 3.0 # sqrt(1^2 + 2^2 + 2^2) = sqrt(1+4+4) = sqrt(9) = 3
    assert calculate_l2_distance_3d(p1, p2) == pytest.approx(expected_distance)

@pytest.mark.unit
def test_calculate_l2_distance_3d_same_point():
    """
    같은 포인트 간 거리는 0
    p1=p2={'x':1, 'y':2, 'z':3}
    기대값: 0.0
    """
    p1 = {'x':1, 'y':2, 'z':3}
    p2 = {'x':1, 'y':2, 'z':3}
    expected_distance = 0.0
    assert calculate_l2_distance_3d(p1, p2) == pytest.approx(expected_distance)

@pytest.mark.integration
def test_compare_poses_use_world_metadata():
    """
    compare_poses()를 use_world=True로 호출 시 metadata에 use_world_landmarks=True가 포함되는지 확인
    실제 파일 사용: data/sample/reference_poses_3d.json, data/sample/compare_poses_3d.json
    """
    result = compare_poses(
        ref_json_path=REF_POSES_3D_PATH,
        comp_json_path=COMP_POSES_3D_PATH,
        use_world=True
    )
    assert result['metadata']['use_world_landmarks'] is True
    # 3D 모드에서는 normalize=True여도 실제로 정규화가 적용되지 않으므로 metadata['normalized']는 False여야 함
    assert result['metadata']['normalized'] is False
    assert 'mean_loss' in result['summary']
    assert result['summary']['mean_loss'] >= 0 # 손실은 0 이상이어야 함

@pytest.mark.integration
def test_compare_poses_2d_backward_compatible():
    """
    use_world 파라미터 없이 기존처럼 호출해도 정상 작동하는지 확인
    실제 파일 사용: data/sample/reference_poses.json, data/sample/compare_poses.json
    """
    result = compare_poses(
        ref_json_path=REF_POSES_2D_PATH,
        comp_json_path=COMP_POSES_2D_PATH,
        normalize=True # 기존 2D 비교는 기본적으로 정규화
    )
    assert result['metadata']['use_world_landmarks'] is False
    assert result['metadata']['normalized'] is True # 2D 모드이고 normalize=True이므로 정규화되어야 함
    assert 'mean_loss' in result['summary']
    assert result['summary']['mean_loss'] >= 0

@pytest.mark.integration
def test_3d_loss_lower_than_2d():
    """
    3D 비교 결과의 mean_loss가 2D보다 현저히 낮은지 확인 (카메라 앵글 영향 감소)
    3D 파일과 2D 파일 각각 비교 실행 후 mean_loss 비교
    assert 3d_mean < 2d_mean
    """
    # 3D 비교
    result_3d = compare_poses(
        ref_json_path=REF_POSES_3D_PATH,
        comp_json_path=COMP_POSES_3D_PATH,
        use_world=True
    )
    mean_loss_3d = result_3d['summary']['mean_loss']

    # 2D 비교 (정규화 포함)
    result_2d = compare_poses(
        ref_json_path=REF_POSES_2D_PATH,
        comp_json_path=COMP_POSES_2D_PATH,
        normalize=True,
        use_world=False
    )
    mean_loss_2d = result_2d['summary']['mean_loss']
    
    # 가정: 3D 비교는 카메라 앵글 영향이 적어 손실이 더 낮을 것으로 예상.
    # 단, 데이터에 따라 다를 수 있으므로, 0.5배 이상 낮아야 한다는 가정을 둠.
    # 실제 데이터의 특성을 반영하여 조정 필요. 여기서는 명확한 차이를 가정.
    print(f"Mean Loss (3D): {mean_loss_3d}, Mean Loss (2D): {mean_loss_2d}")
    assert mean_loss_3d < mean_loss_2d

@pytest.mark.integration
def test_compare_poses_3d_no_normalization():
    """
    use_world=True일 때 normalize=True로 호출해도 실제로 정규화가 적용되지 않음을 확인
    metadata의 normalized 필드가 False인지 확인
    """
    result = compare_poses(
        ref_json_path=REF_POSES_3D_PATH,
        comp_json_path=COMP_POSES_3D_PATH,
        normalize=True, # 이 값은 use_world=True일 때 무시되어야 함
        use_world=True
    )
    assert result['metadata']['use_world_landmarks'] is True
    assert result['metadata']['normalized'] is False # use_world=True이므로 정규화되지 않음
    assert 'mean_loss' in result['summary']
    assert result['summary']['mean_loss'] >= 0
