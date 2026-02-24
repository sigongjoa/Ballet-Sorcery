# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for Ballet Jubeop tests.

This file configures pytest to use SuperClaude PM Agent patterns:
- ConfidenceChecker: Pre-execution confidence assessment
- SelfCheckProtocol: Post-implementation validation
- ReflexionPattern: Cross-session error learning
- TokenBudget: Resource allocation

SuperClaude fixtures are auto-loaded via the pytest plugin entry point.
"""

import pytest
import sys
import os

# Add src directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


# ============================================================================
# Ballet-Specific Fixtures
# ============================================================================

@pytest.fixture
def sample_video_metadata():
    """
    Provides sample video metadata for testing.
    """
    return {
        "fps": 24,
        "width": 1920,
        "height": 1080,
        "duration": 10.0,
        "total_frames": 240
    }


@pytest.fixture
def mediapipe_model_path():
    """
    Provides the path to MediaPipe pose landmarker model.
    """
    return os.path.join("data", "models", "pose_landmarker_lite.task")


@pytest.fixture
def sample_pose_landmarks():
    """
    Provides a sample set of normalized pose landmarks for testing.
    
    This represents a single frame with all 33 MediaPipe pose landmarks.
    """
    # Simplified landmark set for testing
    landmarks = {
        "NOSE": {"x": 0.5, "y": 0.3, "z": 0.0, "visibility": 0.99},
        "LEFT_SHOULDER": {"x": 0.4, "y": 0.5, "z": 0.0, "visibility": 0.95},
        "RIGHT_SHOULDER": {"x": 0.6, "y": 0.5, "z": 0.0, "visibility": 0.95},
        "LEFT_HIP": {"x": 0.45, "y": 0.7, "z": 0.0, "visibility": 0.90},
        "RIGHT_HIP": {"x": 0.55, "y": 0.7, "z": 0.0, "visibility": 0.90},
        "LEFT_KNEE": {"x": 0.45, "y": 0.85, "z": 0.0, "visibility": 0.85},
        "RIGHT_KNEE": {"x": 0.55, "y": 0.85, "z": 0.0, "visibility": 0.85},
        "LEFT_ANKLE": {"x": 0.45, "y": 0.95, "z": 0.0, "visibility": 0.80},
        "RIGHT_ANKLE": {"x": 0.55, "y": 0.95, "z": 0.0, "visibility": 0.80},
    }
    return landmarks


@pytest.fixture
def ballet_pose_grand_battement():
    """
    Provides a sample Grand Battement pose for testing.
    
    Grand Battement is a large, controlled leg kick - ideal for testing
    because it has clear joint angle changes and is easy to measure.
    """
    return {
        "pose_name": "Grand Battement",
        "landmarks": {
            "NOSE": {"x": 0.5, "y": 0.2, "z": 0.0, "visibility": 0.99},
            "LEFT_SHOULDER": {"x": 0.4, "y": 0.4, "z": 0.0, "visibility": 0.95},
            "RIGHT_SHOULDER": {"x": 0.6, "y": 0.4, "z": 0.0, "visibility": 0.95},
            "LEFT_HIP": {"x": 0.45, "y": 0.6, "z": 0.0, "visibility": 0.90},
            "RIGHT_HIP": {"x": 0.55, "y": 0.6, "z": 0.0, "visibility": 0.90},
            # Left leg extended upward (Grand Battement)
            "LEFT_KNEE": {"x": 0.4, "y": 0.3, "z": 0.0, "visibility": 0.85},
            "LEFT_ANKLE": {"x": 0.35, "y": 0.15, "z": 0.0, "visibility": 0.80},
            # Right leg supporting
            "RIGHT_KNEE": {"x": 0.55, "y": 0.75, "z": 0.0, "visibility": 0.85},
            "RIGHT_ANKLE": {"x": 0.55, "y": 0.9, "z": 0.0, "visibility": 0.80},
        },
        "expected_characteristics": {
            "left_leg_angle": 120,  # degrees from vertical
            "right_leg_straight": True,
            "balance_point": "right_foot"
        }
    }


# ============================================================================
# Test Collection Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Automatically add markers based on test location and naming.
    
    This follows SuperClaude's auto-marking pattern:
    - Tests in /unit/ → @pytest.mark.unit
    - Tests in /integration/ → @pytest.mark.integration
    """
    for item in items:
        # Get relative path from tests directory
        rel_path = os.path.relpath(item.fspath, start=config.rootdir / "tests")
        
        # Auto-mark based on directory structure
        if rel_path.startswith("unit"):
            item.add_marker(pytest.mark.unit)
        elif rel_path.startswith("integration"):
            item.add_marker(pytest.mark.integration)


# ============================================================================
# SuperClaude PM Agent Fixtures
# ============================================================================

# Note: The following fixtures are auto-loaded by the SuperClaude pytest plugin:
# - confidence_checker: Pre-execution confidence assessment
# - self_check_protocol: Post-implementation validation
# - reflexion_pattern: Cross-session error learning
# - token_budget: Token allocation based on complexity
# - pm_context: PM Agent context manager

# These fixtures are available in all tests without explicit import.
# See SuperClaude documentation for usage examples.


# ============================================================================
# Shared Fixtures from test_phase0.py
# ============================================================================

@pytest.fixture
def sample_landmarks_basic():
    """
    기본 정규화 테스트를 위한 샘플 랜드마크 데이터를 제공합니다.
    골반 중심이 (0,0,0) 근처로 이동해야 합니다.
    """
    POSE_LANDMARKS_TEST = [
        "NOSE", "MOUTH_LEFT", "RIGHT_SHOULDER", "LEFT_SHOULDER", "RIGHT_HIP", "LEFT_HIP"
    ]
    landmarks_list = [
        [0.5, 0.5, 0.5],  # NOSE
        [0.5, 0.6, 0.5],  # MOUTH_LEFT
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
    POSE_LANDMARKS_TEST = [
        "NOSE", "MOUTH_LEFT", "RIGHT_SHOULDER", "LEFT_SHOULDER", "RIGHT_HIP", "LEFT_HIP"
    ]
    landmarks_list = [
        [0.5, 0.5, 0.5],  # NOSE
        [0.5, 0.6, 0.5],  # MOUTH_LEFT
        [0.0, 0.0, 0.0],  # RIGHT_SHOULDER (missing)
        [0.0, 0.0, 0.0],  # LEFT_SHOULDER (missing)
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
