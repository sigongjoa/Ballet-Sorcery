import pytest
import sys
import os
import numpy as np
import json
from unittest.mock import patch, mock_open, MagicMock

# src 디렉토리를 sys.path에 추가하여 모듈을 임포트할 수 있도록 합니다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# src 모듈 임포트
from beat_loss_analyzer import (
    calculate_frame_loss, 
    calculate_individual_joint_losses, 
    calculate_delta_t_sync_per_beat,
    JOINT_WEIGHTS, # Import JOINT_WEIGHTS for consistent testing
    load_pose_data, # Also test loading behavior
    load_beat_info
)

# Dummy data for testing
DUMMY_REF_FRAME = {
    "world_landmarks": {
        "NOSE": {"x": 0.0, "y": 0.0, "z": 0.0},
        "LEFT_SHOULDER": {"x": 0.1, "y": 0.1, "z": 0.1},
        "RIGHT_SHOULDER": {"x": -0.1, "y": 0.1, "z": 0.1},
        "LEFT_HIP": {"x": 0.05, "y": -0.5, "z": 0.0}, # Lower body joint
        "RIGHT_HIP": {"x": -0.05, "y": -0.5, "z": 0.0}, # Lower body joint
    }
}

DUMMY_CMP_FRAME_PERFECT = {
    "world_landmarks": {
        "NOSE": {"x": 0.0, "y": 0.0, "z": 0.0},
        "LEFT_SHOULDER": {"x": 0.1, "y": 0.1, "z": 0.1},
        "RIGHT_SHOULDER": {"x": -0.1, "y": 0.1, "z": 0.1},
        "LEFT_HIP": {"x": 0.05, "y": -0.5, "z": 0.0},
        "RIGHT_HIP": {"x": -0.05, "y": -0.5, "z": 0.0},
    }
}

DUMMY_CMP_FRAME_DEVIATED = {
    "world_landmarks": {
        "NOSE": {"x": 0.1, "y": 0.0, "z": 0.0}, # Deviated
        "LEFT_SHOULDER": {"x": 0.1, "y": 0.2, "z": 0.1}, # Deviated
        "RIGHT_SHOULDER": {"x": -0.1, "y": 0.1, "z": 0.1},
        "LEFT_HIP": {"x": 0.15, "y": -0.4, "z": 0.0}, # Deviated (lower body)
        "RIGHT_HIP": {"x": -0.05, "y": -0.5, "z": 0.0},
    }
}

DUMMY_LANDMARK_NAMES = list(DUMMY_REF_FRAME["world_landmarks"].keys())

DUMMY_BEAT_INFO = {
    "beat_timestamps_sec": [1.0, 2.0, 3.0],
    "beat_video_frames_24fps": [0, 24, 48, 72], # Corresponds to beats at 1s, 2s, 3s and end frame
    "video_fps_for_mapping": 24
}

@pytest.fixture
def mock_load_pose_data():
    """Fixture to mock load_pose_data."""
    with patch('beat_loss_analyzer.load_pose_data') as mock_func:
        yield mock_func

@pytest.fixture
def mock_load_beat_info():
    """Fixture to mock load_beat_info."""
    with patch('beat_loss_analyzer.load_beat_info') as mock_func:
        yield mock_func

@pytest.mark.unit
def test_calculate_frame_loss_perfect_match():
    """
    Test calculate_frame_loss with perfect matching frames, expecting zero loss.
    """
    loss = calculate_frame_loss(
        DUMMY_REF_FRAME, DUMMY_CMP_FRAME_PERFECT, DUMMY_LANDMARK_NAMES, JOINT_WEIGHTS
    )
    assert loss == pytest.approx(0.0)

@pytest.mark.unit
def test_calculate_frame_loss_deviated_frames():
    """
    Test calculate_frame_loss with deviated frames, expecting non-zero loss.
    """
    loss = calculate_frame_loss(
        DUMMY_REF_FRAME, DUMMY_CMP_FRAME_DEVIATED, DUMMY_LANDMARK_NAMES, JOINT_WEIGHTS
    )
    
    assert loss == pytest.approx(0.0925820068892454)

@pytest.mark.unit
def test_calculate_individual_joint_losses():
    """
    Test calculate_individual_joint_losses with dummy data.
    """
    dummy_ref_poses = [DUMMY_REF_FRAME, DUMMY_REF_FRAME]
    dummy_cmp_poses = [DUMMY_CMP_FRAME_PERFECT, DUMMY_CMP_FRAME_DEVIATED] # First frame perfect, second deviated

    expected_losses = {
        "NOSE": 0.05,
        "LEFT_SHOULDER": 0.05,
        "RIGHT_SHOULDER": 0.0,
        "LEFT_HIP": np.sqrt(0.02) / 2, # L2 distance / number of frames
        "RIGHT_HIP": 0.0,
    }

    avg_losses = calculate_individual_joint_losses(dummy_ref_poses, dummy_cmp_poses, DUMMY_LANDMARK_NAMES)
    
    for joint, expected_loss in expected_losses.items():
        assert avg_losses[joint] == pytest.approx(expected_loss)

@pytest.mark.unit
def test_calculate_delta_t_sync_per_beat():
    """
    Test calculate_delta_t_sync_per_beat with dummy data.
    """
    dummy_frame_losses = np.array([0.1, 0.05, 0.01, 0.1, 0.2, 0.3, 
                                   0.08, 0.04, 0.02, 0.15, 0.25, 
                                   0.03, 0.06, 0.09, 0.12, 0.18]) # total 16 frames
    
    test_beat_info = {
        "beat_timestamps_sec": [2/24, 10/24],
        "beat_video_frames_24fps": [0, 2, 10, 16], # Intervals: [0,1], [2,9], [10,15]
        "video_fps_for_mapping": 24
    }

    # Interval 0: beat_video_frames[0]=0 to beat_video_frames[1]=2. Associated beat timestamp: test_beat_info["beat_timestamps_sec"][0] = 2/24.
    #    frames: [0, 1]. losses: [0.1, 0.05]. Min loss at frame 1. Timestamp = 1/24.
    #    delta_t_sync for interval 0 should be abs(1/24 - 2/24) = 1/24.
    # Interval 1: beat_video_frames[1]=2 to beat_video_frames[2]=10. Associated beat timestamp: test_beat_info["beat_timestamps_sec"][1] = 10/24.
    #    frames: [2, ..., 9]. losses: [0.01, 0.1, 0.2, 0.3, 0.08, 0.04, 0.02, 0.15]. Min loss at frame 2. Timestamp = 2/24.
    #    delta_t_sync for interval 1 should be abs(2/24 - 10/24) = 8/24.
    # Interval 2: beat_video_frames[2]=10 to beat_video_frames[3]=16. No corresponding beat timestamp. Should be 0.0.
    
    delta_t_syncs = calculate_delta_t_sync_per_beat(
        np.array(test_beat_info["beat_video_frames_24fps"]),
        dummy_frame_losses,
        test_beat_info,
        test_beat_info["video_fps_for_mapping"]
    )
    
    expected_delta_t_syncs = [
        pytest.approx(abs(1/24 - 2/24)), # Interval [0,1]
        pytest.approx(abs(2/24 - 10/24)), # Interval [2,9]
        pytest.approx(0.0) # Interval [10,15], no corresponding beat timestamp
    ]
    
    assert len(delta_t_syncs) == len(expected_delta_t_syncs)
    for i in range(len(delta_t_syncs)):
        assert delta_t_syncs[i] == expected_delta_t_syncs[i]

# Mocking file operations for main function test (optional, but good practice)
@pytest.mark.integration
@patch('matplotlib.pyplot.close')                       # top → 6th param = mock_close
@patch('matplotlib.pyplot.figure')                      # → 5th param = mock_figure
@patch('matplotlib.font_manager.FontManager.addfont')   # → 4th param = mock_addfont
@patch('beat_loss_analyzer.plt.savefig')                # → 3rd param = mock_savefig
@patch('beat_loss_analyzer.plt.show')                   # → 2nd param = mock_show
@patch('builtins.open', new_callable=mock_open)         # bottom → 1st param = mock_file_open
def test_main_function_generates_outputs(mock_file_open, mock_show, mock_savefig, mock_addfont, mock_figure, mock_close, mock_load_pose_data, mock_load_beat_info):
    """
    Test that the main function runs without error and calls savefig and json.dump.
    """
    # Configure mocks to return dummy data
    mock_load_pose_data.return_value = [{"frame_idx": i, "world_landmarks": DUMMY_REF_FRAME["world_landmarks"]} for i in range(100)] # Simulate 100 frames
    mock_load_beat_info.return_value = DUMMY_BEAT_INFO
    
    # When plt.figure() is called, it returns 'mock_figure'.
    # We need to ensure that when .savefig() and .close() are called on this 'mock_figure' object,
    # our specific mocks (mock_savefig, mock_close) are called.
    mock_figure.savefig = mock_savefig
    mock_figure.close = mock_close

    # We need to mock os.makedirs if the output_dir is not guaranteed to exist
    with patch('os.makedirs'):
        from beat_loss_analyzer import main
        main()

        # Check that savefig was called for all plots
        # (beat_wise_loss, frame_by_frame_loss, highlighted_error_loss, average_joint_losses, conceptual_loss_jubeop)
        assert mock_savefig.call_count >= 5 # At least 5 plots should be saved

        # Check that json.dump was called for all JSONs
        # (beat_wise_loss.json, average_joint_losses.json, conceptual_loss_jubeop.json)
        write_calls = [call for call in mock_file_open.call_args_list if 'w' in call.args]
        assert len(write_calls) >= 3 # At least 3 JSON files should be written