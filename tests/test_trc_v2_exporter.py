# -*- coding: utf-8 -*-
import os
import sys
import pytest
import numpy as np
from typing import List, Dict

# 프로젝트 루트를 sys.path에 추가하여 src 모듈 임포트 가능하게 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.trc_v2_exporter import export_trc_v2, MARKER_ORDER, NUM_MARKERS

def get_dummy_markers_sequence(n_frames: int, base_xyz: List[float] = None) -> List[Dict[str, List[float]]]:
    """테스트용 56개 마커 더미 데이터 시퀀스 생성 헬퍼 함수."""
    if base_xyz is None:
        base_xyz = [0.1, 1.0, 0.0]
    
    sequence = []
    for _ in range(n_frames):
        frame_data = {name: list(base_xyz) for name in MARKER_ORDER}
        sequence.append(frame_data)
    return sequence

def test_header_format(tmp_path):
    """TRC 파일 헤더의 기본 포맷(1행, 3행) 검증."""
    output_path = tmp_path / "test_header.trc"
    n_frames = 5
    fps = 30.0
    sequence = get_dummy_markers_sequence(n_frames)
    
    export_trc_v2(sequence, str(output_path), fps=fps)
    
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 1행: PathFileType	4	(X/Y/Z)	파일명
    assert lines[0].startswith("PathFileType"), "1행은 PathFileType으로 시작해야 함"
    assert "(X/Y/Z)" in lines[0], "1행에 (X/Y/Z) 포맷 표시가 포함되어야 함"
    
    # 3행: NumMarkers 자리에 56이 포함되어 있는지 확인
    # DataRate CameraRate NumFrames NumMarkers Units ...
    header_values = lines[2].split('	')
    assert int(header_values[3]) == NUM_MARKERS, f"3행의 NumMarkers 값은 {NUM_MARKERS}여야 함"
    assert int(header_values[2]) == n_frames, "3행의 NumFrames 값이 일치하지 않음"

def test_marker_count_in_header(tmp_path):
    """헤더 4행에 MARKER_ORDER의 모든 마커명이 존재하는지 확인."""
    output_path = tmp_path / "test_markers.trc"
    sequence = get_dummy_markers_sequence(1)
    
    export_trc_v2(sequence, str(output_path))
    
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    marker_line = lines[3]
    for marker_name in MARKER_ORDER:
        assert marker_name in marker_line, f"마커명 '{marker_name}'이 헤더 4행에 누락됨"

def test_feet_on_floor_offset(tmp_path):
    """지면 오프셋 계산 검증: HEEL_POST_L y=-0.05m -> y_offset ≈ 0.06."""
    output_path = tmp_path / "test_offset.trc"
    sequence = get_dummy_markers_sequence(5)
    
    # 특정 프레임의 특정 마커 Y 좌표를 낮게 설정
    # export_trc_v2는 모든 프레임의 모든 ground_markers 중 최솟값을 기준으로 오프셋 계산
    for frame in sequence:
        frame['HEEL_POST_L'][1] = -0.05
    
    # min_y = -0.05 일 때, y_offset = -(min_y - 0.01) = -(-0.05 - 0.01) = 0.06
    _, y_offset = export_trc_v2(sequence, str(output_path))
    
    expected_offset = 0.06
    assert abs(y_offset - expected_offset) < 0.001, f"y_offset 기대값 {expected_offset} 대비 오차 과다: {y_offset}"

def test_data_rows_count(tmp_path):
    """데이터 행 개수 검증: 헤더 6행 + 데이터 N행 = 총 N+6행."""
    n_frames = 10
    output_path = tmp_path / "test_rows.trc"
    sequence = get_dummy_markers_sequence(n_frames)
    
    export_trc_v2(sequence, str(output_path))
    
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 헤더 5행 + 빈 줄 1행 + 데이터 10행 = 16행
    expected_total_lines = n_frames + 6
    assert len(lines) == expected_total_lines, f"전체 행 개수가 {expected_total_lines}개가 아님 (실제: {len(lines)})"
    
    # 마지막 데이터 행의 인덱스 확인
    last_row = lines[-1].split('	')
    assert int(last_row[0]) == n_frames, "마지막 행의 Frame# 인덱스가 n_frames와 일치하지 않음"
