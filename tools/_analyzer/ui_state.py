# -*- coding: utf-8 -*-
"""공유 UI 상태 (현재 프레임, 선택 관절, FK 좌표)."""

from dataclasses import dataclass, field


@dataclass
class UIState:
    frame_idx: int = 0          # 현재 프레임 번호
    total_frames: int = 0       # 전체 프레임 수
    selected_body: str = 'tibia_r'  # 클릭/드롭다운으로 선택된 body
    coords: dict = field(default_factory=dict)           # OpenSim DOF 값 (라디안)
    fk_positions: dict = field(default_factory=dict)     # body명 → [x,y,z] (m)
    manual_overrides: dict = field(default_factory=dict) # 슬라이더 수동 조정값
    playing: bool = False       # 재생 중 여부
    fps: float = 30.0           # 영상 FPS
