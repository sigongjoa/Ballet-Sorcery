# -*- coding: utf-8 -*-
"""OpenCV 기반 3D FK 렌더러 (Crash-Free 100% Stable Version).
PyVista(OpenGL)의 세그먼테이션 폴트를 원천 차단하기 위해 순수 수학적 3D->2D 투영을 사용합니다.
"""

import numpy as np
import cv2

COLOR_SELECTED = (50, 50, 255)   # BGR 빨강
COLOR_CHAIN    = (50, 230, 255)  # BGR 노랑
COLOR_DEFAULT  = (150, 150, 150) # BGR 회색
COLOR_GRID     = (50, 50, 50)    # BGR 어두운 회색

class Renderer3D:
    def __init__(self, width: int = 640, height: int = 480):
        self.width  = width
        self.height = height
        self.scale = 200.0 # 미터(m) 단위 좌표를 픽셀 단위로 확대

    def project_3d_to_2d(self, x, y, z):
        """간단한 등각 투영(Orthographic Projection) + 카메라 회전"""
        # 약간의 사선 뷰를 위해 x, z 회전 적용
        theta = np.radians(15) # Y축 회전 (좌우)
        phi = np.radians(10)   # X축 회전 (위아래)
        
        # Y축 회전
        x_rot = x * np.cos(theta) - z * np.sin(theta)
        z_rot = x * np.sin(theta) + z * np.cos(theta)
        
        # X축 회전
        y_rot = y * np.cos(phi) - z_rot * np.sin(phi)
        
        u = int(self.width / 2 + x_rot * self.scale)
        # OpenSim은 Y가 위쪽(+), 화면은 Y가 아래쪽(+)이므로 반전
        v = int(self.height / 2 - (y_rot - 0.9) * self.scale) # 0.9m 골반 높이 기준 보정
        return (u, v)

    def render(self, fk_positions: dict,
               selected_body: str = None,
               chain_bodies: list = None) -> np.ndarray:
        from tools._analyzer.joint_map import FK_SEGMENTS

        # 검은 배경 캔버스
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 1. 바닥 그리드 그리기 (대략적인 평면 투영)
        for i in range(-2, 3):
            p1 = self.project_3d_to_2d(-2, 0, i)
            p2 = self.project_3d_to_2d(2, 0, i)
            cv2.line(img, p1, p2, COLOR_GRID, 1)
            
            p3 = self.project_3d_to_2d(i, 0, -2)
            p4 = self.project_3d_to_2d(i, 0, 2)
            cv2.line(img, p3, p4, COLOR_GRID, 1)

        # 원점 (0,0,0)
        origin_2d = self.project_3d_to_2d(0, 0, 0)
        cv2.circle(img, origin_2d, 4, (255, 255, 255), -1)

        if not fk_positions:
            return self._to_rgba(img)

        chain_bodies = set(chain_bodies or [])

        # 2. 뼈대 세그먼트 그리기
        for a, b in FK_SEGMENTS:
            if a in fk_positions and b in fk_positions:
                pa = fk_positions[a]
                pb = fk_positions[b]
                
                pt1 = self.project_3d_to_2d(pa[0], pa[1], pa[2])
                pt2 = self.project_3d_to_2d(pb[0], pb[1], pb[2])
                
                in_chain = (a in chain_bodies) and (b in chain_bodies)
                color = COLOR_CHAIN if in_chain else COLOR_DEFAULT
                thickness = 3 if in_chain else 1
                cv2.line(img, pt1, pt2, color, thickness)

        # 3. 관절 노드 그리기
        for bname, pos in fk_positions.items():
            pt = self.project_3d_to_2d(pos[0], pos[1], pos[2])
            is_selected = (bname == selected_body)
            in_chain = bname in chain_bodies
            
            radius = 6 if is_selected else (4 if in_chain else 3)
            color = COLOR_SELECTED if is_selected else (COLOR_CHAIN if in_chain else COLOR_DEFAULT)
            
            cv2.circle(img, pt, radius, color, -1)

        return self._to_rgba(img)

    def _to_rgba(self, img_bgr) -> np.ndarray:
        rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
        return rgba

    def close(self):
        pass
