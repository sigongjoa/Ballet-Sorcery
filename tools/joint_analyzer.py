# -*- coding: utf-8 -*-
"""Ballet Joint Analyzer — Integrated Pro Platform (Fixed)."""

import sys
import os
import argparse
import math
import time
import cv2
import numpy as np
import dearpygui.dearpygui as dpg

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools._analyzer.joint_map import BODY_NAMES, DOF_INFO, get_ancestors, get_chain_bodies
from tools._analyzer.ui_state import UIState
from tools._analyzer.pipeline import JointPipeline
from tools._analyzer.renderer3d import Renderer3D

class JointAnalyzerApp:
    def __init__(self, video_path):
        self.state = UIState()
        self.video_path = video_path
        self.output_dir = os.path.join("reports", "joint_analysis_result")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. 오프라인 분석 수행 (창 뜨기 전에 모든 계산 완료)
        self.pipeline = JointPipeline()
        self.pipeline.run_offline_analysis(video_path, self.output_dir)
        
        # 2. 결과 시청용 렌더러 및 캡처 초기화
        self.renderer = Renderer3D(width=640, height=480)
        self.cap = cv2.VideoCapture(video_path)
        self.state.total_frames = len(self.pipeline.analysis_data['frames'])
        self.state.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        print(f"[Platform] 분석 완료. 뷰어 모드를 시작합니다. ({self.state.total_frames} frames)")

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Ballet Joint Analyzer [Pro]", width=1600, height=950)
        
        # 텍스처 등록
        init_data = [0.0] * (640 * 480 * 4)
        with dpg.texture_registry():
            dpg.add_raw_texture(640, 480, init_data, tag="video_tex", format=dpg.mvFormat_Float_rgba)
            dpg.add_raw_texture(640, 480, init_data, tag="fk3d_tex", format=dpg.mvFormat_Float_rgba)

        with dpg.window(tag="main", no_title_bar=True):
            dpg.add_text("STATUS: READY | Professional Dashboard", tag="status_text", color=(100, 255, 100))
            
            with dpg.table(header_row=False, borders_outerV=True, borders_innerV=True):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=640)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=640)
                dpg.add_table_column()

                with dpg.table_row():
                    with dpg.group():
                        dpg.add_text("VIDEO VIEW", color=(150, 255, 150))
                        dpg.add_image("video_tex")
                    with dpg.group():
                        dpg.add_text("OPEN-SIM 3D VIEW", color=(150, 150, 255))
                        dpg.add_image("fk3d_tex")
                    with dpg.group():
                        self._build_inspector_panel()

            dpg.add_separator()
            self._build_timeline_panel()

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)

        # 첫 프레임 로드
        self._update_frame(0)
        self._on_body_selected(self.state.selected_body)

        last_time = time.time()
        while dpg.is_dearpygui_running():
            now = time.time()
            if self.state.playing and (now - last_time) >= (1.0 / self.state.fps):
                self.state.frame_idx = (self.state.frame_idx + 1) % self.state.total_frames
                self._update_frame(self.state.frame_idx)
                last_time = now
            dpg.render_dearpygui_frame()

        self._cleanup()

    def _update_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret: return

        data = self.pipeline.get_frame_data(idx)
        self.state.coords = data.get('coords', {})
        merged_coords = {**self.state.coords, **self.state.manual_overrides}
        fk_pos = self.pipeline.compute_fk_for_coords(merged_coords)

        # 텍스처 업데이트
        skel_frame = self._draw_2d_skeleton(frame, data.get('landmarks_norm'))
        dpg.set_value("video_tex", self._img_to_list(skel_frame))
        
        chain = list(set(get_ancestors(self.state.selected_body)) | set(get_chain_bodies(self.state.selected_body)))
        fk_img = self.renderer.render(fk_pos, selected_body=self.state.selected_body, chain_bodies=chain)
        dpg.set_value("fk3d_tex", self._img_to_list(fk_img))

        dpg.set_value("status_text", f"VIEWING FRAME: {idx} / {self.state.total_frames - 1}")
        dpg.set_value("timeline", idx)
        self._update_inspector_ui(merged_coords, fk_pos)

    def _draw_2d_skeleton(self, frame, landmarks):
        out = frame.copy()
        if landmarks:
            h, w = frame.shape[:2]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            # 기본 관절 연결선
            for s, e in [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]:
                if s < len(pts) and e < len(pts):
                    cv2.line(out, pts[s], pts[e], (0, 255, 0), 2)
        return out

    def _img_to_list(self, img):
        resized = cv2.resize(img, (640, 480))
        rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA) if img.shape[2] == 3 else resized
        return np.ascontiguousarray(rgba, dtype=np.float32).flatten() / 255.0

    def _build_inspector_panel(self):
        dpg.add_text("── JOINT ANALYSIS ──", color=(255, 200, 50))
        dpg.add_combo(BODY_NAMES, default_value=self.state.selected_body, tag="body_selector",
                      callback=lambda s, a: self._on_body_selected(a), width=280)
        dpg.add_separator()
        dpg.add_text("  X: 0.000", tag="pos_x")
        dpg.add_text("  Y: 0.000", tag="pos_y")
        dpg.add_text("  Z: 0.000", tag="pos_z")
        dpg.add_separator()
        dpg.add_text("── DOF OVERRIDE ──", color=(255, 200, 50))
        dpg.add_group(tag="dof_sliders")
        dpg.add_separator()
        with dpg.table(tag="dof_table", header_row=True, height=300, scrollY=True):
            dpg.add_table_column(label="Name", width_fixed=True, init_width_or_weight=140)
            dpg.add_table_column(label="Angle", width_fixed=True, init_width_or_weight=60)
            for dof in sorted(DOF_INFO.keys()):
                with dpg.table_row():
                    dpg.add_text(dof)
                    dpg.add_text("0.0\u00b0", tag=f"val_{dof}")

    def _build_timeline_panel(self):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play/Pause", callback=self._toggle_play)
            dpg.add_slider_int(min_value=0, max_value=max(0, self.state.total_frames - 1),
                               tag="timeline", width=1100, callback=lambda s, a: self._update_frame(a))

    def _update_inspector_ui(self, coords, fk_pos):
        pos = fk_pos.get(self.state.selected_body, [0,0,0])
        dpg.set_value("pos_x", f"  X: {pos[0]:+.3f}"); dpg.set_value("pos_y", f"  Y: {pos[1]:+.3f}"); dpg.set_value("pos_z", f"  Z: {pos[2]:+.3f}")
        for dof, val in coords.items():
            if dof in DOF_INFO:
                try: dpg.set_value(f"val_{dof}", f"{math.degrees(val):+.1f}\u00b0")
                except: pass

    def _on_body_selected(self, body):
        self.state.selected_body = body
        dpg.delete_item("dof_sliders", children_only=True)
        for dof, info in DOF_INFO.items():
            if info.get('body') == body:
                dpg.add_slider_float(label=f" {dof}", min_value=float(info['range'][0]), max_value=float(info['range'][1]), 
                                     default_value=float(self.state.coords.get(dof, 0.0)), parent="dof_sliders", width=280,
                                     callback=lambda s, a, u=dof: self._on_slider_change(u, a))
        self._update_frame(self.state.frame_idx)

    def _on_slider_change(self, dof, val):
        self.state.manual_overrides[dof] = val
        self._update_frame(self.state.frame_idx)

    def _toggle_play(self): self.state.playing = not self.state.playing
    def _cleanup(self):
        self.cap.release()
        dpg.destroy_context()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    JointAnalyzerApp(args.video).run()
