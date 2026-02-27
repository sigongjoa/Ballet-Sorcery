# -*- coding: utf-8 -*-
"""UI가 크래시 없이 최소 프레임을 렌더링하는지 확인하는 스트레스 테스트 (Fix)."""

import sys
import os
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.joint_analyzer import JointAnalyzerApp
import dearpygui.dearpygui as dpg

def run_stress_test(video_path):
    print("\n[Stress-Test] UI Verification Started: " + video_path)
    
    try:
        app = JointAnalyzerApp(video_path)
        dpg.create_context()
        
        init_data = [0.0] * (640 * 480 * 4)
        with dpg.texture_registry():
            dpg.add_raw_texture(640, 480, init_data, tag="video_tex", format=dpg.mvFormat_Float_rgba)
            dpg.add_raw_texture(640, 480, init_data, tag="fk3d_tex", format=dpg.mvFormat_Float_rgba)

        with dpg.window(tag="main"):
            dpg.add_text("Testing...", tag="status_text")
            with dpg.table(header_row=False, tag="main_table"):
                dpg.add_table_column()
                dpg.add_table_column()
                dpg.add_table_column()
                with dpg.table_row():
                    dpg.add_image("video_tex")
                    dpg.add_image("fk3d_tex")
                    with dpg.group():
                        app._build_inspector_panel()
            app._build_timeline_panel()
            dpg.add_text("Log Window Area", tag="log_window")

        dpg.setup_dearpygui()
        print("[Stress-Test] UI Layout Loaded. Starting 10-frame simulation...")
        
        for i in range(10):
            t0 = time.time()
            app._update_frame(i)
            dpg.render_dearpygui_frame()
            dt = (time.time() - t0) * 1000
            print("   - Frame " + str(i) + " rendered in " + str(round(dt, 1)) + "ms")
            
        print("\nSUCCESS: UI is stable for at least 10 frames.")
        return True
        
    except Exception as e:
        print("\nFAILURE: Crash during UI execution: " + str(e))
        return False
    finally:
        dpg.destroy_context()

if __name__ == "__main__":
    video = "my_data/2026_02_25/IMG_2633.MOV"
    if run_stress_test(video):
        sys.exit(0)
    else:
        sys.exit(1)
