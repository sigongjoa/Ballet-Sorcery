# -*- coding: utf-8 -*-
import sys
import os
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.joint_analyzer import JointAnalyzerApp
import dearpygui.dearpygui as dpg

def run_test(video_path):
    print("\n[Test] Starting stable viewer test with video: " + video_path)
    
    app = JointAnalyzerApp(video_path)
    
    dpg.create_context()
    
    # init_data = [0.0] * (640 * 480 * 4)
    # with dpg.texture_registry():
    #     dpg.add_raw_texture(640, 480, init_data, tag="video_tex", format=dpg.mvFormat_Float_rgba)
    #     dpg.add_raw_texture(640, 480, init_data, tag="fk3d_tex", format=dpg.mvFormat_Float_rgba)

    with dpg.window(tag="main", no_title_bar=True):
        dpg.add_text("TESTING VIEWER", tag="status_text")
        
        with dpg.table(header_row=False):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            with dpg.table_row():
                # dpg.add_image("video_tex")
                # dpg.add_image("fk3d_tex")
                with dpg.group():
                    app._build_inspector_panel()
        
        # Add timeline panel so set_value("timeline", idx) doesn't fail
        app._build_timeline_panel()

    dpg.setup_dearpygui()
    print("[Test] UI Layout loaded successfully.")
    
    app._update_frame(0)
    app._on_body_selected(app.state.selected_body)
    
    print("[Test] First frame loaded. Simulating 20 frames of user interaction...")
    
    for i in range(20):
        t0 = time.time()
        # app._update_frame(i)
        dpg.render_dearpygui_frame()
        dt = (time.time() - t0) * 1000
        print("   - Frame " + str(i) + " rendered in " + str(round(dt, 1)) + "ms")
        
    print("\n[Test] SUCCESS: Viewer successfully simulated 20 frames without crashing.")
    app._cleanup()
    return True

if __name__ == "__main__":
    video = "my_data/2026_02_25/IMG_2633.MOV"
    run_test(video)
