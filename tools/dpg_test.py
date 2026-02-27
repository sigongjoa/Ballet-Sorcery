import dearpygui.dearpygui as dpg
import numpy as np
import time

dpg.create_context()
dpg.create_viewport(title='Test', width=800, height=600)

width, height = 640, 480
texture_data = np.ones((width * height * 4), dtype=np.float32)
# Make it red
for i in range(0, width * height * 4, 4):
    texture_data[i] = 1.0
    texture_data[i+1] = 0.0
    texture_data[i+2] = 0.0
    texture_data[i+3] = 1.0

with dpg.texture_registry(show=True):
    dpg.add_raw_texture(width=width, height=height, default_value=texture_data, format=dpg.mvFormat_Float_rgba, tag="texture_tag")

with dpg.window(label="Tutorial"):
    dpg.add_image("texture_tag")

dpg.setup_dearpygui()
dpg.show_viewport()

# Start with a delay before update to simulate our app
start_time = time.time()
updated = False

while dpg.is_dearpygui_running():
    # Change to green after 2 seconds
    if not updated and time.time() - start_time > 2.0:
        new_data = np.zeros((width * height * 4), dtype=np.float32)
        for i in range(0, width * height * 4, 4):
            new_data[i] = 0.0
            new_data[i+1] = 1.0
            new_data[i+2] = 0.0
            new_data[i+3] = 1.0
        dpg.set_value("texture_tag", new_data)
        updated = True
        print("Updated texture to green")

    dpg.render_dearpygui_frame()

dpg.destroy_context()
