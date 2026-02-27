import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import sys
import time

def test_texture_rendering(video_path):
    print(f"[Troubleshoot] 1. 비디오 로드 시도: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 비디오를 열 수 없습니다.")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("❌ 첫 프레임을 읽을 수 없습니다.")
        return
        
    print(f"✅ 비디오 로드 성공. 프레임 형태: {frame.shape}")
    
    # 텍스처 데이터 준비
    print("[Troubleshoot] 2. 텍스처 데이터 변환")
    width, height = 640, 480
    resized = cv2.resize(frame, (width, height))
    rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
    
    # Dear PyGui 요구 포맷 (1D float32 array, 0.0~1.0)
    texture_data = np.ascontiguousarray(rgba, dtype=np.float32) / 255.0
    flat_data = texture_data.flatten()
    
    # 리스트로 변환하는 것이 가장 안전한지 테스트 (가끔 numpy array가 무시됨)
    # flat_data = flat_data.tolist() # 리스트 변환은 느리지만 가장 확실함
    
    print(f"✅ 텍스처 데이터 준비 완료. 길이: {len(flat_data)} (예상: {width*height*4})")

    print("[Troubleshoot] 3. Dear PyGui 창 생성")
    dpg.create_context()
    dpg.create_viewport(title='Troubleshoot DPG Texture', width=800, height=600)

    with dpg.texture_registry(show=True):
        dpg.add_raw_texture(width=width, height=height, default_value=flat_data, format=dpg.mvFormat_Float_rgba, tag="test_tex")

    with dpg.window(label="Test Window", width=800, height=600):
        dpg.add_text("아래에 영상 프레임이 보여야 합니다.")
        dpg.add_image("test_tex")
        dpg.add_text("보이지 않는다면 Dear PyGui 텍스처 업데이트 버그입니다.")

    dpg.setup_dearpygui()
    dpg.show_viewport()

    print("✅ GUI가 실행되었습니다. 창을 확인해주세요.")
    # 업데이트 루프 테스트
    count = 0
    while dpg.is_dearpygui_running():
        # 주기적으로 색상을 바꿔치기 해본다 (업데이트가 되는지)
        if count % 100 == 0:
            ret, frame = cap.read()
            if ret:
                resized = cv2.resize(frame, (width, height))
                rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
                new_data = (np.ascontiguousarray(rgba, dtype=np.float32) / 255.0).flatten()
                dpg.set_value("test_tex", new_data)
        count += 1
        dpg.render_dearpygui_frame()

    dpg.destroy_context()

if __name__ == "__main__":
    test_texture_rendering("my_data/2026_02_25/IMG_2633.MOV")
