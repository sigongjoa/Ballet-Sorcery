import cv2
import numpy as np
import os

def verify_vtp_visual():
    # 설정
    video_path = 'my_data/2026_02_25/IMG_2633_anatomy.mp4'
    output_dir = 'reports/2026-02-28_vtp_visual_verify/assets/'
    os.makedirs(output_dir, exist_ok=True)
    
    target_frames = [100, 250, 400, 600, 850]
    panel_width = 1920
    panel_height = 1080
    
    # 비디오 캡처
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 영상을 열 수 없습니다: {video_path}")
        return

    results = []
    
    print(f"--- 검증 시작: {video_path} ---")
    
    for idx in target_frames:
        # 프레임 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"오류: 프레임 {idx}를 추출할 수 없습니다. 건너뜁니다.")
            continue
            
        # 프레임 크기 확인 (5760x1080)
        h, w, _ = frame.shape
        if w != 5760 or h != 1080:
            print(f"경고: 프레임 {idx}의 크기가 예상(5760x1080)과 다릅니다: {w}x{h}")

        # 패널 분리 (Panel 1: 원본, Panel 2: MediaPipe, Panel 3: VTP)
        panel1 = frame[:, 0:panel_width]
        panel3 = frame[:, 2*panel_width:3*panel_width]
        
        # a. Combined 저장 (5760x1080)
        combined_path = os.path.join(output_dir, f'frame_{idx:04d}_combined.jpg')
        cv2.imwrite(combined_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        # b. Panel 3 (VTP) 저장 (1920x1080)
        vtp_path = os.path.join(output_dir, f'frame_{idx:04d}_panel3_vtp.jpg')
        cv2.imwrite(vtp_path, panel3, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        # c. Compare 저장 (Panel 1 + Panel 3, 3840x1080)
        compare_img = np.hstack((panel1, panel3))
        compare_path = os.path.join(output_dir, f'frame_{idx:04d}_compare.jpg')
        cv2.imwrite(compare_path, compare_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        # VTP 픽셀 통계 분석 (Panel 3에서 RGB 모두 < 30이 아닌 픽셀)
        non_bg_mask = (panel3[:, :, 0] >= 30) | (panel3[:, :, 1] >= 30) | (panel3[:, :, 2] >= 30)
        active_pixel_count = np.sum(non_bg_mask)
        total_pixels = panel_width * panel_height
        active_ratio = (active_pixel_count / total_pixels) * 100
        
        results.append((idx, active_ratio))
        print(f"프레임 {idx:04d} 처리 완료: VTP 활성 픽셀 비율 = {active_ratio:.2f}%")

    cap.release()
    
    # 요약 정보 저장
    summary_path = os.path.join(output_dir, 'verify_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("--- VTP 시각적 검증 요약 ---\n")
        f.write(f"대상 영상: {video_path}\n")
        f.write(f"분석 프레임 수: {len(results)}\n\n")
        
        total_ratio = 0
        for idx, ratio in results:
            f.write(f"Frame {idx:04d}: VTP Active Ratio = {ratio:.2f}%\n")
            total_ratio += ratio
            
        avg_ratio = total_ratio / len(results) if results else 0
        f.write(f"\n평균 활성 비율: {avg_ratio:.2f}%\n")
        
    print(f"\n평균 VTP 활성 비율: {avg_ratio:.2f}%")
    print(f"요약 결과가 저장되었습니다: {summary_path}")
    print("--- 검증 완료 ---")

if __name__ == "__main__":
    verify_vtp_visual()
