import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
import argparse
from tqdm import tqdm

# MediaPipe 관련 설정
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "data/models/pose_landmarker_lite.task"

# 스켈레톤 연결 정보 (src/video_overlay_generator.py 참고)
POSE_CONNECTIONS = [
    (11, 12), # shoulders
    (11, 13), (13, 15), # left arm
    (12, 14), (14, 16), # right arm
    (11, 23), (12, 24), # torso
    (23, 24), # hips
    (23, 25), (25, 27), # left leg
    (24, 26), (26, 28), # right leg
    (27, 29), (28, 30), # heels
    (29, 31), (30, 32), # foot index
    (27, 31), (28, 32)  # ankle to foot index
]

def preprocess_video(input_path, work_path):
    """
    ffmpeg를 사용하여 영상을 전처리합니다.
    - 회전 보정 (자동)
    - 1920x1080 리사이즈
    - H.264 변환 (yuv420p)
    """
    print(f"전처리 중: {input_path} -> {work_path}")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "scale=1920:1080",
        "-c:v", "libx264",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        work_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def draw_skeleton_on_black(landmarks, width, height):
    """
    검은색 배경에 스켈레톤을 그립니다.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    if not landmarks:
        return image

    # 랜드마크 좌표 변환 (정규화 -> 픽셀)
    points = []
    for lm in landmarks:
        points.append((int(lm.x * width), int(lm.y * height)))

    # 뼈대 그리기 (흰색)
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(image, points[start_idx], points[end_idx], (255, 255, 255), 2)

    # 관절점 그리기
    # 좌측: 초록 (0, 255, 0), 우측: 파랑 (255, 100, 0)
    # MediaPipe 기준: 홀수-좌측, 짝수-우측 (일반적으로)
    # 구체적으로: 11, 13, 15, 23, 25, 27, 29, 31 (Left)
    #            12, 14, 16, 24, 26, 28, 30, 32 (Right)
    for i, pt in enumerate(points):
        if i == 0 or (1 <= i <= 10): # 얼굴 등 중심/기타
            color = (200, 200, 200) # 연회색
        elif i % 2 == 1: # 좌측 (홀수)
            color = (0, 255, 0)
        else: # 우측 (짝수)
            color = (255, 100, 0)
        
        cv2.circle(image, pt, 5, color, -1)
        
    return image

def create_study_video(input_mov: str, output_path: str = None):
    if not output_path:
        base, _ = os.path.splitext(input_mov)
        output_path = f"{base}_study.mp4"

    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    work_path = os.path.join(tmp_dir, "_preprocessed.mp4")

    # 1. 전처리
    preprocess_video(input_mov, work_path)

    # 2. 포즈 추출 및 비디오 생성
    cap = cv2.VideoCapture(work_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 결과물은 가로로 두 배
    out_size = (width * 2, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    # MediaPipe 초기화
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        pbar = tqdm(total=total_frames, desc="영상 생성 중")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe 처리
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            detection_result = landmarker.detect(mp_image)
            
            # 스켈레톤 이미지 생성
            landmarks = detection_result.pose_landmarks[0] if detection_result.pose_landmarks else []
            skeleton_frame = draw_skeleton_on_black(landmarks, width, height)

            # 좌우 결합
            combined_frame = np.hstack((frame, skeleton_frame))

            # 텍스트 오버레이
            cv2.putText(combined_frame, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_frame, "Skeleton", (width + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(combined_frame)
            pbar.update(1)

        pbar.close()

    cap.release()
    out.release()
    
    # 임시 파일 삭제 (선택 사항)
    if os.path.exists(work_path):
        os.remove(work_path)

    print(f"완료! 저장된 위치: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="iPhone MOV 영상을 전처리하고 MediaPipe 스켈레톤과 함께 학습용 영상으로 변환합니다.")
    parser.add_argument("--input", required=True, help="입력 MOV 파일 경로")
    parser.add_argument("--output", help="출력 MP4 파일 경로 (기본값: {input}_study.mp4)")

    args = parser.parse_args()
    
    create_study_video(args.input, args.output)
