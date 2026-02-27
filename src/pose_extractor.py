import cv2
import mediapipe as mp
import numpy as np
import json
import argparse
import os
import urllib.request
import math

# 모델 다운로드 관련 설정
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_DIR = "data/models"
MODEL_NAME = "pose_landmarker_lite.task"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

def _download_model_if_not_exists():
    """
    MediaPipe Pose Landmarker 모델 파일을 다운로드합니다.
    이미 존재하면 다운로드하지 않습니다.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Pose Landmarker 모델 파일을 다운로드 중입니다: {MODEL_URL}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"모델 파일 다운로드 완료: {MODEL_PATH}")
        except Exception as e:
            raise IOError(f"모델 파일 다운로드에 실패했습니다: {e}")
    else:
        print(f"모델 파일이 이미 존재합니다: {MODEL_PATH}")

def extract_poses(video_path: str, output_json_path: str, target_fps: int = 24, max_frames: int = -1) -> dict:
    """
    영상 파일에서 MediaPipe Pose Landmarker를 사용하여 포즈 랜드마크를 추출하고 JSON 파일로 저장합니다.

    Args:
        video_path (str): 입력 영상 파일의 경로.
        output_json_path (str): 추출된 포즈 데이터를 저장할 JSON 파일의 경로.
        target_fps (int): 포즈 데이터를 추출할 목표 프레임 속도.
        max_frames (int): 처리할 최대 프레임 수 (-1이면 전체 영상).

    Returns:
        dict: 추출된 포즈 데이터가 담긴 딕셔너리.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    # 모델 다운로드 확인
    _download_model_if_not_exists()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"영상 파일을 열 수 없습니다: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / original_fps if original_fps > 0 else 0
    
    if target_fps <= 0:
        raise ValueError("target_fps는 0보다 커야 합니다.")

    effective_fps = original_fps
    frame_interval = 1 # 기본적으로 모든 프레임 처리

    if original_fps > 0: # 유효한 FPS 정보가 있을 때만 샘플링 로직 적용
        if original_fps > target_fps:
            frame_interval = int(round(original_fps / target_fps))
            effective_fps = original_fps / frame_interval
            print(f"원본 FPS ({original_fps})가 목표 FPS ({target_fps})보다 높으므로, 약 {effective_fps:.2f} FPS로 리샘플링합니다.")
        else:
            effective_fps = original_fps
    else:
        print(f"경고: 원본 영상의 FPS를 확인할 수 없어 모든 프레임을 처리합니다. (video_path: {video_path})")
        # effective_fps는 0으로 유지하거나, target_fps로 가정할 수 있지만, 여기서는 처리된 프레임 수로 나중에 계산하는 것이 안전.
        # 일단은 original_fps 값을 유지하여 metadata에 기록합니다.

    # MediaPipe Pose Landmarker 설정
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    PoseLandmark = mp.tasks.vision.PoseLandmark # 관절 이름 Enum

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO
    )

    landmarker = PoseLandmarker.create_from_options(options)

    frames_data = []
    processed_frame_idx = 0
    current_frame_pos = 0 # 0-based index for current frame in video

    # PoseLandmark enum의 이름을 사용하여 POSE_LANDMARKS 리스트 생성
    # Enum 멤버의 이름(e.g., 'NOSE')만 추출하여 사용
    POSE_LANDMARK_NAMES = [name for name, _ in PoseLandmark.__members__.items()]

    while cap.isOpened():
        if max_frames > 0 and processed_frame_idx >= max_frames:
            print(f"최대 프레임 수({max_frames})에 도달하여 추출을 종료합니다.")
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        # 지정된 간격으로 프레임 처리
        if current_frame_pos % frame_interval == 0:
            # OpenCV 이미지를 MediaPipe Image로 변환
            # timestamp_ms는 프레임의 타임스탬프를 밀리초 단위로 제공해야 합니다.
            # current_frame_pos는 0부터 시작하는 프레임 인덱스이므로,
            # (current_frame_pos / original_fps) * 1000 계산
            timestamp_ms = int((current_frame_pos / (original_fps if original_fps > 0 else target_fps)) * 1000)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # PoseLandmarker로 포즈 추론 실행
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            landmarks_data = {}
            if detection_result.pose_landmarks:
                # 첫 번째 사람의 포즈 랜드마크만 사용
                for i, landmark in enumerate(detection_result.pose_landmarks[0]):
                    # PoseLandmark enum의 이름을 직접 사용
                    if i < len(POSE_LANDMARK_NAMES):
                        landmarks_data[POSE_LANDMARK_NAMES[i]] = {
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.visibility # PoseLandmarker에는 visibility가 직접 제공되지 않을 수 있음.
                                                              # 대신 presence/score를 활용하거나, 기존 필드를 유지하기 위해 기본값 1.0 부여
                        }
                    else:
                        print(f"경고: PoseLandmark Enum 범위를 초과하는 랜드마크 인덱스 {i}가 발견되었습니다.")
            
            world_landmarks_data = {}
            # world_landmarks는 3D 좌표 (미터 단위, 골반 중심 기준)
            if detection_result.pose_world_landmarks:
                for i, landmark in enumerate(detection_result.pose_world_landmarks[0]):
                    if i < len(POSE_LANDMARK_NAMES):
                        world_landmarks_data[POSE_LANDMARK_NAMES[i]] = {
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            # world_landmarks에는 visibility 필드가 불필요
                        }
                    else:
                        print(f"경고: PoseLandmark Enum 범위를 초과하는 월드 랜드마크 인덱스 {i}가 발견되었습니다.")

            timestamp_sec = current_frame_pos / original_fps if original_fps > 0 else processed_frame_idx / effective_fps
            
            frames_data.append({
                "frame_idx": processed_frame_idx,
                "timestamp_sec": timestamp_sec,
                "landmarks": landmarks_data,
                "world_landmarks": world_landmarks_data # 3D world landmarks 추가
            })
            processed_frame_idx += 1
        
        current_frame_pos += 1

    cap.release()
    landmarker.close() # landmarker 리소스 해제

    metadata = {
        "source_video": os.path.basename(video_path),
        "original_fps": original_fps,
        "target_fps": target_fps,
        "effective_fps": effective_fps,
        "total_frames_in_video": frame_count,
        "total_frames_processed": len(frames_data),
        "duration_sec": duration_sec,
        "has_world_landmarks": True # world_landmarks 필드 추가 여부 표시
    }

    output_data = {
        "metadata": metadata,
        "frames": frames_data
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return output_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MediaPipe를 사용하여 영상에서 포즈를 추출하고 JSON으로 저장합니다.")
    parser.add_argument("video_path", type=str, help="처리할 영상 파일의 경로")
    parser.add_argument("output_path", type=str, help="추출된 포즈 데이터를 저장할 JSON 파일의 경로")
    parser.add_argument("--target_fps", type=int, default=24,
                        help="포즈 데이터를 추출할 목표 프레임 속도 (기본값: 24)")

    args = parser.parse_args()

    try:
        print(f"영상 '{args.video_path}'에서 포즈 추출을 시작합니다. JSON 파일은 '{args.output_path}'에 저장됩니다.")
        extracted_data = extract_poses(args.video_path, args.output_path, args.target_fps)
        print(f"포즈 추출 및 JSON 저장 완료: {args.output_path}")
    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"오류 발생: {e}")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")