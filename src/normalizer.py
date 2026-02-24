import json
import numpy as np
import os
from typing import Dict, Any, List, Tuple, Optional

# MediaPipe PoseLandmark 이름을 직접 정의 (pose_extractor.py와 동일하게 유지)
POSE_LANDMARKS = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
    "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
    "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

def load_pose_json(json_path: str) -> Dict[str, Any]:
    """
    JSON 파일에서 포즈 데이터를 로드합니다.

    Args:
        json_path (str): 로드할 JSON 파일의 경로.

    Returns:
        Dict[str, Any]: 로드된 포즈 데이터 딕셔너리.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_pose_json(data: Dict[str, Any], json_path: str):
    """
    포즈 데이터를 JSON 파일로 저장합니다.

    Args:
        data (Dict[str, Any]): 저장할 포즈 데이터 딕셔너리.
        json_path (str): 저장할 JSON 파일의 경로.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def normalize_single_frame(landmarks: Dict[str, Dict[str, float]], prev_scale: Optional[float] = None) -> Tuple[Dict[str, Dict[str, float]], float, List[float]]:
    """
    단일 프레임의 랜드마크 데이터를 정규화합니다.

    골반 중심을 원점으로 이동하고 어깨 너비로 스케일링합니다.
    어깨가 감지되지 않으면 이전 프레임의 스케일 팩터를 사용합니다.

    Args:
        landmarks (Dict[str, Dict[str, float]]): 단일 프레임의 랜드마크 데이터.
                                                각 랜드마크는 'x', 'y', 'z', 'visibility'를 포함합니다.
        prev_scale (Optional[float]): 이전 프레임에서 사용된 스케일 팩터.

    Returns:
        Tuple[Dict[str, Dict[str, float]], float]: 정규화된 랜드마크와 사용된 스케일 팩터.
    """
    if not landmarks:
        return {}, prev_scale if prev_scale is not None else 1.0, [0.0, 0.0, 0.0] # 랜드마크가 없으면 빈 딕셔너리와 이전 스케일 또는 기본 1.0, 0,0,0 골반 중심 반환

    # 필요한 랜드마크 추출 (LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER)
    left_hip = landmarks.get("LEFT_HIP")
    right_hip = landmarks.get("RIGHT_HIP")
    left_shoulder = landmarks.get("LEFT_SHOULDER")
    right_shoulder = landmarks.get("RIGHT_SHOULDER")

    pelvis_center = np.array([0.0, 0.0, 0.0])
    shoulder_width = 1.0 # 기본 스케일 팩터

    # 1. 골반 중심 계산
    if left_hip and right_hip and left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
        pelvis_center_x = (left_hip['x'] + right_hip['x']) / 2
        pelvis_center_y = (left_hip['y'] + right_hip['y']) / 2
        pelvis_center_z = (left_hip['z'] + right_hip['z']) / 2
        pelvis_center = np.array([pelvis_center_x, pelvis_center_y, pelvis_center_z])
    else:
        # 골반이 감지되지 않으면 원점 이동 없이 진행 (또는 다른 전략 필요, 여기서는 0,0,0으로 설정)
        pass # pelvis_center는 기본값 0,0,0 유지

    # 2. 어깨 너비 계산 (스케일 팩터)
    if left_shoulder and right_shoulder and left_shoulder['visibility'] > 0.5 and right_shoulder['visibility'] > 0.5:
        p_left_s = np.array([left_shoulder['x'], left_shoulder['y'], left_shoulder['z']])
        p_right_s = np.array([right_shoulder['x'], right_shoulder['y'], right_shoulder['z']])
        shoulder_width = np.linalg.norm(p_left_s - p_right_s)
        # 스케일 팩터가 너무 작으면 (예: 0에 가까우면) 기본값 사용 또는 예외 처리
        if shoulder_width < 1e-6: # 아주 작은 값으로 나누는 것을 방지
            shoulder_width = prev_scale if prev_scale is not None else 1.0
    else:
        # 어깨가 감지되지 않은 프레임은 이전 프레임의 스케일 팩터를 사용
        # prev_scale이 None이면 기본값 1.0 사용 (첫 프레임이거나 이전 프레임도 어깨 없었을 경우)
        shoulder_width = prev_scale if prev_scale is not None else 1.0

    normalized_landmarks = {}
    for name, lm_data in landmarks.items():
        point = np.array([lm_data['x'], lm_data['y'], lm_data['z']])
        
        # 골반 중심 이동
        translated_point = point - pelvis_center
        
        # 어깨 너비로 스케일링
        scaled_point = translated_point / shoulder_width
        
        normalized_landmarks[name] = {
            "x": scaled_point[0],
            "y": scaled_point[1],
            "z": scaled_point[2],
            "visibility": lm_data.get('visibility', 0.0) # visibility는 그대로 유지
        }
    
    return normalized_landmarks, shoulder_width, pelvis_center.tolist()


def normalize_poses(pose_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    추출된 모든 포즈 데이터를 정규화합니다.

    Args:
        pose_data (Dict[str, Any]): pose_extractor.py에서 생성된 원본 포즈 데이터.

    Returns:
        Dict[str, Any]: 정규화된 포즈 데이터.
    """
    if "frames" not in pose_data:
        raise ValueError("입력 pose_data에 'frames' 키가 없습니다.")

    normalized_output_data = pose_data.copy() # 원본 데이터를 복사하여 수정

    prev_scale_factor: Optional[float] = None
    for frame_idx, frame in enumerate(normalized_output_data["frames"]):
        current_landmarks = frame.get("landmarks", {})
        
        normalized_lm, current_scale_factor, current_pelvis_center = normalize_single_frame(current_landmarks, prev_scale_factor)
        
        frame["landmarks"] = normalized_lm # 랜드마크 데이터를 정규화된 랜드마크로 교체
        frame["normalized_scale"] = current_scale_factor
        frame["normalized_pelvis_center"] = current_pelvis_center
        
        # 다음 프레임을 위해 현재 스케일 팩터 저장
        if current_scale_factor is not None:
            prev_scale_factor = current_scale_factor

    # metadata에 normalized: true 필드 추가
    if "metadata" not in normalized_output_data:
        normalized_output_data["metadata"] = {}
    normalized_output_data["metadata"]["normalized"] = True

    return normalized_output_data

if __name__ == '__main__':
    # 이 부분은 테스트 및 사용 예시를 위한 코드입니다.
    # 실제 사용 시에는 이 스크립트를 모듈로 임포트하여 사용합니다.

    # 임시 JSON 파일 생성 (pose_extractor.py의 출력 형태를 모방)
    sample_pose_data = {
        "metadata": {
            "source_video": "sample.mp4",
            "original_fps": 30.0,
            "target_fps": 24,
            "effective_fps": 24.0,
            "total_frames_in_video": 300,
            "total_frames_processed": 240,
            "duration_sec": 10.0
        },
        "frames": [
            # Frame 0: 정상적인 어깨/골반
            {
                "frame_idx": 0,
                "timestamp_sec": 0.0,
                "landmarks": {
                    "NOSE": {"x": 0.5, "y": 0.1, "z": 0.0, "visibility": 0.9},
                    "LEFT_SHOULDER": {"x": 0.4, "y": 0.2, "z": 0.1, "visibility": 0.9},
                    "RIGHT_SHOULDER": {"x": 0.6, "y": 0.2, "z": 0.1, "visibility": 0.9},
                    "LEFT_HIP": {"x": 0.45, "y": 0.5, "z": 0.0, "visibility": 0.9},
                    "RIGHT_HIP": {"x": 0.55, "y": 0.5, "z": 0.0, "visibility": 0.9},
                    "LEFT_KNEE": {"x": 0.4, "y": 0.7, "z": 0.0, "visibility": 0.9},
                    "RIGHT_KNEE": {"x": 0.6, "y": 0.7, "z": 0.0, "visibility": 0.9},
                }
            },
            # Frame 1: 어깨가 감지되지 않는 경우 (이전 스케일 사용해야 함)
            {
                "frame_idx": 1,
                "timestamp_sec": 0.04,
                "landmarks": {
                    "NOSE": {"x": 0.5, "y": 0.1, "z": 0.0, "visibility": 0.9},
                    # 어깨 랜드마크 없음
                    "LEFT_HIP": {"x": 0.45, "y": 0.51, "z": 0.01, "visibility": 0.9},
                    "RIGHT_HIP": {"x": 0.55, "y": 0.51, "z": 0.01, "visibility": 0.9},
                    "LEFT_KNEE": {"x": 0.4, "y": 0.71, "z": 0.01, "visibility": 0.9},
                    "RIGHT_KNEE": {"x": 0.6, "y": 0.71, "z": 0.01, "visibility": 0.9},
                }
            },
            # Frame 2: 다시 어깨가 감지되는 경우, 다른 포즈
            {
                "frame_idx": 2,
                "timestamp_sec": 0.08,
                "landmarks": {
                    "NOSE": {"x": 0.3, "y": 0.1, "z": 0.0, "visibility": 0.9},
                    "LEFT_SHOULDER": {"x": 0.2, "y": 0.2, "z": 0.1, "visibility": 0.9},
                    "RIGHT_SHOULDER": {"x": 0.4, "y": 0.2, "z": 0.1, "visibility": 0.9},
                    "LEFT_HIP": {"x": 0.25, "y": 0.5, "z": 0.0, "visibility": 0.9},
                    "RIGHT_HIP": {"x": 0.35, "y": 0.5, "z": 0.0, "visibility": 0.9},
                    "LEFT_KNEE": {"x": 0.2, "y": 0.7, "z": 0.0, "visibility": 0.9},
                    "RIGHT_KNEE": {"x": 0.4, "y": 0.7, "z": 0.0, "visibility": 0.9},
                }
            }
        ]
    }

    input_json_path = os.path.join(os.path.dirname(__file__), "sample_pose_data.json")
    output_json_path = os.path.join(os.path.dirname(__file__), "normalized_pose_data.json")

    try:
        print(f"샘플 포즈 데이터를 '{input_json_path}'에 저장합니다.")
        save_pose_json(sample_pose_data, input_json_path)

        print(f"'{input_json_path}'에서 포즈 데이터를 로드합니다.")
        loaded_data = load_pose_json(input_json_path)

        print("포즈 데이터 정규화를 시작합니다...")
        normalized_data = normalize_poses(loaded_data)
        print("포즈 데이터 정규화 완료.")

        print(f"정규화된 포즈 데이터를 '{output_json_path}'에 저장합니다.")
        save_pose_json(normalized_data, output_json_path)
        
        print("\n정규화 결과 (첫 프레임 랜드마크 예시):")
        if normalized_data["frames"]:
            first_frame_landmarks = normalized_data["frames"][0]["landmarks"]
            # 골반 중심과 어깨 너비로 정규화되었는지 확인하기 위한 간단한 출력
            print(f"  LEFT_HIP: {first_frame_landmarks.get('LEFT_HIP')}")
            print(f"  RIGHT_HIP: {first_frame_landmarks.get('RIGHT_HIP')}")
            print(f"  LEFT_SHOULDER: {first_frame_landmarks.get('LEFT_SHOULDER')}")
            print(f"  RIGHT_SHOULDER: {first_frame_landmarks.get('RIGHT_SHOULDER')}")
        
        print(f"\n메타데이터에 'normalized': {normalized_data['metadata'].get('normalized')} 추가됨.")

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        # 생성된 임시 파일 정리 (선택 사항)
        # if os.path.exists(input_json_path):
        #     os.remove(input_json_path)
        # if os.path.exists(output_json_path):
        #     os.remove(output_json_path)
        pass
