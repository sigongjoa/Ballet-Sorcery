import cv2
import mediapipe as mp
import numpy as np
import json
import os
import sys
import time
import collections
import argparse

# 로컬 모듈 임포트 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normalizer
from comparator import calculate_l2_distance_3d, DEFAULT_WEIGHTS

# MediaPipe PoseLandmark 이름 정의 (33개)
POSE_LANDMARK_NAMES = [
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

MODEL_PATH = 'data/models/pose_landmarker_lite.task'

class RealtimeComparator:
    def __init__(self, ref_json_path: str, *, use_world: bool = True, window_size: int = 5, camera_id: int = 0):
        """
        실시간 포즈 비교기 초기화

        Args:
            ref_json_path (str): 참조 포즈 데이터 JSON 파일 경로
            use_world (bool): 3D world_landmarks 사용 여부 (기본값: True)
            window_size (int): 손실 값 이동 평균을 위한 윈도우 크기
            camera_id (int): 사용할 카메라 장치 ID
        """
        if not os.path.exists(ref_json_path):
            raise FileNotFoundError(f"참조 JSON 파일을 찾을 수 없습니다: {ref_json_path}")

        with open(ref_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.ref_frames = data.get("frames", [])
        
        if not self.ref_frames:
            raise ValueError("참조 JSON 파일에 'frames' 데이터가 없습니다.")

        self.use_world = use_world
        self.ref_frame_idx = 0  # 현재 비교 중인 기준 프레임 인덱스
        self.loss_history = collections.deque(maxlen=window_size)  # 이동평균용 큐
        self.camera_id = camera_id
        self._latest_result = None  # 최신 감지 결과 저장용

        # MediaPipe PoseLandmarker 초기화
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        if not os.path.exists(MODEL_PATH):
            print(f"경고: 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._on_result
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms):
        """MediaPipe 비동기 감지 결과 콜백"""
        self._latest_result = result

    def _extract_landmarks(self, detection_result) -> dict:
        """감지 결과에서 랜드마크 딕셔너리 추출"""
        if not detection_result or not detection_result.pose_landmarks:
            return None

        landmarks_dict = {}
        world_landmarks_dict = {}

        # 첫 번째 사람의 포즈만 사용 (index 0)
        for i, name in enumerate(POSE_LANDMARK_NAMES):
            # 일반 랜드마크 (화면 좌표계)
            lm = detection_result.pose_landmarks[0][i]
            landmarks_dict[name] = {
                'x': lm.x, 'y': lm.y, 'z': lm.z, 
                'visibility': lm.visibility
            }
            
            # 월드 랜드마크 (미터 단위 3D 좌표계)
            wlm = detection_result.pose_world_landmarks[0][i]
            world_landmarks_dict[name] = {
                'x': wlm.x, 'y': wlm.y, 'z': wlm.z, 
                'visibility': wlm.visibility
            }

        return {
            'landmarks': landmarks_dict,
            'world_landmarks': world_landmarks_dict
        }

    def _compute_frame_loss(self, live_landmarks: dict, ref_frame: dict) -> float:
        """라이브 랜드마크와 참조 프레임 간의 가중 손실 계산"""
        field = 'world_landmarks' if self.use_world else 'landmarks'
        live_pts = live_landmarks[field]
        ref_pts = ref_frame.get(field, {})

        # 2D 랜드마크를 사용하는 경우 실시간 정규화 적용
        if not self.use_world:
            live_pts, _, _ = normalizer.normalize_single_frame(live_pts)

        total_loss = 0.0
        weighted_count = 0.0
        
        # 공통 관절에 대해 거리 계산
        common_joints = set(live_pts.keys()) & set(ref_pts.keys())
        for joint in common_joints:
            p1 = live_pts[joint]
            p2 = ref_pts[joint]
            
            # 가중 L2 거리 계산
            dist = calculate_l2_distance_3d(p1, p2)
            weight = DEFAULT_WEIGHTS.get(joint, 1.0)
            
            total_loss += dist * weight
            weighted_count += weight

        return total_loss / weighted_count if weighted_count > 0 else 0.0

    def run(self):
        """실시간 카메라 피드 처리 및 비교 루프"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"오류: 카메라를 열 수 없습니다 (ID: {self.camera_id}). WSL 환경인 경우 USB 패스스루 설정을 확인하세요.")
            return

        print(f"실시간 비교 시작 (사용 모드: {'3D World' if self.use_world else '2D Normalized'})")
        print("'q' 키를 누르면 종료합니다.")
        
        start_time = time.time()
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # 이미지를 RGB로 변환 (MediaPipe 요구사항)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # 비동기 감지 실행 (밀리초 단위 타임스탬프 전달)
                timestamp_ms = int((time.time() - start_time) * 1000)
                self.landmarker.detect_async(mp_image, timestamp_ms)

                # 최신 감지 결과가 있으면 처리
                if self._latest_result:
                    live_landmarks = self._extract_landmarks(self._latest_result)
                    
                    if live_landmarks:
                        # 참조 프레임 순환 참조
                        ref_frame = self.ref_frames[self.ref_frame_idx % len(self.ref_frames)]
                        
                        # 손실 계산 및 기록
                        loss = self._compute_frame_loss(live_landmarks, ref_frame)
                        self.loss_history.append(loss)
                        self.ref_frame_idx += 1
                        
                        # 이동 평균 손실 계산
                        avg_loss = sum(self.loss_history) / len(self.loss_history)
                        
                        # 화면 출력 설정 (초록색: 우수, 빨간색: 보정 필요)
                        color = (0, 255, 0) if avg_loss <= 0.3 else (0, 0, 255)
                        cv2.putText(frame, f'Loss: {avg_loss:.3f}', (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                        
                        # 간단한 스켈레톤 시각화
                        self._draw_simple_skeleton(frame, live_landmarks['landmarks'])

                # 화면 표시
                cv2.imshow('芭蕾呪法 - 실시간 비교', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmarker.close()

    def _draw_simple_skeleton(self, frame, landmarks):
        """OpenCV를 사용하여 기본 스켈레톤을 그립니다."""
        h, w, _ = frame.shape
        # 주요 연결 부위 정의
        connections = [
            ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
            ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
            ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
            ("LEFT_HIP", "RIGHT_HIP"),
            ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
            ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE")
        ]
        
        for start_joint, end_joint in connections:
            if start_joint in landmarks and end_joint in landmarks:
                pt1_data = landmarks[start_joint]
                pt2_data = landmarks[end_joint]
                
                # 가시성이 낮은 경우 그리지 않음
                if pt1_data['visibility'] > 0.5 and pt2_data['visibility'] > 0.5:
                    pt1 = (int(pt1_data['x'] * w), int(pt1_data['y'] * h))
                    pt2 = (int(pt2_data['x'] * w), int(pt2_data['y'] * h))
                    cv2.line(frame, pt1, pt2, (255, 100, 0), 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="발레 주법 실시간 포즈 비교 도구")
    parser.add_argument("--ref", type=str, required=True, help="비교 기준이 될 포즈 JSON 파일 경로")
    parser.add_argument("--camera", type=int, default=0, help="카메라 장치 ID (기본값: 0)")
    parser.add_argument("--no-world", action="store_true", help="3D world_landmarks 대신 2D 정규화 좌표 사용")
    parser.add_argument("--window", type=int, default=5, help="평균 손실 계산을 위한 프레임 윈도우 크기")
    
    args = parser.parse_args()
    
    try:
        comparator = RealtimeComparator(
            args.ref, 
            use_world=not args.no_world, 
            window_size=args.window,
            camera_id=args.camera
        )
        comparator.run()
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
        sys.exit(1)
