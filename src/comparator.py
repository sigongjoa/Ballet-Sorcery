import json
import os
import sys
import argparse
import math
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Any, List, Optional, Tuple

# sys.path에 현재 스크립트의 디렉토리를 추가하여 normalizer 모듈을 임포트할 수 있도록 합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import normalizer

# 기본 관절 가중치 정의
# 하체(힙, 무릎, 발목, 발뒤꿈치, 발끝) = 2.0, 나머지 = 1.0
DEFAULT_WEIGHTS = {
    "NOSE": 1.0,
    "LEFT_EYE_INNER": 1.0, "LEFT_EYE": 1.0, "LEFT_EYE_OUTER": 1.0,
    "RIGHT_EYE_INNER": 1.0, "RIGHT_EYE": 1.0, "RIGHT_EYE_OUTER": 1.0,
    "LEFT_EAR": 1.0, "RIGHT_EAR": 1.0, "MOUTH_LEFT": 1.0, "MOUTH_RIGHT": 1.0,
    "LEFT_SHOULDER": 1.0, "RIGHT_SHOULDER": 1.0,
    "LEFT_ELBOW": 1.0, "RIGHT_ELBOW": 1.0,
    "LEFT_WRIST": 1.0, "RIGHT_WRIST": 1.0,
    "LEFT_PINKY": 1.0, "RIGHT_PINKY": 1.0,
    "LEFT_INDEX": 1.0, "RIGHT_INDEX": 1.0,
    "LEFT_THUMB": 1.0, "RIGHT_THUMB": 1.0,
    # 하체 부위
    "LEFT_HIP": 2.0, "RIGHT_HIP": 2.0,
    "LEFT_KNEE": 2.0, "RIGHT_KNEE": 2.0,
    "LEFT_ANKLE": 2.0, "RIGHT_ANKLE": 2.0,
    "LEFT_HEEL": 2.0, "RIGHT_HEEL": 2.0,
    "LEFT_FOOT_INDEX": 2.0, "RIGHT_FOOT_INDEX": 2.0,
}

def calculate_l2_distance(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    """
    두 2D 포인트 간의 유클리드 (L2) 거리를 계산합니다.
    z 좌표는 비교에서 제외됩니다.
    """
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def calculate_l2_distance_3d(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    """
    두 3D 포인트 간의 유클리드 (L2) 거리를 계산합니다.
    """
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)

def align_with_dtw(
    ref_json_path: str,
    comp_json_path: str,
    *,
    use_world: bool = False
) -> Tuple[int, int]:
    """
    DTW(Dynamic Time Warping)를 사용하여 두 포즈 시퀀스 간의 최적 시작 프레임 오프셋을 계산합니다.

    Args:
        ref_json_path (str): 참조 포즈 데이터 JSON 파일 경로.
        comp_json_path (str): 비교할 포즈 데이터 JSON 파일 경로.
        use_world (bool): 3D world_landmarks를 사용할지 여부.

    Returns:
        Tuple[int, int]: (start_frame_ref, start_frame_comp) 최적 오프셋.
    """
    # 1. 데이터 로드 및 정규화
    ref_data = normalizer.load_pose_json(ref_json_path)
    comp_data = normalizer.load_pose_json(comp_json_path)

    if not use_world:
        # 2D인 경우 정규화 적용 (DTW 피처의 일관성을 위해)
        ref_data = normalizer.normalize_poses(ref_data)
        comp_data = normalizer.normalize_poses(comp_data)

    # 2. 피처 추출 (주요 관절 x, y, z 좌표 flatten)
    feat_joints = [
        "LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
    ]

    def extract_features(data: Dict[str, Any], use_world: bool) -> np.ndarray:
        frames = data.get("frames", [])
        field = "world_landmarks" if use_world else "landmarks"
        features = []
        for frame in frames:
            landmarks = frame.get(field, {})
            f_vec = []
            for joint in feat_joints:
                p = landmarks.get(joint, {"x": 0.0, "y": 0.0, "z": 0.0})
                f_vec.extend([p["x"], p["y"], p["z"]])
            features.append(f_vec)
        return np.array(features)

    feat_ref = extract_features(ref_data, use_world)
    feat_comp = extract_features(comp_data, use_world)

    if len(feat_ref) == 0 or len(feat_comp) == 0:
        return 0, 0

    # 3. 거리 행렬 계산
    dist_mat = cdist(feat_ref, feat_comp, metric='euclidean')
    N, M = dist_mat.shape

    # 4. DTW 비용 행렬 계산 (Open-Start DTW)
    cost_mat = np.full((N, M), np.inf)

    # 시작 오프셋 탐색 범위 제한 (각 시퀀스 앞부분의 30%)
    limit_ref = max(1, int(N * 0.3))
    limit_comp = max(1, int(M * 0.3))

    # 초기값 설정: 첫 행/열의 일부를 시작점으로 허용
    for i in range(limit_ref):
        cost_mat[i, 0] = dist_mat[i, 0]
    for j in range(limit_comp):
        cost_mat[0, j] = dist_mat[0, j]

    # DP 루프
    for i in range(1, N):
        for j in range(1, M):
            cost_mat[i, j] = dist_mat[i, j] + min(
                cost_mat[i-1, j],    # insertion
                cost_mat[i, j-1],    # deletion
                cost_mat[i-1, j-1]   # match
            )

    # 5. 최적 경로 역추적 (Backtracking)
    # 마지막 프레임에서 시작하여 시작점까지 추적
    curr_i, curr_j = N - 1, M - 1
    
    while curr_i > 0 and curr_j > 0:
        # 시작 가능 지점에 도달하면 중단
        if (curr_i < limit_ref and curr_j == 0) or (curr_i == 0 and curr_j < limit_comp):
            break
            
        choices = [
            cost_mat[curr_i-1, curr_j],
            cost_mat[curr_i, curr_j-1],
            cost_mat[curr_i-1, curr_j-1]
        ]
        best = np.argmin(choices)
        if best == 0:
            curr_i -= 1
        elif best == 1:
            curr_j -= 1
        else:
            curr_i -= 1
            curr_j -= 1
            
    return int(curr_i), int(curr_j)

def compare_poses(
    ref_json_path: str,
    comp_json_path: str,
    start_frame_ref: int = 0,
    start_frame_comp: int = 0,
    normalize: bool = True,
    weights: Optional[Dict[str, float]] = None,
    *, # 다음부터는 keyword-only arguments
    use_world: bool = False,
    use_dtw: bool = False
) -> Dict[str, Any]:
    """
    두 포즈 JSON 데이터를 비교하여 프레임별 L2 Loss를 계산합니다.

    Args:
        ref_json_path (str): 참조 포즈 데이터 JSON 파일 경로.
        comp_json_path (str): 비교할 포즈 데이터 JSON 파일 경로.
        start_frame_ref (int): 참조 포즈 시퀀스의 시작 프레임 오프셋.
        start_frame_comp (int): 비교 포즈 시퀀스의 시작 프레임 오프셋.
        normalize (bool): 2D 포즈 데이터를 정규화할지 여부. use_world=True인 경우 무시됩니다.
        weights (Optional[Dict[str, float]]): 관절별 가중치 딕셔너리. None이면 DEFAULT_WEIGHTS 사용.
        use_world (bool): 3D world_landmarks를 사용할지 여부. True인 경우 3D 유클리드 거리가 계산됩니다.
        use_dtw (bool): DTW를 사용하여 최적 시작 오프셋을 자동으로 계산할지 여부.

    Returns:
        Dict[str, Any]: 비교 결과 (메타데이터, 프레임별 손실, 요약 통계).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # DTW를 사용하는 경우 시작 오프셋 자동 계산
    if use_dtw:
        start_frame_ref, start_frame_comp = align_with_dtw(
            ref_json_path, comp_json_path, use_world=use_world
        )

    # 1. 두 JSON 로드
    ref_data = normalizer.load_pose_json(ref_json_path)
    comp_data = normalizer.load_pose_json(comp_json_path)

    # 2. 2D 모드이고 normalize=True이면 둘 다 normalize_poses() 적용
    # 3D world_landmarks는 이미 골반 중심 기준이므로 정규화하지 않습니다.
    if not use_world and normalize:
        ref_data = normalizer.normalize_poses(ref_data)
        comp_data = normalizer.normalize_poses(comp_data)

    ref_frames = ref_data.get("frames", [])
    comp_frames = comp_data.get("frames", [])

    # 3. start_frame_ref, start_frame_comp로 시작 프레임 오프셋 지정
    ref_frames_offset = ref_frames[start_frame_ref:]
    comp_frames_offset = comp_frames[start_frame_comp:]

    # 4. 두 시퀀스 중 짧은 쪽 길이만큼만 비교
    comparison_length = min(len(ref_frames_offset), len(comp_frames_offset))

    per_frame_loss_list: List[Dict[str, Any]] = []
    all_total_losses: List[float] = []

    for i in range(comparison_length):
        ref_frame_data = ref_frames_offset[i]
        comp_frame_data = comp_frames_offset[i]

        # use_world 설정에 따라 사용할 랜드마크 필드를 선택
        if use_world:
            ref_landmarks = ref_frame_data.get("world_landmarks", {})
            comp_landmarks = comp_frame_data.get("world_landmarks", {})
            distance_calculator = calculate_l2_distance_3d
        else:
            ref_landmarks = ref_frame_data.get("landmarks", {})
            comp_landmarks = comp_frame_data.get("landmarks", {})
            distance_calculator = calculate_l2_distance

        frame_joint_losses: Dict[str, float] = {}
        frame_total_loss: float = 0.0
        weighted_joint_count: float = 0.0 # 가중치가 적용된 관절 수

        # 공통 관절에 대해서만 비교
        common_joints = set(ref_landmarks.keys()) & set(comp_landmarks.keys())

        for joint_name in common_joints:
            ref_point = ref_landmarks.get(joint_name)
            comp_point = comp_landmarks.get(joint_name)

            # 유효한 포인트가 아니면 스킵
            if not ref_point or not comp_point:
                continue
            
            # 선택된 거리 계산 함수 사용
            distance = distance_calculator(ref_point, comp_point)
            
            weight = weights.get(joint_name, 1.0)
            weighted_distance = distance * weight
            
            frame_joint_losses[joint_name] = weighted_distance
            frame_total_loss += weighted_distance
            weighted_joint_count += weight

        # 프레임의 총 손실은 가중 평균으로 계산
        if weighted_joint_count > 0:
            frame_total_loss /= weighted_joint_count
        else:
            frame_total_loss = 0.0 # 비교할 관절이 없으면 손실은 0

        per_frame_loss_list.append({
            "frame_idx": i, # 비교 시작부터의 인덱스
            "original_ref_frame_idx": ref_frame_data.get("frame_idx"), # 원본 참조 영상의 프레임 인덱스
            "original_comp_frame_idx": comp_frame_data.get("frame_idx"), # 원본 비교 영상의 프레임 인덱스
            "total_loss": frame_total_loss,
            "joint_losses": frame_joint_losses
        })
        all_total_losses.append(frame_total_loss)

    # 5. 요약 통계 계산
    summary: Dict[str, Any] = {
        "mean_loss": 0.0,
        "max_loss": 0.0,
        "max_loss_frame": -1,
        "min_loss": float('inf'),
        "min_loss_frame": -1,
        "std_loss": 0.0
    }

    if all_total_losses:
        summary["mean_loss"] = sum(all_total_losses) / len(all_total_losses)
        summary["max_loss"] = max(all_total_losses)
        summary["min_loss"] = min(all_total_losses)

        # 최대/최소 손실 프레임 인덱스 찾기
        max_loss_index = all_total_losses.index(summary["max_loss"])
        min_loss_index = all_total_losses.index(summary["min_loss"])
        summary["max_loss_frame"] = per_frame_loss_list[max_loss_index]["frame_idx"]
        summary["min_loss_frame"] = per_frame_loss_list[min_loss_index]["frame_idx"]
        
        # 표준 편차 계산 (numpy 없이)
        if len(all_total_losses) > 1:
            variance = sum([(x - summary["mean_loss"]) ** 2 for x in all_total_losses]) / (len(all_total_losses) - 1)
            summary["std_loss"] = math.sqrt(variance)
        else:
            summary["std_loss"] = 0.0 # 단일 프레임은 표준편차 0

    # 출력 dict 구조
    result = {
        "metadata": {
            "ref_video": os.path.basename(ref_json_path),
            "comp_video": os.path.basename(comp_json_path),
            "total_compared_frames": comparison_length,
            "normalized": normalize and not use_world, # 3D 모드에서는 정규화되지 않음
            "start_frame_ref": start_frame_ref,
            "start_frame_comp": start_frame_comp,
            "use_world_landmarks": use_world, # 3D 랜드마크 사용 여부
            "dtw_aligned": use_dtw,
            "dtw_offsets": {"ref": start_frame_ref, "comp": start_frame_comp}
        },
        "per_frame_loss": per_frame_loss_list,
        "summary": summary
    }

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="두 포즈 JSON 데이터를 비교하여 프레임별 L2 Loss를 계산합니다."
    )
    parser.add_argument(
        "ref_path",
        type=str,
        help="참조 포즈 데이터 JSON 파일 경로."
    )
    parser.add_argument(
        "comp_path",
        type=str,
        help="비교할 포즈 데이터 JSON 파일 경로."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="결과를 저장할 JSON 파일 경로. 지정하지 않으면 콘솔에 출력."
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="포즈 데이터를 정규화하지 않습니다."
    )
    parser.add_argument(
        "--start-frame-ref",
        type=int,
        default=0,
        help="참조 포즈 시퀀스의 시작 프레임 오프셋 (기본값: 0)."
    )
    parser.add_argument(
        "--start-frame-comp",
        type=int,
        default=0,
        help="비교 포즈 시퀀스의 시작 프레임 오프셋 (기본값: 0)."
    )
    parser.add_argument(
        "--use-world",
        action="store_true",
        help="2D 랜드마크 대신 3D world_landmarks를 사용하여 비교합니다."
    )
    parser.add_argument(
        "--use-dtw",
        action="store_true",
        help="DTW를 사용하여 최적 시작 프레임을 자동으로 찾습니다."
    )
    
    args = parser.parse_args()

    try:
        comparison_result = compare_poses(
            ref_json_path=args.ref_path,
            comp_json_path=args.comp_path,
            start_frame_ref=args.start_frame_ref,
            start_frame_comp=args.start_frame_comp,
            normalize=not args.no_normalize,
            weights=None,
            use_world=args.use_world,
            use_dtw=args.use_dtw
        )

        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            print(f"비교 결과가 '{args.output}'에 저장되었습니다.")
        else:
            print(json.dumps(comparison_result, indent=2, ensure_ascii=False))

    except FileNotFoundError as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"오류: JSON 파일 파싱 중 문제 발생 - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            print(f"비교 결과가 '{args.output}'에 저장되었습니다.")
        else:
            print(json.dumps(comparison_result, indent=2, ensure_ascii=False))

    except FileNotFoundError as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"오류: JSON 파일 파싱 중 문제 발생 - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)
