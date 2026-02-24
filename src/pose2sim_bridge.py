# -*- coding: utf-8 -*-
"""
Pose2Sim + OpenSim 바이오메카닉스 분석 브리지.

우선순위:
  1. OpenSim이 설치된 경우 → Pose2Sim.kinematics() 호출 후 .mot 파일 파싱
  2. Fallback → joint_angles.py 순수 Python 계산

반환 데이터 포맷:
  {
    'trc_path': str,              # 생성된 .trc 파일 경로
    'mot_path': str | None,       # OpenSim 생성 .mot 파일 경로 (없으면 None)
    'method': 'opensim'|'python', # 사용된 계산 방법
    'frames': [
      {
        'frame_idx': int,
        'angles': {
          'left_knee_flexion': float,    # 도
          'right_knee_flexion': float,
          'left_hip_flexion': float,
          'right_hip_flexion': float,
          'left_turnout': float,
          'right_turnout': float,
          'left_ankle_plantarflexion': float,
          'right_ankle_plantarflexion': float,
          'spine_lateral': float,
          'shoulder_symmetry': float,
        }
      }, ...
    ],
    'summary': {
      'mean_angles': dict,
      'std_angles': dict,
      'ballet_scores': {
        'turnout_score': float,
        'knee_alignment_score': float,
        'spine_alignment_score': float,
      }
    }
  }
"""

import json
import os
import sys
from typing import Optional

# src 디렉토리를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import opensim_ik
    import opensim_dynamics
except ImportError:
    opensim_ik = None
    opensim_dynamics = None

from trc_exporter import export_trc
from joint_angles import extract_sequence_angles, compute_summary


def _try_pose2sim_opensim(trc_path: str, output_dir: str) -> Optional[str]:
    """
    Pose2Sim + OpenSim을 사용한 IK 계산을 시도합니다.

    Returns:
        .mot 파일 경로 (성공 시), None (실패 시)
    """
    try:
        import opensim  # noqa: F401 (설치 여부 확인용)
    except ImportError:
        return None

    try:
        import pose2sim
        config_path = _create_pose2sim_config(trc_path, output_dir)
        pose2sim.kinematics(config_path)
        mot_path = _find_mot_file(output_dir)
        return mot_path
    except Exception as e:
        print(f"[pose2sim_bridge] Pose2Sim/OpenSim 실패: {e}")
        return None


def _create_pose2sim_config(trc_path: str, output_dir: str) -> str:
    """Pose2Sim Config.toml을 생성합니다."""
    import toml  # Pose2Sim과 함께 설치됨
    config = {
        'project': {
            'project_dir': output_dir,
            'frame_rate': 30,
            'frame_range': [],
        },
        'kinematics': {
            'use_augmentation': False,
            'right_to_left_models': False,
            'default_height': 1.7,
        },
    }
    config_path = os.path.join(output_dir, 'Config.toml')
    with open(config_path, 'w') as f:
        toml.dump(config, f)
    return config_path


def _find_mot_file(output_dir: str) -> Optional[str]:
    """output_dir에서 .mot 파일을 탐색합니다."""
    for fname in os.listdir(output_dir):
        if fname.endswith('.mot'):
            return os.path.join(output_dir, fname)
    return None


def run_biomechanics_analysis(
    poses_json_path: str,
    output_dir: str,
    use_opensim: bool = True,
    body_mass_kg: float = 60.0,
    fps: float = 30.0,
) -> dict:
    """
    바이오메카닉스 분석을 실행합니다 (IK -> GRF -> ID -> SO -> JRA).

    Args:
        poses_json_path: MediaPipe 포즈 JSON 파일 경로.
        output_dir: 결과 파일 저장 디렉토리.
        use_opensim: OpenSim 사용 시도 여부 (False면 Python 직접 계산).
        body_mass_kg: 대상 체중 (ID, SO 등에 사용).
        fps: 데이터 프레임 레이트.

    Returns:
        분석 결과 dict (모듈 docstring 참조).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: TRC 파일 생성
    base_name = os.path.splitext(os.path.basename(poses_json_path))[0]
    trc_path = os.path.abspath(os.path.join(output_dir, f"{base_name}.trc"))
    print(f"[pose2sim_bridge] TRC 파일 생성 중: {trc_path}")
    export_trc(poses_json_path, trc_path)

    # 초기화
    ik_mot_path = None
    scaled_model_path = None
    grf_path = None
    id_sto_path = None
    so_result = {}
    jr_result = {}
    method = 'python'
    frame_angles = None

    # Step 2: OpenSim 파이프라인 시도
    if use_opensim and opensim_ik and opensim_dynamics:
        try:
            print("[pose2sim_bridge] OpenSim 파이프라인 시작...")
            
            # a) Scaling
            print("[pose2sim_bridge] OpenSim Scaling 실행 중...")
            scaled_model_path = opensim_ik.run_scaling(trc_path, output_dir, subject_mass_kg=body_mass_kg)
            
            # b) IK
            print("[pose2sim_bridge] OpenSim IK 실행 중...")
            ik_mot_path = opensim_ik.run_ik(trc_path, scaled_model_path, output_dir, fps=fps)
            
            if ik_mot_path and os.path.exists(ik_mot_path):
                print(f"[pose2sim_bridge] OpenSim IK 성공: {ik_mot_path}")
                method = 'opensim'
                frame_angles = opensim_ik.parse_mot_file(ik_mot_path)
                
                # c) GRF Estimation (quasi-static fallback for plie)
                print("[pose2sim_bridge] OpenSim GRF 추정 중...")
                grf_path = opensim_dynamics.estimate_grf(ik_mot_path, output_dir, body_mass_kg=body_mass_kg, trc_path=trc_path)
                
                if grf_path:
                    # d) Inverse Dynamics
                    print("[pose2sim_bridge] OpenSim Inverse Dynamics 실행 중...")
                    id_sto_path = opensim_dynamics.run_id(ik_mot_path, grf_path, scaled_model_path, output_dir)
                    
                    # e) Static Optimization (Muscles)
                    print("[pose2sim_bridge] OpenSim Static Optimization 실행 중...")
                    so_result = opensim_dynamics.run_static_optimization(ik_mot_path, grf_path, scaled_model_path, output_dir)
                
                # f) Joint Reaction Analysis
                print("[pose2sim_bridge] OpenSim Joint Reaction Analysis 실행 중...")
                # so_forces_path는 현재 SO 결과에서 추출하거나 None으로 전달
                jr_result = opensim_dynamics.run_joint_reaction(ik_mot_path, None, scaled_model_path, output_dir)
                
            else:
                print("[pose2sim_bridge] OpenSim IK 실패, Python fallback 사용")

        except Exception as e:
            print(f"[pose2sim_bridge] OpenSim 파이프라인 실행 중 오류: {e}")

    # Step 3: Fallback — 순수 Python 각도 계산
    if frame_angles is None:
        if method == 'python':
            print("[pose2sim_bridge] Python 관절 각도 계산 사용 (OpenSim 없음/실패)")
        frame_angles = extract_sequence_angles(poses_json_path)

    # Step 4: 요약 통계
    summary = compute_summary(frame_angles)

    result = {
        'trc_path': trc_path,
        'mot_path': ik_mot_path,        # 하위 호환성 유지
        'ik_mot_path': ik_mot_path,
        'scaled_model_path': scaled_model_path,
        'grf_path': grf_path,
        'id_sto_path': id_sto_path,
        'so_result': so_result,
        'jr_result': jr_result,
        'method':   method,
        'frames':   frame_angles,
        'summary':  summary,
    }

    return result


def save_biomechanics_json(result: dict, output_path: str) -> str:
    """
    바이오메카닉스 분석 결과를 JSON 파일로 저장합니다.

    Returns:
        저장된 파일의 절대 경로.
    """
    # JSON 직렬화 가능한 형태로 변환
    serializable = {
        'trc_path': result.get('trc_path'),
        'mot_path': result.get('mot_path'),
        'ik_mot_path': result.get('ik_mot_path'),
        'scaled_model_path': result.get('scaled_model_path'),
        'grf_path': result.get('grf_path'),
        'id_sto_path': result.get('id_sto_path'),
        'so_result': result.get('so_result', {}),
        'jr_result': result.get('jr_result', {}),
        'method':   result.get('method'),
        'frames':   result.get('frames', []),
        'summary':  result.get('summary', {}),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    abs_path = os.path.abspath(output_path)
    print(f"[pose2sim_bridge] 바이오메카닉스 결과 저장: {abs_path}")
    return abs_path


if __name__ == "__main__":
    import sys
    json_path  = sys.argv[1] if len(sys.argv) > 1 else "data/e2e_output/reference_poses.json"
    out_dir    = sys.argv[2] if len(sys.argv) > 2 else "data/e2e_output"
    result = run_biomechanics_analysis(json_path, out_dir, use_opensim=False)
    out_json = os.path.join(out_dir, "reference_biomechanics.json")
    save_biomechanics_json(result, out_json)
    print(f"\n발레 스코어:")
    for k, v in result['summary'].get('ballet_scores', {}).items():
        print(f"  {k}: {v:.3f}" if v is not None else f"  {k}: N/A")
