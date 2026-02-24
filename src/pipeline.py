# -*- coding: utf-8 -*-
"""
Ballet Jubeop(芭蕾呪法) 전체 파이프라인 통합 실행 스크립트.
포즈 추출, 비교, 시각화 단계를 하나의 명령으로 실행합니다.
"""

import os
import sys
import json
import argparse
import traceback
from typing import Dict, Any, List

# src 디렉토리를 sys.path에 추가하여 내부 모듈 임포트 가능하게 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 내부 모듈 임포트
from pose_extractor import extract_poses
from comparator import compare_poses
from visualizer import generate_loss_graph, generate_skeleton_comparison, generate_joint_heatmap
from pose2sim_bridge import run_biomechanics_analysis, save_biomechanics_json
from joint_angles import compare_angle_sequences

def run_pipeline(
    ref_video: str,
    comp_video: str,
    output_dir: str,
    *,
    use_world: bool = True,
    use_dtw: bool = True,
    target_fps: int = 24
) -> Dict[str, Any]:
    """
    전체 파이프라인을 실행합니다: 포즈 추출 -> 비교 -> 시각화.

    Args:
        ref_video (str): 기준(Reference) 영상 파일 경로.
        comp_video (str): 비교(Comparison) 영상 파일 경로.
        output_dir (str): 결과물 저장 디렉토리.
        use_world (bool): 3D 월드 좌표계 사용 여부.
        use_dtw (bool): DTW 알고리즘을 통한 시간축 정렬 사용 여부.
        target_fps (int): 추출할 목표 FPS.

    Returns:
        Dict[str, Any]: 생성된 파일 경로 정보를 포함한 결과 요약.
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    ref_poses_path = os.path.join(output_dir, "reference_poses.json")
    comp_poses_path = os.path.join(output_dir, "compare_poses.json")
    comparison_result_path = os.path.join(output_dir, "comparison_result.json")

    print(f"=== [Step 1] 포즈 추출 시작 ===")
    print(f"기준 영상: {ref_video}")
    extract_poses(ref_video, ref_poses_path, target_fps=target_fps)
    print(f"기준 포즈 추출 완료: {ref_poses_path}")

    print(f"비교 영상: {comp_video}")
    extract_poses(comp_video, comp_poses_path, target_fps=target_fps)
    print(f"비교 포즈 추출 완료: {comp_poses_path}")

    print(f"\n=== [Step 2] 포즈 비교 시작 (DTW: {use_dtw}, 3D: {use_world}) ===")
    result = compare_poses(
        ref_poses_path,
        comp_poses_path,
        use_world=use_world,
        use_dtw=use_dtw
    )
    
    with open(comparison_result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"비교 완료 및 결과 저장: {comparison_result_path}")

    print(f"\n=== [Step 3] 시각화 생성 시작 ===")
    
    # 1. Loss 그래프 생성
    loss_graph_path = os.path.join(output_dir, "loss_graph.png")
    generate_loss_graph(result, loss_graph_path)
    print(f"- Loss 그래프 생성 완료: {loss_graph_path}")

    # 2. 최대 오차 프레임 스켈레톤 비교 생성
    max_loss_frame = result.get('summary', {}).get('max_loss_frame', 0)
    skeleton_max_loss_path = os.path.join(output_dir, "skeleton_max_loss.png")
    generate_skeleton_comparison(
        ref_poses_path, 
        comp_poses_path, 
        max_loss_frame, 
        skeleton_max_loss_path,
        normalize=(not use_world) # 3D가 아닐 때만 시각화 시 정규화 적용
    )
    print(f"- 최대 오차 프레임({max_loss_frame}) 스켈레톤 비교 생성 완료: {skeleton_max_loss_path}")

    # 3. 관절별 히트맵(바 차트) 생성
    joint_heatmap_path = os.path.join(output_dir, "joint_heatmap.png")
    generate_joint_heatmap(result, joint_heatmap_path)
    print(f"- 관절별 오차 히트맵 생성 완료: {joint_heatmap_path}")

    print(f"\n=== [Step 4] 실행 결과 요약 ===")
    summary = result.get('summary', {})
    print(f"평균 오차 (Mean Loss): {summary.get('mean_loss', 0):.4f}")
    print(f"최대 오차 (Max Loss): {summary.get('max_loss', 0):.4f} (Frame: {max_loss_frame})")
    
    if use_dtw:
        meta = result.get('metadata', {})
        print(f"DTW 정렬 결과: Ref Start Frame {meta.get('start_frame_ref', 0)}, Comp Start Frame {meta.get('start_frame_comp', 0)}")
    
    print(f"\n=== [Step 5] 바이오메카닉스 분석 (Pose2Sim) ===")
    ref_bio_path  = os.path.join(output_dir, "reference_biomechanics.json")
    comp_bio_path = os.path.join(output_dir, "compare_biomechanics.json")
    bio_comparison_path = os.path.join(output_dir, "biomechanics_comparison.json")

    try:
        ref_bio  = run_biomechanics_analysis(ref_poses_path,  output_dir, use_opensim=True)
        comp_bio = run_biomechanics_analysis(comp_poses_path, output_dir, use_opensim=True)
        save_biomechanics_json(ref_bio,  ref_bio_path)
        save_biomechanics_json(comp_bio, comp_bio_path)

        # 두 시퀀스 관절 각도 비교
        bio_cmp = compare_angle_sequences(ref_bio['frames'], comp_bio['frames'])
        with open(bio_comparison_path, 'w', encoding='utf-8') as f:
            json.dump(bio_cmp, f, indent=2, ensure_ascii=False)

        method = ref_bio.get('method', 'python')
        print(f"바이오메카닉스 분석 완료 (method={method})")
        ref_scores  = ref_bio.get('summary', {}).get('ballet_scores', {})
        comp_scores = comp_bio.get('summary', {}).get('ballet_scores', {})
        for k in ref_scores:
            rv = ref_scores.get(k)
            cv = comp_scores.get(k)
            rv_s = f"{rv:.3f}" if rv is not None else "N/A"
            cv_s = f"{cv:.3f}" if cv is not None else "N/A"
            print(f"  {k}: Ref={rv_s}, Comp={cv_s}")
    except Exception as e:
        print(f"[경고] 바이오메카닉스 분석 실패 (무시하고 계속): {e}")
        ref_bio_path = None
        comp_bio_path = None
        bio_comparison_path = None

    print(f"\n모든 공정이 완료되었습니다. 결과물 확인: {output_dir}")

    return {
        'ref_poses': ref_poses_path,
        'comp_poses': comp_poses_path,
        'comparison': comparison_result_path,
        'visualizations': [
            loss_graph_path,
            skeleton_max_loss_path,
            joint_heatmap_path
        ],
        'biomechanics': {
            'ref':        ref_bio_path,
            'comp':       comp_bio_path,
            'comparison': bio_comparison_path,
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Ballet Jubeop 파이프라인 통합 실행 스크립트")
    parser.add_argument("--ref", required=True, help="기준(Reference) 영상 파일 경로")
    parser.add_argument("--comp", required=True, help="비교(Comparison) 영상 파일 경로")
    parser.add_argument("--output", default="output", help="결과물 저장 디렉토리 (기본값: output)")
    parser.add_argument("--no-world", action="store_true", help="3D 월드 좌표계 대신 2D 화면 좌표계 사용")
    parser.add_argument("--no-dtw", action="store_true", help="DTW 시간축 정렬 사용 안 함")
    parser.add_argument("--fps", type=int, default=24, help="포즈 추출 대상 FPS (기본값: 24)")

    args = parser.parse_args()

    try:
        run_pipeline(
            ref_video=args.ref,
            comp_video=args.comp,
            output_dir=args.output,
            use_world=not args.no_world,
            use_dtw=not args.no_dtw,
            target_fps=args.fps
        )
    except Exception as e:
        print(f"\n[오류] 파이프라인 실행 중 에러가 발생했습니다: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
