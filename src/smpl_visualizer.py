# src/smpl_visualizer.py
# SMPL mesh visualization using joints2smpl's proven 2-stage optimization + GMM prior
# Based on: https://github.com/wangsen1312/joints2smpl

import numpy as np
import torch
import pyvista as pv
import json
import os
import sys
import pickle
from typing import List

pv.start_xvfb()

# Monkey-patch numpy for chumpy compatibility
np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.complex = np.complex128
np.object = np.object_
np.unicode = np.str_
np.str = np.str_

import smplx

# ============================================================
# GMM Pose Prior (from joints2smpl/VIBE)
# ============================================================
class MaxMixturePrior(torch.nn.Module):
    """Gaussian Mixture Model prior over SMPL body pose (from VIBE/SMPLify)."""
    def __init__(self, prior_folder, num_gaussians=8, dtype=torch.float32):
        super().__init__()
        with open(os.path.join(prior_folder, f'gmm_{num_gaussians:02d}.pkl'), 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        means = gmm['means'].astype(np.float32)
        covs = gmm['covars'].astype(np.float32)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))
        precisions = np.stack([np.linalg.inv(c) for c in covs]).astype(np.float32)
        self.register_buffer('precisions', torch.tensor(precisions, dtype=dtype))

        sqrdets = np.array([np.sqrt(max(np.linalg.det(c), 1e-300)) for c in covs])
        const = (2 * np.pi) ** (69 / 2.)
        nll_weights = gmm['weights'] / (const * (sqrdets / max(sqrdets.min(), 1e-300)))
        self.register_buffer('nll_weights', torch.tensor(nll_weights, dtype=dtype).unsqueeze(0))

    def forward(self, pose, betas=None):
        diff = pose.unsqueeze(1) - self.means  # (B, K, 69)
        prec_diff = torch.einsum('mij,bmj->bmi', self.precisions, diff)
        quadratic = (prec_diff * diff).sum(dim=-1)  # (B, K)
        curr_ll = 0.5 * quadratic - torch.log(self.nll_weights)
        return torch.min(curr_ll, dim=1)[0]


# ============================================================
# Angle prior (penalizes unnatural knee/elbow bending)
# ============================================================
def angle_prior(pose):
    return torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] *
        torch.tensor([1., -1., -1, -1.], device=pose.device)
    ) ** 2


# ============================================================
# MediaPipe → SMPL joint mapping
# ============================================================
# MediaPipe 33 landmarks → SMPL 24 joints (22 used by joints2smpl AMASS format)
# We map what we can and set confidence=0 for unmapped joints
MEDIAPIPE_TO_SMPL_22 = {
    # MediaPipe name → SMPL AMASS index (0-21)
    'LEFT_HIP': 1, 'RIGHT_HIP': 2,
    'LEFT_KNEE': 4, 'RIGHT_KNEE': 5,
    'LEFT_ANKLE': 7, 'RIGHT_ANKLE': 8,
    'LEFT_FOOT_INDEX': 10, 'RIGHT_FOOT_INDEX': 11,
    'LEFT_SHOULDER': 16, 'RIGHT_SHOULDER': 17,
    'LEFT_ELBOW': 18, 'RIGHT_ELBOW': 19,
    'LEFT_WRIST': 20, 'RIGHT_WRIST': 21,
    'NOSE': 15,  # Head
}


def mediapipe_to_smpl_joints(landmarks_dict, num_joints=22):
    """Convert MediaPipe world_landmarks dict to (22, 3) numpy array for SMPL fitting.
    Returns joints array and confidence array."""
    joints = np.zeros((num_joints, 3), dtype=np.float32)
    confidence = np.zeros(num_joints, dtype=np.float32)

    # Pelvis (joint 0) = midpoint of hips
    lh = landmarks_dict.get('LEFT_HIP')
    rh = landmarks_dict.get('RIGHT_HIP')
    if lh and rh:
        joints[0] = [(lh['x'] + rh['x']) / 2,
                      -(lh['y'] + rh['y']) / 2,   # Flip Y: MediaPipe Y-down → SMPL Y-up
                      (lh['z'] + rh['z']) / 2]
        confidence[0] = 1.0

    for mp_name, smpl_idx in MEDIAPIPE_TO_SMPL_22.items():
        if mp_name in landmarks_dict:
            p = landmarks_dict[mp_name]
            joints[smpl_idx] = [p['x'], -p['y'], p['z']]  # Flip Y
            confidence[smpl_idx] = 1.0

    return joints, confidence


# ============================================================
# 2-Stage SMPLify3D fitting (from joints2smpl)
# ============================================================
def fit_smpl_to_joints(target_joints, confidence, smpl_model, pose_prior,
                       device='cuda', num_iters=150):
    """
    Fit SMPL model to 3D joint targets using 2-stage optimization.
    Stage 1: Global orientation + translation (body frozen)
    Stage 2: Full body pose + shape (with GMM prior)

    Returns: vertices (6890,3), joints (22,3), pose (72,), betas (10,), translation (3,)
    """
    batch_size = 1

    # Prepare inputs
    j3d = torch.tensor(target_joints, dtype=torch.float32, device=device).unsqueeze(0)  # (1, J, 3)
    conf = torch.tensor(confidence, dtype=torch.float32, device=device)  # (J,)

    # Initialize parameters
    global_orient = torch.zeros(1, 3, device=device, dtype=torch.float32)
    body_pose = torch.zeros(1, 69, device=device, dtype=torch.float32)
    betas = torch.zeros(1, 10, device=device, dtype=torch.float32)
    transl = torch.zeros(1, 3, device=device, dtype=torch.float32)

    # Initialize translation from torso joints
    smpl_out = smpl_model(global_orient=global_orient, body_pose=body_pose, betas=betas)
    model_joints = smpl_out.joints[:, :22]  # First 22 are body joints

    # Use hip+shoulder average to estimate initial translation
    torso_idx = [0, 1, 2, 16, 17]  # pelvis, hips, shoulders
    valid_torso = [i for i in torso_idx if confidence[i] > 0]
    if valid_torso:
        init_t = (j3d[0, valid_torso] - model_joints[0, valid_torso]).mean(dim=0)
        transl[0] = init_t.detach()

    # ---- Stage 1: Optimize global orientation + translation ----
    global_orient.requires_grad_(True)
    transl.requires_grad_(True)
    body_pose.requires_grad_(False)
    betas.requires_grad_(False)

    smpl_idx = list(range(22))
    opt1 = torch.optim.Adam([global_orient, transl], lr=0.02)

    for i in range(50):
        opt1.zero_grad()
        out = smpl_model(global_orient=global_orient, body_pose=body_pose, betas=betas)
        mj = out.joints[:, :22] + transl.unsqueeze(1)
        diff = (mj[:, smpl_idx] - j3d[:, smpl_idx]) ** 2
        loss = (conf[smpl_idx].unsqueeze(0).unsqueeze(-1) * diff).sum()
        loss.backward()
        opt1.step()
    print(f"  Stage 1 done: orient_loss={loss.item():.4f}")

    # ---- Stage 2: Optimize full body (with GMM prior) ----
    global_orient.requires_grad_(True)
    body_pose.requires_grad_(True)
    betas.requires_grad_(True)
    transl.requires_grad_(True)

    preserve_pose = body_pose.detach().clone()
    opt2 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=0.01)

    for step in range(num_iters):
        opt2.zero_grad()
        out = smpl_model(global_orient=global_orient, body_pose=body_pose, betas=betas)
        mj = out.joints[:, :22] + transl.unsqueeze(1)

        # Joint loss (weighted by confidence)
        diff = (mj[:, smpl_idx] - j3d[:, smpl_idx]) ** 2
        joint_loss = 600.0 * (conf[smpl_idx].unsqueeze(0).unsqueeze(-1) * diff).sum()

        # GMM pose prior
        prior_loss = 7.17 * pose_prior(body_pose)

        # Angle prior (knees/elbows)
        angle_loss = 15.2 ** 2 * angle_prior(body_pose).sum(dim=-1)

        # Shape regularization
        shape_loss = 5.0 ** 2 * (betas ** 2).sum(dim=-1)

        # Pose preservation
        preserve_loss = 5.0 * ((body_pose - preserve_pose) ** 2).sum(dim=-1)

        total = joint_loss + prior_loss.sum() + angle_loss.sum() + shape_loss.sum() + preserve_loss.sum()
        total.backward()
        opt2.step()

        if step % 50 == 0:
            print(f"  Stage 2 iter {step}: joint={joint_loss.item():.2f}, prior={prior_loss.sum().item():.2f}, total={total.item():.2f}")

    # Get final output
    with torch.no_grad():
        final_out = smpl_model(global_orient=global_orient, body_pose=body_pose, betas=betas)
        verts = (final_out.vertices[0] + transl).cpu().numpy()
        joints_np = (final_out.joints[0, :22] + transl[0]).cpu().numpy()
        pose = torch.cat([global_orient[0], body_pose[0]]).cpu().numpy()

    return verts, joints_np, pose, betas[0].detach().cpu().numpy(), transl[0].detach().cpu().numpy()


def normalize_smpl_mesh(verts, joints):
    """
    Normalize SMPL mesh based on pelvis and shoulder width.
    Args:
        verts: (6890, 3) vertices in world coordinates (with translation)
        joints: (22, 3) joints in world coordinates (with translation)
    Returns:
        verts_norm, joints_norm, scale
    """
    pelvis = joints[0]
    shoulder_width = np.linalg.norm(joints[16] - joints[17])
    scale = 1.0 / max(shoulder_width, 1e-6)

    verts_norm = (verts - pelvis) * scale
    joints_norm = (joints - pelvis) * scale

    return verts_norm, joints_norm, scale


# ============================================================
# Rendering functions
# ============================================================
def render_smpl_sidebyside(ref_verts, comp_verts, faces, output_path, title=''):
    """
    Render Reference and Comparison normalized meshes side-by-side.
    """
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[1920, 960])
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).flatten()

    # Left: Reference
    plotter.subplot(0, 0)
    plotter.add_mesh(pv.PolyData(ref_verts, pv_faces), color='royalblue', smooth_shading=True)
    plotter.camera_position = [(0, 0, 4), (0, 0, 0), (0, 1, 0)]
    plotter.add_text('Reference (Normalized)', position='upper_left', color='black', font_size=12)
    plotter.set_background('white')

    # Right: Compare
    plotter.subplot(0, 1)
    plotter.add_mesh(pv.PolyData(comp_verts, pv_faces), color='tomato', smooth_shading=True)
    plotter.camera_position = [(0, 0, 4), (0, 0, 0), (0, 1, 0)]
    plotter.add_text('Compare (Normalized)', position='upper_left', color='black', font_size=12)
    plotter.set_background('white')

    plotter.add_title(title, font_size=12, color='black')
    plotter.show(screenshot=output_path)
    plotter.close()


def render_mesh_comparison(ref_verts, comp_verts, faces, output_path, title=''):
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).flatten()

    plotter.add_mesh(pv.PolyData(ref_verts, pv_faces), color='royalblue', opacity=0.5, smooth_shading=True)
    plotter.add_mesh(pv.PolyData(comp_verts, pv_faces), color='tomato', opacity=0.5, smooth_shading=True)

    center = (ref_verts.mean(0) + comp_verts.mean(0)) / 2
    ext = max(np.ptp(ref_verts, axis=0).max(), np.ptp(comp_verts, axis=0).max())
    plotter.camera_position = [
        (center[0], center[1], center[2] + ext * 2.5),
        tuple(center), (0, 1, 0)
    ]
    plotter.add_text(title, position='upper_left', color='black', font_size=12)
    plotter.set_background('white')
    plotter.show(screenshot=output_path)
    plotter.close()


def render_mesh_multiview(verts, faces, output_path, title='', color='lightgray'):
    plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=[1920, 1080])
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).flatten()
    mesh = pv.PolyData(verts, pv_faces)

    center = verts.mean(0)
    d = np.ptp(verts, axis=0).max() * 2.5
    views = [
        ("Front", (center[0], center[1], center[2] + d)),
        ("Side", (center[0] - d, center[1], center[2])),
        ("Back", (center[0], center[1], center[2] - d)),
        ("3/4 View", (center[0] - d*0.7, center[1] + d*0.5, center[2] + d*0.7)),
    ]
    for idx, (label, cam) in enumerate(views):
        plotter.subplot(idx // 2, idx % 2)
        plotter.add_mesh(mesh, color=color, smooth_shading=True, show_edges=False)
        plotter.camera_position = [cam, tuple(center), (0, 1, 0)]
        plotter.add_text(label, position='lower_left', font_size=10, color='black')
        plotter.set_background('white')

    plotter.add_title(title, font_size=14, color='black')
    plotter.show(screenshot=output_path)
    plotter.close()


# ============================================================
# Main pipeline
# ============================================================
def generate_smpl_comparison(ref_json_path, comp_json_path, frame_idx, output_dir,
                              smpl_model_dir="data/models/smpl",
                              num_iters=150):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load SMPL model
    smpl_model = smplx.create(smpl_model_dir, model_type='smpl', gender='neutral',
                               ext='pkl', batch_size=1).to(device)
    faces = smpl_model.faces

    # Load GMM pose prior
    pose_prior = MaxMixturePrior(prior_folder=smpl_model_dir, num_gaussians=8).to(device)
    print("SMPL model + GMM prior loaded")

    # Load pose data
    with open(ref_json_path) as f:
        ref_data = json.load(f)
    with open(comp_json_path) as f:
        comp_data = json.load(f)

    ref_lm = ref_data['frames'][frame_idx]['world_landmarks']
    comp_lm = comp_data['frames'][frame_idx]['world_landmarks']

    # Convert MediaPipe → SMPL joint format
    ref_joints, ref_conf = mediapipe_to_smpl_joints(ref_lm)
    comp_joints, comp_conf = mediapipe_to_smpl_joints(comp_lm)

    print(f"Mapped joints: ref={int(ref_conf.sum())}, comp={int(comp_conf.sum())}")

    # Fit SMPL
    print(f"Fitting reference pose (frame {frame_idx})...")
    ref_verts, ref_joints_fit, ref_pose, _, _ = fit_smpl_to_joints(
        ref_joints, ref_conf, smpl_model, pose_prior, device, num_iters)
    print(f"  pose norm: {np.linalg.norm(ref_pose):.3f}")

    print(f"Fitting comparison pose (frame {frame_idx})...")
    comp_verts, comp_joints_fit, comp_pose, _, _ = fit_smpl_to_joints(
        comp_joints, comp_conf, smpl_model, pose_prior, device, num_iters)
    print(f"  pose norm: {np.linalg.norm(comp_pose):.3f}")

    os.makedirs(output_dir, exist_ok=True)

    # Normalize
    ref_verts_n, _, _ = normalize_smpl_mesh(ref_verts, ref_joints_fit)
    comp_verts_n, _, _ = normalize_smpl_mesh(comp_verts, comp_joints_fit)

    # Render
    print("Rendering comparison overlay...")
    render_mesh_comparison(ref_verts, comp_verts, faces,
        os.path.join(output_dir, f"smpl_comparison_frame_{frame_idx}.png"),
        f"Frame {frame_idx}: Reference (Blue) vs Comparison (Red)")

    print("Rendering normalized side-by-side...")
    render_smpl_sidebyside(ref_verts_n, comp_verts_n, faces,
        os.path.join(output_dir, f"smpl_normalized_frame_{frame_idx}.png"),
        f"Frame {frame_idx}: Normalized Reference vs Compare")

    print("Rendering reference multiview...")
    render_mesh_multiview(ref_verts, faces,
        os.path.join(output_dir, f"smpl_ref_multiview_frame_{frame_idx}.png"),
        f"Frame {frame_idx}: Reference", color='royalblue')

    print("Rendering comparison multiview...")
    render_mesh_multiview(comp_verts, faces,
        os.path.join(output_dir, f"smpl_comp_multiview_frame_{frame_idx}.png"),
        f"Frame {frame_idx}: Comparison", color='tomato')

    print("Done!")


def generate_smpl_sequence(ref_json_path: str, comp_json_path: str, output_dir: str, *,
                           frame_indices: List[int] = None, max_frames: int = 10,
                           smpl_model_dir: str = 'data/models/smpl',
                           num_iters: int = 100) -> List[str]:
    """
    여러 프레임에 대해 SMPL 피팅 및 시퀀스 이미지를 생성합니다.
    OpenCV가 설치되어 있으면 MP4 애니메이션도 생성합니다.

    Args:
        ref_json_path: 참조 3D 포즈 JSON 경로
        comp_json_path: 비교 3D 포즈 JSON 경로
        output_dir: 결과 저장 디렉토리
        frame_indices: 처리할 프레임 인덱스 목록 (None이면 max_frames만큼 샘플링)
        max_frames: 균등 샘플링할 최대 프레임 수
        smpl_model_dir: SMPL 모델 파일 경로
        num_iters: 최적화 반복 횟수

    Returns:
        생성된 파일 경로 목록
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # SMPL 모델 및 GMM prior 로드 (한 번만)
    smpl_model = smplx.create(smpl_model_dir, model_type='smpl', gender='neutral',
                               ext='pkl', batch_size=1).to(device)
    faces = smpl_model.faces
    pose_prior = MaxMixturePrior(prior_folder=smpl_model_dir, num_gaussians=8).to(device)
    print("SMPL model + GMM prior loaded")

    with open(ref_json_path) as f:
        ref_data = json.load(f)
    with open(comp_json_path) as f:
        comp_data = json.load(f)

    ref_frames = ref_data['frames']
    comp_frames = comp_data['frames']
    total_frames = min(len(ref_frames), len(comp_frames))

    if frame_indices is None:
        step = max(1, total_frames // max_frames)
        frame_indices = list(range(0, total_frames, step))[:max_frames]

    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    for i, frame_idx in enumerate(frame_indices):
        print(f'[{i+1}/{len(frame_indices)}] Frame {frame_idx} 처리 중...')

        ref_lm = ref_frames[frame_idx]['world_landmarks']
        comp_lm = comp_frames[frame_idx]['world_landmarks']

        # Convert MediaPipe → SMPL joint format
        ref_joints, ref_conf = mediapipe_to_smpl_joints(ref_lm)
        comp_joints, comp_conf = mediapipe_to_smpl_joints(comp_lm)

        # Fit SMPL
        ref_verts, ref_joints_fit, _, _, _ = fit_smpl_to_joints(
            ref_joints, ref_conf, smpl_model, pose_prior, device, num_iters)
        comp_verts, comp_joints_fit, _, _, _ = fit_smpl_to_joints(
            comp_joints, comp_conf, smpl_model, pose_prior, device, num_iters)

        # Normalize
        ref_verts_n, _, _ = normalize_smpl_mesh(ref_verts, ref_joints_fit)
        comp_verts_n, _, _ = normalize_smpl_mesh(comp_verts, comp_joints_fit)

        # Render comparison image
        output_path = os.path.join(output_dir, f"smpl_seq_frame_{frame_idx:04d}.png")
        render_smpl_sidebyside(ref_verts_n, comp_verts_n, faces, output_path,
                               f"Frame {frame_idx}: Ref vs Comp (Normalized)")
        generated_files.append(output_path)

    # OpenCV를 이용한 MP4 애니메이션 생성 (선택사항)
    try:
        import cv2
        if generated_files:
            video_path = os.path.join(output_dir, "smpl_sequence.mp4")
            first_img = cv2.imread(generated_files[0])
            if first_img is not None:
                height, width, _ = first_img.shape
                # fps=4: 각 포즈를 충분히 관찰할 수 있는 속도
                video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 4, (width, height))
                for img_path in generated_files:
                    img = cv2.imread(img_path)
                    if img is not None:
                        video.write(img)
                video.release()
                generated_files.append(video_path)
                print(f"Video saved to {video_path}")
    except ImportError:
        print("OpenCV (cv2) not found. Skipping MP4 generation.")
    except Exception as e:
        print(f"Failed to generate video: {e}")

    return generated_files


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SMPL mesh visualization from 3D pose data")
    parser.add_argument("--ref", required=True, help="Reference 3D pose JSON")
    parser.add_argument("--comp", required=True, help="Comparison 3D pose JSON")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--iters", type=int, default=150, help="Optimization iterations")
    parser.add_argument("--seq", action="store_true", help="Generate multi-frame sequence")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames for sequence")
    args = parser.parse_args()

    if args.seq:
        generate_smpl_sequence(args.ref, args.comp, args.output,
                               max_frames=args.max_frames, num_iters=args.iters)
    else:
        generate_smpl_comparison(args.ref, args.comp, args.frame, args.output,
                                  num_iters=args.iters)
