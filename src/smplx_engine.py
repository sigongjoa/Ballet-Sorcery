# -*- coding: utf-8 -*-
"""SMPL-X Fitting Engine (SMPLify-3D for Ballet).
MediaPipe 3D 관측값을 가이드로 하여 SMPL 파라메터를 최적화하고
고밀도 가상 마커를 추출합니다. (SMPL 6890 정점 모델 사용)

Phase 1 개선사항:
  - 2-stage 최적화 (Stage1: global_orient+transl, Stage2: 전체)
  - 200 iteration (30 + 170)
  - 발레 허용 각도 제한 prior (물리 불가능 포즈 억제, 발레 극단 포즈 허용)
  - VPoser 자동 활성화 (data/models/vposer_v1_0/ 존재 시)
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np

# Chumpy/SMPL compatibility monkey-patch
np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.complex = np.complex128
np.object = np.object_
np.unicode = np.str_
np.str = np.str_

import smplx
from typing import Dict, Optional, Tuple


# ── 발레 허용 관절 최대 회전각 (radian) ─────────────────────────────────────
# SMPL body_pose joint 순서 (1-indexed = global_orient 제외):
#   0=L_Hip, 1=R_Hip, 2=spine1, 3=L_Knee, 4=R_Knee, 5=spine2,
#   6=L_Ankle, 7=R_Ankle, 8=spine3, 9=L_Foot, 10=R_Foot, 11=neck,
#   12=L_Collar, 13=R_Collar, 14=head, 15=L_Shoulder, 16=R_Shoulder,
#   17=L_Elbow, 18=R_Elbow, 19=L_Wrist, 20=R_Wrist, 21=jaw, 22=L_Eye
_BALLET_MAX_ANGLES = torch.tensor([
    3.0,  # 0  L_Hip      — 아라베스크 170°+, 매우 관대
    3.0,  # 1  R_Hip      —  "
    1.2,  # 2  spine1     — 척추 굴곡 69°
    2.9,  # 3  L_Knee     — 플리에 165°
    2.9,  # 4  R_Knee     —  "
    1.2,  # 5  spine2     — 척추 굴곡
    1.6,  # 6  L_Ankle    — 발목 92°
    1.6,  # 7  R_Ankle    —  "
    1.2,  # 8  spine3     — 척추 회전
    1.0,  # 9  L_Foot     — 발 57°
    1.0,  # 10 R_Foot     —  "
    1.0,  # 11 neck       — 목 57°
    0.8,  # 12 L_Collar   — 쇄골 46°
    0.8,  # 13 R_Collar   —  "
    0.9,  # 14 head       — 머리 52°
    3.0,  # 15 L_Shoulder — 포르드브라 170°+
    3.0,  # 16 R_Shoulder —  "
    2.6,  # 17 L_Elbow    — 팔꿈치 149°
    2.6,  # 18 R_Elbow    —  "
    1.0,  # 19 L_Wrist    — 손목 57°
    1.0,  # 20 R_Wrist    —  "
    0.5,  # 21 jaw        — 턱 29°
    0.5,  # 22 L_Eye      — 눈 (무시용)
], dtype=torch.float32)


def _load_vposer(vposer_dir: str):
    """VPoser 체크포인트 로드. 파일 없으면 None 반환."""
    try:
        import glob
        snapshots = glob.glob(os.path.join(vposer_dir, 'snapshots', '*.pt'))
        if not snapshots:
            return None, None
        from human_body_prior.tools.model_loader import expid2model
        from human_body_prior.train.vposer_smpl import VPoser
        ps, best_model_fname = expid2model(vposer_dir)
        vposer = VPoser(ps)
        state = torch.load(best_model_fname, map_location='cpu')
        vposer.load_state_dict(state)
        vposer.eval()
        print(f"[SMPLX-Engine] VPoser loaded: {best_model_fname}")
        return vposer, ps
    except Exception as e:
        print(f"[SMPLX-Engine] VPoser not available: {e}")
        return None, None


class SMPLXEngine:
    """SMPL-X 엔진 클래스명을 유지하되 내부적으로 SMPL 모델을 사용하여 하위 호환성 유지."""

    def __init__(self, model_dir: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_dir = model_dir
        print(f"[SMPLX-Engine] Initializing SMPL model on {device}...")

        self.model = smplx.create(model_dir, model_type='smpl', gender='neutral',
                                  ext='pkl').to(device)
        self.model.eval()

        # VPoser 자동 로드 (체크포인트 존재 시)
        vposer_dir = os.path.join(model_dir, '..', 'vposer_v1_0')
        vposer_dir = os.path.normpath(vposer_dir)
        self.vposer, self.vposer_ps = _load_vposer(vposer_dir)
        if self.vposer is not None:
            self.vposer = self.vposer.to(device)

        # MediaPipe (33 landmarks) → SMPL joint 인덱스 매핑
        self.mp_to_smplx_idx = {
            11: 16, 12: 17,  # Shoulders
            13: 18, 14: 19,  # Elbows
            15: 20, 16: 21,  # Wrists
            23: 1,  24: 2,   # Hips
            25: 4,  26: 5,   # Knees
            27: 7,  28: 8,   # Ankles
            0:  15,          # Head/Nose
        }

    # ── 내부: 각도 제한 prior ────────────────────────────────────────────────
    def _angle_limit_loss(self, body_pose: torch.Tensor) -> torch.Tensor:
        """발레 허용 범위를 초과하는 관절 회전에 패널티."""
        angles = body_pose.view(1, 23, 3).norm(dim=-1)          # (1, 23)
        max_a = _BALLET_MAX_ANGLES.to(body_pose.device)          # (23,)
        excess = F.relu(angles - max_a.unsqueeze(0))             # (1, 23)
        return (excess ** 2).sum()

    # ── 내부: VPoser 기반 body_pose 디코딩 ──────────────────────────────────
    def _vposer_decode(self, z: torch.Tensor) -> torch.Tensor:
        """z (1, 32) → body_pose (1, 63) axis-angle (21 joints)."""
        with torch.no_grad():
            body_pose_aa = self.vposer.decode(z, output_type='aa')
        # VPoser는 21개 관절 출력 → SMPL 23관절에 맞게 패딩
        bp = body_pose_aa.view(1, -1)   # (1, 63)
        if bp.shape[1] < 69:
            pad = torch.zeros(1, 69 - bp.shape[1], device=bp.device)
            bp = torch.cat([bp, pad], dim=1)
        return bp

    # ── 공개: 단일 프레임 피팅 ───────────────────────────────────────────────
    def fit_frame(self, mp_world_landmarks: Dict,
                  num_iters: int = 200) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        2-stage SMPLify-3D 최적화.

        Stage 1 (30 iter):  global_orient + transl — 전체 위치/방향 정렬
        Stage 2 (num_iters-30 iter): 전체 파라미터 — 세부 포즈 피팅
        """
        # ── 1. 타겟 텐서 구성 ────────────────────────────────────────────────
        target_joints = torch.zeros(1, 45, 3, device=self.device)
        conf = torch.zeros(1, 45, 1, device=self.device)

        landmark_names = list(mp_world_landmarks.keys())
        for mp_idx, sm_idx in self.mp_to_smplx_idx.items():
            if mp_idx < len(landmark_names):
                name = landmark_names[mp_idx]
                p = mp_world_landmarks[name]
                target_joints[0, sm_idx] = torch.tensor(
                    [p['x'], -p['y'], p['z']], device=self.device)
                conf[0, sm_idx] = 1.0

        # ── 2. 최적화 변수 초기화 ────────────────────────────────────────────
        body_pose    = torch.zeros(1, 69, device=self.device, requires_grad=True)
        betas        = torch.zeros(1, 10, device=self.device, requires_grad=True)
        global_orient = torch.zeros(1, 3, device=self.device, requires_grad=True)
        transl       = torch.zeros(1, 3, device=self.device, requires_grad=True)

        # VPoser 사용 시: z를 최적화, body_pose는 decode 결과로 대체
        use_vposer = self.vposer is not None
        if use_vposer:
            z = torch.zeros(1, 32, device=self.device, requires_grad=True)

        # ── Stage 1: 전체 위치/방향 정렬 (global_orient + transl) ────────────
        s1_params = [global_orient, transl]
        opt1 = torch.optim.Adam(s1_params, lr=0.05)
        s1_iters = min(30, num_iters)

        for i in range(s1_iters):
            opt1.zero_grad()
            if use_vposer:
                bp = self._vposer_decode(z).detach()
            else:
                bp = body_pose.detach()

            out = self.model(betas=betas.detach(),
                             body_pose=bp,
                             global_orient=global_orient,
                             transl=transl)
            mj = out.joints[:, :45]
            loss = (conf * (mj - target_joints) ** 2).sum()
            loss.backward()
            opt1.step()

        # ── Stage 2: 전체 파라미터 피팅 ─────────────────────────────────────
        if use_vposer:
            s2_params = [z, betas, global_orient, transl]
        else:
            s2_params = [body_pose, betas, global_orient, transl]

        opt2 = torch.optim.Adam(s2_params, lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=num_iters - s1_iters, eta_min=1e-4)

        total_loss = torch.tensor(0.0)
        for i in range(num_iters - s1_iters):
            opt2.zero_grad()

            if use_vposer:
                bp = self._vposer_decode(z)
                bp.requires_grad_(True)
            else:
                bp = body_pose

            out = self.model(betas=betas, body_pose=bp,
                             global_orient=global_orient, transl=transl)
            mj = out.joints[:, :45]

            loss_joint = (conf * (mj - target_joints) ** 2).sum()
            loss_shape = (betas ** 2).sum() * 0.1

            if use_vposer:
                loss_pose = (z ** 2).sum() * 0.1          # z latent 정규화
            else:
                loss_pose  = (bp ** 2).sum() * 0.005      # L2 (낮게)
                loss_pose += self._angle_limit_loss(bp) * 2.0  # 발레 각도 제한

            total_loss = loss_joint + loss_shape + loss_pose
            total_loss.backward()
            opt2.step()
            scheduler.step()

        # ── 3. 결과 반환 ─────────────────────────────────────────────────────
        with torch.no_grad():
            if use_vposer:
                final_bp = self._vposer_decode(z).detach()
            else:
                final_bp = body_pose

            final_out = self.model(betas=betas, body_pose=final_bp,
                                   global_orient=global_orient, transl=transl)
            vertices = final_out.vertices[0].cpu().numpy()
            joints   = final_out.joints[0].cpu().numpy()

        return vertices, joints, {
            "loss": total_loss.item(),
            "joints": joints,
            "vposer_used": use_vposer,
        }


if __name__ == "__main__":
    pass
