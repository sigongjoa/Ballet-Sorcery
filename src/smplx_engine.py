# -*- coding: utf-8 -*-
"""SMPL-X Fitting Engine (SMPLify-3D for Ballet).
MediaPipe 3D 관측값을 가이드로 하여 SMPL 파라메터를 최적화하고
고밀도 가상 마커를 추출합니다. (SMPL 6890 정점 모델 사용)

Phase 1: 2-stage 최적화 + 발레 각도 제한 prior
Phase 1+: VPoser v2.0 통합 (V02_05.zip)
"""

import os, glob
import torch
import torch.nn as nn
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
from typing import Dict, Tuple


# ── 발레 허용 관절 최대 회전각 (radian) ─────────────────────────────────────
_BALLET_MAX_ANGLES = torch.tensor([
    3.0, 3.0,  # L_Hip, R_Hip   — 아라베스크 허용
    1.2,       # spine1
    2.9, 2.9,  # L_Knee, R_Knee — 플리에 허용
    1.2,       # spine2
    1.6, 1.6,  # L_Ankle, R_Ankle
    1.2,       # spine3
    1.0, 1.0,  # L_Foot, R_Foot
    1.0,       # neck
    0.8, 0.8,  # L_Collar, R_Collar
    0.9,       # head
    3.0, 3.0,  # L_Shoulder, R_Shoulder — 포르드브라 허용
    2.6, 2.6,  # L_Elbow, R_Elbow
    1.0, 1.0,  # L_Wrist, R_Wrist
    0.5, 0.5,  # jaw, L_Eye
], dtype=torch.float32)


# ── VPoser v2.0 Decoder (순수 PyTorch 구현) ──────────────────────────────────
class _VPoserV2Decoder(nn.Module):
    """
    VPoser v2.0 decoder: z (32) → body_pose (69, axis-angle).
    V02_05 .ckpt 에서 'vp_model.' prefix 제거한 state_dict로 로드.

    Architecture (state_dict에서 역추출):
      decoder_net = Sequential(
        Linear(32→512),  # index 0
        ELU(),           # index 1  (no params)
        Dropout(0.1),    # index 2  (no params)
        Linear(512→512), # index 3
        ELU(),           # index 4  (no params)
        Linear(512→126), # index 5  → 21 joints × 6D rotation
      )
    """
    def __init__(self):
        super().__init__()
        self.decoder_net = nn.Sequential(
            nn.Linear(32, 512),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 126),   # 21 × 6
        )

    def _sixd_to_rotmat(self, x: torch.Tensor) -> torch.Tensor:
        """6D rotation → 3×3 rotation matrix (Gram-Schmidt).
        x: (..., 6) → out: (..., 3, 3)
        """
        a1 = x[..., :3]
        a2 = x[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        dot = (b1 * a2).sum(dim=-1, keepdim=True)
        b2 = F.normalize(a2 - dot * b1, dim=-1)
        b3 = torch.linalg.cross(b1, b2)
        return torch.stack([b1, b2, b3], dim=-1)   # (..., 3, 3)

    def _rotmat_to_aa(self, R: torch.Tensor) -> torch.Tensor:
        """3×3 rotation matrix → axis-angle (pure PyTorch).
        R: (..., 3, 3) → out: (..., 3)
        """
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6))
        # axis = skew-symmetric part / (2 sin θ)
        skew = torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ], dim=-1)
        sin_theta = torch.sin(theta).unsqueeze(-1).clamp(min=1e-6)
        axis = skew / (2 * sin_theta)
        return axis * theta.unsqueeze(-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, 32) → body_pose: (B, 69) axis-angle."""
        x = self.decoder_net(z)                    # (B, 126)
        x = x.view(-1, 21, 6)                      # (B, 21, 6)
        R = self._sixd_to_rotmat(x)                # (B, 21, 3, 3)
        aa = self._rotmat_to_aa(R)                 # (B, 21, 3)
        aa = aa.view(-1, 63)                       # (B, 63)
        # SMPL body_pose=69 (23 joints): 나머지 2개(jaw, eye) zeros 패딩
        pad = torch.zeros(aa.shape[0], 6, device=aa.device)
        return torch.cat([aa, pad], dim=1)         # (B, 69)


def _load_vposer_v2(vposer_dir: str) -> '_VPoserV2Decoder | None':
    """VPoser v2.0 .ckpt 로드. 경로 없으면 None 반환."""
    pattern = os.path.join(vposer_dir, 'V02_05', 'snapshots', '*.ckpt')
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        return None
    ckpt_path = ckpts[0]   # val_loss 낮은 epoch=08 우선 (파일명 정렬)
    try:
        raw = torch.load(ckpt_path, map_location='cpu')
        sd = raw['state_dict']
        # 'vp_model.' prefix 제거 후 decoder_net 만 추출
        decoder_sd = {k.replace('vp_model.', ''): v
                      for k, v in sd.items()
                      if k.startswith('vp_model.decoder_net')}
        decoder = _VPoserV2Decoder()
        # Dropout은 state_dict에 없으니 strict=False
        missing, unexpected = decoder.load_state_dict(decoder_sd, strict=False)
        decoder.eval()
        print(f"[SMPLX-Engine] VPoser v2.0 loaded: {os.path.basename(ckpt_path)}")
        if missing:
            print(f"  missing keys: {missing}")
        return decoder
    except Exception as e:
        print(f"[SMPLX-Engine] VPoser v2.0 load failed: {e}")
        return None


# ── 메인 엔진 ────────────────────────────────────────────────────────────────
class SMPLXEngine:
    """SMPL-X 엔진 클래스명 유지. 내부적으로 SMPL(6890 정점) 사용."""

    def __init__(self, model_dir: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 use_vposer: bool = False):
        """
        use_vposer: VPoser v2.0 활성화 여부.
          기본값 False — VPoser는 2D 재투영 손실 기반으로 학습됨.
          우리의 3D 월드 좌표 직접 매칭에는 각도 제한 prior가 더 효과적.
          향후 이미지 재투영 손실 도입 시 True로 전환.
        """
        self.device = device
        self.model_dir = model_dir
        print(f"[SMPLX-Engine] Initializing SMPL model on {device}...")

        self.model = smplx.create(model_dir, model_type='smpl',
                                  gender='neutral', ext='pkl').to(device)
        self.model.eval()

        # VPoser v2.0 — 명시적으로 활성화할 때만 로드
        self.vposer = None
        if use_vposer:
            vp_dir = os.path.normpath(os.path.join(model_dir, '..', 'vposer_v2_0'))
            self.vposer = _load_vposer_v2(vp_dir)
            if self.vposer is not None:
                self.vposer = self.vposer.to(device)
            else:
                print("[SMPLX-Engine] VPoser 로드 실패 — 각도 제한 prior 사용")

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

    # ── 각도 제한 prior ──────────────────────────────────────────────────────
    def _angle_limit_loss(self, body_pose: torch.Tensor) -> torch.Tensor:
        angles = body_pose.view(-1, 23, 3).norm(dim=-1)         # (1, 23)
        max_a = _BALLET_MAX_ANGLES.to(body_pose.device)
        excess = F.relu(angles - max_a.unsqueeze(0))
        return (excess ** 2).sum()

    # ── 단일 프레임 피팅 ─────────────────────────────────────────────────────
    def fit_frame(self, mp_world_landmarks: Dict,
                  num_iters: int = 200) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        2-stage SMPLify-3D.
        Stage 1 (30 iter): global_orient + transl 정렬
        Stage 2 (170 iter): 전체 파라미터 (VPoser z or body_pose)
        """
        # ── 타겟 구성 ────────────────────────────────────────────────────────
        target_joints = torch.zeros(1, 45, 3, device=self.device)
        conf = torch.zeros(1, 45, 1, device=self.device)
        landmark_names = list(mp_world_landmarks.keys())
        for mp_idx, sm_idx in self.mp_to_smplx_idx.items():
            if mp_idx < len(landmark_names):
                p = mp_world_landmarks[landmark_names[mp_idx]]
                target_joints[0, sm_idx] = torch.tensor(
                    [p['x'], -p['y'], p['z']], device=self.device)
                conf[0, sm_idx] = 1.0

        # ── 파라미터 초기화 ──────────────────────────────────────────────────
        betas         = torch.zeros(1, 10, device=self.device, requires_grad=True)
        global_orient = torch.zeros(1, 3,  device=self.device, requires_grad=True)
        transl        = torch.zeros(1, 3,  device=self.device, requires_grad=True)

        use_vposer = self.vposer is not None
        if use_vposer:
            z = torch.zeros(1, 32, device=self.device, requires_grad=True)
        else:
            body_pose = torch.zeros(1, 69, device=self.device, requires_grad=True)

        # ── Stage 1: global_orient + transl (T-pose 기준으로 정렬) ──────────
        s1_iters = min(30, num_iters)
        opt1 = torch.optim.Adam([global_orient, transl], lr=0.05)
        # Stage 1은 항상 T-pose(zeros)로 정렬 — VPoser z=0이 T-pose가 아니므로
        bp_frozen = torch.zeros(1, 69, device=self.device)

        for _ in range(s1_iters):
            opt1.zero_grad()
            out = self.model(betas=betas.detach(), body_pose=bp_frozen,
                             global_orient=global_orient, transl=transl)
            loss = (conf * (out.joints[:, :45] - target_joints) ** 2).sum()
            loss.backward()
            opt1.step()

        # ── Stage 2: 전체 파라미터 ───────────────────────────────────────────
        s2_iters = num_iters - s1_iters
        if use_vposer:
            # z는 더 높은 lr로, 나머지는 낮게
            opt2 = torch.optim.Adam([
                {'params': [z],             'lr': 0.05},
                {'params': [betas],         'lr': 0.01},
                {'params': [global_orient], 'lr': 0.01},
                {'params': [transl],        'lr': 0.01},
            ])
        else:
            opt2 = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=max(1, s2_iters), eta_min=1e-4)

        total_loss = torch.tensor(0.0)
        for _ in range(s2_iters):
            opt2.zero_grad()

            if use_vposer:
                bp = self.vposer(z)              # 그래디언트 통과
            else:
                bp = body_pose

            out = self.model(betas=betas, body_pose=bp,
                             global_orient=global_orient, transl=transl)
            mj = out.joints[:, :45]

            loss_joint = (conf * (mj - target_joints) ** 2).sum()
            loss_shape = (betas ** 2).sum() * 0.1
            if use_vposer:
                # z 정규화 매우 작게 — data term 우선
                loss_pose = (z ** 2).sum() * 0.001
            else:
                loss_pose  = (bp ** 2).sum() * 0.005
                loss_pose += self._angle_limit_loss(bp) * 2.0

            total_loss = loss_joint + loss_shape + loss_pose
            total_loss.backward()
            opt2.step()
            scheduler.step()

        # ── 결과 반환 ────────────────────────────────────────────────────────
        with torch.no_grad():
            final_bp = self.vposer(z) if use_vposer else body_pose
            final_out = self.model(betas=betas, body_pose=final_bp,
                                   global_orient=global_orient, transl=transl)
            vertices = final_out.vertices[0].cpu().numpy()
            joints   = final_out.joints[0].cpu().numpy()

        return vertices, joints, {
            "loss":        total_loss.item(),
            "joints":      joints,
            "vposer_used": use_vposer,
            "body_pose":   final_bp.detach().cpu().numpy(),   # (1, 69) GMM용
        }


if __name__ == "__main__":
    pass
