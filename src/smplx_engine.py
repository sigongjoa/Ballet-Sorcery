# -*- coding: utf-8 -*-
"""SMPL-X Fitting Engine (SMPLify-3D for Ballet).
MediaPipe 3D 관측값을 가이드로 하여 SMPL 파라메터를 최적화하고
고밀도 가상 마커를 추출합니다. (SMPL 6890 정점 모델 사용)
"""

import os
import torch
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

class SMPLXEngine:
    """SMPL-X 엔진 클래스명을 유지하되 내부적으로 SMPL 모델을 사용하여 하위 호환성 유지."""
    def __init__(self, model_dir: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"[SMPLX-Engine] Initializing SMPL model on {device}...")
        
        # Neutral SMPL 모델 로드 (data/models/smpl/ SMPL_NEUTRAL.pkl 등)
        self.model = smplx.create(model_dir, model_type='smpl', gender='neutral', 
                                  ext='pkl').to(device)
        self.model.eval()
        
        # MediaPipe (33 joints) -> SMPL (23 joints system) 매핑
        # SMPL 23관절: 0=pelvis, 1=L_Hip, 2=R_Hip, 3=spine1, 4=L_Knee, 5=R_Knee, 
        # 6=spine2, 7=L_Ankle, 8=R_Ankle, 9=spine3, 10=L_Foot, 11=R_Foot, 12=neck, 
        # 13=L_Collar, 14=R_Collar, 15=head, 16=L_Shoulder, 17=R_Shoulder, 
        # 18=L_Elbow, 19=R_Elbow, 20=L_Wrist, 21=R_Wrist, 22=jaw
        self.mp_to_smplx_idx = {
            11: 16, 12: 17, # Shoulders
            13: 18, 14: 19, # Elbows
            15: 20, 16: 21, # Wrists
            23: 1,  24: 2,  # Hips
            25: 4,  26: 5,  # Knees
            27: 7,  28: 8,  # Ankles
            0: 15,          # Head/Nose
        }

    def fit_frame(self, mp_world_landmarks: Dict, num_iters: int = 100) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """단일 프레임에 대해 SMPL 파라메터 최적화 수행."""
        # 1. MediaPipe 데이터를 Torch Tensor로 변환 (Y-up 반전 포함)
        # SMPL output joints는 기본적으로 45개
        target_joints = torch.zeros(1, 45, 3, device=self.device)
        conf = torch.zeros(1, 45, 1, device=self.device)
        
        landmark_names = list(mp_world_landmarks.keys())
        for mp_idx, sm_idx in self.mp_to_smplx_idx.items():
            if mp_idx < len(landmark_names):
                name = landmark_names[mp_idx]
                p = mp_world_landmarks[name]
                # MediaPipe world coordinates (meters)
                target_joints[0, sm_idx] = torch.tensor([p['x'], -p['y'], p['z']], device=self.device)
                conf[0, sm_idx] = 1.0

        # 2. 최적화 변수 설정
        # SMPL body_pose: (24-1)*3 = 69 (23 body joints)
        body_pose = torch.zeros(1, 69, device=self.device, requires_grad=True)
        betas = torch.zeros(1, 10, device=self.device, requires_grad=True)
        global_orient = torch.zeros(1, 3, device=self.device, requires_grad=True)
        transl = torch.zeros(1, 3, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([body_pose, betas, global_orient, transl], lr=0.01)

        # 3. 최적화 루프
        for i in range(num_iters):
            optimizer.zero_grad()
            output = self.model(betas=betas, body_pose=body_pose, 
                                global_orient=global_orient, transl=transl)
            
            # Joint Loss (MediaPipe 가이드라인 준수)
            # SMPL output joints count is 45
            mj = output.joints[:, :45]
            diff = (mj - target_joints) ** 2
            loss_joint = (conf * diff).sum()
            
            # Prior Loss (Regularization)
            loss_shape = (betas ** 2).sum() * 0.1
            loss_pose = (body_pose ** 2).sum() * 0.01
            
            total_loss = loss_joint + loss_shape + loss_pose
            total_loss.backward()
            optimizer.step()

        # 4. 결과 반환
        with torch.no_grad():
            final_output = self.model(betas=betas, body_pose=body_pose, 
                                      global_orient=global_orient, transl=transl)
            vertices = final_output.vertices[0].cpu().numpy()
            joints = final_output.joints[0].cpu().numpy()
            
        return vertices, joints, {"loss": total_loss.item(), "joints": joints}

if __name__ == "__main__":
    pass
