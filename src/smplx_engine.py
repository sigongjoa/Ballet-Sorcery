# -*- coding: utf-8 -*-
"""SMPL-X Fitting Engine (SMPLify-3D for Ballet).
MediaPipe 3D 관측값을 가이드로 하여 SMPL-X 파라메터를 최적화하고
56개의 고밀도 가상 마커를 추출합니다.
"""

import os
import torch
import numpy as np
import smplx
from typing import Dict, Tuple

class SMPLXEngine:
    def __init__(self, model_dir: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"[SMPLX-Engine] Initializing on {device}...")
        
        # Neutral SMPL-X 모델 로드
        self.model = smplx.create(model_dir, model_type='smplx', gender='neutral', 
                                  ext='pkl', use_pca=True, num_pca_comps=12).to(device)
        self.model.eval()
        
        # SMPL-X (54 joints) -> MediaPipe (33 joints) 매핑 테이블
        # 핵심 16개 관절 위주로 가이드라인 설정
        self.mp_to_smplx_idx = {
            11: 16, 12: 17, # Shoulders
            13: 18, 14: 19, # Elbows
            15: 20, 16: 21, # Wrists
            23: 1,  24: 2,  # Hips
            25: 4,  26: 5,  # Knees
            27: 7,  28: 8,  # Ankles
            0: 15,          # Head/Nose
        }

    def fit_frame(self, mp_world_landmarks: Dict, num_iters: int = 100) -> Tuple[np.ndarray, Dict]:
        """단일 프레임에 대해 SMPL-X 파라메터 최적화 수행."""
        # 1. MediaPipe 데이터를 Torch Tensor로 변환 (Y-up 반전 포함)
        target_joints = torch.zeros(1, 54, 3, device=self.device)
        conf = torch.zeros(1, 54, 1, device=self.device)
        
        for mp_idx, sm_idx in self.mp_to_smplx_idx.items():
            name = list(mp_world_landmarks.keys())[mp_idx] # MP Index 기반 접근
            p = mp_world_landmarks[name]
            target_joints[0, sm_idx] = torch.tensor([p['x'], -p['y'], p['z']], device=self.device)
            conf[0, sm_idx] = 1.0

        # 2. 최적화 변수 설정
        body_pose = torch.zeros(1, 32, device=self.device, requires_grad=True) # PCA 기반
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
            mj = output.joints[:, :54]
            diff = (mj - target_joints) ** 2
            loss_joint = (conf * diff).sum()
            
            # Prior Loss (Regularization)
            loss_shape = (betas ** 2).sum() * 0.1
            loss_pose = (body_pose ** 2).sum() * 0.01
            
            total_loss = loss_joint + loss_shape + loss_pose
            total_loss.backward()
            optimizer.step()

        # 4. 결과 반환 (Vertices)
        with torch.no_grad():
            final_output = self.model(betas=betas, body_pose=body_pose, 
                                      global_orient=global_orient, transl=transl)
            vertices = final_output.vertices[0].cpu().numpy()
            
        return vertices, {"loss": total_loss.item()}

if __name__ == "__main__":
    # 유닛 테스트 코드 생략 (별도 파일로 작성)
    pass
