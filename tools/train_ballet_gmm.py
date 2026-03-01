# -*- coding: utf-8 -*-
"""
Ballet-GMM 학습기 — Ballet-GMM Phase 2

data/ballet_poses.npy 로부터 sklearn GaussianMixture를 학습하고
data/models/smpl/ballet_gmm.pkl 로 저장합니다.

사용:
    python3 tools/train_ballet_gmm.py [--n_components N]
"""

import os, sys, argparse, pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_components', type=int, default=16)
    parser.add_argument('--input',  type=str, default='data/ballet_poses.npy')
    parser.add_argument('--output', type=str, default='data/models/smpl/ballet_gmm.pkl')
    args = parser.parse_args()

    print(f"[gmm] Loading {args.input}...")
    body_poses = np.load(args.input)   # (N, 69)
    print(f"[gmm] Dataset shape: {body_poses.shape}")
    print(f"[gmm] Range: [{body_poses.min():.3f}, {body_poses.max():.3f}]")

    # 이상값 필터링 (|body_pose| > 2π 인 샘플 제거 — 최적화 발산 결과물)
    angles = np.linalg.norm(body_poses.reshape(-1, 23, 3), axis=-1)  # (N, 23)
    valid_mask = (angles.max(axis=1) < 3.2)   # 3.2 rad ≈ 183°
    body_poses = body_poses[valid_mask]
    print(f"[gmm] After outlier filter: {body_poses.shape[0]} samples "
          f"(removed {(~valid_mask).sum()})")

    if body_poses.shape[0] < args.n_components * 5:
        print(f"[gmm] WARNING: 샘플 수({body_poses.shape[0]})가 "
              f"n_components({args.n_components}) × 5보다 적습니다.")
        args.n_components = max(2, body_poses.shape[0] // 5)
        print(f"[gmm] n_components 자동 조정: {args.n_components}")

    print(f"[gmm] Training GMM: n_components={args.n_components}, "
          f"covariance_type='full'...")
    gmm = GaussianMixture(
        n_components=args.n_components,
        covariance_type='full',
        max_iter=500,
        random_state=42,
        verbose=1,
        verbose_interval=50,
    )
    gmm.fit(body_poses)
    print(f"[gmm] Converged: {gmm.converged_}, iterations: {gmm.n_iter_}")

    # gmm_08.pkl 호환 형식으로 저장
    out_dict = {
        'weights': gmm.weights_,          # (K,)
        'means':   gmm.means_,            # (K, 69)
        'covars':  gmm.covariances_,      # (K, 69, 69)
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(out_dict, f)

    print(f"[gmm] Saved → {args.output}")
    print(f"[gmm] Log-likelihood on training data: {gmm.score(body_poses):.4f}")

    # 간단한 샘플 검증
    samples = gmm.sample(5)[0]   # (5, 69)
    sample_angles = np.linalg.norm(samples.reshape(-1, 23, 3), axis=-1)
    print(f"[gmm] Sample max angles: {sample_angles.max(axis=1)}")
    print(f"[gmm] All samples physically plausible (< π): "
          f"{(sample_angles.max(axis=1) < np.pi).all()}")

if __name__ == '__main__':
    main()
