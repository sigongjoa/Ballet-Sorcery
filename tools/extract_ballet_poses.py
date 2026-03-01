# -*- coding: utf-8 -*-
"""
Ballet body_pose 추출기 — Ballet-GMM Phase 2 데이터 수집

reference_poses.json의 모든 유효 프레임에서 SMPL body_pose(69차원)를 추출하여
data/ballet_poses.npy 로 저장합니다.

사용:
    python3 tools/extract_ballet_poses.py [--step N] [--iters N]
    --step N   : 프레임 간격 (기본 5, 즉 5프레임마다 1개 추출)
    --iters N  : fit_frame 최적화 횟수 (기본 50)
"""

import os, sys, json, argparse, time
import numpy as np

# monkey-patch (smplx_engine 임포트 전)
np.bool = np.bool_; np.int = np.int_; np.float = np.float64
np.complex = np.complex128; np.object = np.object_; np.unicode = np.str_; np.str = np.str_

sys.path.insert(0, '.')
from src.smplx_engine import SMPLXEngine

MP_LANDMARK_NAMES = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT',
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL',
    'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step',  type=int, default=5,  help='프레임 간격')
    parser.add_argument('--iters', type=int, default=50, help='최적화 반복 횟수')
    parser.add_argument('--out',   type=str, default='data/ballet_poses.npy')
    args = parser.parse_args()

    ref_path  = 'data/e2e_output/reference_poses.json'
    model_dir = 'data/models/smpl'

    print(f"[extract] Loading {ref_path}...")
    with open(ref_path) as f:
        data = json.load(f)

    valid_frames = [fr for fr in data['frames'] if fr.get('world_landmarks')]
    sampled     = valid_frames[::args.step]
    print(f"[extract] Total valid: {len(valid_frames)}, sampled (step={args.step}): {len(sampled)}")
    print(f"[extract] Optimization: {args.iters} iter/frame")

    engine = SMPLXEngine(model_dir)

    body_poses = []
    t_start = time.time()

    for i, frame_data in enumerate(sampled):
        frame_idx = frame_data['frame_idx']
        mp_world  = frame_data['world_landmarks']
        ordered   = {k: mp_world[k] for k in MP_LANDMARK_NAMES if k in mp_world}

        _, _, meta = engine.fit_frame(ordered, num_iters=args.iters)
        bp = meta['body_pose'].reshape(69)   # (69,)
        body_poses.append(bp)

        # 진행률 출력 (10프레임마다)
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            per_frame = elapsed / (i + 1)
            remaining = per_frame * (len(sampled) - i - 1)
            print(f"  [{i+1}/{len(sampled)}] frame {frame_idx} | "
                  f"elapsed {elapsed/60:.1f}m | ETA {remaining/60:.1f}m | "
                  f"loss {meta['loss']:.4f}")

    body_poses_arr = np.stack(body_poses)   # (N, 69)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, body_poses_arr)
    print(f"\n[extract] Saved {body_poses_arr.shape} → {args.out}")
    print(f"[extract] Total time: {(time.time()-t_start)/60:.1f} min")

if __name__ == '__main__':
    main()
