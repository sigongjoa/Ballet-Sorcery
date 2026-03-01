# -*- coding: utf-8 -*-
"""
시각적 검증 도구 — SMPL 피팅 결과를 실제 영상 위에 투영하여 눈으로 확인

검증 패널 구성 (프레임당):
  [원본] | [MediaPipe 스켈레톤] | [SMPL 스켈레톤 + 가상 마커]

사용:
    python3 tools/visual_verification.py [--n N] [--iters N] [--out_dir DIR]
    --n N       : 검증 프레임 수 (기본 8, 영상 전체에 균등 분포)
    --iters N   : SMPL 최적화 반복 (기본 150)
    --out_dir   : 출력 디렉토리 (기본 reports/2026-03-01_e2e_validation/assets/visual)
"""

import os, sys, json, argparse
import numpy as np
import cv2

# numpy monkey-patch
np.bool = np.bool_; np.int = np.int_; np.float = np.float64
np.complex = np.complex128; np.object = np.object_
np.unicode = np.str_; np.str = np.str_

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '.')
from src.smplx_engine import SMPLXEngine
from src.smplx_mapper import SMPLXToOpenSimMapper

# ── 상수 ────────────────────────────────────────────────────────────
VIDEO_PATH  = 'my_data/2026_02_25/IMG_2633.MOV'
REF_JSON    = 'data/e2e_output/reference_poses.json'
MODEL_DIR   = 'data/models/smpl'
DISPLAY_W, DISPLAY_H = 1280, 720   # 출력 해상도

MP_NAMES = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT',
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL',
    'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

# MP landmark name → SMPL joint index (matches engine.mp_to_smplx_idx)
MP_TO_SMPL = {
    'NOSE': 15,
    'LEFT_SHOULDER': 16, 'RIGHT_SHOULDER': 17,
    'LEFT_ELBOW': 18,    'RIGHT_ELBOW': 19,
    'LEFT_WRIST': 20,    'RIGHT_WRIST': 21,
    'LEFT_HIP': 1,       'RIGHT_HIP': 2,
    'LEFT_KNEE': 4,      'RIGHT_KNEE': 5,
    'LEFT_ANKLE': 7,     'RIGHT_ANKLE': 8,
}

# MediaPipe 스켈레톤 연결선 (landmark index 쌍)
MP_SKELETON = [
    (11, 13), (13, 15),  # 왼팔
    (12, 14), (14, 16),  # 오른팔
    (11, 12),            # 어깨
    (11, 23), (12, 24),  # 몸통
    (23, 24),            # 엉덩이
    (23, 25), (25, 27), (27, 29), (27, 31),  # 왼다리
    (24, 26), (26, 28), (28, 30), (28, 32),  # 오른다리
]

# SMPL 스켈레톤 연결선 (joint index 쌍)
SMPL_SKELETON = [
    (0, 1), (1, 4), (4, 7), (7, 10),   # 왼다리
    (0, 2), (2, 5), (5, 8), (8, 11),   # 오른다리
    (0, 3), (3, 6), (6, 9),            # 척추
    (9, 13), (13, 16), (16, 18), (18, 20),  # 왼팔
    (9, 14), (14, 17), (17, 19), (19, 21),  # 오른팔
    (9, 12), (12, 15),                 # 목-머리
]

# 핵심 가상 마커 (과도하면 복잡하므로 대표 14개)
KEY_MARKERS = [
    'ACROMION_L', 'ACROMION_R',
    'ELBOW_LAT_L', 'ELBOW_LAT_R',
    'WRIST_RAD_L', 'WRIST_RAD_R',
    'GTROCHANTER_L', 'GTROCHANTER_R',
    'KNEE_LAT_L', 'KNEE_LAT_R',
    'ANKLE_LAT_L', 'ANKLE_LAT_R',
    'HEEL_POST_L', 'HEEL_POST_R',
]


def fit_affine_from_joints(joints, landmarks_img, img_w, img_h):
    """
    SMPL joint 위치 ↔ MP 이미지 좌표 대응점으로 affine 투영 행렬 학습.
    x_px = A @ [x,y,z,1]^T,  y_px = B @ [x,y,z,1]^T

    핵심: MP world 좌표 대신 실제 SMPL joint 위치를 3D 입력으로 사용.
    → affine이 SMPL 좌표계(transl 포함)에 맞춰지므로 가상 마커 투영이 정확해짐.
    """
    pts2d, pts3d = [], []

    for mp_name, sm_idx in MP_TO_SMPL.items():
        if mp_name not in landmarks_img or sm_idx >= len(joints):
            continue
        lm2 = landmarks_img[mp_name]
        if lm2.get('visibility', 1.0) < 0.5:
            continue
        pts2d.append([lm2['x'] * img_w, lm2['y'] * img_h])
        pts3d.append(joints[sm_idx])   # SMPL joint 실제 위치 (transl 포함)

    if len(pts2d) < 6:
        return None

    pts2d = np.array(pts2d, dtype=np.float64)
    pts3d = np.array(pts3d, dtype=np.float64)

    ones = np.ones((len(pts3d), 1))
    A_mat = np.hstack([pts3d, ones])   # (N, 4)

    coef_x, _, _, _ = np.linalg.lstsq(A_mat, pts2d[:, 0], rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(A_mat, pts2d[:, 1], rcond=None)

    px_pred = A_mat @ coef_x
    py_pred = A_mat @ coef_y
    reproj_err = np.mean(np.sqrt(
        (px_pred - pts2d[:, 0])**2 + (py_pred - pts2d[:, 1])**2
    ))

    return coef_x, coef_y, reproj_err


def project_3d_affine(pts3d, coef_x, coef_y):
    """3D 점 배열 → 2D 이미지 좌표 (affine 투영)."""
    pts3d = np.array(pts3d, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((len(pts3d), 1))
    A_mat = np.hstack([pts3d, ones])
    px = A_mat @ coef_x
    py = A_mat @ coef_y
    return np.column_stack([px, py])


def draw_mp_skeleton(img, landmarks_img, img_w, img_h, color=(0, 255, 0)):
    """MediaPipe 스켈레톤 그리기."""
    # 이름→인덱스 매핑
    name_to_idx = {name: i for i, name in enumerate(MP_NAMES)}
    pts = {}
    for name, lm in landmarks_img.items():
        if name in name_to_idx:
            x = int(lm['x'] * img_w)
            y = int(lm['y'] * img_h)
            pts[name_to_idx[name]] = (x, y)

    for (i, j) in MP_SKELETON:
        if i in pts and j in pts:
            cv2.line(img, pts[i], pts[j], color, 2)

    for idx, (x, y) in pts.items():
        cv2.circle(img, (x, y), 4, color, -1)

    return img


def draw_smpl_overlay(img, joints, virtual_markers, coef_x, coef_y, img_w, img_h):
    """SMPL 스켈레톤 + 가상 마커 그리기 (affine 투영)."""
    # SMPL joints → 2D
    try:
        jpts = project_3d_affine(joints, coef_x, coef_y)
    except Exception:
        return img

    # 스켈레톤 연결선
    for (i, j) in SMPL_SKELETON:
        if i < len(jpts) and j < len(jpts):
            p1 = (int(jpts[i][0]), int(jpts[i][1]))
            p2 = (int(jpts[j][0]), int(jpts[j][1]))
            if (0 <= p1[0] < img_w and 0 <= p1[1] < img_h and
                    0 <= p2[0] < img_w and 0 <= p2[1] < img_h):
                cv2.line(img, p1, p2, (255, 100, 0), 2)

    # SMPL 관절 점 (파란색) — 기본 22개 관절만 (extra regression joint 제외)
    for pt in jpts[:22]:
        px, py = int(pt[0]), int(pt[1])
        if 0 <= px < img_w and 0 <= py < img_h:
            cv2.circle(img, (px, py), 5, (0, 120, 255), -1)

    # 가상 마커 (핵심 14개, 청록색)
    if virtual_markers:
        marker_pts_3d, marker_names = [], []
        for name in KEY_MARKERS:
            if name in virtual_markers:
                marker_pts_3d.append(virtual_markers[name])
                marker_names.append(name)

        if marker_pts_3d:
            try:
                mpts = project_3d_affine(marker_pts_3d, coef_x, coef_y)
                for mname, mpt in zip(marker_names, mpts):
                    mx, my = int(mpt[0]), int(mpt[1])
                    if 0 <= mx < img_w and 0 <= my < img_h:
                        cv2.circle(img, (mx, my), 8, (0, 255, 180), -1)
                        cv2.circle(img, (mx, my), 8, (0, 140, 100), 2)
                        short = mname.replace('_L', 'L').replace('_R', 'R')
                        cv2.putText(img, short, (mx + 9, my - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 230, 180), 1)
            except Exception:
                pass

    return img


def make_panel(orig_frame, mp_frame, smpl_frame, title, loss, mpjpe_cm=None):
    """3패널 합성 이미지 생성."""
    h, w = orig_frame.shape[:2]
    panel = np.zeros((h, w * 3, 3), dtype=np.uint8)

    # 패널 배치
    panel[:, 0:w] = orig_frame
    panel[:, w:2*w] = mp_frame
    panel[:, 2*w:3*w] = smpl_frame

    # 구분선
    cv2.line(panel, (w, 0), (w, h), (80, 80, 80), 2)
    cv2.line(panel, (2*w, 0), (2*w, h), (80, 80, 80), 2)

    # 상단 라벨
    for col, label in enumerate(['원본 영상', 'MediaPipe 스켈레톤', 'SMPL 피팅 + 가상 마커']):
        cv2.putText(panel, label, (col*w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 제목 + 수치
    info = title
    if mpjpe_cm is not None:
        info += f'  |  MPJPE={mpjpe_cm:.1f}cm'
    info += f'  |  loss={loss:.4f}'
    cv2.putText(panel, info, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',       type=int, default=8,
                        help='검증 프레임 수')
    parser.add_argument('--iters',   type=int, default=150,
                        help='SMPL 최적화 반복')
    parser.add_argument('--out_dir', type=str,
                        default='reports/2026-03-01_e2e_validation/assets/visual')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 데이터 로드 ────────────────────────────────────────────────
    print(f'[visual] 데이터 로드: {REF_JSON}')
    with open(REF_JSON) as f:
        data = json.load(f)

    # 비디오 정보
    cap = cv2.VideoCapture(VIDEO_PATH)
    vid_fps  = cap.get(cv2.CAP_PROP_FPS)
    vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_duration = vid_total / vid_fps
    cap.release()
    print(f'[visual] 비디오: {vid_total}프레임, {vid_fps}fps, {vid_duration:.1f}s')

    # 비디오 기간 내 유효 reference 프레임 수집
    valid = [(fr, int(fr['timestamp_sec'] * vid_fps))
             for fr in data['frames']
             if fr.get('world_landmarks') and fr.get('landmarks')
             and fr['timestamp_sec'] < vid_duration]
    print(f'[visual] 비디오 기간 내 유효 프레임: {len(valid)}개')

    if len(valid) < args.n:
        args.n = len(valid)

    # 균등 간격으로 샘플링
    step = len(valid) // args.n
    sampled = [valid[i * step] for i in range(args.n)]
    print(f'[visual] 선택된 프레임: {[f["frame_idx"] for f, _ in sampled]}')

    # ── SMPL 엔진 초기화 ─────────────────────────────────────────
    engine = SMPLXEngine(MODEL_DIR)
    mapper = SMPLXToOpenSimMapper()

    # ── MP→SMPL 오차 계산용 매핑 ─────────────────────────────────
    mp_to_smpl_idx = {
        'LEFT_SHOULDER': 16, 'RIGHT_SHOULDER': 17,
        'LEFT_ELBOW': 18, 'RIGHT_ELBOW': 19,
        'LEFT_HIP': 1, 'RIGHT_HIP': 2,
        'LEFT_KNEE': 4, 'RIGHT_KNEE': 5,
        'LEFT_ANKLE': 7, 'RIGHT_ANKLE': 8,
    }

    panels = []
    per_frame_results = []
    cap = cv2.VideoCapture(VIDEO_PATH)

    for idx, (frame_data, vid_frame_no) in enumerate(sampled):
        ref_fidx = frame_data['frame_idx']
        ts = frame_data['timestamp_sec']
        lm_img   = frame_data['landmarks']        # 정규화 이미지 좌표
        lm_world = frame_data['world_landmarks']  # 3D 월드 좌표

        print(f'\n[{idx+1}/{args.n}] ref_frame={ref_fidx} (ts={ts:.1f}s, vid={vid_frame_no})')

        # ── 영상 프레임 추출 ──────────────────────────────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_frame_no)
        ret, frame_bgr = cap.read()
        if not ret:
            print(f'  경고: vid_frame={vid_frame_no} 추출 실패, 건너뜀')
            continue

        # 리사이즈
        frame_disp = cv2.resize(frame_bgr, (DISPLAY_W, DISPLAY_H))

        # ── SMPL 피팅 ─────────────────────────────────────────────
        ordered = {k: lm_world[k] for k in MP_NAMES if k in lm_world}
        vertices, joints, meta = engine.fit_frame(ordered, num_iters=args.iters)
        virtual_markers = mapper.extract_virtual_markers(vertices)
        loss = meta['loss']
        print(f'  SMPL loss={loss:.4f}')

        # ── Affine 투영 행렬 추정 ───────────────────────────────
        # SMPL joint 위치 ↔ MP 이미지 좌표 대응점으로 최소제곱 affine 피팅
        # (MP world 좌표 대신 실제 SMPL joint 위치 사용 → transl 오프셋 자동 반영)
        proj_result = fit_affine_from_joints(joints, lm_img, DISPLAY_W, DISPLAY_H)
        if proj_result is None:
            print(f'  경고: affine 피팅 실패, 건너뜀')
            continue
        coef_x, coef_y, reproj_err = proj_result
        print(f'  affine 재투영 오차 (SMPL→image): {reproj_err:.1f}px')

        # ── MPJPE 계산 ────────────────────────────────────────────
        errors = []
        for mp_name, sm_idx in mp_to_smpl_idx.items():
            if mp_name in lm_world:
                p_mp = lm_world[mp_name]
                target = np.array([p_mp['x'], -p_mp['y'], p_mp['z']])
                err = np.linalg.norm(target - joints[sm_idx]) * 100
                errors.append(err)
        mpjpe = np.mean(errors) if errors else None
        if mpjpe:
            print(f'  MPJPE={mpjpe:.2f}cm')

        # ── 패널 1: 원본 ──────────────────────────────────────────
        panel_orig = frame_disp.copy()

        # ── 패널 2: MediaPipe 스켈레톤 ───────────────────────────
        panel_mp = frame_disp.copy()
        panel_mp = draw_mp_skeleton(panel_mp, lm_img, DISPLAY_W, DISPLAY_H, color=(0, 230, 0))

        # ── 패널 3: SMPL 스켈레톤 + 가상 마커 ────────────────────
        panel_smpl = frame_disp.copy()
        panel_smpl = draw_smpl_overlay(
            panel_smpl, joints, virtual_markers, coef_x, coef_y, DISPLAY_W, DISPLAY_H
        )

        # ── 합성 ─────────────────────────────────────────────────
        title = f'Frame {ref_fidx} ({ts:.1f}s)'
        panel = make_panel(panel_orig, panel_mp, panel_smpl, title, loss, mpjpe)

        # 저장
        out_path = os.path.join(args.out_dir, f'verify_frame{ref_fidx:04d}.jpg')
        cv2.imwrite(out_path, panel, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f'  저장: {out_path}')

        panels.append(panel)
        per_frame_results.append({
            'ref_frame': ref_fidx,
            'ts': ts,
            'vid_frame': vid_frame_no,
            'loss': loss,
            'mpjpe_cm': mpjpe,
        })

    cap.release()

    # ── 그리드 요약 이미지 ─────────────────────────────────────────
    if panels:
        # 각 패널을 축소해서 그리드로 합침
        thumb_w = 960
        thumb_h = int(thumb_w * panels[0].shape[0] / panels[0].shape[1])
        thumbs = [cv2.resize(p, (thumb_w, thumb_h)) for p in panels]

        cols = 2
        rows = (len(thumbs) + cols - 1) // cols
        grid_h = rows * thumb_h
        grid_w = cols * thumb_w
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i, thumb in enumerate(thumbs):
            r, c = divmod(i, cols)
            grid[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb

        grid_path = os.path.join(args.out_dir, 'verify_grid.jpg')
        cv2.imwrite(grid_path, grid, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print(f'\n[visual] 그리드 저장: {grid_path}')

    # ── 텍스트 요약 ───────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, 'verify_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('SMPL 시각적 검증 요약\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'검증 프레임: {len(per_frame_results)}개\n')
        f.write(f'SMPL 최적화: {args.iters}iter/frame\n\n')
        f.write(f'{"프레임":>6}  {"시간":>6}  {"loss":>8}  {"MPJPE(cm)":>10}\n')
        f.write('-' * 40 + '\n')
        mpjpe_vals = []
        for r in per_frame_results:
            mpjpe_str = f'{r["mpjpe_cm"]:.2f}' if r['mpjpe_cm'] else 'N/A'
            f.write(f'{r["ref_frame"]:>6}  {r["ts"]:>5.1f}s  {r["loss"]:>8.4f}  {mpjpe_str:>10}\n')
            if r['mpjpe_cm']:
                mpjpe_vals.append(r['mpjpe_cm'])
        if mpjpe_vals:
            f.write(f'\n평균 MPJPE: {np.mean(mpjpe_vals):.2f}cm\n')
            f.write(f'최대 MPJPE: {np.max(mpjpe_vals):.2f}cm\n')
            f.write(f'최소 MPJPE: {np.min(mpjpe_vals):.2f}cm\n')

    print(f'[visual] 요약: {summary_path}')
    print(f'\n[visual] 완료! 결과 확인: {args.out_dir}/')
    print(f'         ▶ verify_grid.jpg — 전체 개요')
    print(f'         ▶ verify_frame*.jpg — 프레임별 상세')


if __name__ == '__main__':
    main()
