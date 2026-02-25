# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import subprocess
import math
import pyvista as pv
from tqdm import tqdm
from typing import Optional, List, Dict

# MediaPipe 관련 설정
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "data/models/pose_landmarker_heavy.task"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "data/models/pose_landmarker_lite.task"

# VTP 관련 설정
GEOM_DIR = '/tmp/opensim_geometry/'
OPENSIM_MODEL_PATH = '/mnt/d/progress/芭蕾呪法/data/e2e_output/opensim_output/scaled_model.osim'

BODY_VTP = {
    'pelvis': ['r_pelvis.vtp', 'l_pelvis.vtp', 'sacrum.vtp'],
    'femur_r': ['femur_r.vtp'], 'patella_r': ['r_patella.vtp'],
    'tibia_r': ['tibia_r.vtp', 'fibula_r.vtp'], 'talus_r': ['talus_rv.vtp'],
    'calcn_r': ['foot.vtp'], 'toes_r': ['bofoot.vtp'],
    'femur_l': ['femur_l.vtp'], 'patella_l': ['l_patella.vtp'],
    'tibia_l': ['tibia_l.vtp', 'fibula_l.vtp'], 'talus_l': ['talus_lv.vtp'],
    'calcn_l': ['l_foot.vtp'], 'toes_l': ['l_bofoot.vtp'],
    'torso': ['hat_ribs_scap.vtp'],
    'humerus_r': ['humerus_rv.vtp'], 'radius_r': ['radius_rv.vtp', 'ulna_rv.vtp'],
    'humerus_l': ['humerus_lv.vtp'], 'radius_l': ['radius_lv.vtp', 'ulna_lv.vtp'],
}

LOWER = {'femur_r','tibia_r','talus_r','calcn_r','toes_r','femur_l','tibia_l','talus_l','calcn_l','toes_l','patella_r','patella_l'}
UPPER = {'humerus_r','radius_r','humerus_l','radius_l'}

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

pv.start_xvfb()


def calc_angle(p1, vertex, p2):
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def body_color(name):
    if name in LOWER: return '#4488ff'
    if name in UPPER: return '#ff8844'
    if name == 'pelvis': return '#d4a0a0'
    return '#c0c0c0'


def landmarks_to_opensim_coords(landmarks):
    """MediaPipe 랜드마크 → OpenSim 좌표 (라디안) 매핑."""
    def pt(i):
        return np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z])

    lh, rh = pt(23), pt(24)
    lk, rk = pt(25), pt(26)
    la, ra = pt(27), pt(28)
    ls, rs = pt(11), pt(12)
    le, re = pt(13), pt(14)
    lw, rw = pt(15), pt(16)
    lf, rf = pt(31), pt(32)

    coords = {
        'pelvis_tilt': 0.0, 'pelvis_list': 0.0, 'pelvis_rotation': 0.0,
        'pelvis_tx': 0.0, 'pelvis_ty': 1.0, 'pelvis_tz': 0.0,
        'knee_angle_r': math.radians(calc_angle(rh, rk, ra) - 180),
        'knee_angle_l': math.radians(calc_angle(lh, lk, la) - 180),
        'hip_flexion_r': math.radians(180 - calc_angle(rs, rh, rk)),
        'hip_flexion_l': math.radians(180 - calc_angle(ls, lh, lk)),
        'ankle_angle_r': math.radians(90 - calc_angle(rk, ra, rf)),
        'ankle_angle_l': math.radians(90 - calc_angle(lk, la, lf)),
        'arm_flex_r': math.radians(180 - calc_angle(rh, rs, re)),
        'arm_flex_l': math.radians(180 - calc_angle(lh, ls, le)),
        'elbow_flex_r': math.radians(180 - calc_angle(rs, re, rw)),
        'elbow_flex_l': math.radians(180 - calc_angle(ls, le, lw)),
    }
    return coords


def compute_fk(model, state, coords_dict):
    """pyopensim FK: 좌표 설정 후 각 뼈의 위치·회전 4x4 행렬 반환."""
    cs = model.getCoordinateSet()
    for i in range(cs.getSize()):
        coord = cs.get(i)
        name = coord.getName()
        if name in coords_dict:
            coord.setValue(state, coords_dict[name])
    model.realizePosition(state)

    transforms = {}
    bs = model.getBodySet()
    for i in range(bs.getSize()):
        body = bs.get(i)
        name = body.getName()
        try:
            pos = body.getPositionInGround(state)
            R = body.getTransformInGround(state).R()
            T = np.eye(4)
            for r in range(3):
                for c in range(3):
                    T[r, c] = float(R.get(r, c))
                T[r, 3] = float(pos[r])
            transforms[name] = T
        except Exception:
            pass
    return transforms


def fallback_vtp_frame(landmarks, width=1920, height=1080):
    """VTP 렌더 실패 시 두꺼운 캡슐 뼈대 반환."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if not landmarks:
        return img
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for s, e in POSE_CONNECTIONS:
        if s < len(pts) and e < len(pts):
            cv2.line(img, pts[s], pts[e], (80, 80, 200), 10)
    for pt in pts:
        cv2.circle(img, pt, 7, (200, 200, 200), -1)
    return img


def prerender_vtp_frames(all_landmarks, model, state):
    """단일 PyVista Plotter로 전체 프레임 VTP 배치 렌더링."""
    width, height = 1920, 1080

    # VTP 메시 사전 로드
    cached_meshes = {}
    for bname, vtps in BODY_VTP.items():
        meshes = [pv.read(os.path.join(GEOM_DIR, vf))
                  for vf in vtps if os.path.exists(os.path.join(GEOM_DIR, vf))]
        if meshes:
            cached_meshes[bname] = meshes

    results = []
    try:
        pl = pv.Plotter(off_screen=True, window_size=[960, 540])
        pl.set_background('#0a0a1a')

        for landmarks in tqdm(all_landmarks, desc="VTP 뼈 렌더링"):
            pl.clear()
            rendered = False
            if landmarks and model and state and cached_meshes:
                try:
                    coords = landmarks_to_opensim_coords(landmarks)
                    bt = compute_fk(model, state, coords)
                    for bname, meshes in cached_meshes.items():
                        if bname not in bt:
                            continue
                        T = bt[bname]
                        for mesh in meshes:
                            pl.add_mesh(mesh.transform(T, inplace=False),
                                        color=body_color(bname), smooth_shading=True)
                    pl.camera_position = [(0, 1, 3), (0, 0.8, 0), (0, 1, 0)]
                    img = pl.screenshot(None, return_img=True)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    results.append(cv2.resize(img_bgr, (width, height)))
                    rendered = True
                except Exception:
                    pass
            if not rendered:
                results.append(fallback_vtp_frame(landmarks, width, height))

        pl.close()
    except Exception as e:
        print(f"PyVista 배치 렌더링 실패, fallback 사용: {e}")
        results = [fallback_vtp_frame(lm, width, height) for lm in all_landmarks]

    return results


def draw_overlay_panel(frame, landmarks):
    """원본 위에 스켈레톤 + 관절 각도 텍스트 오버레이."""
    panel = frame.copy()
    if not landmarks:
        return panel

    w, h = frame.shape[1], frame.shape[0]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for s, e in POSE_CONNECTIONS:
        if s < len(pts) and e < len(pts):
            cv2.line(panel, pts[s], pts[e], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(panel, pt, 4, (255, 255, 255), -1)

    def draw_angle(i1, iv, i2, label=''):
        p1 = np.array([landmarks[i1].x, landmarks[i1].y, landmarks[i1].z])
        v  = np.array([landmarks[iv].x, landmarks[iv].y, landmarks[iv].z])
        p2 = np.array([landmarks[i2].x, landmarks[i2].y, landmarks[i2].z])
        ang = calc_angle(p1, v, p2)
        pos = (pts[iv][0] + 10, pts[iv][1] - 10)
        cv2.putText(panel, f"{int(ang)}", pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    draw_angle(24, 26, 28)  # R 무릎
    draw_angle(23, 25, 27)  # L 무릎
    draw_angle(12, 24, 26)  # R 고관절
    draw_angle(11, 23, 25)  # L 고관절
    draw_angle(26, 28, 32)  # R 발목
    draw_angle(25, 27, 31)  # L 발목
    return panel


def preprocess_video(input_path, work_path):
    """HEVC→H264, 1920x1080, 회전 자동 보정."""
    print(f"전처리 중: {input_path}")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "scale=1920:1080",
        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
        work_path
    ], check=True, capture_output=True)


def create_anatomy_video(input_mov, output_path=None):
    """3패널(원본|오버레이|VTP뼈) 학습 영상 생성."""
    if not output_path:
        base, _ = os.path.splitext(input_mov)
        output_path = f"{base}_anatomy.mp4"

    os.makedirs("tmp", exist_ok=True)
    tmp_preproc = "tmp/_anat_preproc.mp4"

    # 1. 전처리
    preprocess_video(input_mov, tmp_preproc)

    # 2. OpenSim 모델 로드
    import pyopensim
    try:
        model = pyopensim.Model(OPENSIM_MODEL_PATH)
        state = model.initSystem()
        print("OpenSim 모델 로드 완료")
    except Exception as e:
        print(f"OpenSim 모델 로드 실패 (fallback 사용): {e}")
        model, state = None, None

    # 3. 1패스: MediaPipe 포즈 추출 + 프레임 저장
    cap = cv2.VideoCapture(tmp_preproc)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE
    )

    all_frames = []
    all_landmarks = []

    print("1패스: MediaPipe 포즈 추출...")
    with PoseLandmarker.create_from_options(options) as landmarker:
        for _ in tqdm(range(total_frames), desc="포즈 추출"):
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)
            lm = result.pose_landmarks[0] if result.pose_landmarks else None
            all_frames.append(frame)
            all_landmarks.append(lm)
    cap.release()

    # 4. 2패스: VTP 배치 렌더링
    print("2패스: VTP 뼈 렌더링...")
    vtp_frames = prerender_vtp_frames(all_landmarks, model, state)

    # 5. 3패스: 3패널 결합 → MP4 출력
    print("3패스: 3패널 결합...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920 * 3, 1080))

    for frame, landmarks, vtp_frame in tqdm(
            zip(all_frames, all_landmarks, vtp_frames),
            total=len(all_frames), desc="영상 합성"):
        p1 = frame
        p2 = draw_overlay_panel(frame, landmarks)
        p3 = vtp_frame

        combined = np.hstack((p1, p2, p3))
        cv2.putText(combined, "Original",        (50,          80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.putText(combined, "Overlay",          (1920 + 50,   80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.putText(combined, "OpenSim VTP",      (3840 + 50,   80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        out.write(combined)

    out.release()
    if os.path.exists(tmp_preproc):
        os.remove(tmp_preproc)
    print(f"완료! 저장 위치: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3패널 해부학 오버레이 뷰어 (원본|오버레이|VTP뼈)")
    parser.add_argument("--input", required=True, help="입력 MOV/MP4 파일")
    parser.add_argument("--output", help="출력 MP4 파일 경로")
    args = parser.parse_args()
    create_anatomy_video(args.input, args.output)
