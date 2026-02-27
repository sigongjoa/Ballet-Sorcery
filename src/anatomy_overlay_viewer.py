# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import subprocess
import math
os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')  # Xvfb 대신 EGL 오프스크린 (GPU)
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

# EGL 오프스크린 모드 사용 (PYVISTA_OFF_SCREEN=true → Xvfb 불필요)
# pv.start_xvfb()  # 소프트웨어 렌더링 → EGL GPU 렌더링으로 전환


def calc_angle(p1, vertex, p2):
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


# EMA 상태 (프레임 간 시간 평탄화)
_prev_coords: dict = {}


def body_color(name):
    if name in LOWER: return '#4488ff'
    if name in UPPER: return '#ff8844'
    if name == 'pelvis': return '#d4a0a0'
    return '#c0c0c0'


def landmarks_to_opensim_coords(landmarks, world_lm=None):
    """MediaPipe 랜드마크 → OpenSim 좌표 (라디안) 매핑.

    world_lm (pose_world_landmarks): 미터 단위 진짜 3D 좌표, Z 안정적.
      - 없으면 normalized landmarks의 Z=0 처리(2D 투영)로 대체.
    hip_flexion: 어깨 단일점 대신 양쪽 어깨 중점(body axis) 기준
      → 팔 동작에 의한 오염 제거.
    """
    # 각도 계산용 소스: world_lm 우선, 없으면 XY만 사용(Z=0 → 노이즈 차단)
    if world_lm is not None:
        def pt(i):
            return np.array([world_lm[i].x, world_lm[i].y, world_lm[i].z])
    else:
        def pt(i):
            return np.array([landmarks[i].x, landmarks[i].y, 0.0])

    lh, rh = pt(23), pt(24)
    lk, rk = pt(25), pt(26)
    la, ra = pt(27), pt(28)
    ls, rs = pt(11), pt(12)
    le, re = pt(13), pt(14)
    lw, rw = pt(15), pt(16)
    lf, rf = pt(31), pt(32)

    # hip_flexion 기준: 어깨 중점 (팔 동작에 무관)
    shoulder_mid = (ls + rs) / 2.0
    pelvis_mid   = (lh + rh) / 2.0

    # 대퇴 벡터 (hip→knee, world_lm Y-DOWN)
    # dancer faces AWAY: world_lm +X = dancer LEFT
    # → right leg adduction = knee toward dancer's left = +X
    # → left  leg adduction = knee toward dancer's right = -X
    thigh_r = rk - rh
    thigh_l = lk - lh
    # 점프 감지: 무릎이 힙 근처 또는 위 (Y-DOWN에서 thigh Y < 0.05)
    # → 대퇴 벡터 기하 붕괴 → hip_adduction 비활성, EMA 강화
    is_jump = (thigh_r[1] < 0.05) or (thigh_l[1] < 0.05)

    def snorm(val, lo, hi):
        """Soft normalization: tanh 기반 smooth saturation.
        범위 내 값은 거의 그대로 통과, 범위 초과 값은 부드럽게 포화.
        하드 클램핑처럼 특정 값에서 딱 멈추는 현상 없음.
        """
        center = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        return center + half * math.tanh((val - center) / half)

    raw = {
        'pelvis_tilt': 0.0, 'pelvis_list': 0.0, 'pelvis_rotation': 0.0,
        'pelvis_tx': 0.0, 'pelvis_ty': 0.9, 'pelvis_tz': 0.0,
        # 무릎: Rajagopal -120°~10° → soft saturation
        'knee_angle_r': snorm(math.radians(calc_angle(rh, rk, ra) - 180), -2.0, 0.17),
        'knee_angle_l': snorm(math.radians(calc_angle(lh, lk, la) - 180), -2.0, 0.17),
        # 힙: -30°~120°
        'hip_flexion_r': snorm(math.radians(180 - calc_angle(shoulder_mid, rh, rk)), -0.52, 2.09),
        'hip_flexion_l': snorm(math.radians(180 - calc_angle(shoulder_mid, lh, lk)), -0.52, 2.09),
        # 발목: -70°~40°
        'ankle_angle_r': snorm(math.radians(90 - calc_angle(rk, ra, rf)), -1.0, 0.6),
        'ankle_angle_l': snorm(math.radians(90 - calc_angle(lk, la, lf)), -1.0, 0.6),
        # 힙 내전/외전: 정상 자세에서만 계산 (점프 시 기하 붕괴 → 0 설정)
        # atan2(lateral_X, vertical_Y) → frontal-plane angle
        # is_jump 아닐 때 thigh_r[1] > 0.05 보장 → denominator 양수
        'hip_adduction_r': snorm(
            math.atan2(thigh_r[0], max(thigh_r[1], 0.05)), -0.87, 0.35) if not is_jump else 0.0,
        'hip_adduction_l': snorm(
            math.atan2(-thigh_l[0], max(thigh_l[1], 0.05)), -0.87, 0.35) if not is_jump else 0.0,
        # 팔·팔꿈치
        'arm_flex_r': snorm(math.radians(180 - calc_angle(rh, rs, re)), -0.5, 2.5),
        'arm_flex_l': snorm(math.radians(180 - calc_angle(lh, ls, le)), -0.5, 2.5),
        'elbow_flex_r': snorm(math.radians(180 - calc_angle(rs, re, rw)), -0.1, 2.5),
        'elbow_flex_l': snorm(math.radians(180 - calc_angle(ls, le, lw)), -0.1, 2.5),
    }

    # EMA 시간 평탄화: 점프 시 α=0.25 (이전 75% 유지), 평상시 α=0.6
    # 점프 프레임 FK 발산을 이전 안정 프레임으로 강하게 끌어당김
    global _prev_coords
    EMA_ALPHA = 0.25 if is_jump else 0.6
    if _prev_coords:
        coords = {k: EMA_ALPHA * v + (1 - EMA_ALPHA) * _prev_coords.get(k, v)
                  for k, v in raw.items()}
    else:
        coords = raw
    _prev_coords = coords
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


def prerender_vtp_frames(all_landmarks, all_world_landmarks, model, state):
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

        for landmarks, world_lm in tqdm(
                zip(all_landmarks, all_world_landmarks), total=len(all_landmarks),
                desc="VTP 뼈 렌더링"):
            pl.clear()
            rendered = False
            if landmarks and model and state and cached_meshes:
                try:
                    coords = landmarks_to_opensim_coords(landmarks, world_lm)
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

    # GPU delegate 우선 시도; 실패 시 CPU fallback
    try:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=MODEL_PATH,
                delegate=BaseOptions.Delegate.GPU,
            ),
            running_mode=RunningMode.IMAGE,
        )
        # 실제로 생성해봐야 GPU 지원 여부 확인 가능
        _test = PoseLandmarker.create_from_options(options)
        _test.close()
        print("MediaPipe: GPU delegate 사용")
    except Exception:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
        )
        print("MediaPipe: CPU fallback")

    all_frames = []
    all_landmarks = []
    all_world_landmarks = []

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
            # world_landmarks: 미터 단위 3D 좌표 (Z 안정적) — 각도 계산에 사용
            wlm = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
            all_frames.append(frame)
            all_landmarks.append(lm)
            all_world_landmarks.append(wlm)
    cap.release()

    # 4. 2패스: VTP 배치 렌더링
    global _prev_coords
    _prev_coords = {}  # EMA 상태 리셋 (새 영상 처리 시작)
    print("2패스: VTP 뼈 렌더링...")
    vtp_frames = prerender_vtp_frames(all_landmarks, all_world_landmarks, model, state)

    # 5. 3패스: 3패널 결합 → ffmpeg 파이프로 H.264 직접 출력
    print("3패스: 3패널 결합 (H.264 출력)...")
    out_w, out_h = 1920 * 3, 1080
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
        "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    for frame, landmarks, vtp_frame in tqdm(
            zip(all_frames, all_landmarks, vtp_frames),
            total=len(all_frames), desc="영상 합성"):
        p1 = frame
        p2 = draw_overlay_panel(frame, landmarks)
        p3 = vtp_frame

        combined = np.hstack((p1, p2, p3))
        cv2.putText(combined, "Original",    (50,        80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.putText(combined, "Overlay",     (1920 + 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.putText(combined, "OpenSim VTP", (3840 + 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        ffmpeg_proc.stdin.write(combined.tobytes())

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    if os.path.exists(tmp_preproc):
        os.remove(tmp_preproc)
    print(f"완료! 저장 위치: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3패널 해부학 오버레이 뷰어 (원본|오버레이|VTP뼈)")
    parser.add_argument("--input", required=True, help="입력 MOV/MP4 파일")
    parser.add_argument("--output", help="출력 MP4 파일 경로")
    args = parser.parse_args()
    create_anatomy_video(args.input, args.output)
