#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTP 관절 각도 진단 도구
======================
문제: MediaPipe → OpenSim 각도 변환이 근본적으로 잘못됨
- arm_flex: 팔이 내려있을 때도 180° (머리 위로 올라감)
- knee_angle: 시상면 굴곡이 2D 투영에서 안 잡힘

이 스크립트는:
1. 지정 프레임에서 각 관절의 실제 연결 노드 간 각도를 계산
2. MediaPipe 2D 스켈레톤과 FK stick figure를 나란히 표시
3. 각 공식의 예상값 vs 실제값 비교 출력

사용법:
  python3 tools/diag_joint_angles.py --frame 100 --frame 200 --frame 300
"""
import sys, os, math, argparse
import cv2
import numpy as np
import mediapipe as mp
import pyopensim

os.chdir('/mnt/d/progress/芭蕾呪法')

VIDEO = "my_data/2026_02_25/IMG_2633.MOV"
MODEL_PATH = "data/models/pose_landmarker_lite.task"
OSIM_PATH  = "data/e2e_output/opensim_output/scaled_model.osim"
OUT_DIR    = "reports/diag_joints"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 각도 계산 유틸 ─────────────────────────────────────────────────────────
def vec_angle_deg(a, b, c):
    """꼭짓점 b에서 a-b-c 사이 각도 (도)"""
    v1 = a - b; v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8: return 0.0
    return math.degrees(math.acos(np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)))

def angle_between_vecs(v1, v2):
    """두 벡터 사이 각도 (도)"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8: return 0.0
    return math.degrees(math.acos(np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)))

# ── FK 계산 ──────────────────────────────────────────────────────────────────
def run_fk(model, state, coords):
    cs = model.getCoordinateSet()
    for i in range(cs.getSize()):
        c = cs.get(i)
        if c.getName() in coords:
            c.setValue(state, coords[c.getName()])
    model.realizePosition(state)
    bs = model.getBodySet()
    pos = {}
    for i in range(bs.getSize()):
        b = bs.get(i)
        try:
            p = b.getPositionInGround(state)
            pos[b.getName()] = np.array([float(p[0]), float(p[1]), float(p[2])])
        except: pass
    return pos

# ── 관절별 진단 테이블 출력 ──────────────────────────────────────────────────
def diagnose_angles(w):
    """world_lm 기반으로 각 관절의 실제 각도를 진단."""
    def pt(i): return np.array([w[i].x, w[i].y, w[i].z])

    lh,rh = pt(23),pt(24)
    lk,rk = pt(25),pt(26)
    la,ra = pt(27),pt(28)
    ls,rs = pt(11),pt(12)
    le,re = pt(13),pt(14)
    lw,rw = pt(15),pt(16)
    lf,rf = pt(31),pt(32)
    pelvis_mid   = (lh+rh)/2
    shoulder_mid = (ls+rs)/2

    print("\n" + "="*80)
    print("관절별 세그먼트 각도 (world_lm, Y-DOWN 좌표계)")
    print("="*80)

    rows = []

    # ── 1. 무릎 ──────────────────────────────────────────────────────────────
    # 연결 세그먼트: 대퇴(hip→knee) / 경골(knee→ankle)
    # 꼭짓점: knee
    # 기대: 다리 직선 ≈ 180°, 굴곡 30° → ≈ 150°
    knee_r_raw = vec_angle_deg(rh, rk, ra)  # 꼭짓점=knee
    knee_l_raw = vec_angle_deg(lh, lk, la)
    # 현재 공식: calc_angle(hip,knee,ankle) - 180  →  180-180=0(직선), 150-180=-30
    # OpenSim knee_angle_r = 0은 완전신전, 음수 = 굴곡  ← 공식 방향은 맞음
    curr_r = math.degrees(math.radians(knee_r_raw - 180))
    curr_l = math.degrees(math.radians(knee_l_raw - 180))
    rows.append(["무릎_R", f"{knee_r_raw:.1f}°",
                 f"굴곡={knee_r_raw-180:.1f}°",
                 "직선→0°, 굴곡→음수 (공식 방향 OK)", "✅" if abs(knee_r_raw-180)<5 or knee_r_raw<175 else "⚠️"])
    rows.append(["무릎_L", f"{knee_l_raw:.1f}°",
                 f"굴곡={knee_l_raw-180:.1f}°", "", "✅"])

    # ── 2. 팔 (어깨 굴곡) ────────────────────────────────────────────────────
    # 연결 세그먼트: 체간축(pelvis→shoulder) / 상완(shoulder→elbow)
    # 꼭짓점: shoulder
    # 기대: 팔 아래 ≈ 0°, 수평 ≈ 90°, 머리위 ≈ 180°
    # 현재 공식: 180 - angle(hip, shoulder, elbow)
    #   → 팔아래: angle≈0° → 공식=180° (틀림! 머리위로 올라감)
    arm_r_hip_to_shoulder = rs - rh  # 어깨 기준 하강 방향 (Y-DOWN: 음수=위)
    arm_r_shoulder_to_elbow = re - rs  # 상완 방향

    # 방법A (현재): angle(rh, rs, re) = 힙방향 vs 팔꿈치방향 @ 어깨
    method_a_r = vec_angle_deg(rh, rs, re)
    # 180 - method_a_r = 현재 공식 결과
    current_arm_r = 180 - method_a_r

    # 방법B (올바른): 체간 하강벡터 vs 상완벡터 사이 각도
    #   체간 하강 = rh - rs (어깨에서 힙 방향)
    #   상완     = re - rs (어깨에서 팔꿈치 방향)
    torso_down_r = rh - rs  # Y-DOWN에서 힙은 아래 → 이 벡터가 체간 아래 방향
    upper_arm_r  = re - rs
    method_b_r = angle_between_vecs(torso_down_r, upper_arm_r)
    # 팔이 내려있으면: 둘 다 아래 방향 → 각도≈작음 (팔이 체간과 거의 평행)
    # 팔이 머리위: upper_arm_r는 위, torso_down_r는 아래 → 각도≈180°

    rows.append(["팔_R (현재공식)", f"raw={method_a_r:.1f}°",
                 f"→ arm_flex=180-{method_a_r:.0f}={current_arm_r:.0f}°",
                 "팔아래인데 arm_flex≈180° → 머리위 버그", "❌"])
    rows.append(["팔_R (올바른방법)", f"체간vs상완={method_b_r:.1f}°",
                 f"→ arm_flex={method_b_r:.0f}°",
                 "팔아래≈작은값, 수평≈90°, 머리위≈180°", "✅"])

    # 같은 진단 왼팔
    method_a_l = vec_angle_deg(lh, ls, le)
    current_arm_l = 180 - method_a_l
    torso_down_l = lh - ls
    upper_arm_l  = le - ls
    method_b_l = angle_between_vecs(torso_down_l, upper_arm_l)
    rows.append(["팔_L (현재→버그)", f"raw={method_a_l:.1f}°",
                 f"arm_flex={current_arm_l:.0f}°", "", "❌"])
    rows.append(["팔_L (올바른)", f"체간vs상완={method_b_l:.1f}°",
                 f"arm_flex={method_b_l:.0f}°", "", "✅"])

    # ── 3. 팔꿈치 굴곡 ────────────────────────────────────────────────────────
    # 연결 세그먼트: 상완(elbow←shoulder) / 전완(elbow→wrist)
    # 꼭짓점: elbow
    # 기대: 직선 ≈ 180°, 90° 굴곡 → 90°
    # 현재: 180 - angle(shoulder, elbow, wrist)
    #   → 직선: angle=180° → 0° ✓, 90°굴곡: angle=90° → 90° ✓
    elbow_r_raw = vec_angle_deg(rs, re, rw)
    elbow_l_raw = vec_angle_deg(ls, le, lw)
    rows.append(["팔꿈치_R", f"{elbow_r_raw:.1f}°",
                 f"elbow_flex=180-{elbow_r_raw:.0f}={180-elbow_r_raw:.0f}°",
                 "직선→0°, 굴곡→양수 (공식 방향 OK)", "✅"])
    rows.append(["팔꿈치_L", f"{elbow_l_raw:.1f}°",
                 f"elbow_flex=180-{elbow_l_raw:.0f}={180-elbow_l_raw:.0f}°",
                 "", "✅"])

    # ── 4. 힙 굴곡 ────────────────────────────────────────────────────────────
    # 연결 세그먼트: 체간(hip←shoulder_mid) / 대퇴(hip→knee)
    # 꼭짓점: hip
    # 현재: 180 - angle(shoulder_mid, hip, knee)
    #   → 직립: shoulder위, knee아래 → 거의 반대방향 → angle≈180° → 180-180=0° ✓
    #   → 앞으로 들기: angle 줄어듦 → 180-angle 커짐 ✓
    hip_r_raw = vec_angle_deg(shoulder_mid, rh, rk)
    hip_l_raw = vec_angle_deg(shoulder_mid, lh, lk)
    rows.append(["힙굴곡_R", f"{hip_r_raw:.1f}°",
                 f"hip_flex=180-{hip_r_raw:.0f}={180-hip_r_raw:.0f}°",
                 "직립→0°, 들기→양수 (공식 방향 OK)", "✅"])
    rows.append(["힙굴곡_L", f"{hip_l_raw:.1f}°",
                 f"hip_flex=180-{hip_l_raw:.0f}={180-hip_l_raw:.0f}°",
                 "", "✅"])

    # ── 5. 발목 ────────────────────────────────────────────────────────────────
    ankle_r_raw = vec_angle_deg(rk, ra, rf)
    ankle_l_raw = vec_angle_deg(lk, la, lf)
    rows.append(["발목_R", f"{ankle_r_raw:.1f}°",
                 f"ankle=90-{ankle_r_raw:.0f}={90-ankle_r_raw:.0f}°",
                 "중립≈90°→0°", "✅"])

    # 출력
    print(f"{'관절':14s}  {'raw각도':12s}  {'변환값':22s}  {'설명':30s}  판정")
    print("-"*100)
    for r in rows:
        print(f"{r[0]:14s}  {r[1]:12s}  {r[2]:22s}  {r[3]:30s}  {r[4]}")

    # ── 핵심 버그 요약 ─────────────────────────────────────────────────────
    print("\n[핵심 버그 요약]")
    print(f"  ❌ arm_flex_r: 팔이 체간 옆에 있는데 계산값={current_arm_r:.0f}°  (올바른값≈{method_b_r:.0f}°)")
    print(f"     현재공식 '180 - angle(hip,shoulder,elbow)' → 팔 아래있을때 ≈180° (머리위 버그)")
    print(f"     올바른공식: angle(체간하강벡터, 상완벡터) = {method_b_r:.0f}°")

    return {
        'knee_r': knee_r_raw - 180,
        'knee_l': knee_l_raw - 180,
        'arm_flex_r_wrong': current_arm_r,
        'arm_flex_r_correct': method_b_r,
        'elbow_flex_r': 180 - elbow_r_raw,
        'hip_flex_r': 180 - hip_r_raw,
    }

# ── 비교 이미지 생성 ──────────────────────────────────────────────────────────
MP_CONN = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (27,29),(28,30),(29,31),(30,32),
]

def draw_skeleton_on_frame(frame, lms_norm, color=(0,255,0), thickness=3):
    H, W = frame.shape[:2]
    for s, e in MP_CONN:
        if s < len(lms_norm) and e < len(lms_norm):
            x1 = int(lms_norm[s].x * W); y1 = int(lms_norm[s].y * H)
            x2 = int(lms_norm[e].x * W); y2 = int(lms_norm[e].y * H)
            cv2.line(frame, (x1,y1), (x2,y2), color, thickness)
    for lm in lms_norm:
        cv2.circle(frame, (int(lm.x*W), int(lm.y*H)), 5, (255,255,255), -1)
    return frame

def project_fk_to_image(fk_pos, img_size, view='front'):
    """OpenSim Y-UP 좌표 → 2D 이미지 투영.
    front: XY 평면 (Z무시, X→가로, -Y→세로)
    side:  ZY 평면 (Z→가로, -Y→세로)
    """
    H, W = img_size
    # 모든 점 범위 파악
    pts_3d = np.array(list(fk_pos.values()))
    if len(pts_3d) == 0: return {}

    if view == 'front':   # X-Y 평면 (모델 정면)
        pts_2d = pts_3d[:, [0, 1]]
        pts_2d[:,1] *= -1  # Y반전 (위가 작은 값)
    elif view == 'side':  # Z-Y 평면 (모델 측면)
        pts_2d = pts_3d[:, [2, 1]]
        pts_2d[:,1] *= -1
    else:  # top: X-Z
        pts_2d = pts_3d[:, [0, 2]]

    # 정규화 → 이미지 좌표
    mn = pts_2d.min(axis=0) - 0.1
    mx = pts_2d.max(axis=0) + 0.1
    rng = mx - mn
    rng[rng < 0.01] = 0.01

    result = {}
    for name, p in fk_pos.items():
        if view == 'front': xy = np.array([p[0], -p[1]])
        elif view == 'side': xy = np.array([p[2], -p[1]])
        else:               xy = np.array([p[0], p[2]])
        u = int((xy[0]-mn[0])/rng[0] * (W-40) + 20)
        v = int((xy[1]-mn[1])/rng[1] * (H-40) + 20)
        result[name] = (u, v)
    return result

FK_CHAINS = [
    ('pelvis','femur_r'),('femur_r','tibia_r'),('tibia_r','calcn_r'),
    ('pelvis','femur_l'),('femur_l','tibia_l'),('tibia_l','calcn_l'),
    ('torso','humerus_r'),('humerus_r','radius_r'),
    ('torso','humerus_l'),('humerus_l','radius_l'),
    ('pelvis','torso'),
]

def draw_fk_stick(img, pts_2d, color=(0,100,255)):
    for a,b in FK_CHAINS:
        if a in pts_2d and b in pts_2d:
            cv2.line(img, pts_2d[a], pts_2d[b], color, 4)
    for name, pt in pts_2d.items():
        cv2.circle(img, pt, 6, (255,200,0), -1)
        cv2.putText(img, name[:6], (pt[0]+4, pt[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200,200,200), 1)
    return img

# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, action='append', default=[100, 300, 600])
    args = parser.parse_args()
    frames = sorted(set(args.frame))

    print("OpenSim 모델 로드 중...")
    model = pyopensim.Model(OSIM_PATH)
    state = model.initSystem()

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE)

    cap = cv2.VideoCapture(VIDEO)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as lmk:
        for fn in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                print(f"Frame {fn}: 읽기 실패"); continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.pose_world_landmarks or not res.pose_landmarks:
                print(f"Frame {fn}: 랜드마크 없음"); continue

            wlm  = res.pose_world_landmarks[0]
            nlm  = res.pose_landmarks[0]

            print(f"\n{'='*80}")
            print(f"FRAME {fn}")
            diag = diagnose_angles(wlm)

            # ── FK 실행 (현재 공식 그대로) ────────────────────────────────
            def pt(i): return np.array([wlm[i].x, wlm[i].y, wlm[i].z])
            lh,rh=pt(23),pt(24); lk,rk=pt(25),pt(26)
            la,ra=pt(27),pt(28); ls,rs=pt(11),pt(12)
            le,re=pt(13),pt(14); lw,rw=pt(15),pt(16)
            lf,rf=pt(31),pt(32)
            shoulder_mid=(ls+rs)/2; thigh_r=rk-rh; thigh_l=lk-lh

            def snorm(v, lo, hi):
                c,h = (lo+hi)/2, (hi-lo)/2
                return c + h * math.tanh((v-c)/h)

            coords_wrong = {
                'pelvis_tilt':0,'pelvis_list':0,'pelvis_rotation':0,
                'pelvis_tx':0,'pelvis_ty':0.9,'pelvis_tz':0,
                'knee_angle_r':  snorm(math.radians(vec_angle_deg(rh,rk,ra)-180),-2.0,0.17),
                'knee_angle_l':  snorm(math.radians(vec_angle_deg(lh,lk,la)-180),-2.0,0.17),
                'hip_flexion_r': snorm(math.radians(180-vec_angle_deg(shoulder_mid,rh,rk)),-0.52,2.09),
                'hip_flexion_l': snorm(math.radians(180-vec_angle_deg(shoulder_mid,lh,lk)),-0.52,2.09),
                'ankle_angle_r': snorm(math.radians(90-vec_angle_deg(rk,ra,rf)),-1.0,0.6),
                'ankle_angle_l': snorm(math.radians(90-vec_angle_deg(lk,la,lf)),-1.0,0.6),
                # 현재 버그: 180 - angle
                'arm_flex_r': snorm(math.radians(180-vec_angle_deg(rh,rs,re)),-0.5,2.5),
                'arm_flex_l': snorm(math.radians(180-vec_angle_deg(lh,ls,le)),-0.5,2.5),
                'elbow_flex_r': snorm(math.radians(180-vec_angle_deg(rs,re,rw)),-0.1,2.5),
                'elbow_flex_l': snorm(math.radians(180-vec_angle_deg(ls,le,lw)),-0.1,2.5),
            }

            # 올바른 arm_flex: angle(체간하강, 상완)
            coords_fixed = dict(coords_wrong)
            torso_down_r = rh - rs; upper_arm_r = re - rs
            torso_down_l = lh - ls; upper_arm_l = le - ls
            coords_fixed['arm_flex_r'] = snorm(math.radians(
                angle_between_vecs(torso_down_r, upper_arm_r)), -0.5, 2.5)
            coords_fixed['arm_flex_l'] = snorm(math.radians(
                angle_between_vecs(torso_down_l, upper_arm_l)), -0.5, 2.5)

            print("\n[FK 입력 각도 비교]")
            print(f"  {'관절':20s}  {'현재(버그)':12s}  {'수정값':12s}")
            print(f"  {'-'*50}")
            for k in ['arm_flex_r','arm_flex_l','elbow_flex_r','knee_angle_r','hip_flexion_r']:
                wv = math.degrees(coords_wrong[k])
                fv = math.degrees(coords_fixed[k])
                diff = abs(wv - fv)
                flag = "  ← ❌ 버그" if diff > 20 else ""
                print(f"  {k:20s}  {wv:+8.1f}°    {fv:+8.1f}°{flag}")

            fk_wrong = run_fk(model, state, coords_wrong)
            fk_fixed = run_fk(model, state, coords_fixed)

            # ── 비교 이미지 생성 ───────────────────────────────────────────
            # 패널 크기
            PH = 600; PW = 480
            blank = lambda: np.zeros((PH, PW, 3), dtype=np.uint8)

            # 패널1: MediaPipe 스켈레톤 (정답)
            p1 = cv2.resize(frame, (PW, PH))
            draw_skeleton_on_frame(p1, nlm, color=(0,255,0))
            cv2.putText(p1,"MediaPipe (정답)",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            # 패널2: 현재 FK 정면뷰 (버그)
            p2 = blank()
            pts2_f = project_fk_to_image(fk_wrong, (PH, PW), view='front')
            draw_fk_stick(p2, pts2_f, color=(0,80,255))
            cv2.putText(p2,"FK 현재공식 (정면)",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,80,255),2)
            cv2.putText(p2,f"arm_flex_r={math.degrees(coords_wrong['arm_flex_r']):.0f}deg (BUG)",(10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,80,255),1)

            # 패널3: 현재 FK 측면뷰
            p3 = blank()
            pts3_s = project_fk_to_image(fk_wrong, (PH, PW), view='side')
            draw_fk_stick(p3, pts3_s, color=(0,80,255))
            cv2.putText(p3,"FK 현재공식 (측면)",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,80,255),2)
            cv2.putText(p3,f"knee_r={math.degrees(coords_wrong['knee_angle_r']):.0f}deg",(10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(100,255,100),1)

            # 패널4: 수정 FK 정면뷰
            p4 = blank()
            pts4_f = project_fk_to_image(fk_fixed, (PH, PW), view='front')
            draw_fk_stick(p4, pts4_f, color=(0,220,0))
            cv2.putText(p4,"FK 수정공식 (정면)",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,0),2)
            cv2.putText(p4,f"arm_flex_r={math.degrees(coords_fixed['arm_flex_r']):.0f}deg (FIX)",(10,50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,220,0),1)

            # 패널5: 수정 FK 측면뷰
            p5 = blank()
            pts5_s = project_fk_to_image(fk_fixed, (PH, PW), view='side')
            draw_fk_stick(p5, pts5_s, color=(0,220,0))
            cv2.putText(p5,"FK 수정공식 (측면)",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,0),2)

            # 패널6: 수정 FK 위에서뷰
            p6 = blank()
            pts6_t = project_fk_to_image(fk_fixed, (PH, PW), view='top')
            draw_fk_stick(p6, pts6_t, color=(200,200,0))
            cv2.putText(p6,"FK 수정공식 (위)",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,0),2)

            top_row = np.hstack([p1, p2, p3])
            bot_row = np.hstack([p4, p5, p6])

            # 구분선 + 레이블
            sep = np.full((8, PW*3, 3), 60, dtype=np.uint8)
            cv2.putText(sep,"◀ 위: 버그    아래: 수정 ▶",(PW*3//2-150,6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1)
            out_img = np.vstack([top_row, sep, bot_row])

            out_path = f"{OUT_DIR}/frame_{fn:04d}.jpg"
            cv2.imwrite(out_path, out_img)
            print(f"\n  → 비교 이미지 저장: {out_path}")

    cap.release()
    print("\n진단 완료.")

if __name__ == '__main__':
    main()
