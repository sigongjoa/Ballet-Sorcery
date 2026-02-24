import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import urllib.request
import warnings
warnings.filterwarnings('ignore')

# VTP 다운로드
BASE_URL = 'https://raw.githubusercontent.com/opensim-org/opensim-models/master/Models/Rajagopal/Geometry/'
VTP_FILES = ['femur_r.vtp','tibia_r.vtp','fibula_r.vtp','talus_rv.vtp','foot.vtp','bofoot.vtp',
             'femur_l.vtp','tibia_l.vtp','fibula_l.vtp','talus_lv.vtp','l_foot.vtp','l_bofoot.vtp',
             'r_pelvis.vtp','l_pelvis.vtp','sacrum.vtp','hat_ribs_scap.vtp',
             'humerus_rv.vtp','radius_rv.vtp','ulna_rv.vtp',
             'humerus_lv.vtp','radius_lv.vtp','ulna_lv.vtp','r_patella.vtp','l_patella.vtp']
GEOM_DIR = '/tmp/opensim_geometry'
os.makedirs(GEOM_DIR, exist_ok=True)

print('Downloading VTP files...')
for vtp in VTP_FILES:
    path = os.path.join(GEOM_DIR, vtp)
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(BASE_URL + vtp, path)
            print(f'  Downloaded: {vtp}')
        except Exception as e:
            print(f'  Failed: {vtp} - {e}')

# pyopensim FK
import pyopensim
MODEL_PATH = '/mnt/d/progress/芭蕾呪法/data/e2e_output/opensim_output/scaled_model.osim'
MOT_PATH = '/mnt/d/progress/芭蕾呪法/data/e2e_output/opensim_output/reference_poses_ik.mot'

print('Loading model...')
model = pyopensim.Model(MODEL_PATH)
state = model.initSystem()

with open(MOT_PATH) as f:
    skip = 0
    for i, line in enumerate(f):
        if 'endheader' in line:
            skip = i + 1
            break
ik = pd.read_csv(MOT_PATH, sep='\t', skiprows=skip)
print(f'IK loaded: {len(ik)} frames')

N = len(ik)
frame_indices = [int(N * p / 11) for p in range(11)] + [N - 1]

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
def body_color(name):
    if name in LOWER: return '#4488ff'
    if name in UPPER: return '#ff8844'
    if name == 'pelvis': return '#d4a0a0'
    return '#c0c0c0'

def compute_fk(row):
    cs = model.getCoordinateSet()
    for j in range(cs.getSize()):
        coord = cs.get(j)
        name = coord.getName()
        if name in row.index:
            val = float(row[name])
            try:
                if coord.getMotionType() == 1:
                    val = float(np.radians(val))
                coord.setValue(state, val)
            except:
                pass
    model.realizePosition(state)
    result = {}
    bs = model.getBodySet()
    for j in range(bs.getSize()):
        b = bs.get(j)
        n = b.getName()
        try:
            p = b.getPositionInGround(state)
            R = b.getTransformInGround(state).R()
            result[n] = {
                'pos': np.array([float(p[0]), float(p[1]), float(p[2])]),
                'R': np.array([[float(R.get(r,c)) for c in range(3)] for r in range(3)])
            }
        except:
            pass
    return result

import pyvista as pv
pv.start_xvfb()

def render_frame(bt, t, knee, hip):
    pl = pv.Plotter(off_screen=True, window_size=[400, 600])
    pl.set_background('#1a1a2e')
    for bname, vtps in BODY_VTP.items():
        if bname not in bt:
            continue
        pos = bt[bname]['pos']
        R = bt[bname]['R']
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = pos
        color = body_color(bname)
        added = False
        for vf in vtps:
            vpath = os.path.join(GEOM_DIR, vf)
            if os.path.exists(vpath):
                try:
                    mesh = pv.read(vpath)
                    mesh = mesh.transform(T)
                    pl.add_mesh(mesh, color=color, smooth_shading=True, opacity=0.9)
                    added = True
                except:
                    pass
        if not added:
            pl.add_mesh(pv.Sphere(radius=0.03, center=pos.tolist()), color=color)
    pl.camera_position = 'yz'
    pl.add_text(f't={t:.1f}s knee={knee:.0f}d hip={hip:.0f}d', position='upper_left', font_size=7, color='white')
    img = pl.screenshot(None, return_img=True)
    pl.close()
    return img

print('Rendering 12 frames...')
fig, axes = plt.subplots(4, 3, figsize=(12, 16), facecolor='#1a1a2e')
fig.suptitle('OpenSim Ballet - 3D Bone Rendering via Forward Kinematics', color='white', fontsize=13, fontweight='bold')

for idx, fi in enumerate(frame_indices):
    row = ik.iloc[fi]
    print(f'  Frame {idx+1}/12 t={row["time"]:.1f}s')
    bt = compute_fk(row)
    knee = float(row.get('knee_angle_r', 0))
    hip = float(row.get('hip_flexion_r', 0))
    img = render_frame(bt, float(row['time']), knee, hip)
    ax = axes[idx // 3][idx % 3]
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f't={row["time"]:.1f}s', color='white', fontsize=9, pad=2)

plt.tight_layout()
OUT = '/tmp/opensim_3d_bones_montage.png'
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f'Saved: {OUT}')
