# -*- coding: utf-8 -*-
import streamlit as st
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D

# 페이지 설정
st.set_page_config(page_title="芭蕾呪法 — 포즈 비교 분석기 v2", layout="wide")

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1a1a2e;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e9eff !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# MediaPipe 관절 연결선 정의
NAMED_CONNECTIONS = [
    ("LEFT_SHOULDER","RIGHT_SHOULDER"),
    ("LEFT_SHOULDER","LEFT_ELBOW"),("RIGHT_SHOULDER","RIGHT_ELBOW"),
    ("LEFT_ELBOW","LEFT_WRIST"),("RIGHT_ELBOW","RIGHT_WRIST"),
    ("LEFT_SHOULDER","LEFT_HIP"),("RIGHT_SHOULDER","RIGHT_HIP"),
    ("LEFT_HIP","RIGHT_HIP"),
    ("LEFT_HIP","LEFT_KNEE"),("RIGHT_HIP","RIGHT_KNEE"),
    ("LEFT_KNEE","LEFT_ANKLE"),("RIGHT_KNEE","RIGHT_ANKLE"),
    ("LEFT_ANKLE","LEFT_HEEL"),("RIGHT_ANKLE","RIGHT_HEEL"),
    ("LEFT_HEEL","LEFT_FOOT_INDEX"),("RIGHT_HEEL","RIGHT_FOOT_INDEX"),
]

# 차이 연결선 대상 주요 관절 (8개)
KEY_JOINTS = [
    "LEFT_SHOULDER","RIGHT_SHOULDER",
    "LEFT_HIP","RIGHT_HIP",
    "LEFT_KNEE","RIGHT_KNEE",
    "LEFT_ANKLE","RIGHT_ANKLE",
]

@st.cache_data
def load_data(data_dir):
    """데이터 디렉토리에서 JSON 파일 3개를 로드하고 캐싱합니다."""
    paths = {
        "comparison_result": os.path.join(data_dir, "comparison_result.json"),
        "reference_poses":   os.path.join(data_dir, "reference_poses.json"),
        "compare_poses":     os.path.join(data_dir, "compare_poses.json"),
    }
    data = {}
    try:
        for key, path in paths.items():
            with open(path, 'r', encoding='utf-8') as f:
                data[key] = json.load(f)
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None
    return data


@st.cache_data
def load_biomechanics(data_dir):
    """바이오메카닉스 JSON 파일을 로드합니다. 없으면 None 반환."""
    paths = {
        "ref":  os.path.join(data_dir, "reference_biomechanics.json"),
        "comp": os.path.join(data_dir, "compare_biomechanics.json"),
        "cmp":  os.path.join(data_dir, "biomechanics_comparison.json"),
    }
    result = {}
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                result[key] = json.load(f)
        else:
            result[key] = None
    return result if any(v is not None for v in result.values()) else None


def normalize_landmarks(lm_dict):
    """
    world_landmarks dict를 공통 좌표계로 정규화합니다.
    - 골반 중심화: pelvis = (LEFT_HIP + RIGHT_HIP) / 2 → (0, 0, 0)
    - 어깨너비 스케일: shoulder_width → 1.0
    반환: (normalized_dict, scale_factor)
    """
    if not isinstance(lm_dict, dict) or len(lm_dict) == 0:
        return lm_dict, 1.0

    needed = ["LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
    if not all(k in lm_dict for k in needed):
        return lm_dict, 1.0

    def vec(name):
        lm = lm_dict[name]
        return np.array([lm['x'], lm['y'], lm['z']])

    pelvis = (vec("LEFT_HIP") + vec("RIGHT_HIP")) / 2.0
    shoulder_width = np.linalg.norm(vec("LEFT_SHOULDER") - vec("RIGHT_SHOULDER"))
    scale = 1.0 / shoulder_width if shoulder_width > 1e-6 else 1.0

    normalized = {}
    for name, lm in lm_dict.items():
        p = np.array([lm['x'], lm['y'], lm['z']])
        n = (p - pelvis) * scale
        normalized[name] = {"x": float(n[0]), "y": float(n[1]), "z": float(n[2])}

    return normalized, scale


def find_nearest_smpl_frame(smpl_dir, target_idx):
    """캐시된 SMPL 프레임 중 가장 가까운 것을 찾습니다."""
    if not os.path.exists(smpl_dir):
        return None, None, None
    files = [f for f in os.listdir(smpl_dir)
             if f.startswith("smpl_seq_frame_") and f.endswith(".png")]
    if not files:
        return None, None, None
    indices = sorted([int(f.split('_')[-1].split('.')[0]) for f in files])
    nearest = min(indices, key=lambda x: abs(x - target_idx))
    diff = abs(nearest - target_idx)
    path = os.path.join(smpl_dir, f"smpl_seq_frame_{nearest:04d}.png")
    return path, nearest, diff


def draw_skeleton(ax, lm_dict, color, title="", lims=None):
    """
    단일 matplotlib axes에 스켈레톤을 그립니다.
    world_landmarks 기준 → y 반전 적용
    """
    ax.set_facecolor('#1a1a2e')
    if not isinstance(lm_dict, dict) or len(lm_dict) == 0:
        ax.text(0.5, 0.5, "포즈 데이터 없음", color='white',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title, color='white', fontsize=13)
        ax.axis('off')
        return

    # world coords: y↓ → 화면 y↑ 로 반전
    coord = {}
    for name, lm in lm_dict.items():
        if isinstance(lm, dict) and 'x' in lm and 'y' in lm:
            coord[name] = (lm['x'], -lm['y'])

    if not coord:
        ax.text(0.5, 0.5, "좌표 없음", color='white',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return

    xs = [v[0] for v in coord.values()]
    ys = [v[1] for v in coord.values()]

    # 점
    ax.scatter(xs, ys, c=color, s=45, zorder=4, alpha=0.9)

    # 연결선
    for a, b in NAMED_CONNECTIONS:
        if a in coord and b in coord:
            ax.plot([coord[a][0], coord[b][0]],
                    [coord[a][1], coord[b][1]],
                    c=color, linewidth=2.5, alpha=0.75)

    if lims:
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])

    ax.set_title(title, color='white', fontsize=13, pad=8)
    ax.set_aspect('equal')
    ax.grid(True, color='#2a2a4a', linestyle='--', alpha=0.4)
    ax.tick_params(colors='#666688')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a4a')


def render_joint_bar(joint_losses, color, title):
    """관절별 손실 수평 bar chart (Plotly)."""
    top10 = sorted(joint_losses.items(), key=lambda x: x[1], reverse=True)[:10]
    joints = [x[0] for x in top10][::-1]
    values = [x[1] for x in top10][::-1]
    fig = go.Figure(go.Bar(x=values, y=joints, orientation='h',
                           marker_color=color))
    fig.update_layout(
        title=title, height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='#1a1a2e', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=11),
        xaxis=dict(gridcolor='#33334d', title="Loss"),
        yaxis=dict(gridcolor='#33334d')
    )
    return fig


def render_loss_timeline(per_frame_loss, current_frame, start_frame, end_frame):
    """전체 손실 타임라인 (Plotly)."""
    frames = [item['frame_idx'] for item in per_frame_loss]
    losses = [item['total_loss'] for item in per_frame_loss]

    mask = [start_frame <= f <= end_frame for f in frames]
    act_f = [f for f, m in zip(frames, mask) if m]
    act_l = [l for l, m in zip(losses, mask) if m]

    cur_loss = next((item['total_loss'] for item in per_frame_loss
                     if item['frame_idx'] == current_frame), 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=act_f, y=act_l, mode='lines',
                             line=dict(color='#00d4ff', width=2.5),
                             fill='tozeroy', fillcolor='rgba(0,212,255,0.08)'))
    fig.add_trace(go.Scatter(x=[current_frame], y=[cur_loss], mode='markers',
                             marker=dict(color='#ff4e4e', size=12,
                                         line=dict(color='white', width=2))))
    fig.add_vline(x=current_frame, line_dash="dash", line_color="#ff4e4e", opacity=0.6)
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='#1a1a2e', paper_bgcolor='#0d0d1a',
        font=dict(color='white'), showlegend=False,
        xaxis=dict(title="Frame Index", gridcolor='#33334d'),
        yaxis=dict(title="Loss", gridcolor='#33334d'),
    )
    return fig


# ── 발레 지표 메타 정보 ──────────────────────────────────────────────
ANGLE_META = {
    'left_knee_flexion':          {'label': '좌 무릎 굴곡',      'unit': '°', 'ideal': 0.0},
    'right_knee_flexion':         {'label': '우 무릎 굴곡',      'unit': '°', 'ideal': 0.0},
    'left_hip_flexion':           {'label': '좌 고관절 굴곡',    'unit': '°', 'ideal': 0.0},
    'right_hip_flexion':          {'label': '우 고관절 굴곡',    'unit': '°', 'ideal': 0.0},
    'left_turnout':               {'label': '좌 턴아웃',         'unit': '°', 'ideal': 90.0},
    'right_turnout':              {'label': '우 턴아웃',         'unit': '°', 'ideal': 90.0},
    'left_ankle_plantarflexion':  {'label': '좌 발목 저굴',      'unit': '°', 'ideal': 90.0},
    'right_ankle_plantarflexion': {'label': '우 발목 저굴',      'unit': '°', 'ideal': 90.0},
    'spine_lateral':              {'label': '척추 측굴',         'unit': '°', 'ideal': 0.0},
    'shoulder_symmetry':          {'label': '어깨 비대칭',       'unit': '',  'ideal': 0.0},
}

ANGLE_GROUPS = [
    ('무릎·고관절', ['left_knee_flexion', 'right_knee_flexion',
                     'left_hip_flexion',  'right_hip_flexion']),
    ('턴아웃·발목', ['left_turnout',  'right_turnout',
                     'left_ankle_plantarflexion', 'right_ankle_plantarflexion']),
    ('척추·어깨',   ['spine_lateral', 'shoulder_symmetry']),
]


def render_angle_timeseries(bio_ref, bio_comp, angle_keys, title):
    """
    관절 각도 시계열 그래프를 렌더링합니다.
    ref=파란선, comp=빨간선.
    """
    fig = make_subplots(
        rows=len(angle_keys), cols=1,
        shared_xaxes=True,
        subplot_titles=[ANGLE_META.get(k, {}).get('label', k) for k in angle_keys],
        vertical_spacing=0.06,
    )

    for row, key in enumerate(angle_keys, start=1):
        # Reference
        if bio_ref:
            ref_frames = [f['frame_idx'] for f in bio_ref['frames']]
            ref_vals   = [f['angles'].get(key) for f in bio_ref['frames']]
            fig.add_trace(go.Scatter(
                x=ref_frames, y=ref_vals,
                mode='lines', name=f'Ref {ANGLE_META.get(key, {}).get("label", key)}',
                line=dict(color='#4e9eff', width=1.5),
                showlegend=(row == 1),
            ), row=row, col=1)

        # Compare
        if bio_comp:
            cmp_frames = [f['frame_idx'] for f in bio_comp['frames']]
            cmp_vals   = [f['angles'].get(key) for f in bio_comp['frames']]
            fig.add_trace(go.Scatter(
                x=cmp_frames, y=cmp_vals,
                mode='lines', name=f'Comp {ANGLE_META.get(key, {}).get("label", key)}',
                line=dict(color='#ff7f4e', width=1.5),
                showlegend=(row == 1),
            ), row=row, col=1)

        # 이상값 수평선
        ideal = ANGLE_META.get(key, {}).get('ideal')
        if ideal is not None:
            fig.add_hline(
                y=ideal, line_dash="dot", line_color="#00ff88",
                opacity=0.4, row=row, col=1,
            )

        unit = ANGLE_META.get(key, {}).get('unit', '°')
        fig.update_yaxes(title_text=unit, row=row, col=1,
                         gridcolor='#33334d', title_font_size=10)

    fig.update_layout(
        title=title,
        height=200 * len(angle_keys),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#0d0d1a',
        font=dict(color='white', size=11),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=20, t=60, b=20),
    )
    fig.update_xaxes(gridcolor='#33334d', title_text="Frame")
    return fig


def render_ballet_scores_cards(ref_scores, comp_scores):
    """발레 스코어 카드를 Plotly 테이블로 렌더링합니다."""
    score_labels = {
        'turnout_score':        '턴아웃 스코어',
        'knee_alignment_score': '무릎 정렬 스코어',
        'spine_alignment_score':'척추 정렬 스코어',
    }

    rows = []
    for key, label in score_labels.items():
        rv = ref_scores.get(key)
        cv = comp_scores.get(key) if comp_scores else None
        rv_s = f"{rv:.3f}" if rv is not None else "—"
        cv_s = f"{cv:.3f}" if cv is not None else "—"
        rows.append((label, rv_s, cv_s))

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['지표', 'Reference', 'Compare'],
            fill_color='#2a2a4a',
            font=dict(color='white', size=13),
            align='center',
            height=35,
        ),
        cells=dict(
            values=list(zip(*rows)) if rows else [[], [], []],
            fill_color='#1a1a2e',
            font=dict(color='white', size=12),
            align='center',
            height=30,
        ),
    )])
    fig.update_layout(
        height=180, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def render_mean_angle_table(bio_ref, bio_comp):
    """평균 각도 비교 테이블을 Plotly로 렌더링합니다."""
    ref_means  = bio_ref.get('summary', {}).get('mean_angles', {})  if bio_ref  else {}
    comp_means = bio_comp.get('summary', {}).get('mean_angles', {}) if bio_comp else {}

    rows = []
    for key, meta in ANGLE_META.items():
        rv = ref_means.get(key)
        cv = comp_means.get(key)
        ideal = meta.get('ideal')
        rows.append((
            meta['label'],
            f"{ideal:.1f}{meta['unit']}" if ideal is not None else "—",
            f"{rv:.1f}{meta['unit']}"   if rv    is not None else "—",
            f"{cv:.1f}{meta['unit']}"   if cv    is not None else "—",
        ))

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['관절 지표', '이상값', 'Ref 평균', 'Comp 평균'],
            fill_color='#2a2a4a',
            font=dict(color='white', size=12),
            align='center',
            height=35,
        ),
        cells=dict(
            values=list(zip(*rows)) if rows else [[], [], [], []],
            fill_color='#1a1a2e',
            font=dict(color='white', size=11),
            align='center',
            height=28,
        ),
    )])
    fig.update_layout(
        height=360, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ─────────────────────── 메인 ───────────────────────
def main():
    st.title("芭蕾呪法 — 포즈 비교 분석기 v2")

    # ── 사이드바 ──
    st.sidebar.header("설정")
    data_dir = st.sidebar.text_input("데이터 디렉토리", value="data/e2e_output")

    data = load_data(data_dir)
    if not data:
        st.warning("데이터를 불러올 수 없습니다.")
        return

    pfl = data['comparison_result'].get('per_frame_loss', [])
    if not pfl:
        st.error("per_frame_loss 없음")
        return

    max_frame = pfl[-1]['frame_idx']

    if 'start_frame' not in st.session_state:
        st.session_state.start_frame = 0
    if 'end_frame' not in st.session_state:
        st.session_state.end_frame = max_frame

    start_in = st.sidebar.number_input("시작 프레임", 0, max_frame, int(st.session_state.start_frame))
    end_in   = st.sidebar.number_input("종료 프레임", 0, max_frame, int(st.session_state.end_frame))

    if st.sidebar.button("적용 (Apply Range)", use_container_width=True):
        st.session_state.start_frame = start_in
        st.session_state.end_frame   = end_in
        st.rerun()

    # ── 프레임 슬라이더 ──
    cur = st.slider(
        "프레임 선택",
        min_value=int(st.session_state.start_frame),
        max_value=int(st.session_state.end_frame),
        value=int(st.session_state.start_frame),
        key="frame_slider",
    )

    # 현재 프레임 데이터
    loss_data = next((x for x in pfl if x['frame_idx'] == cur), None)
    if not loss_data:
        st.warning(f"Frame {cur} 데이터 없음")
        return

    ref_idx  = loss_data['original_ref_frame_idx']
    comp_idx = loss_data['original_comp_frame_idx']

    st.info(
        f"**Frame {cur}** | "
        f"Loss: `{loss_data['total_loss']:.4f}` | "
        f"Ref 원본: `{ref_idx}` | Comp 원본: `{comp_idx}`"
    )

    # 포즈 데이터 조회
    ref_frames  = data['reference_poses'].get('frames', [])
    comp_frames = data['compare_poses'].get('frames', [])
    ref_fd  = next((f for f in ref_frames  if f['frame_idx'] == ref_idx),  None)
    comp_fd = next((f for f in comp_frames if f['frame_idx'] == comp_idx), None)

    ref_wl  = ref_fd.get("world_landmarks")  if ref_fd  else None
    comp_wl = comp_fd.get("world_landmarks") if comp_fd else None

    # SMPL 이미지 (exact match만 표시)
    smpl_dir = os.path.join(data_dir, "smpl_seq")
    smpl_path, nearest, diff = find_nearest_smpl_frame(smpl_dir, cur)
    if smpl_path and diff == 0:
        _, sc, _ = st.columns([1, 8, 1])
        with sc:
            st.image(smpl_path,
                     caption=f"SMPL 3D Mesh — Frame {cur}  |  정규화: 골반 원점 · 어깨폭=1.0",
                     width=1400)
            st.caption("📐 Reference(파랑)와 Compare(빨강) 모두 동일한 좌표계로 정규화되어 표시됩니다.")

    # 정규화
    norm_ref,  scale_ref  = normalize_landmarks(ref_wl)
    norm_comp, scale_comp = normalize_landmarks(comp_wl)
    LIMS = [-1.8, 1.8, -2.0, 1.6]  # 정규화 좌표 범위

    # 바이오메카닉스 데이터 로드 (탭 렌더링 전에 준비)
    bio_data = load_biomechanics(data_dir)

    # ── 4탭 ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 개별 상세 보기", "⚖️ 정규화 나란히", "🎯 정규화 오버레이", "🦴 바이오메카닉스"
    ])

    # ─ Tab1: 개별 상세 보기 ─
    with tab1:
        c_ref, c_comp = st.columns(2)

        with c_ref:
            st.markdown("#### Reference")
            fig_r, ax_r = plt.subplots(figsize=(7, 7))
            fig_r.patch.set_facecolor('#1a1a2e')
            draw_skeleton(ax_r, ref_wl, color='#4e9eff',
                          title=f"Reference — Frame {ref_idx}")
            st.pyplot(fig_r)
            plt.close(fig_r)
            if loss_data.get('joint_losses'):
                st.plotly_chart(
                    render_joint_bar(loss_data['joint_losses'], '#4e9eff', "관절 오차 (Ref 기준)"),
                    use_container_width=True)

        with c_comp:
            st.markdown("#### Compare")
            fig_c, ax_c = plt.subplots(figsize=(7, 7))
            fig_c.patch.set_facecolor('#1a1a2e')
            draw_skeleton(ax_c, comp_wl, color='#ff7f4e',
                          title=f"Compare — Frame {comp_idx}")
            st.pyplot(fig_c)
            plt.close(fig_c)
            if loss_data.get('joint_losses'):
                st.plotly_chart(
                    render_joint_bar(loss_data['joint_losses'], '#ff7f4e', "관절 오차 (Comp 기준)"),
                    use_container_width=True)

    # ─ Tab2: 정규화 나란히 ─
    with tab2:
        st.caption("골반을 원점(0,0)으로 이동하고 어깨 너비를 1.0으로 스케일한 공통 좌표계입니다.")
        fig_n, (ax_n1, ax_n2) = plt.subplots(1, 2, figsize=(14, 7))
        fig_n.patch.set_facecolor('#1a1a2e')

        draw_skeleton(ax_n1, norm_ref,  color='#4e9eff',
                      title=f"Ref (Normalized) — Frame {ref_idx}",  lims=LIMS)
        draw_skeleton(ax_n2, norm_comp, color='#ff7f4e',
                      title=f"Comp (Normalized) — Frame {comp_idx}", lims=LIMS)

        plt.tight_layout(pad=2)
        st.pyplot(fig_n)
        plt.close(fig_n)

        ci1, ci2 = st.columns(2)
        with ci1:
            st.code(f"[Ref]  Scale: {scale_ref:.4f}x  |  Pelvis → (0, 0, 0)")
        with ci2:
            st.code(f"[Comp] Scale: {scale_comp:.4f}x  |  Pelvis → (0, 0, 0)")

    # ─ Tab3: 정규화 오버레이 ─
    with tab3:
        st.caption("두 포즈를 같은 좌표계 위에 겹쳐 그립니다. 빨간 점선 = 주요 관절 간 차이.")
        fig_ov, ax_ov = plt.subplots(figsize=(9, 9))
        fig_ov.patch.set_facecolor('#1a1a2e')

        draw_skeleton(ax_ov, norm_ref,  color='#4e9eff', lims=LIMS)
        draw_skeleton(ax_ov, norm_comp, color='#ff7f4e', lims=LIMS)

        # 주요 관절 차이 점선
        if isinstance(norm_ref, dict) and isinstance(norm_comp, dict):
            for j in KEY_JOINTS:
                if j in norm_ref and j in norm_comp:
                    rx, ry = norm_ref[j]['x'],  -norm_ref[j]['y']
                    cx, cy = norm_comp[j]['x'], -norm_comp[j]['y']
                    ax_ov.plot([rx, cx], [ry, cy],
                               color='#ff3333', linestyle='--',
                               linewidth=1.2, alpha=0.65, zorder=2)

        ax_ov.set_title("Normalized Overlay", color='white', fontsize=14, pad=10)

        legend_handles = [
            Line2D([0], [0], color='#4e9eff', lw=3, label='Reference'),
            Line2D([0], [0], color='#ff7f4e', lw=3, label='Compare'),
            Line2D([0], [0], color='#ff3333', lw=1.5,
                   linestyle='--', label='Joint Diff'),
        ]
        ax_ov.legend(handles=legend_handles, facecolor='#1a1a2e',
                     edgecolor='#444', labelcolor='white',
                     loc='upper right', fontsize=11)

        st.pyplot(fig_ov)
        plt.close(fig_ov)

    # ─ Tab4: 바이오메카닉스 ─
    with tab4:
        if bio_data is None:
            st.info(
                "바이오메카닉스 데이터가 없습니다. "
                "파이프라인을 실행하거나 `reference_biomechanics.json`을 생성하세요.\n\n"
                "```bash\n"
                "python src/pipeline.py --ref <ref_video> --comp <comp_video> "
                "--output data/e2e_output\n"
                "```"
            )
        else:
            bio_ref  = bio_data.get('ref')
            bio_comp = bio_data.get('comp')

            # ── 발레 스코어 카드 ──
            st.markdown("#### 발레 스코어")
            ref_scores  = bio_ref.get('summary', {}).get('ballet_scores', {})  if bio_ref  else {}
            comp_scores = bio_comp.get('summary', {}).get('ballet_scores', {}) if bio_comp else {}
            st.plotly_chart(
                render_ballet_scores_cards(ref_scores, comp_scores),
                use_container_width=True,
            )

            method = bio_ref.get('method', '?') if bio_ref else '?'
            trc_path = bio_ref.get('trc_path', '') if bio_ref else ''
            st.caption(
                f"계산 방법: `{method}` | "
                f"TRC 파일: `{os.path.basename(trc_path)}` | "
                f"초록 점선 = 발레 이상값"
            )

            # ── 평균 각도 비교 테이블 ──
            st.markdown("#### 평균 관절 각도 비교")
            st.plotly_chart(
                render_mean_angle_table(bio_ref, bio_comp),
                use_container_width=True,
            )

            # ── 시계열 그래프 (그룹별) ──
            st.markdown("#### 관절 각도 시계열")
            for group_name, keys in ANGLE_GROUPS:
                with st.expander(f"📈 {group_name}", expanded=(group_name == '무릎·고관절')):
                    fig_ts = render_angle_timeseries(bio_ref, bio_comp, keys, group_name)
                    st.plotly_chart(fig_ts, use_container_width=True)

    # ── 하단 고정: 손실 타임라인 ──
    st.markdown("---")
    st.subheader("손실 타임라인")
    st.plotly_chart(
        render_loss_timeline(pfl, cur,
                             st.session_state.start_frame,
                             st.session_state.end_frame),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
