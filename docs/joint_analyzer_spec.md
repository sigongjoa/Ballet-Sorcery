# Ballet Joint Analyzer — 구현 명세

## 개요
MediaPipe → OpenSim FK 파이프라인의 체계적 디버깅/분석 도구.
Dear PyGui 네이티브 앱으로 관절 각도·3D FK·비디오를 동시에 표시.

---

## 레이아웃 (1600×900px 고정)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Ballet Joint Analyzer                                              [─][□][✕]            │
├───────────────────────┬───────────────────────┬─────────────────────────────────────────┤
│  VIDEO  (640×480)     │  FK 3D  (640×480)      │  JOINT INSPECTOR  (300px)               │
│                       │                        │                                         │
│  [원본 프레임]         │  [PyVista offscreen     │  ┌─ 선택된 관절 ──────────────────┐    │
│  + MediaPipe 초록선    │   → DPG texture]        │  │  knee_r                        │    │
│  + 관절 노드 노란점    │                        │  │  Position (ground)             │    │
│                       │  선택관절 = 빨간구체    │  │    X: -0.066 m                 │    │
│                       │  연결체인 = 노란선(굵)  │  │    Y:  0.456 m                 │    │
│                       │  나머지   = 회색선      │  │    Z:  0.077 m                 │    │
│                       │                        │  ├────────────────────────────────┤    │
│                       │                        │  │  DOF: knee_angle_r             │    │
│                       │                        │  │  값: 42.4°  ✅ [0°~155°]      │    │
│                       │                        │  │  [──────────●──────] slider    │    │
│                       │                        │  ├────────────────────────────────┤    │
│                       │                        │  │  Connected Chain               │    │
│                       │                        │  │  pelvis                        │    │
│                       │                        │  │    └─ femur_r                  │    │
│                       │                        │  │         └─ [tibia_r] ◄선택    │    │
│                       │                        │  │              └─ calcn_r        │    │
│                       │                        │  └────────────────────────────────┘    │
│                       │                        │                                         │
│                       │                        │  ┌─ 전체 DOF 상태 ────────────────┐    │
│                       │                        │  │ knee_r    42° ████░░░  ✅      │    │
│                       │                        │  │ knee_l    36° ███░░░░  ✅      │    │
│                       │                        │  │ hip_r     21° ██░░░░░  ✅      │    │
│                       │                        │  │ arm_flex_r 6° █░░░░░░  ✅      │    │
│                       │                        │  │ arm_add_r 22° ██░░░░░  ✅      │    │
│                       │                        │  │ ankle_r  -16° ░░░█░░░  ⚠️     │    │
│                       │                        │  │ elbow_r   36° ███░░░░  ✅      │    │
│                       │                        │  └────────────────────────────────┘    │
├───────────────────────┴───────────────────────┴─────────────────────────────────────────┤
│  [◀◀] [◀] [▶/⏸] [▶▶]   Frame [═══════════●═══════════════════════════════]  300/1019  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 파일 구조

```
tools/
  joint_analyzer.py          ← 진입점: python3 tools/joint_analyzer.py --video <path>
  _analyzer/
    __init__.py
    pipeline.py              ← MediaPipe + FK 계산 (anatomy_overlay_viewer.py 재사용)
    renderer3d.py            ← PyVista offscreen → numpy RGBA
    joint_map.py             ← 관절 체인 정의, DOF 범위, 표시명
    ui_state.py              ← 공유 상태 (현재 프레임, 선택 관절, coords)
```

---

## 각 모듈 명세

### `joint_map.py`
```python
# 관절 체인 (부모→자식)
CHAIN = {
    'pelvis':    ['femur_r', 'femur_l', 'torso'],
    'femur_r':   ['tibia_r'],
    'tibia_r':   ['calcn_r'],
    'torso':     ['humerus_r', 'humerus_l'],
    'humerus_r': ['radius_r'],
    ...
}

# MediaPipe에서 계산되는 DOF와 범위
DOF_INFO = {
    'knee_angle_r':  {'range': (0.0, 2.7),   'unit': 'rad', 'body': 'tibia_r'},
    'knee_angle_l':  {'range': (0.0, 2.7),   'unit': 'rad', 'body': 'tibia_l'},
    'hip_flexion_r': {'range': (-0.52, 2.09),'unit': 'rad', 'body': 'femur_r'},
    'hip_flexion_l': {'range': (-0.52, 2.09),'unit': 'rad', 'body': 'femur_l'},
    'hip_adduction_r':{'range':(-0.87, 0.35),'unit': 'rad', 'body': 'femur_r'},
    'hip_adduction_l':{'range':(-0.87, 0.35),'unit': 'rad', 'body': 'femur_l'},
    'ankle_angle_r': {'range': (-1.0, 0.6),  'unit': 'rad', 'body': 'calcn_r'},
    'ankle_angle_l': {'range': (-1.0, 0.6),  'unit': 'rad', 'body': 'calcn_l'},
    'arm_flex_r':    {'range': (-0.5, 2.5),  'unit': 'rad', 'body': 'humerus_r'},
    'arm_flex_l':    {'range': (-0.5, 2.5),  'unit': 'rad', 'body': 'humerus_l'},
    'arm_add_r':     {'range': (-2.09, 1.22),'unit': 'rad', 'body': 'humerus_r'},
    'arm_add_l':     {'range': (-2.09, 1.22),'unit': 'rad', 'body': 'humerus_l'},
    'elbow_flex_r':  {'range': (0.0, 2.6),   'unit': 'rad', 'body': 'radius_r'},
    'elbow_flex_l':  {'range': (0.0, 2.6),   'unit': 'rad', 'body': 'radius_l'},
}

# 3D 렌더용 체인 (a→b 세그먼트)
FK_SEGMENTS = [
    ('pelvis','femur_r'), ('femur_r','tibia_r'), ('tibia_r','calcn_r'),
    ('pelvis','femur_l'), ('femur_l','tibia_l'), ('tibia_l','calcn_l'),
    ('pelvis','torso'),
    ('torso','humerus_r'), ('humerus_r','radius_r'),
    ('torso','humerus_l'), ('humerus_l','radius_l'),
]
```

### `pipeline.py`
```python
class JointPipeline:
    def __init__(self, osim_path, model_path):
        # pyopensim 모델 + MediaPipe landmarker 초기화

    def process_frame(self, bgr_frame) -> dict:
        # 반환: {
        #   'landmarks_norm': [...],   # 화면 스켈레톤용
        #   'landmarks_world': [...],  # 각도 계산용
        #   'coords': {...},           # OpenSim DOF 값
        #   'fk_positions': {...},     # body명 → [x,y,z]
        # }

    def compute_fk_for_coords(self, coords) -> dict:
        # coords 직접 받아서 FK 위치 반환 (슬라이더 수동 조정용)
```

### `renderer3d.py`
```python
class Renderer3D:
    def __init__(self, width=640, height=480):
        # PyVista offscreen plotter

    def render(self, fk_positions, selected_body=None) -> np.ndarray:
        # fk_positions: body명 → [x,y,z]
        # selected_body: 하이라이트할 body (빨간색)
        # 반환: (H, W, 4) RGBA uint8
        # 색상: 선택체인=노랑, 선택관절=빨강, 나머지=회색
        # 카메라: (1.8, 1.2, 2.5) → (0, 0.8, 0) 3/4뷰
```

### `ui_state.py`
```python
@dataclass
class UIState:
    frame_idx: int = 0
    total_frames: int = 0
    selected_body: str = 'tibia_r'   # 클릭으로 변경
    coords: dict = field(default_factory=dict)
    fk_positions: dict = field(default_factory=dict)
    manual_overrides: dict = field(default_factory=dict)  # 슬라이더 수동값
    playing: bool = False
    fps: float = 30.0
```

### `joint_analyzer.py` (메인)

#### DPG 윈도우 구성
```python
dpg.create_context()
dpg.create_viewport(title="Ballet Joint Analyzer", width=1600, height=900)

# 텍스처 등록 (비디오 + 3D)
dpg.add_raw_texture(640, 480, video_data, tag="video_tex", format=dpg.mvFormat_Float_rgb)
dpg.add_raw_texture(640, 480, fk3d_data,  tag="fk3d_tex",  format=dpg.mvFormat_Float_rgba)

with dpg.window(tag="main", no_title_bar=True, no_resize=True):
    with dpg.table(header_row=False):
        # 3열: video | fk3d | inspector
        with dpg.table_column(width_fixed=True, init_width_or_weight=640):
            dpg.add_image("video_tex")
        with dpg.table_column(width_fixed=True, init_width_or_weight=640):
            dpg.add_image("fk3d_tex")
            # 3D 이미지 클릭 → 관절 선택 (향후 구현)
        with dpg.table_column():
            _build_inspector_panel()   # 오른쪽 패널

    # 타임라인
    _build_timeline_panel()
```

#### Inspector 패널 내용
```python
def _build_inspector_panel():
    # 선택 관절 드롭다운
    dpg.add_combo(list(BODY_NAMES), default_value='tibia_r',
                  callback=on_body_selected, tag="body_selector")

    # Position in ground
    with dpg.group(tag="position_group"):
        dpg.add_text("Position (ground frame)", color=(200,200,200))
        dpg.add_text("X: ---", tag="pos_x")
        dpg.add_text("Y: ---", tag="pos_y")
        dpg.add_text("Z: ---", tag="pos_z")

    dpg.add_separator()

    # 해당 관절의 DOF 슬라이더들
    with dpg.group(tag="dof_sliders"):
        pass  # on_body_selected에서 동적 생성

    dpg.add_separator()

    # 연결 체인 트리
    with dpg.tree_node(label="Connected Chain", default_open=True):
        dpg.add_text("", tag="chain_text")

    dpg.add_separator()

    # 전체 DOF 상태 테이블
    with dpg.table(tag="dof_table", header_row=True):
        dpg.add_table_column(label="DOF")
        dpg.add_table_column(label="값")
        dpg.add_table_column(label="범위내")
        # 행은 update_dof_table()에서 동적 추가
```

#### 타임라인 패널
```python
def _build_timeline_panel():
    with dpg.group(horizontal=True):
        dpg.add_button(label="◀◀", callback=lambda: seek(-10))
        dpg.add_button(label="◀",  callback=lambda: seek(-1))
        dpg.add_button(label="▶",  tag="play_btn", callback=toggle_play)
        dpg.add_button(label="▶▶", callback=lambda: seek(+10))
        dpg.add_slider_int(min_value=0, max_value=1019, tag="timeline",
                           callback=on_timeline_change, width=1100)
        dpg.add_text("0/1019", tag="frame_label")
```

---

## 업데이트 루프

```python
def update_frame(frame_idx):
    ret, frame = cap.read(frame_idx)
    result = pipeline.process_frame(frame)

    # 1. 비디오 텍스처 업데이트 (MediaPipe 스켈레톤 오버레이)
    video_with_skeleton = draw_skeleton(frame, result['landmarks_norm'])
    dpg.set_value("video_tex", frame_to_dpg(video_with_skeleton))

    # 2. 수동 오버라이드 적용
    coords = {**result['coords'], **state.manual_overrides}
    fk_pos = pipeline.compute_fk_for_coords(coords)

    # 3. 3D 렌더
    fk_img = renderer.render(fk_pos, selected_body=state.selected_body)
    dpg.set_value("fk3d_tex", img_to_dpg(fk_img))

    # 4. Inspector 업데이트
    update_inspector(fk_pos, coords)
    update_dof_table(coords)

# 렌더 루프 (30fps 목표)
while dpg.is_dearpygui_running():
    if state.playing:
        state.frame_idx = (state.frame_idx + 1) % state.total_frames
        update_frame(state.frame_idx)
    dpg.render_dearpygui_frame()
```

---

## 구현 순서

1. `joint_map.py` — 데이터 정의 (의존성 없음)
2. `ui_state.py` — 상태 클래스
3. `pipeline.py` — `anatomy_overlay_viewer.py`에서 `landmarks_to_opensim_coords`, `compute_fk` 재사용
4. `renderer3d.py` — PyVista offscreen 렌더러
5. `joint_analyzer.py` — DPG 메인 앱, 위 모듈들 조합

---

## 실행 방법

```bash
python3 tools/joint_analyzer.py --video my_data/2026_02_25/IMG_2633.MOV
```

## 의존성

```
dearpygui>=2.0
pyvista
mediapipe
pyopensim (로컬)
opencv-python
numpy
```
