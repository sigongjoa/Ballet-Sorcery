#set document(title: "芭蕾呪法 — SMPL 메시 정규화 리포트")
#set page(margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 22pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 15pt)[SMPL 메시 정규화 — 포즈 추정 + SMPL 공통 좌표계 통합 리포트]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-23 | /sc:duo — Claude 설계 + Gemini 구현]
]

#line(length: 100%)
#v(0.5em)

= 작업 개요

== 해결한 문제

이번 작업에서는 포즈 추정 스켈레톤(v2에서 해결)에 이어 SMPL 3D 메시 출력에도 정규화를 적용하였다.

#table(
  columns: (1fr, 1.5fr, 1.5fr),
  inset: 8pt,
  align: (center, left, left),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if calc.odd(row) { rgb("#f8f8ff") } else { white },
  table.header(
    text(fill: white, weight: "bold")[레이어],
    text(fill: white, weight: "bold")[정규화 이전],
    text(fill: white, weight: "bold")[정규화 이후],
  ),
  [포즈 스켈레톤 (v2)], [카메라 좌표계 의존, 스케일 불일치], [골반 원점 + 어깨폭=1.0 공통 좌표계 ✅],
  [SMPL 3D 메시 (이번)], [단순 오버레이, 위치/스케일 불일치], [동일 좌표계 side-by-side 렌더링 ✅],
)

= 핵심 구현

== fit\_smpl\_to\_joints() — 관절 좌표 반환 추가

SMPL 피팅 후 최종 관절 좌표(joints\_np)를 함께 반환하도록 수정하였다.
이 관절 좌표가 정규화 기준(골반, 어깨)으로 사용된다.

```python
# 기존 반환 (4개)
return verts, pose, betas, transl

# 변경 후 반환 (5개)
with torch.no_grad():
    final_out = smpl_model(global_orient=global_orient,
                           body_pose=body_pose, betas=betas)
    verts = (final_out.vertices[0] + transl).cpu().numpy()
    joints_np = (final_out.joints[0, :22] + transl[0]).cpu().numpy()
    pose = torch.cat([global_orient[0], body_pose[0]]).cpu().numpy()

return verts, joints_np, pose, betas[0].detach().cpu().numpy(), transl[0].detach().cpu().numpy()
```

== normalize\_smpl\_mesh() — 정규화 함수

```python
def normalize_smpl_mesh(verts, joints):
    """
    SMPL 메시를 공통 좌표계로 정규화.
    - pelvis (joint[0]) → 원점 (0, 0, 0)
    - shoulder_width (joint[16]↔joint[17]) → 1.0
    """
    pelvis = joints[0]                                    # SMPL joint 0 = 골반
    shoulder_width = np.linalg.norm(joints[16] - joints[17])  # 어깨 간 거리
    scale = 1.0 / max(shoulder_width, 1e-6)

    verts_norm  = (verts  - pelvis) * scale
    joints_norm = (joints - pelvis) * scale

    return verts_norm, joints_norm, scale
```

*정규화 기준:*
- *관절 인덱스:* SMPL AMASS 포맷 기준 — joint\[0\] = 골반, joint\[16\] = 왼어깨, joint\[17\] = 오른어깨
- *좌표 일관성:* MediaPipe → SMPL 매핑 시 이미 Y축 반전 적용 (MediaPipe Y↓ → SMPL Y↑)
- *포즈 추정 정규화와 동일한 기준* 사용 → 스켈레톤과 SMPL 메시가 같은 좌표계 공유

== render\_smpl\_sidebyside() — Side-by-Side 렌더링

```python
def render_smpl_sidebyside(ref_verts, comp_verts, faces, output_path, title=''):
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[1920, 960])
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).flatten()

    # 왼쪽: Reference (파랑)
    plotter.subplot(0, 0)
    plotter.add_mesh(pv.PolyData(ref_verts, pv_faces),
                     color='royalblue', smooth_shading=True)
    plotter.camera_position = [(0, 0, 4), (0, 0, 0), (0, 1, 0)]  # 고정 카메라
    plotter.add_text('Reference (Normalized)', ...)

    # 오른쪽: Compare (빨강)
    plotter.subplot(0, 1)
    plotter.add_mesh(pv.PolyData(comp_verts, pv_faces),
                     color='tomato', smooth_shading=True)
    plotter.camera_position = [(0, 0, 4), (0, 0, 0), (0, 1, 0)]  # 동일 카메라
    plotter.add_text('Compare (Normalized)', ...)
```

*핵심 포인트:* 두 서브플롯에 *완전히 동일한 카메라 위치*를 사용 → 동일한 시점에서 비교 가능.
기존 오버레이 방식은 두 메시의 평균 중심에 카메라를 배치하여 비교가 왜곡될 수 있었다.

#pagebreak()

= Before / After 비교

== Frame 519 — 정규화 전 (Raw Overlay)

#figure(
  image("assets/smpl_raw_frame0519.png", width: 90%),
  caption: [정규화 이전 (Frame 519): 두 메시를 단순 오버레이. 스케일 차이와 카메라 왜곡으로 비교 어려움]
)

*문제점:*
- 두 메시가 겹쳐져 구분 어려움
- 다른 카메라/촬영 거리로 인한 스케일 차이 내포
- 카메라가 두 메시의 평균 중심에 배치 → 한 메시가 더 크게 보임

#pagebreak()

== Frame 519 — 정규화 후 (Normalized Side-by-Side)

#figure(
  image("assets/smpl_norm_frame0519.png", width: 100%),
  caption: [정규화 이후 (Frame 519): Reference(파랑)와 Compare(빨강)을 동일 스케일로 나란히 표시]
)

*개선 효과:*
- 두 메시가 명확히 분리되어 각각 관찰 가능
- 골반을 원점으로, 어깨폭을 1.0으로 정규화하여 체형 차이 제거
- 동일한 카메라 위치(z=4) → 동등한 시점에서 비교
- Frame 519에서 Reference는 왼팔을 내리고 오른쪽으로 무게 중심, Compare는 정면 서기

#pagebreak()

= 정규화 SMPL 시퀀스 결과

== Frame 0 (시작 프레임)

#figure(
  image("assets/smpl_norm_frame0000.png", width: 100%),
  caption: [Frame 0: Reference(파랑) — 팔을 측면으로 뻗은 자세. Compare(빨강) — 직립 서기. Loss: 0.3630]
)

#pagebreak()

== Frame 865 (후반부)

#figure(
  image("assets/smpl_norm_frame0865.png", width: 100%),
  caption: [Frame 865: Reference(파랑) — 복잡한 팔 자세 (loss 높음). Compare(빨강) — 안정적 서기. Loss: 0.8]
)

#pagebreak()

= 뷰어 통합 결과

== 정규화 SMPL 이미지가 로드된 뷰어 (Frame 0)

#figure(
  image("assets/ss01_frame0_full.png", width: 100%),
  caption: [뷰어 v2에 정규화 SMPL 메시가 통합된 모습. Reference(파랑)와 Compare(빨강)이 동일 스케일로 표시됨]
)

#pagebreak()

== Tab 2 — 정규화 나란히 (스켈레톤)

#figure(
  image("assets/ss03_tab2_norm_skeleton.png", width: 100%),
  caption: [Tab 2: 스켈레톤 정규화 나란히 보기. SMPL 메시(위)와 스켈레톤(아래) 모두 동일 좌표계 적용]
)

== Tab 3 — 정규화 오버레이

#figure(
  image("assets/ss04_tab3_overlay.png", width: 100%),
  caption: [Tab 3: 정규화 오버레이. 빨간 점선이 주요 관절 차이를 표시]
)

#pagebreak()

= 변경 사항 요약

== 수정된 파일

#table(
  columns: (1.5fr, 1fr, 2fr),
  inset: 8pt,
  align: (left, center, left),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if calc.odd(row) { rgb("#f8f8ff") } else { white },
  table.header(
    text(fill: white, weight: "bold")[파일],
    text(fill: white, weight: "bold")[상태],
    text(fill: white, weight: "bold")[변경 내용],
  ),
  [`src/smpl_visualizer.py`], [✅ DONE], [
    - `fit_smpl_to_joints()` 반환값에 `joints_np (22,3)` 추가\
    - `normalize_smpl_mesh(verts, joints)` 함수 신규 추가\
    - `render_smpl_sidebyside()` 함수 신규 추가\
    - `generate_smpl_sequence()` 정규화 파이프라인 적용\
    - `generate_smpl_comparison()` 정규화 렌더링 추가
  ],
  [`src/viewer_app.py`], [✅ DONE], [
    - SMPL 이미지 캡션에 정규화 정보 표시\
    - `use_column_width` deprecated → `width=1400` 수정
  ],
  [`data/e2e_output/smpl_seq/`], [✅ 재생성], [
    - 6개 프레임 × 정규화된 side-by-side PNG 재생성\
    - 기존 비정규화 이미지는 `smpl_seq_before_norm/`에 백업
  ],
)

== 정규화 파이프라인 전체 흐름

```
MediaPipe world_landmarks (dict)
         ↓
mediapipe_to_smpl_joints()   # MediaPipe → SMPL 22 joints, Y축 반전
         ↓
fit_smpl_to_joints()         # 2-stage 최적화 (Stage1: 방향/위치, Stage2: 전체 포즈)
         ↓  returns: verts(6890,3), joints(22,3), pose, betas, transl
normalize_smpl_mesh()        # pelvis 원점, shoulder_width=1.0 스케일
         ↓  returns: verts_norm(6890,3), joints_norm(22,3), scale
render_smpl_sidebyside()     # PyVista shape=(1,2), 고정 카메라 (0,0,4)
         ↓
smpl_seq_frame_{idx:04d}.png # 정규화된 side-by-side 이미지
```

= 결론 및 다음 단계

== 달성한 것

이번 작업으로 芭蕾呪法 시스템의 *전체 시각화 레이어*에 정규화가 적용되었다.

- *포즈 스켈레톤 (v2):* 골반 원점 + 어깨폭 1.0 → MediaPipe 좌표계 정규화 ✅
- *SMPL 3D 메시 (이번):* 동일한 기준으로 SMPL 메시 정규화 + Side-by-side 렌더링 ✅
- *일관성:* 스켈레톤과 SMPL 메시가 동일한 정규화 기준 사용 → 레이어 간 좌표 일관성 확보

== 관찰 — 정규화 전후 차이

Frame 519를 기준으로:
- *정규화 전:* 두 메시가 겹쳐진 복잡한 오버레이 — 어느 팔이 누구의 것인지 불분명
- *정규화 후:* Reference(파랑)는 왼팔을 내린 자세, Compare(빨강)는 양손을 허리에 올린 자세가 명확히 구분됨

== 개선 가능 사항

+ *카메라 각도:* 현재 정면(z=4)만 표시 → 다중 뷰(front/side/top) 제공 시 더 완전한 비교
+ *골반 이하 바디:* 현재 카메라(focal=(0,0,0))가 골반 기준 → 발끝이 짤릴 수 있음. focal=(0,-0.5,0) 조정 검토
+ *On-demand 생성:* 뷰어에서 "이 프레임 SMPL 생성" 버튼으로 임의 프레임 즉시 정규화 렌더링
+ *전체 시퀀스:* 현재 6프레임 → 1,041개 전체 프레임 배치 처리 (수 시간 소요 예상)

#v(1em)
#line(length: 100%)
#align(center)[
  #text(size: 8pt, fill: gray)[
    /sc:duo 구현 | Claude 설계 + Gemini 구현 + Claude 검증 | PyVista + smplx | 2026-02-23
  ]
]
