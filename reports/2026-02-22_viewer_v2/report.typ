#set document(title: "芭蕾呪法 — 프레임 뷰어 v2 리포트")
#set page(margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 22pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 15pt)[프레임 뷰어 v2 — 개별 상세 뷰 & 공통 축 정규화 리포트]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-22 | Streamlit + 정규화 스켈레톤 비교]
]

#line(length: 100%)
#v(0.5em)

= 개요

기존 뷰어(v1)의 두 가지 핵심 문제를 해결한 인터랙티브 프레임 뷰어 v2를 구현하였다.

*해결한 문제:*
1. *개별 상세 뷰 부재*: 기준(Reference) / 비교(Compare) 포즈를 각각 충분히 크게 볼 수 없었음
2. *공통 좌표축 없음*: 두 영상의 카메라 위치·스케일·체형이 달라 스켈레톤 직접 비교가 어려웠음

*해결 방법:*
- 3개 탭 뷰 모드 구현: 개별 상세 보기 / 정규화 나란히 / 정규화 오버레이
- `normalize_landmarks()`: 골반 중심점 → 원점, 어깨 폭 → 1.0 스케일로 공통 좌표계 구성

= 핵심 구현 — 공통 축 정규화

== normalize\_landmarks() 함수

```python
def normalize_landmarks(lm_dict):
    """세계 좌표계 랜드마크를 공통 좌표계로 정규화.
    - 원점: 골반 중심 (LEFT_HIP + RIGHT_HIP) / 2
    - 스케일: 어깨 폭(LEFT_SHOULDER ↔ RIGHT_SHOULDER) = 1.0
    """
    # 골반 중심 계산
    lh = np.array([lm_dict['LEFT_HIP']['x'],
                   -lm_dict['LEFT_HIP']['y'],
                    lm_dict['LEFT_HIP']['z']])
    rh = np.array([lm_dict['RIGHT_HIP']['x'],
                   -lm_dict['RIGHT_HIP']['y'],
                    lm_dict['RIGHT_HIP']['z']])
    pelvis = (lh + rh) / 2.0

    # 어깨 폭 계산 (스케일 기준)
    ls = np.array([lm_dict['LEFT_SHOULDER']['x'],
                   -lm_dict['LEFT_SHOULDER']['y'],
                    lm_dict['LEFT_SHOULDER']['z']])
    rs = np.array([lm_dict['RIGHT_SHOULDER']['x'],
                   -lm_dict['RIGHT_SHOULDER']['y'],
                    lm_dict['RIGHT_SHOULDER']['z']])
    shoulder_width = np.linalg.norm(ls - rs)
    scale = 1.0 / max(shoulder_width, 1e-6)

    # 모든 랜드마크 정규화
    normalized = {}
    for name, lm in lm_dict.items():
        pos = np.array([lm['x'], -lm['y'], lm['z']])
        normalized[name] = (pos - pelvis) * scale

    return normalized, scale
```

*정규화 효과:*
- 두 사람의 골반이 동일한 원점(0,0)에 배치됨
- 어깨 폭 기준으로 신체 크기가 통일됨
- 카메라 위치·촬영 거리 차이가 제거되어 순수 포즈 비교 가능

= 3탭 뷰 모드

== Tab 1 — 개별 상세 보기

각 수행자(Reference, Compare)의 스켈레톤을 7×7 인치 크기의 독립된 뷰로 표시한다.
오른쪽 컬럼에는 해당 수행자의 관절별 손실 기여도를 Plotly 가로 막대 차트로 표시한다.

*특징:*
- Reference (파란색): 좌측 2/3, Compare (주황색): 우측 2/3 각각 전용 공간
- 관절 연결선 16개 (어깨-팔꿈치-손목, 힙-무릎-발목-발뒤꿈치-발끝 등)
- Y축 반전 적용 (`world_landmarks` Y↓ → 화면 Y↑)
- 비정규화 좌표 — 실제 월드 좌표계 그대로 표시 (스케일 보정 없음)

#figure(
  image("assets/ss01_tab1_individual_frame0.png", width: 100%),
  caption: [Tab 1 — Frame 0: Reference(좌, 파랑)와 Compare(우, 주황) 개별 상세 뷰. 각각 독립된 대형 패널에 스켈레톤 렌더링]
)

#pagebreak()

#figure(
  image("assets/ss04_tab1_frame456_maxloss.png", width: 100%),
  caption: [Tab 1 — Frame 456 (최대 손실 프레임, Loss=0.7730). Reference는 전방 굴곡 자세, Compare는 직립 서기 자세]
)

#pagebreak()

== Tab 2 — 정규화 나란히

두 수행자의 스켈레톤을 공통 좌표계(`normalize_landmarks()` 적용)로 변환한 뒤 동일한 축 범위에 나란히 표시한다.

*특징:*
- 공통 표시 범위: X ∈ \[-1.8, 1.8\], Y ∈ \[-2.0, 1.6\] (어깨 폭 = 1.0 기준)
- 두 패널의 축 범위가 완전히 동일하여 직관적인 자세 비교 가능
- 어깨 폭 스케일 값(예: ×1.42)을 패널 상단에 표시
- SMPL 이미지가 있는 프레임에서는 탭 위에 3D 메시 비교 이미지 표시

#figure(
  image("assets/ss02_tab2_normalized_sidebyside.png", width: 100%),
  caption: [Tab 2 — 정규화 나란히 보기. 상단: SMPL 3D 메시 비교(Frame 0). 하단: 공통 좌표축에 정렬된 Reference(파랑)와 Compare(주황) 스켈레톤]
)

#pagebreak()

== Tab 3 — 정규화 오버레이

두 수행자의 정규화된 스켈레톤을 단일 플롯에 겹쳐서 표시하며, 주요 관절 8개 사이에 빨간 점선으로 차이를 시각화한다.

*특징:*
- Reference(파랑) + Compare(주황)를 단일 좌표계에 오버레이
- 빨간 점선("Joint Diff"): 양쪽 어깨, 힙, 무릎, 발목 — 8개 주요 관절 연결
- 점선 길이가 포즈 차이를 즉각적으로 시각화
- 좌측 부정보 패널: 현재 프레임 손실 + 관절별 TOP 5 오차

#figure(
  image("assets/ss03_tab3_normalized_overlay.png", width: 100%),
  caption: [Tab 3 — 정규화 오버레이 (Frame 0). Reference(파랑)와 Compare(주황)이 동일 좌표계에서 겹쳐짐. 빨간 점선: 주요 관절 위치 차이]
)

#pagebreak()

#figure(
  image("assets/ss05_tab3_frame456_overlay.png", width: 100%),
  caption: [Tab 3 — 최대 손실 프레임 456 오버레이 (Loss=0.7730). 빨간 점선이 크게 늘어져 대폭 자세 차이를 명확히 표시. 발끝 관절 오차 최대]
)

#pagebreak()

= 구현 상세

== 뷰어 구조

```
viewer_app.py (~370줄)
├── load_json_data()          # @st.cache_data, 3개 JSON 로드
├── normalize_landmarks()     # 골반 중심 + 어깨 폭 정규화
├── draw_skeleton()           # Matplotlib 스켈레톤 렌더링
│   └── CONNECTIONS = [       # 16개 관절 연결 정의
│       ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
│       ('LEFT_SHOULDER', 'LEFT_ELBOW'),
│       ... (14개 추가)
│       ]
├── render_joint_bar()        # Plotly 관절별 손실 가로 막대
├── render_loss_timeline()    # Plotly 손실 타임라인 (현재 프레임 하이라이트)
└── main()
    ├── 사이드바: 데이터 디렉토리 + 프레임 구간 설정
    ├── 현재 프레임 정보 표시 (Loss, Ref/Comp 원본 프레임)
    ├── SMPL 이미지 (정확히 일치 시 표시)
    ├── st.tabs([Tab1, Tab2, Tab3])
    │   ├── Tab1: ref_col + comp_col (7×7 개별 뷰 + 관절 바)
    │   ├── Tab2: ref_col + comp_col (정규화 동일 축 + 스케일 표시)
    │   └── Tab3: overlay_col + info_col (오버레이 + 점선 + TOP5)
    └── 손실 타임라인 (하단 전체 너비)
```

== v1 → v2 변경 사항

#table(
  columns: (1fr, 1.5fr, 1.5fr),
  inset: 8pt,
  align: (center, left, left),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if calc.odd(row) { rgb("#f8f8ff") } else { white },
  table.header(
    text(fill: white, weight: "bold")[항목],
    text(fill: white, weight: "bold")[v1],
    text(fill: white, weight: "bold")[v2],
  ),
  [개별 뷰], [작은 side-by-side], [7×7 인치 전용 패널 (Tab1)],
  [좌표계], [비정규화 (카메라 의존)], [골반 중심 + 어깨 폭 정규화 (Tab2/3)],
  [오버레이], [없음], [단일 좌표계 + 관절 diff 점선 (Tab3)],
  [관절 바], [없음], [각 수행자별 TOP10 관절 손실 (Tab1/3)],
  [SMPL 표시], [항상 시도 (없으면 힌트)], [정확 일치 시만 탭 위에 표시],
  [뷰 모드], [단일 뷰], [3탭 (개별 / 정규화 나란히 / 오버레이)],
)

== 핵심 기술 선택

#table(
  columns: (1.5fr, 1fr, 2fr),
  inset: 8pt,
  align: (left, center, left),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if calc.odd(row) { rgb("#fff8f0") } else { white },
  table.header(
    text(fill: white, weight: "bold")[항목],
    text(fill: white, weight: "bold")[선택],
    text(fill: white, weight: "bold")[이유],
  ),
  [스켈레톤 렌더링], [Matplotlib], [정확한 좌표 제어, 커스텀 색상/축 범위],
  [차트], [Plotly], [인터랙티브, 다크 테마, 현재 프레임 마커],
  [정규화 기준], [어깨 폭], [발레에서 어깨 폭이 안정적·의미 있는 기준],
  [Y축 처리], [\-lm\['y'\]], [MediaPipe Y↓ → 화면 Y↑ 변환],
  [캐싱], [`@st.cache_data`], [JSON 파일 최초 로드 후 즉시 재사용],
)

= 분석 관찰

== Frame 0 (초기 자세)

- Reference (프레임 793, 전문가): 양팔을 약간 벌린 준비 자세
- Compare (프레임 0, 초보자): 팔을 옆에 붙인 직립 자세
- Loss: 0.28 — 기본 자세의 팔 위치 차이

정규화 오버레이에서 두 골반이 동일한 원점에 정렬되어 팔 위치 차이가 명확히 드러남.

== Frame 456 (최대 손실 — Loss: 0.7730)

- Reference (프레임 1089): 전방으로 크게 굴곡된 플리에 자세
- Compare (프레임 456): 직립 서기 자세
- 오버레이에서 빨간 점선이 특히 무릎·발목 관절에서 크게 늘어짐
- 발끝(RIGHT\_FOOT\_INDEX) 오차 2.16으로 전체 최고값

이 프레임은 전문가의 깊은 플리에 동작 대비 초보자의 부족한 무릎 굴곡을 명확히 보여준다.

= 결론

*해결된 문제:*
- 개별 대형 뷰를 통해 각 수행자의 자세를 충분히 상세하게 분석 가능
- 공통 좌표계 정규화로 서로 다른 카메라 설정의 영상 간 의미 있는 비교 가능
- 오버레이 + 빨간 점선으로 포즈 차이의 위치와 크기를 직관적으로 파악 가능

*향후 개선 가능 사항:*
+ 3D 오버레이: 현재 2D(X-Y 평면) 투영 → 3D scatter로 전환 시 깊이 방향(Z) 차이도 시각화
+ 프레임 애니메이션: 슬라이더 자동 재생(Play/Pause) 버튼 추가
+ 관절 필터: 표시할 관절 그룹(상체/하체/손가락)을 체크박스로 선택
+ SMPL 온디맨드 생성: "이 프레임 SMPL 생성" 버튼으로 캐시 확장

#v(1em)
#line(length: 100%)
#align(center)[
  #text(size: 8pt, fill: gray)[
    /sc:duo 구현 | Claude 직접 수정 | Streamlit + Plotly + Matplotlib | 2026-02-22
  ]
]
