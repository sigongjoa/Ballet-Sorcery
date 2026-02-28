#set document(title: "芭蕾呪法 - VTP vs MediaPipe 벡터 정렬 진단")
#set page(paper: "a4", margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 20pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 14pt)[VTP 포즈 추종 실패 근본 원인 진단 리포트]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-28 | 이슈 \#6/\#7 | tools/diag_vtp_vs_mp.py]
]

#line(length: 100%)
#v(0.5em)

= 진단 목적

VTP 뼈 렌더링이 실제 포즈를 따라오지 못하는 원인을 파이프라인 단계별로 추적한다.

#rect(fill: rgb("#fff3cd"), radius: 4pt, inset: 10pt)[
  *핵심 질문*: MediaPipe 관절 벡터와 OpenSim FK 결과 벡터 간 차이가 얼마나 크며, 어느 단계에서 오차가 발생하는가?
]

#v(0.3em)

진단 파이프라인:
#align(center)[
  ```
  MediaPipe world_landmarks (3D, m)
       ↓ landmarks_to_opensim_coords()
  OpenSim DOF 값 (라디안)
       ↓ compute_fk()
  뼈 위치 행렬 (ground frame, 4x4)  ← 여기에서 VTP 렌더링
       ↓ 비교
  MediaPipe 관절 방향벡터 (ground truth)
  ```
]

= 진단 방법

#table(
  columns: (auto, 1fr),
  inset: 8pt,
  align: (left, left),
  [대상 영상], [`IMG_2633.MOV` — 실제 발레 촬영본],
  [분석 프레임], [100, 300, 600 (초기·중간·후반 포즈)],
  [비교 체인], [우측 대퇴 / 좌측 대퇴 / 우측 하퇴 / 좌측 하퇴 / 우측 상완],
  [오차 지표], [MP 방향벡터 vs FK 방향벡터 간 각도 오차 (도)],
  [허용 기준], [15도 이내 (임상 모션캡처 기준)],
)

= STEP 2: 방향벡터 각도 오차

== 결과 그래프

#figure(
  image("assets/angular_error_comparison.png", width: 100%),
  caption: [MediaPipe vs OpenSim FK 방향벡터 각도 오차. 빨간 점선 = 15도 허용 기준.]
)

== 수치 요약

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  inset: 6pt,
  align: center,
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if col == 0 { rgb("#f5f5f5") } else { white },
  table.header(
    text(fill: white)[프레임],
    text(fill: white)[우측 대퇴],
    text(fill: white)[좌측 대퇴],
    text(fill: white)[우측 하퇴],
    text(fill: white)[좌측 하퇴],
    text(fill: white)[우측 상완],
    text(fill: white)[평균],
  ),
  [100], [22.7°], [44.8°], [*74.5° ❌*], [16.9°], [36.1°], [39.0°],
  [300], [40.0°], [9.1°], [29.6°], [47.6° ❌], [47.1°], [34.7°],
  [600], [60.2° ❌], [29.4°], [13.8°], [24.6°], [12.3°], [28.1°],
  [*전체 평균*], [41.0°], [27.8°], [39.3°], [29.7°], [31.8°], [*33.9°*],
)

#rect(fill: rgb("#ffe0e0"), radius: 4pt, inset: 8pt)[
  *판정: FAIL* — 전체 평균 오차 33.9도 (허용 기준 15도의 2.3배). 특히 우측 하퇴(Frame 100: 74.5도)가 가장 심각. VTP가 실제 포즈를 따라오지 못하는 것이 수치로 확인됨.
]

= STEP 3: 체인 순서(Ordering) 보존 검증

#figure(
  image("assets/chain_ordering_check.png", width: 90%),
  caption: [Pelvis→Femur→Tibia→Talus→Calcn Y좌표 체인. 단조감소(초록) = 순서 정상.]
)

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  inset: 6pt,
  align: center,
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white)[프레임],
    text(fill: white)[Pelvis Y],
    text(fill: white)[Femur_r Y],
    text(fill: white)[Tibia_r Y],
    text(fill: white)[Talus_r Y],
    text(fill: white)[Calcn_r Y],
    text(fill: white)[순서 판정],
  ),
  [100], [0.90m], [0.84m], [0.50m], [0.10m], [0.10m], [PASS ✅],
  [300], [0.90m], [0.84m], [0.50m], [0.10m], [0.10m], [PASS ✅],
  [600], [0.90m], [0.84m], [0.56m], [0.13m], [0.12m], [PASS ✅],
)

#rect(fill: rgb("#e8f5e9"), radius: 4pt, inset: 8pt)[
  *판정: PASS* — 운동학 체인 순서(위→아래) 위반 없음. 단, FK Y좌표가 프레임 100, 300, 600에서 거의 동일. 포즈 변화에 따른 세그먼트 위치 업데이트가 불충분함을 시사.
]

= 근본 원인 분석

== 확인된 문제 1: DOF→FK 방향벡터 매핑 오류

#table(
  columns: (auto, 1fr, 1fr),
  inset: 8pt,
  align: (left, left, left),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white)[원인],
    text(fill: white)[설명],
    text(fill: white)[증거],
  ),
  [좌표계 불일치],
    [MediaPipe Z축(깊이)이 OpenSim 좌표계와 다름. `landmarks_to_opensim_coords()`가 3D 벡터를 YZ·XY 평면으로 투영하여 깊이 정보 손실],
    [하퇴 오차 최대 74.5° — 무릎 굴곡은 주로 시상면(XZ) 운동이므로 Z 손실 영향 극대화],
  [2D 투영 기반 각도 계산],
    [힙/무릎 각도를 3D 벡터 내적으로 계산하지만, MediaPipe world_lm의 Z 신뢰도가 낮아 3D 각도 자체가 부정확],
    [프레임별 오차 패턴 비일관: 같은 체인이 프레임마다 오차 크기가 크게 달라짐],
  [EMA 과도 평탄화],
    [시간적 EMA(α=0.6)가 DOF 값을 이전 프레임 쪽으로 당겨 포즈 변화 속도를 억제],
    [FK Y좌표가 3개 프레임(100→600)에서 거의 변화 없음 (Pelvis=0.90m 고정)],
)

== 확인된 문제 2: MediaPipe 33점 스파스 데이터의 구조적 한계

MediaPipe 33개 랜드마크는 표면 관측점(surface marker)이 아니라 관절 중심 추정값이다. 특히:
- 카메라 뒤쪽(Z방향) 정확도 낮음 → 3D 회전 계산 불확실
- 발목·발 관절의 회전(pronation/supination) 정보 없음
- 힙 내전/외전 분리 계산이 thigh Y벡터 부호에 의존 → 불안정

= DOF 값 분포

#figure(
  image("assets/dof_values_frame100.png", width: 100%),
  caption: [Frame 100 DOF 값 (파랑) vs OpenSim 유효 범위 (회색). hip_adduction 값이 0에 가까워 측면 동작이 FK에 반영되지 않음.]
)

= 결론 및 해결 방향

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  align: (left, left, center),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white)[진단 항목],
    text(fill: white)[결과],
    text(fill: white)[판정],
  ),
  [방향벡터 평균 오차], [33.9도 (허용 15도 초과)], [FAIL ❌],
  [체인 순서 보존], [3개 프레임 모두 단조감소 유지], [PASS ✅],
  [포즈 변화 반응성], [FK Y좌표가 프레임간 거의 변화 없음], [FAIL ❌],
  [hip_adduction 반영], [값이 ~0으로 측면 동작 미반영], [FAIL ❌],
)

#v(0.5em)

#rect(fill: rgb("#e3f2fd"), radius: 4pt, inset: 10pt)[
  *해결 방향 — SMPL-X 메시 기반 접근 (이슈 \#7)*

  현재 방식인 MediaPipe 33점 → DOF 수동 계산은 구조적 한계가 있음.

  SMPL-X 피팅(이슈 \#7)을 통해:
  1. 10,475개 정점으로 전신 3D 형태 복원 → Z축 깊이 손실 없음
  2. SMPL-X body_pose (54개 관절 × axis-angle) → 직접 OpenSim DOF 변환 가능
  3. 56개 가상 마커를 OpenSim IK에 주입 → DOF 수동 계산 우회
  4. 예상 오차 감소: 33.9° → 목표 15° 이내
]

#v(1cm)
#text(size: 9pt, fill: gray)[
  진단 도구: tools/diag_vtp_vs_mp.py |
  분석 영상: IMG_2633.MOV (Frame 100, 300, 600) |
  생성일: 2026-02-28
]
