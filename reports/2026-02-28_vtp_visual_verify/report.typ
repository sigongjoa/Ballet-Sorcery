#set document(title: "芭蕾呪法 - VTP 시각적 검증 리포트")
#set page(paper: "a4", margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 20pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 14pt)[OpenSim VTP 시각적 검증 리포트]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-28 | 이슈 \#6 검증 | IMG_2633.MOV 실제 데이터]
]

#line(length: 100%)
#v(0.5em)

= 검증 개요

이슈 \#6에서 구현된 3패널 해부학 오버레이 뷰어(`src/anatomy_overlay_viewer.py`)의 VTP 뼈 렌더링이 실제 영상에서 올바르게 동작하는지 시각적으로 검증하였다.

#table(
  columns: (1fr, 2fr),
  inset: 8pt,
  align: (left, left),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white, weight: "bold")[항목],
    text(fill: white, weight: "bold")[내용],
  ),
  [원본 영상], [`IMG_2633.MOV` — 실제 발레/동작 촬영본],
  [검증 대상], [`IMG_2633_anatomy.mp4` — 3패널 합성 영상 (5760×1080, 60fps, 1019프레임)],
  [추출 프레임], [100, 250, 400, 600, 850 (5개 대표 시점)],
  [검증 도구], [`tools/verify_vtp_visual.py` — 프레임 추출 + 픽셀 통계 분석],
)

= VTP 픽셀 활성 비율 분석

패널 3(OpenSim VTP)에서 배경(RGB < 30)을 제외한 유효 렌더링 픽셀의 비율을 측정하였다.

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: (center, center, center),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if row == 6 { rgb("#e8f5e9") } else { white },
  table.header(
    text(fill: white, weight: "bold")[프레임 인덱스],
    text(fill: white, weight: "bold")[VTP 활성 픽셀 비율],
    text(fill: white, weight: "bold")[판정],
  ),
  [Frame 0100], [3.29%], [PASS ✅],
  [Frame 0250], [3.51%], [PASS ✅],
  [Frame 0400], [3.32%], [PASS ✅],
  [Frame 0600], [3.74%], [PASS ✅],
  [Frame 0850], [3.92%], [PASS ✅],
  [*평균*], [*3.56%*], [*렌더링 정상*],
)

#text(size: 9pt, fill: gray)[
  활성 픽셀 3~4%는 어두운 배경 위 3D 뼈 메시의 전형적인 비율임. 전신 VTP가 화면에 차지하는 면적 자체가 작기 때문에 정상 수치.
]

= 프레임별 시각적 분석

== Frame 0100 — 초기 자세 (보행 준비)

#figure(
  image("assets/frame_0100_compare.jpg", width: 100%),
  caption: [Frame 0100: 원본(좌) vs OpenSim VTP(우). 흉곽(흰), 골반(분홍), 오른쪽 다리(파랑), 팔(주황) 확인.]
)

#table(
  columns: (auto, 1fr),
  inset: 6pt,
  align: (center, left),
  [항목], [관찰 결과],
  [VTP 색상 코딩], [하지(파랑), 상지(주황/적), 흉곽(흰), 골반(분홍) — 정의대로 렌더링됨],
  [뼈 형태], [femur, tibia, calcn, torso, humerus 형상 모두 인식 가능],
  [자세 대응], [원본의 한쪽 다리 앞으로 나온 자세 → VTP에 다리 하나 연장 반영],
)

#v(0.5em)

== Frame 0400 — 중간 동작

#figure(
  image("assets/frame_0400_compare.jpg", width: 100%),
  caption: [Frame 0400: 직립 자세. VTP에서 골반-흉곽 분리 시각 확인.]
)

== Frame 0600 — 다리 벌림 동작 (핵심 검증)

#figure(
  image("assets/frame_0600_compare.jpg", width: 100%),
  caption: [Frame 0600: 다리 벌림 동작. 원본의 양 발 분리가 VTP에서 좌우 다리로 정확히 반영됨.]
)

#text(fill: rgb("#0066cc"), weight: "bold")[
  핵심 성공 케이스: Frame 0600에서 원본 영상의 넓은 스탠스가 VTP의 두 다리 분기 형태로 정확히 재현됨. FK 계산이 실제 동작을 올바르게 추종하고 있음을 증명.
]

= 검증 결과 요약

#table(
  columns: (1fr, auto, 1fr),
  inset: 8pt,
  align: (left, center, left),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white, weight: "bold")[검증 항목],
    text(fill: white, weight: "bold")[결과],
    text(fill: white, weight: "bold")[비고],
  ),
  [VTP 렌더링 동작 여부], [PASS ✅], [모든 5개 프레임에서 뼈 형상 렌더링 확인],
  [색상 코딩 정확성], [PASS ✅], [하지=파랑, 상지=주황, 흉곽=흰, 골반=분홍],
  [동작 추종 (FK 반응성)], [PASS ✅], [Frame 0600 다리 벌림 포즈 정확 반영],
  [뼈 형태 식별 가능성], [PASS ✅], [femur, tibia, torso, pelvis 형상 인식 가능],
  [카메라 뷰 최적화], [개선 필요 ⚠️], [전신이 우상단에 치우침, 하체가 화면 밖으로 벗어남],
  [흉곽-골반 연속성], [개선 필요 ⚠️], [두 세그먼트 간 시각적 갭 존재 (카메라 투영 각도 문제)],
  [전신 스케일 배치], [개선 필요 ⚠️], [활성 픽셀 3.56% → 화면 대비 골격 면적이 작음],
)

= 식별된 문제점 및 개선 방향

*문제 1: 카메라 뷰 치우침*
VTP 골격이 화면 우상단에 집중되어 있음. `prerender_vtp_frames()` 내 PyVista 카메라 위치를 골격 중심점(pelvis FK 위치) 기준으로 동적으로 조정 필요.

*문제 2: 흉곽-골반 갭*
`torso`와 `pelvis` 세그먼트 사이 시각적 단절이 보임. OpenSim FK에서 `torso`의 부모가 `pelvis`로 연결되어 있지만, 카메라 투영 각도에 따라 깊이 방향으로 분리되어 보이는 현상. 정면 뷰 카메라로 전환 시 개선 기대.

*문제 3: 전신 스케일*
현재 고정 카메라 거리로 인해 전신이 작게 보임. 동적 줌(zoom-to-fit) 또는 가까운 카메라 거리 적용 필요.

= 결론

OpenSim VTP 렌더링 파이프라인은 실제 영상(IMG_2633.MOV)에서 *정상 동작* 확인됨. 핵심 기능인 FK 기반 뼈 형태 렌더링 및 동작 추종은 성공적으로 구현됨. 카메라 뷰 최적화는 이슈 \#6 후속 작업으로 추진한다.

#v(1cm)
#text(size: 9pt, fill: gray)[
  검증 도구: tools/verify_vtp_visual.py |
  분석 영상: my_data/2026_02_25/IMG_2633_anatomy.mp4 |
  생성일: 2026-02-28
]
