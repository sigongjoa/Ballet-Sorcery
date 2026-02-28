#set page(paper: "a4", margin: 2cm)
#set text(font: "Noto Sans CJK KR", size: 10pt)

= Phase 4 Milestone: SMPLify-3D Fitting Engine & Marker Extraction

== 1. 검증 목적
MediaPipe 3D 관측값을 가이드로 하여 SMPL(6,890 정점) 모델의 자세를 최적화하고, 복원된 인체 표면에서 OpenSim 분석용 고밀도 가상 마커를 추출하는 파이프라인의 수치적 안정성을 증명함.

== 2. 파이프라인 정합성 검증 결과 (Verification Results)
#table(
  columns: (1fr, 1.5fr, 1fr),
  inset: 8pt,
  align: (center, left, center),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white, weight: "bold")[검증 항목],
    text(fill: white, weight: "bold")[측정 수치 / 로그],
    text(fill: white, weight: "bold")[판정],
  ),
  [좌표축 정합성], [Shoulder Y > Hip Y (Y-up 반전 성공)], [PASS ✅],
  [최적화 수렴], [Total Loss: 3551.2 → 1492.8 (수렴 확인)], [PASS ✅],
  [포즈 복원력], [Pose Parameter Norm: 2.1085], [PASS ✅],
  [마커 추출], [56개 가상 마커 좌표 산출 성공], [PASS ✅],
)

== 3. 시각적 정합성 증명 (Visual Proof)
#figure(
  image("assets/smplify_fitting_visual.png", width: 80%),
  caption: [SMPL Mesh + 56 Virtual Markers (Yellow). 신체 표면 전체에 걸친 조밀한 마커 분포를 확인할 수 있음.]
)

== 4. 결론
실제 데이터를 활용한 SMPLify-3D 엔진이 성공적으로 구동됨을 수치와 시각 자료로 증명함. 56개 가상 마커는 인체의 3차원 회전을 잡아내기에 충분한 밀도를 가지고 있으며, 이를 통해 OpenSim IK 분석의 정밀도를 획기적으로 높일 준비가 완료됨.

#v(2cm)
*검증 엔진:* tests/test_smplify_pipeline.py
*작성일:* 2026-02-27
