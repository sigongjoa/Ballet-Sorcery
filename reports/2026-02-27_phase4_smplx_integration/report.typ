#set page(paper: "a4", margin: 2cm)
#set text(font: "Noto Sans CJK KR", size: 10pt)

= Phase 4: SMPL-X Mesh 기반 고정밀 바이오메카닉 파이프라인 통합 리포트

== 1. 개요
기존 MediaPipe 기반의 희소(Sparse)한 관절 추적 방식을 SMPL-X 파라메트릭 모델 기반의 조밀(Dense)한 메쉬 복원 방식으로 전환함. 

== 2. 데이터 정합성 정밀 검증 (Real-world Data Test)
실제 발레 분석 데이터(`extracted_poses.json`)를 활용하여 MediaPipe 월드 좌표계와 SMPL-X 모델 간의 정합성을 수치로 증명함.

#table(
  columns: (1fr, 1.5fr, 1fr),
  inset: 8pt,
  align: (center, left, center),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white, weight: "bold")[항목],
    text(fill: white, weight: "bold")[측정 수치 / 상태],
    text(fill: white, weight: "bold")[판정],
  ),
  [매핑 관절 수], [16개 (핵심 관절 100% 매핑)], [PASS ✅],
  [어깨/골반 비율], [1.58 (표준 범위: 1.2 ~ 1.6)], [PASS ✅],
  [수직 방향성], [Knee rel Y: -0.070m (Y-up 반전 성공)], [PASS ✅],
)

== 3. 기술적 구현 현황
1. *MediaPipe to SMPL-X:* Y축 반전 및 미터 스케일 일치 로직 검증 완료.
2. *Virtual Marker Extraction:* `smplx_mapper.py`를 통해 10,475개 정점 중 가상 마커 추출 엔진 구축.
3. *Optimization:* GMM Prior 기반의 2단계 최적화 루프 (Pose & Shape) 준비 완료.

== 4. 향후 계획
- SMPL-X 가상 마커 50개 세트 확정 및 OpenSim .osim 모델 마커 매핑.
- 메쉬 피팅 결과의 3D 시각화 (뼈-피부 오버레이).

#v(2cm)
*최종 업데이트:* 2026년 2월 27일 14:15
*검증 엔진:* tools/verify_mp_smplx_alignment.py
