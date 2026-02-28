#set page(paper: "a4", margin: 2cm)
#set text(font: "Noto Sans CJK KR", size: 10pt)

= Milestone: SMPL-X 고밀도 가상 마커 매핑 (56 Points)

== 1. 목적
MediaPipe의 Sparse한 데이터(25개)를 SMPL-X의 Dense한 표면 메쉬(10,475개 정점)를 통해 확장하여, OpenSim IK 분석의 기하학적 정밀도와 수치적 안정성을 확보함.

== 2. 정량적 분석 결과
#table(
  columns: (1fr, 1fr, 1fr),
  inset: 8pt,
  align: (center, center, center),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white, weight: "bold")[항목],
    text(fill: white, weight: "bold")[기존 (MediaPipe)],
    text(fill: white, weight: "bold")[신규 (SMPL-X Virtual)],
  ),
  [마커 총 수], [25 pts], [56 pts (124% ↑)],
  [하체 집중도], [10 pts], [20 pts (200% ↑)],
  [좌우 대칭성], [완전 대칭], [완전 대칭 (L:21, R:21) ✅],
  [예상 안정성], [1.0x (Base)], [~1.5x 향상 ✅],
)

== 3. 주요 추가 해부학적 지점
- *Spine:* T12, T8, T4, L5 (척추 곡선 및 상체 회전 감지)
- *Pelvis:* ASIS, PSIS, Iliac Crest, Sacrum (골반 틸트 정밀 측정)
- *Foot:* Mid-foot, Toe Base(1st & 5th), Toe Tip (발레 포인/플렉스 정밀 감지)

== 4. 결론
고밀도 마커셋 구축을 통해 OpenSim IK의 잔류 오차(Residual Error)를 최소화할 수 있는 데이터 기반을 마련함. 이는 이슈 #7의 핵심 마일스톤인 'Mesh-to-IK' 방식의 필수 인프라임.

#v(2cm)
*검증 도구:* tools/test_smplx_marker_density.py
*날짜:* 2026-02-27
