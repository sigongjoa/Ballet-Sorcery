#set document(title: "芭蕾呪法 - SMPL 메시 기반 관절 추정 정밀도 개선 리포트")
#set page(paper: "a4", margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 20pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 14pt)[SMPL 메시 피팅 기반 관절 벡터 정밀도 개선 리포트]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-28 | 이슈 \#7 | /sc:duo]
]

#line(length: 100%)
#v(0.5em)

= 개요

이전 VTP 진단(reports/2026-02-28_vtp_mp_diagnosis)에서 MediaPipe 33점 → DOF 수동 계산 방식의 평균 각도 오차가 *33.9°* 로 측정되었다. 본 리포트는 SMPL 메시 피팅(이슈 \#7)을 도입하여 해당 오차를 목표치(15°) 이내로 감소시킨 결과를 기록한다.

#table(
  columns: (1fr, 1fr),
  inset: 8pt,
  align: (left, left),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white, weight: "bold")[구 방식 (DOF 수동)],
    text(fill: white, weight: "bold")[신 방식 (SMPL 피팅)],
  ),
  [MediaPipe 33점 → 각도 수식으로 DOF 계산],
    [MediaPipe 16관절 → SMPL 최적화 → 관절 위치],
  [Z축(깊이) 정보 투영 손실],
    [SMPL 형상 프라이어로 Z축 복원],
  [EMA 과도 평탄화로 동작 반응 둔화],
    [프레임별 독립 피팅, 순수 측정 기반],
  [평균 각도 오차: 33.9°],
    [평균 각도 오차: 12.7° (목표 달성)],
)

= 구현 내용

== 수정된 파일

#table(
  columns: (1.2fr, 1fr, 2fr),
  inset: 7pt,
  align: (left, left, left),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white)[파일],
    text(fill: white)[변경 유형],
    text(fill: white)[내용],
  ),
  [`src/smplx_engine.py`], [수정], [SMPL-X → SMPL 전환. numpy monkey-patch 추가. body_pose 크기 66→69. fit_frame() 반환에 joints 추가],
  [`src/smplx_mapper.py`], [수정], [VIRTUAL_MARKER_MAP 전체 교체. 인덱스 범위 10475→6890 이내. 해부학 위치 재매핑],
  [`tests/test_smplx_pipeline.py`], [수정], [마커명 업데이트 (SHOULDER_L→ACROMION_L, KNEE_R_LAT→KNEE_LAT_R)],
  [`tools/diag_smplx_vs_mp.py`], [신규], [SMPL 피팅 후 5개 체인 각도 오차 측정 및 이전 방식과 비교],
)

== 핵심 수정: SMPLXEngine (SMPL 모드)

```python
# numpy monkey-patch (chumpy 호환)
np.bool = np.bool_; np.int = np.int_; np.float = np.float64
np.complex = np.complex128; np.object = np.object_
np.unicode = np.str_; np.str = np.str_

# SMPL 모델 로드 (6890 정점, 23 관절)
self.model = smplx.create(model_dir, model_type='smpl',
                           gender='neutral', ext='pkl')

# MediaPipe → SMPL 관절 인덱스 매핑 (16개 핵심 관절)
self.mp_to_smplx_idx = {
    11:16, 12:17,  # Shoulders
    13:18, 14:19,  # Elbows
    15:20, 16:21,  # Wrists
    23:1,  24:2,   # Hips
    25:4,  26:5,   # Knees
    27:7,  28:8,   # Ankles
    0:15,          # Head
}
```

= 각도 오차 비교 결과

#figure(
  image("assets/smplx_vs_mp_error.png", width: 100%),
  caption: [DOF 수동 계산(파랑) vs SMPL 피팅(주황) 방향벡터 각도 오차 비교. 빨간 점선 = 15° 허용 기준.]
)

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, center),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if row == 6 { rgb("#e8f5e9") } else { white },
  table.header(
    text(fill: white)[뼈 체인],
    text(fill: white)[이전 DOF 방식],
    text(fill: white)[SMPL 피팅],
    text(fill: white)[개선율],
  ),
  [R-Thigh (우측 대퇴)], [41.0°], [13.0° ✅], [68.3%],
  [L-Thigh (좌측 대퇴)], [27.8°], [12.9° ✅], [53.4%],
  [R-Shank (우측 하퇴)], [39.3°], [13.8° ✅], [64.8%],
  [L-Shank (좌측 하퇴)], [29.7°], [14.5° ✅], [51.1%],
  [R-Arm (우측 상완)],   [31.8°], [ 9.0° ✅], [71.6%],
  [*평균*], [*33.9°*], [*12.7° ✅*], [*62.6%*],
)

#rect(fill: rgb("#e8f5e9"), radius: 4pt, inset: 10pt)[
  *목표 달성*: 5개 뼈 체인 모두 15° 이내. 평균 12.7° (62.6% 개선). SMPL 메시 기반 접근이 MediaPipe 33점 DOF 수동 계산 대비 구조적으로 우월함이 수치로 증명됨.
]

= 테스트 결과

#raw(block: true, lang: "text", read("assets/pytest_output.txt"))

= 이슈 \#7 달성 현황

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  align: (left, left, center),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white)[마일스톤],
    text(fill: white)[내용],
    text(fill: white)[상태],
  ),
  [SMPL 환경 구축], [`src/smplx_engine.py` SMPL 로드 정상], [완료 ✅],
  [56개 가상 마커 추출], [`src/smplx_mapper.py` SMPL 6890 기반 재매핑], [완료 ✅],
  [최적화 루프], [GMM 없이 Adam 50iter, 평균 수렴 확인], [완료 ✅],
  [TRC v2 생성기], [`src/trc_v2_exporter.py` 56마커 → TRC 출력], [완료 ✅],
  [방향벡터 오차 15° 이내], [평균 12.7° (이전 33.9°)], [완료 ✅],
  [OpenSim IK 주입 비교], [TRC v2 → IK 잔류 오차 비교 리포트], [미완료],
  [3D 뷰어 메시 오버레이], [anatomy_overlay_viewer에 SMPL 메시 레이어], [미완료],
)

= 다음 단계

1. *OpenSim IK 주입*: `src/trc_v2_pipeline.py` 구현 → 56마커 TRC를 OpenSim IK에 주입하고 잔류 오차(RMSE) 측정. 기존 25마커 TRC 대비 비교.
2. *3D 뷰어 통합*: `anatomy_overlay_viewer.py`의 패널 3에 SMPL 메시를 VTP 뼈와 함께 오버레이.
3. *이슈 \#7 완료 기준 충족*: PoC 수치 리포트 생성 후 GitHub 이슈 close.

#v(1cm)
#text(size: 9pt, fill: gray)[
  진단 도구: tools/diag_smplx_vs_mp.py |
  분석 영상: IMG_2633.MOV (Frame 100, 300, 600) |
  생성일: 2026-02-28
]
