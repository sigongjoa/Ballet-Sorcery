#set document(title: "芭蕾呪法 - TRC v2 생성기 구현 리포트")
#set page(paper: "a4", margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 20pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 14pt)[TRC v2 고밀도 마커 생성기 구현 리포트]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-28 | /sc:duo 자동 생성 | 이슈 \#7]
]

#line(length: 100%)
#v(0.5em)

= 작업 요약

이슈 \#7의 핵심 마일스톤인 **TRC v2 생성기**를 구현하였다. 기존 `src/trc_exporter.py`의 25개 MediaPipe 마커 방식에서 SMPL-X 피팅 엔진이 생성한 **56개 고밀도 가상 마커**를 OpenSim TRC 포맷으로 출력하는 기능을 추가하였다.

#table(
  columns: (1fr, 2fr),
  inset: 8pt,
  align: (left, left),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white, weight: "bold")[구분],
    text(fill: white, weight: "bold")[내용],
  ),
  [기존 방식], [`trc_exporter.py` — MediaPipe 25개 마커, Y-DOWN→UP 반전 필요],
  [신규 방식], [`trc_v2_exporter.py` — SMPL-X 56개 마커, Y-UP 유지 (반전 불필요)],
  [생성 파일], [`src/trc_v2_exporter.py`, `tests/test_trc_v2_exporter.py`],
  [구현 도구], [Gemini CLI (2개 서브태스크), Claude 검증],
)

= 구현 상세

== 핵심 함수

```python
def export_trc_v2(
    markers_sequence: List[Dict[str, List[float]]],
    output_trc_path: str,
    fps: float = 30.0
) -> Tuple[str, float]:
```

- `markers_sequence`: 프레임별 56개 마커 딕셔너리 리스트 (`extract_virtual_markers()` 반환값)
- 반환값: `(생성된 파일 절대경로, 적용된 y_offset)`

== 좌표계 처리

SMPL-X 피팅 엔진은 MediaPipe Y-DOWN 값을 최적화 과정에서 이미 Y-UP으로 변환하므로, TRC v2 생성기에서는 **Y축 반전을 수행하지 않는다** (기존 trc_exporter.py와의 핵심 차이점).

== feet_on_floor 오프셋

발 마커 6개 (`HEEL_POST_L`, `HEEL_POST_R`, `ANKLE_LAT_L`, `ANKLE_LAT_R`, `TOE_TIP_L`, `TOE_TIP_R`)의 최솟값을 기준으로 y_offset을 계산하여, 발이 항상 y=0.01m에 위치하도록 정렬한다.

```
y_offset = -(min_foot_y - 0.01)
```

= 테스트 결과

== pytest 실행 결과 (4/4 통과)

#table(
  columns: (auto, 1fr, auto, auto),
  inset: 8pt,
  align: (center, left, center, center),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if row == 5 { rgb("#f0fff0") } else { white },
  table.header(
    text(fill: white, weight: "bold")[\#],
    text(fill: white, weight: "bold")[테스트 항목],
    text(fill: white, weight: "bold")[결과],
    text(fill: white, weight: "bold")[확인 내용],
  ),
  [1], [`test_header_format`], [PASS ✅], [PathFileType 행, NumMarkers=56 포함],
  [2], [`test_marker_count_in_header`], [PASS ✅], [56개 마커명 모두 헤더에 존재],
  [3], [`test_feet_on_floor_offset`], [PASS ✅], [y_offset ≈ 0.06m (±0.001 이내)],
  [4], [`test_data_rows_count`], [PASS ✅], [헤더 6행 + 데이터 10행 = 총 16행],
  [*합계*], [*4개 테스트*], [*4 PASSED*], [*0.27s*],
)

#raw(block: true, lang: "text", read("assets/pytest_output.txt"))

= 마커 구성 및 TRC v2 파일 구조

#figure(
  image("assets/trc_v2_structure.png", width: 100%),
  caption: [고밀도 가상 마커 56개 카테고리 분포 (좌) 및 TRC v2 파일 구조 다이어그램 (우)]
)

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  inset: 6pt,
  align: center,
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Head & Neck],
    text(fill: white, weight: "bold")[Torso & Spine],
    text(fill: white, weight: "bold")[Pelvis],
    text(fill: white, weight: "bold")[Upper Limbs],
    text(fill: white, weight: "bold")[Lower Limbs],
  ),
  [4개], [8개], [8개], [16개], [20개],
)

= 결론 및 다음 단계

*구현 완료*: `src/trc_v2_exporter.py` 및 테스트 4개 작성·통과. 이슈 \#7의 TRC v2 생성 마일스톤 달성.

*다음 마일스톤*:
1. *OpenSim IK 주입 비교*: 기존 25개 마커 TRC vs. 56개 마커 TRC v2의 IK 잔류 오차 비교 리포트
2. *3D 뷰어 메시 오버레이*: anatomy_overlay_viewer.py에 SMPL-X 피부 메시 레이어 추가
3. *이슈 \#7 최종 완료 기준* 충족을 위한 PoC 수치 리포트 생성
