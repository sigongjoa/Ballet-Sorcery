#set document(title: "파뢰주법 - Phase 0 테스트 리포트")
#set page(margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 24pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 14pt)[Phase 0: 환경 세팅 + 포즈 추출 파이프라인]
  #v(0.2em)
  #text(size: 14pt)[테스트 리포트]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-14 | /sc:duo 자동 생성]
]

#v(1em)
#line(length: 100%)
#v(1em)

= 작업 요약

본 Phase에서는 파뢰주법 프로젝트의 기반이 되는 포즈 추출 파이프라인을 구현하였다.

- *구현 방식*: Claude (오케스트레이터) + Gemini CLI (구현자) 분업
- *Gemini 재시도 횟수*: 0회 (모든 서브태스크 1회 성공)

=== 구현된 모듈

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  align: (center, left, center),
  table.header([*파일*], [*역할*], [*구현자*]),
  [`src/pose_extractor.py`], [영상 → MediaPipe Pose → 프레임별 관절 좌표 JSON 저장. 24fps 리샘플링 포함.], [Gemini],
  [`src/normalizer.py`], [골반 중심 원점 이동 + 어깨 너비 스케일링 정규화.], [Gemini],
  [`tests/test_phase0.py`], [5개 테스트 케이스 (포즈 추출 에러 처리 + 정규화 검증).], [Gemini],
)

= 테스트 결과

#table(
  columns: (auto, 1fr, auto, auto),
  inset: 8pt,
  align: (center, left, center, center),
  table.header([*\#*], [*테스트 항목*], [*결과*], [*소요시간*]),
  [1], [파일 미존재 시 FileNotFoundError 발생], [PASS], [< 1ms],
  [2], [단일 프레임 정규화 - 골반 중심 원점 이동], [PASS], [< 1ms],
  [3], [어깨 미감지 시 이전 프레임 스케일 팩터 사용], [PASS], [< 1ms],
  [4], [다중 프레임 순차 정규화 검증], [PASS], [< 1ms],
  [5], [정규화 후 metadata에 normalized=true 추가 확인], [PASS], [< 1ms],
)

#align(center)[
  #text(size: 12pt, weight: "bold", fill: rgb("#2d8a4e"))[전체 결과: 5/5 PASS (0.84초)]
]

= 상세 결과

== pytest 출력

#raw(block: true, lang: "text", read("assets/pytest_output.txt"))

== 정규화 검증 수치

정규화 알고리즘의 핵심 지표:

#table(
  columns: (1fr, auto, auto),
  inset: 8pt,
  align: (left, center, center),
  table.header([*지표*], [*측정값*], [*기대값*]),
  [골반 중심 X (정규화 후)], [0.0000], [0.0],
  [골반 중심 Y (정규화 후)], [0.0000], [0.0],
  [어깨 너비 (정규화 후)], [1.0000], [1.0],
)

== 시각화: 원본 vs 정규화 좌표

#figure(
  image("assets/normalization_comparison.png", width: 90%),
  caption: [원본 좌표(좌)와 정규화 좌표(우) 비교. 정규화 후 골반 중심이 원점으로 이동하고 어깨 너비가 1.0으로 스케일링된다.]
)

= 프로젝트 구조

#raw(block: true, lang: "text",
"芭蕾呪法/
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── POC_SPEC.md
│   └── TECH_RISKS.md
├── src/
│   ├── __init__.py
│   ├── pose_extractor.py
│   └── normalizer.py
├── tests/
│   ├── __init__.py
│   └── test_phase0.py
├── data/sample/
├── reports/
│   └── 2026-02-14_phase0_pose_extraction/
│       ├── report.typ
│       ├── report.pdf
│       └── assets/
├── requirements.txt
└── google_gemini (4)/     # Gemini 대화 원본"
)

= 결론 및 다음 단계

=== 완료된 작업
- Python 개발 환경 구성 (MediaPipe, OpenCV, NumPy, matplotlib, pytest)
- 포즈 추출기 구현: 영상 → 24fps 리샘플링 → JSON
- 좌표 정규화기 구현: 골반 중심 이동 + 어깨 너비 스케일링
- 5개 테스트 케이스 전체 통과

=== 다음 단계 (Phase 1)
+ 테스트용 유튜브 발레 영상(그랑 바뜨망) 다운로드
+ 실제 영상에 포즈 추출기 적용 → JSON 데이터 확보
+ 두 영상(기준 vs 비교)의 프레임별 L2 Loss 비교 엔진 구현
+ 시각화: 두 영상 나란히 재생 + 스켈레톤 오버레이 + Loss 그래프
