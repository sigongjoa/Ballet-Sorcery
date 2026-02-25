#set document(title: "芭蕾呪法 — 실제 데이터 테스트 리포트")
#set page(margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 20pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 14pt)[실제 데이터 테스트 — 학습용 병렬 뷰어]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-25 | /sc:duo 자동 생성 | 이슈 \#2]
]

#line(length: 100%)
#v(0.5em)

= 작업 요약

iPhone 13으로 촬영한 실제 운동 영상(IMG_2633.MOV)을 입력으로 받아, 원본 영상과 MediaPipe 스켈레톤 오버레이를 좌우로 나란히 배치한 학습용 MP4 영상을 생성하는 `src/live_study_viewer.py` 파이프라인을 구현하고 검증하였다.

= 입력 데이터 명세

#table(
  columns: (auto, 1fr),
  inset: 8pt,
  align: (left, left),
  [*항목*], [*값*],
  [파일명], [`IMG_2633.MOV`],
  [촬영 기기], [Apple iPhone 13],
  [해상도], [3840×2160 (4K UHD)],
  [프레임레이트], [60 fps],
  [코덱], [HEVC (H.265), 10-bit HDR],
  [회전 메타데이터], [rotate: 180 (거꾸로 촬영)],
  [비트레이트], [53,890 kb/s],
  [길이], [16.98초],
)

= 파이프라인 설계

== 처리 흐름

```
MOV (4K HEVC) → ffmpeg 전처리 → H.264 1920×1080
                                      ↓
                          MediaPipe PoseLandmarker (LITE)
                                      ↓
                     프레임별 스켈레톤 추출 (33 랜드마크)
                                      ↓
                   np.hstack([원본 프레임, 검정배경+스켈레톤])
                                      ↓
                          출력 MP4 (3840×1080, 60fps)
```

== 핵심 기술 결정

- *ffmpeg 자동 회전*: `-noautorotate` 없이 실행하여 rotate:180 메타데이터를 ffmpeg이 자동 처리
- *MediaPipe IMAGE 모드*: 배치 처리로 프레임별 독립 추론
- *스켈레톤 색상 체계*: 뼈대 흰색, 좌측 관절 초록, 우측 관절 파랑, 얼굴/중심 연회색

= 테스트 결과

#table(
  columns: (auto, 1fr, auto, auto),
  inset: 8pt,
  align: (center, left, center, center),
  [*\#*], [*테스트 항목*], [*결과*], [*비고*],
  [1], [MOV 파일 인식 및 ffmpeg 전처리], [✅ PASS], [rotate:180 자동 보정],
  [2], [HEVC 10-bit → H.264 변환], [✅ PASS], [1920×1080 리사이즈],
  [3], [MediaPipe 포즈 추출 (1019프레임 전체)], [✅ PASS], [~26fps 처리 속도],
  [4], [스켈레톤 좌우 배치 영상 생성], [✅ PASS], [3840×1080 출력],
  [5], [원본/Skeleton 레이블 표시], [✅ PASS], [좌상단 흰색 텍스트],
  [6], [tqdm 진행률 표시], [✅ PASS], [1019it 완료],
)

= 출력 결과

== 처리 성능

#table(
  columns: (auto, 1fr),
  inset: 8pt,
  align: (left, left),
  [*항목*], [*값*],
  [입력 해상도], [3840×2160 → 1920×1080 (전처리 후)],
  [출력 해상도], [3840×1080 (좌우 병렬)],
  [처리 프레임 수], [1,019프레임],
  [평균 처리 속도], [~26 fps],
  [총 처리 시간], [약 38초],
  [출력 파일 크기], [59 MB],
  [출력 경로], [`my_data/2026_02_25/IMG_2633_study.mp4`],
)

== 스크린샷 — 초반부 (30프레임)

#figure(
  image("assets/frame_01.jpg", width: 100%),
  caption: [프레임 30: 점프 동작 — 공중에서 스켈레톤 정확히 추적]
)

== 스크린샷 — 중반부 (509프레임)

#figure(
  image("assets/frame_02.jpg", width: 100%),
  caption: [프레임 509: 착지 후 자세 — 팔다리 관절 모두 정상 감지]
)

= 관찰 사항

- *거울 반사 미감지*: 배경에 거울이 있으나 MediaPipe가 인물 1명만 정확히 추적
- *회전 보정 정상*: rotate:180 메타데이터가 ffmpeg에 의해 올바르게 처리됨
- *조명 환경 대응*: 실내 형광등/자연광 혼합 환경에서도 안정적 추적
- *배경 인물 미감지*: 뒤편의 다른 인물은 자동으로 제외 (주요 인물 포커스)

= 결론 및 다음 단계

== 결론
iPhone 13 MOV 영상(4K HEVC 10-bit, rotate:180)에 대해 파이프라인이 정상 동작함을 확인하였다. `src/live_study_viewer.py`는 실제 촬영 조건에서의 스켈레톤 학습 도구로 활용 가능하다.

== 다음 단계
- 이슈 \#3: 근육 활성도 색상을 스켈레톤 위에 오버레이
- 이슈 \#5: 촬영 각도(정면/측면)가 다른 영상에서의 동작 검증
- 발레 동작 영상 추가 확보 및 테스트
