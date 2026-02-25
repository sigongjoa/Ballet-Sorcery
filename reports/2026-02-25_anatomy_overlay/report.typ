#set document(title: "芭蕾呪法 — 3패널 해부학 오버레이 뷰어")
#set page(margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 20pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 14pt)[3패널 해부학 오버레이 뷰어 — OpenSim VTP 통합]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-25 | /sc:duo 자동 생성 | 이슈 \#6]
]

#line(length: 100%)
#v(0.5em)

= 작업 요약

`src/anatomy_overlay_viewer.py`를 구현하여 3패널 그리드 학습 영상을 생성하였다.
패널 1(원본), 패널 2(스켈레톤 오버레이 + 관절 각도), 패널 3(OpenSim VTP 뼈 형상)을 가로로 이어붙인 `5760×1080` 출력을 IMG_2633.MOV에서 성공적으로 생성하였다.

= 출력 레이아웃

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  align: (center, left, center),
  [*패널*], [*내용*], [*기술*],
  [패널 1], [원본 영상 (무수정)], [OpenCV 그대로],
  [패널 2], [원본 위 MediaPipe 스켈레톤 + 무릎·고관절·발목 각도 수치], [MediaPipe + OpenCV putText],
  [패널 3], [OpenSim pyopensim FK → PyVista VTP 뼈 형상 렌더링], [pyopensim + PyVista 오프스크린],
)

= 처리 파이프라인 (3패스)

```
[1패스] HEVC→H264 전처리 + MediaPipe 포즈 추출 → 프레임·랜드마크 메모리 저장
            ↓
[2패스] MediaPipe 관절각도 → OpenSim 좌표 매핑 → pyopensim FK
        → 단일 PyVista Plotter.clear() 재사용 → VTP 뼈 배치 렌더링
            ↓
[3패스] np.hstack([원본, 오버레이, VTP]) → MP4 출력
```

핵심 설계: PyVista Plotter를 매 프레임 생성·삭제하지 않고, *단일 인스턴스*에 `pl.clear()`를 반복 호출하여 SIGABRT 충돌 방지.

= MediaPipe → OpenSim 좌표 매핑

#table(
  columns: (1fr, 1fr),
  inset: 8pt,
  [*OpenSim 좌표*], [*MediaPipe 계산*],
  [`knee_angle_r/l`], [`radians(calc_angle(hip,knee,ankle) - 180)`],
  [`hip_flexion_r/l`], [`radians(180 - calc_angle(shoulder,hip,knee))`],
  [`ankle_angle_r/l`], [`radians(90 - calc_angle(knee,ankle,foot))`],
  [`arm_flex_r/l`], [`radians(180 - calc_angle(hip,shoulder,elbow))`],
  [`elbow_flex_r/l`], [`radians(180 - calc_angle(shoulder,elbow,wrist))`],
  [`pelvis_ty`], [1.0m 고정 (시각화용)],
)

= 테스트 결과

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  align: (center, left, center),
  [*\#*], [*테스트 항목*], [*결과*],
  [1], [HEVC MOV 전처리 및 H264 변환], [✅ PASS],
  [2], [MediaPipe 1,019프레임 포즈 추출 (~38fps)], [✅ PASS],
  [3], [pyopensim FK 좌표 계산 (매 프레임)], [✅ PASS],
  [4], [PyVista VTP 뼈 배치 렌더링 (단일 Plotter)], [✅ PASS],
  [5], [원본+오버레이+VTP 3패널 결합 (5760×1080)], [✅ PASS],
  [6], [출력 MP4 생성], [✅ PASS],
)

= 출력 정보

#table(
  columns: (auto, 1fr),
  inset: 8pt,
  [출력 해상도], [5760×1080 (1920×3 패널)],
  [FPS], [60fps],
  [총 프레임], [1,019],
  [파일 크기], [114 MB],
  [출력 경로], [`my_data/2026_02_25/IMG_2633_anatomy.mp4`],
  [VTP 렌더 속도], [~1.3초/프레임 → 약 22분 (배치)],
)

= 스크린샷

== 프레임 60 — 점프 자세

#figure(
  image("assets/frame_01.jpg", width: 100%),
  caption: [패널1: 원본 | 패널2: 스켈레톤 오버레이 | 패널3: OpenSim VTP 뼈 (하지 파랑·상지 주황)]
)

== 프레임 509 — 착지 자세

#figure(
  image("assets/frame_02.jpg", width: 100%),
  caption: [하지 뼈(대퇴골·경골·족골)가 자세 변화에 따라 각도 변경 확인]
)

= 관찰 및 개선 과제

== 성공 사항
- VTP 뼈 형상(대퇴골, 경골, 비골, 족골, 골반, 흉곽, 상완골 등) 렌더링 정상
- 자세 변화에 따른 하지 각도 변화 반영 확인
- 패널2 관절 각도 수치(무릎·고관절·발목) 실시간 표시

== 알려진 이슈
- *VTP 뼈 분리*: 골반 `pelvis_ty=1.0` 고정값으로 인해 상체-하체 간격이 분리되어 보임
  → 개선: MediaPipe 랜드마크 기반 pelvis_ty 동적 계산 필요
- *VTP 렌더 속도*: ~1.3초/프레임 → 실시간 불가, 배치 전처리 방식 유지

= 다음 단계
- 이슈 \#3: 패널3에 근육 경로 + 활성도 색상 추가
- `pelvis_ty` 동적 계산으로 상체-하체 연결 보정
- 발레 동작 영상 추가 테스트
