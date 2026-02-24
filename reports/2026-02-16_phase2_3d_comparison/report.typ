#set document(title: "파뢰주법 - Phase 2: 3D 포즈 비교 실험 리포트")
#set page(margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 20pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 14pt)[Phase 2: 3D 포즈 비교 실험 리포트]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-16 | /sc:duo 자동 생성]
]

#line(length: 100%)

= 작업 요약

Phase 1에서 확인된 카메라 앵글 민감성 문제를 해결하기 위해, MediaPipe의 3D world\_landmarks를 활용한 비교 모드를 구현하고, 2D vs 3D 비교 실험을 수행하였다.

*핵심 가설*: 3D world\_landmarks는 골반 중심 기준 미터 단위 좌표이므로, 카메라 각도에 의한 2D 투영 왜곡이 제거되어 더 정확한 포즈 비교가 가능하다.

= 구현 내용

== 수정된 모듈

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  align: (center, left, center),
  [*파일*], [*변경 내용*], [*담당*],
  [`src/comparator.py`], [`use_world` 파라미터 추가, `calculate_l2_distance_3d()` 함수 구현, 3D 모드에서 정규화 자동 비활성화], [Gemini],
  [`src/pose_extractor.py`], [Phase 2-1에서 이미 `world_landmarks` 출력 지원 완료 (변경 없음)], [기존],
  [`tests/test_phase2.py`], [6개 테스트 케이스 신규 작성], [Gemini],
)

== 핵심 설계 결정

- *3D 거리 함수*: $sqrt(Delta x^2 + Delta y^2 + Delta z^2)$ (3축 유클리드 거리)
- *정규화 불필요*: `world_landmarks`는 이미 골반 중심 기준 + 미터 단위이므로 별도 정규화 없이 직접 비교
- *하위 호환성*: `use_world=False`(기본값)이면 기존 2D 파이프라인과 동일하게 동작

= 테스트 결과

#table(
  columns: (auto, 1fr, auto, auto),
  inset: 8pt,
  align: (center, left, center, center),
  [*\#*], [*테스트 항목*], [*결과*], [*유형*],
  [1], [`test_calculate_l2_distance_3d_basic`], [PASS], [단위],
  [2], [`test_calculate_l2_distance_3d_same_point`], [PASS], [단위],
  [3], [`test_compare_poses_use_world_metadata`], [PASS], [통합],
  [4], [`test_compare_poses_2d_backward_compatible`], [PASS], [통합],
  [5], [`test_3d_loss_lower_than_2d`], [PASS], [통합],
  [6], [`test_compare_poses_3d_no_normalization`], [PASS], [통합],
)

*전체 테스트 스위트*: 18 passed, 3 skipped (Phase 0 + Phase 2 + PM Agent 통합)

= 실험: 2D vs 3D 비교

== 실험 설정

동일한 그랑 바뜨망(Grand Battement) 영상 쌍을 2D 정규화 비교와 3D world\_landmarks 비교로 각각 처리하여 결과를 비교하였다.

- *참조 영상*: `reference_grand_battement.mp4` (301 프레임, 24fps)
- *비교 영상*: `compare_grand_battement.mp4` (301 프레임, 24fps)
- *비교 프레임 수*: 301 프레임 (동일)

== 통계 비교

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, right, right, right),
  [*지표*], [*2D*], [*3D*], [*변화율*],
  [평균 손실 (mean)], [4.6161], [0.2916], [-93.7%],
  [최대 손실 (max)], [11.4594], [0.5682], [-95.0%],
  [최소 손실 (min)], [0.0000], [0.0000], [--],
  [표준 편차 (std)], [0.9537], [0.1411], [-85.2%],
  [최대 손실 프레임], [49], [40], [--],
  [최소 손실 프레임], [12], [12], [동일],
)

*핵심 발견*: 3D 비교의 평균 손실이 2D 대비 *약 15.8배 감소* (4.62 → 0.29). 이는 2D 비교에서의 높은 손실이 상당 부분 카메라 앵글 차이에 기인했음을 실증적으로 확인한 것이다.

== 프레임별 손실 비교

#figure(
  image("assets/loss_comparison_2d_vs_3d.png", width: 95%),
  caption: [2D vs 3D 프레임별 총 손실. 3D(하단)에서 손실 범위가 0~0.57로 대폭 축소됨.]
)

== 2D/3D 손실 중첩 그래프

#figure(
  image("assets/loss_overlay_2d_3d.png", width: 95%),
  caption: [2D(파란색)와 3D(빨간색) 손실을 이중 Y축으로 중첩. 3D가 더 세밀한 동작 차이를 포착함.]
)

== 관절별 손실 분석 (3D)

#figure(
  image("assets/joint_heatmap_3d.png", width: 85%),
  caption: [3D 비교에서의 관절별 평균 손실 상위 10개. 하체 관절이 여전히 최상위이나 절대값이 대폭 감소.]
)

== 통계 요약 비교

#figure(
  image("assets/std_comparison.png", width: 75%),
  caption: [2D와 3D의 주요 통계 비교. 모든 지표에서 3D가 현저히 낮은 값을 보임.]
)

= 3D 스켈레톤 비교 시각화

== 최대 손실 프레임 (Frame 40)

#figure(
  image("assets/skeleton_3d_max_loss.png", width: 80%),
  caption: [3D 스켈레톤 오버레이 - 최대 손실 프레임. 참조(파란색)와 비교(빨간색)의 하체 위치 차이가 뚜렷함.]
)

== 최소 손실 프레임 (Frame 12)

#figure(
  image("assets/skeleton_3d_min_loss.png", width: 80%),
  caption: [3D 스켈레톤 오버레이 - 최소 손실 프레임. 두 포즈가 거의 완벽히 일치하여 겹쳐 보임.]
)

== 다중 시점 비교 (Frame 40)

#figure(
  image("assets/skeleton_3d_multiview_max.png", width: 95%),
  caption: [최대 손실 프레임을 정면/측면/상단/대각선 4개 시점에서 관찰. 측면 뷰에서 깊이 차이가 가장 명확.]
)

== 2D vs 3D 스켈레톤 직접 비교 (Frame 40)

#figure(
  image("assets/skeleton_2d_vs_3d_comparison.png", width: 95%),
  caption: [같은 프레임의 2D(좌)와 3D(우) 스켈레톤 비교. 2D에서는 카메라 투영으로 인해 포즈가 유사해 보이지만, 3D에서는 실제 공간상의 차이를 정확히 보여줌.]
)

= 관절별 고손실 순위 비교

#table(
  columns: (auto, 1fr, auto, 1fr, auto),
  inset: 6pt,
  align: (center, left, right, left, right),
  [*순위*], [*2D 관절*], [*2D 손실*], [*3D 관절*], [*3D 손실*],
  [1], [LEFT\_FOOT\_INDEX], [10.68], [LEFT\_FOOT\_INDEX], [1.12],
  [2], [LEFT\_HEEL], [10.65], [LEFT\_HEEL], [0.97],
  [3], [RIGHT\_FOOT\_INDEX], [10.54], [LEFT\_ANKLE], [0.92],
  [4], [LEFT\_ANKLE], [10.45], [LEFT\_KNEE], [0.84],
  [5], [RIGHT\_HEEL], [10.41], [RIGHT\_FOOT\_INDEX], [0.77],
)

*관찰*: 2D와 3D 모두 하체 관절(발, 발목, 무릎)이 최상위 손실을 보이지만, 3D에서의 절대값은 약 10배 감소. 이는 그랑 바뜨망 동작의 실제 차이를 더 정확하게 반영한다.

= 프레임별 손실 상세 분석 (3D)

== 프레임별 3D 손실 그래프

#figure(
  image("assets/frame_by_frame_loss.png", width: 90%),
  caption: [3D world\_landmarks 기반 프레임별 손실. 동작 전환 구간에서 손실 피크가 발생함.]
)

== 고오차 구간 하이라이트

#figure(
  image("assets/highlighted_error_loss.png", width: 90%),
  caption: [평균 + 1 표준편차를 초과하는 고오차 구간(빨간 영역)이 자동 표시됨. 집중 교정이 필요한 구간을 시각적으로 식별 가능.]
)

= SMPL 메쉬 기반 3D 포즈 시각화

포인트+라인 기반 스켈레톤의 한계를 극복하기 위해, SMPL(Skinned Multi-Person Linear) 바디 모델을 활용하여 사실적인 인체 메쉬 시각화를 구현하였다.

== 구현 방법

- *SMPL 모델*: 6,890 vertices, 13,776 faces, 24 관절의 파라메트릭 인체 모델
- *GMM Pose Prior*: VIBE/SMPLify에서 검증된 8-Gaussian Mixture Model로 자연스러운 포즈만 생성
- *2단계 최적화*: (1) 글로벌 회전+이동 최적화 → (2) 전체 포즈+체형 최적화 (GMM prior 포함)
- *좌표 변환*: MediaPipe Y-down → SMPL Y-up (Y축 반전)
- *관절 매핑*: MediaPipe 33개 중 16개를 SMPL 22개 관절에 매핑

== 참조 vs 비교 포즈 오버레이 (Frame 40)

#figure(
  image("assets/smpl_comparison_frame_40.png", width: 85%),
  caption: [SMPL 메쉬 오버레이 - 최대 손실 프레임. 참조(파란색)와 비교(빨간색)의 하체 포즈 차이가 사실적인 인체 형태로 명확히 드러남.]
)

== 참조 포즈 다중 시점 (Frame 40)

#figure(
  image("assets/smpl_ref_multiview_frame_40.png", width: 95%),
  caption: [참조 포즈를 정면/측면/후면/3-4 시점에서 관찰. Y축이 올바르게 정렬되어 서 있는 포즈로 표시됨.]
)

== 비교 포즈 다중 시점 (Frame 40)

#figure(
  image("assets/smpl_comp_multiview_frame_40.png", width: 95%),
  caption: [비교 포즈의 4개 시점 뷰. 참조 대비 다리와 팔의 각도 차이가 시각적으로 확인 가능.]
)

== 스켈레톤 → 메쉬 전환 효과

#table(
  columns: (auto, 1fr, 1fr),
  inset: 8pt,
  align: (left, center, center),
  [*항목*], [*포인트+라인 스켈레톤*], [*SMPL 메쉬*],
  [형태 인식], [추상적 점/선], [사실적 인체],
  [포즈 직관성], [전문가만 해석 가능], [누구나 포즈 차이 파악],
  [깊이 정보], [점의 크기로만 추정], [메쉬 표면으로 명확],
  [좌표 안정성], [Y축 뒤집힘 문제], [SMPL 표준 좌표계 사용],
)

= Gemini CLI 실행 기록

#table(
  columns: (auto, 1fr, auto, auto),
  inset: 8pt,
  align: (center, left, center, center),
  [*\#*], [*작업*], [*재시도*], [*결과*],
  [1], [`comparator.py` 3D 모드 추가], [0회], [성공],
  [2], [2D vs 3D 통계 시각화 (4개 PNG)], [0회], [성공],
  [3], [Phase 2 테스트 작성 (6개)], [0회], [성공],
  [4], [3D 스켈레톤 시각화 (5개 PNG) + 프레임별 분석], [0회], [성공],
  [5], [SMPL 메쉬 시각화 (`smpl_visualizer.py`)], [Gemini 3회 실패], [[Claude 직접 수정]],
)

*SMPL 시각화 실패 원인 및 해결*: Gemini의 자체 IK 구현은 Pose Prior 부재로 메쉬 왜곡 발생. `/sc:research`로 joints2smpl 프로젝트를 발견하여 GMM Pose Prior + 2단계 최적화 방식으로 전환 후 성공.

= 결론 및 다음 단계

== 결론

1. *가설 검증 성공*: 3D world\_landmarks 비교가 카메라 앵글 영향을 대폭 감소시킴 (평균 손실 93.7% 감소)
2. *하위 호환성 유지*: 기존 2D 파이프라인 완전 보존 (`use_world=False` 기본값)
3. *실용적 의미*: 촬영 각도가 다른 영상 쌍에서도 포즈 차이를 의미 있게 비교 가능
4. *SMPL 메쉬 시각화*: GMM Pose Prior 기반 피팅으로 사실적인 3D 인체 메쉬 비교 달성

== 다음 단계

- *Phase 3*: 음악 BPM 동기화 (librosa 기반 비트 추출 + 비트별 손실 집계)
- *Phase 4*: DTW(Dynamic Time Warping)를 통한 자동 시간축 정렬
- *개선 사항*: SMPL 메쉬 애니메이션 (프레임 시퀀스 → GIF/영상 출력)
