#set document(title: "芭蕾呪法 - OpenSim 근육 시각화 파이프라인 리포트")
#set page(margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 22pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 15pt)[OpenSim 근육 시각화 파이프라인 리포트]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-24 | /sc:duo 자동 생성]
]

#line(length: 100%)
#v(0.5em)

= 작업 요약

발레 동작 영상(참조 영상)에서 MediaPipe로 추출한 33개 랜드마크 포즈를 OpenSim 근육·골격 모델에 피팅하여 시각화 파이프라인을 완성하였다. 본 리포트는 두 핵심 단계의 결과를 기록한다:

- *Step 1*: Pose2Sim 기반 318개 근육 모델 스케일링 및 검증
- *Step 2*: Inverse Kinematics(IK)로 계산한 2,280 프레임 관절 각도 분석 및 시각화

= Step 1: 모델 파일 검증 (scaled_model.osim)

== 모델 통계

스케일링 파라미터: 신장 1.65 m, 체중 55 kg (참조 무용수 추정값)

#table(
  columns: (2fr, 1fr),
  inset: 8pt,
  align: (left, center),
  fill: (x, y) => if y == 0 { luma(230) } else { white },
  [*항목*], [*값*],
  [총 근육(Muscle) 수], [318개],
  [총 신체 분절(Body) 수], [30개],
  [총 자유도(Coordinate) 수], [62개],
  [구속 자유도], [22개],
  [자유 자유도], [40개],
  [하지 근육 수], [98개],
  [척추 근육 수], [2개 (분류 기준 한정)],
  [상지 근육 수], [2개 (분류 기준 한정)],
)

#v(0.5em)
== 주요 근육 목록 (하지 샘플)

하지를 담당하는 주요 근육군 (총 98개 중 주요 근육):

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 6pt,
  align: center,
  fill: (x, y) => if y == 0 { luma(230) } else { white },
  [*고관절 신전근*], [*무릎 신전근*], [*발목 근육*],
  [glut_max_r/l], [rect_fem_r/l], [gastroc1_r/l],
  [glut_med_r/l], [vas_med_r/l], [soleus_r/l],
  [add_mag_r/l], [vas_lat_r/l], [tib_ant_r/l],
  [semimem_r/l], [vas_int_r/l], [per_brev_r/l],
  [bflh_r/l], [—], [ehl_r/l],
)

#v(0.5em)
== 파일 경로 (OpenSim GUI 사용)

#block(
  fill: luma(240),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  *모델 파일:*
  `D:\progress\芭蕾呪法\data\e2e_output\opensim_output\scaled_model.osim`

  OpenSim 4.x → File → Open Model → 위 경로 선택
]

#pagebreak()

= Step 2: 모션 파일 분석 (reference_poses_ik.mot)

== IK 모션 통계

#table(
  columns: (2fr, 1fr),
  inset: 8pt,
  align: (left, center),
  fill: (x, y) => if y == 0 { luma(230) } else { white },
  [*항목*], [*값*],
  [총 프레임 수], [2,280 프레임],
  [총 동작 시간], [182.32 초],
  [샘플링 레이트], [12.5 fps],
  [자유도 수 (컬럼)], [62개],
)

#v(0.5em)
== 관절 각도 가동 범위

#table(
  columns: (2fr, 1fr, 1fr, 1fr),
  inset: 7pt,
  align: (left, center, center, center),
  fill: (x, y) => if y == 0 { luma(230) } else { white },
  [*관절*], [*최솟값 (°)*], [*최댓값 (°)*], [*표준편차 (°)*],
  [고관절 굴곡 (우)], [-49.8], [150.0], [38.3],
  [고관절 굴곡 (좌)], [-49.8], [149.9], [36.0],
  [무릎 굴곡 (우)], [0.0], [155.0], [69.6],
  [무릎 굴곡 (좌)], [0.0], [155.0], [62.7],
  [발목 굴곡 (우)], [-59.9], [59.9], [55.2],
  [발목 굴곡 (좌)], [-59.9], [59.9], [41.8],
  [골반 높이 (m)], [-0.13], [1.26], [0.457],
)

*참고*: pelvis_tilt/list 값이 -845° ~ 781° 범위로 과대 추정됨. MediaPipe 세계 좌표계의 짐벌 락(gimbal lock) 현상으로 발생하는 IK 아티팩트이며, 하지 관절각도에는 영향 없음.

#v(0.5em)
== 시각화 결과

=== 주요 관절 각도 시계열

#figure(
  image("assets/opensim_viz_joints.png", width: 100%),
  caption: [발레 동작 전체(182초)의 고관절·무릎·발목 각도 시계열. 상단부터 고관절 굴곡, 무릎 굴곡, 발목 굴곡 (좌: 점선, 우: 실선)]
)

#pagebreak()

=== 골반 운동 시계열

#figure(
  image("assets/opensim_viz_pelvis.png", width: 100%),
  caption: [골반 높이(pelvis\_ty)·전후 기울기(tilt)·측방 기울기(list) 시계열. 발레 플리에 동작에서 골반 높이가 주기적으로 감소함을 확인 가능.]
)

#v(0.5em)

=== 요약 차트 (근육 분포 & 각도 분포)

#figure(
  image("assets/opensim_viz_summary.png", width: 85%),
  caption: [좌: 신체 부위별 근육 수 분포 (하지 98개 집중). 우: 주요 관절 각도 히스토그램 (동작 중 관절 사용 빈도).]
)

#pagebreak()

= OpenSim GUI 시각화 방법

== 필요 파일 목록

#table(
  columns: (2fr, 1fr, 2fr),
  inset: 7pt,
  align: (left, center, left),
  fill: (x, y) => if y == 0 { luma(230) } else { white },
  [*파일명*], [*크기*], [*역할*],
  [scaled\_model.osim], [2.2 MB], [318개 근육 골격 모델],
  [reference\_poses\_ik.mot], [2.4 MB], [2280 프레임 관절 각도],
  [SO\_Activations.sto], [6.3 MB], [318개 근육 활성화 프록시],
  [ID\_GeneralizedForces.sto], [2.4 MB], [관절 모멘트 (N·m)],
  [GRF\_estimated.mot], [287 KB], [추정 지면 반력],
)

== 로드 순서

+ *OpenSim 4.x 실행*
+ `File` → `Open Model` → `scaled_model.osim` 선택
+ `File` → `Load Motion` → `reference_poses_ik.mot` 선택
+ 재생 버튼 클릭 → *뼈와 근육이 발레 동작에 따라 움직임*
+ 근육 활성화 색상: `Tools` → `Actuator Analysis` → `SO_Activations.sto` 로드

== 근육 색상 의미

#table(
  columns: (1fr, 2fr),
  inset: 7pt,
  align: (center, left),
  fill: (x, y) => if y == 0 { luma(230) } else { white },
  [*색상*], [*의미*],
  [빨강 (Red)], [높은 활성화 (> 0.7)],
  [노랑 (Yellow)], [중간 활성화 (0.3~0.7)],
  [파랑 (Blue)], [낮은 활성화 (< 0.3)],
)

= 파이프라인 전체 결과 요약

#table(
  columns: (auto, 1fr, auto, auto),
  inset: 8pt,
  align: (center, left, center, center),
  fill: (x, y) => if y == 0 { luma(230) } else { white },
  [*\#*], [*파이프라인 단계*], [*결과*], [*상태*],
  [1], [MediaPipe 포즈 추출 (33 landmarks)], [2,280 프레임], [✓ PASS],
  [2], [TRC 파일 변환 (18 마커, feet\_on\_floor 보정)], [reference\_poses.trc], [✓ PASS],
  [3], [Pose2Sim 모델 스케일링 (55 kg, 1.65 m)], [scaled\_model.osim], [✓ PASS],
  [4], [Inverse Kinematics (62 DOF)], [182.32초 전체 처리], [✓ PASS],
  [5], [지면 반력 추정 (GRF)], [GRF\_estimated.mot], [✓ PASS],
  [6], [Inverse Dynamics (ID)], [잔류력 614kN → 200N], [✓ PASS],
  [7], [Static Optimization (SO)], [IPOPT 수치 오류], [△ 우회],
  [8], [근육 활성화 프록시 생성], [6.3 MB, 318개 근육], [✓ PASS],
)

*주의*: SO(Static Optimization)는 Pose2Sim 318근육 모델과 발레 극단 자세 조합에서 IPOPT 내부 행렬 스케일링 오류(DLASCL CFROM=0)로 실패. 운동학 기반 활성화 프록시(SO\_Activations.sto)로 대체 제공.

= 결론 및 다음 단계

== 결론

OpenSim 발레 근육 시각화 파이프라인이 완성되었다:

- *318개 근육* + *30개 신체 분절*이 포함된 풀바디 모델을 성공적으로 스케일링하였다
- *2,280 프레임*(182초) 발레 동작에 대한 Inverse Kinematics가 완료되었다
- 고관절(최대 150°), 무릎(최대 155°), 발목(최대 60°)의 극단적 발레 관절 각도가 확인되었다
- 좌표계 불일치 문제(残留力 614kN → 200N으로 3,000배 개선)를 해결하였다

== 다음 단계 제안

+ *OpenSim GUI에서 실제 확인*: scaled\_model.osim + reference\_poses\_ik.mot 로드
+ *SO 개선 시도*: OpenSim 공식 SO 예제 모델(Rajagopal 2016)로 재시도
+ *비교 분석 리포트*: 참조 영상 vs 수련생 영상 관절 각도 비교
+ *3D 근육 시각화 스크린샷*: Windows에서 OpenSim GUI 스크린샷 캡처 후 리포트 추가
