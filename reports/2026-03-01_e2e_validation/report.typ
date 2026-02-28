#set document(title: "芭蕾呪法 - E2E 파이프라인 검증 리포트")
#set page(margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 20pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 14pt)[E2E 파이프라인 검증 리포트]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026‑03‑01 | 이슈 \#8 Phase 2+3 | /sc:duo 자동 생성]
]

#line(length: 100%)

= 검증 개요

MediaPipe world landmarks를 ground truth로 삼아 전체 파이프라인을 단계별로 측정하였다.
이전까지의 검증(대칭성 테스트)은 "L/R이 서로 거울인가"만 측정했으나,
이번에는 _"SMPL 출력이 MediaPipe 입력을 얼마나 잘 재현하는가"_ 라는 올바른 질문으로 전환하였다.

#table(
  columns: (auto, 1fr, auto, auto),
  inset: 8pt,
  align: (center, left, center, center),
  [*단계*], [*검증 항목*], [*측정값*], [*목표*],
  [1], [SMPL 피팅 MPJPE (평균, 5개 프레임)], [9.83 cm], [< 5 cm],
  [2], [SMPL 피팅 MPJPE (일반 포즈, 4개 프레임)], [5.87 cm], [< 5 cm],
  [3], [재투영 오차 — Frame 100], [32.9 px], [< 30 px],
  [4], [재투영 오차 — Frame 300], [35.9 px], [< 30 px],
  [5], [재투영 오차 — Frame 600 (다리 벌림)], [73.7 px], [< 30 px],
)

= SMPL 피팅 오차 측정 (MPJPE)

== 방법

`reference_poses.json` (2280 프레임, MediaPipe world landmarks) 에서 5개 프레임을 선택하고
각 프레임에 대해 SMPL 피팅(50 iteration)을 수행하였다.
`mp_to_smplx_idx` 매핑으로 대응하는 관절 쌍을 연결하고 유클리드 거리를 측정하였다.

#raw(block: true, lang: "text", read("assets/smpl_mpjpe.txt"))

== 오차 분포

#figure(
  image("assets/smpl_error_bar.png", width: 85%),
  caption: [관절별 평균 SMPL 피팅 오차. 빨간 점선 = 목표 5cm.]
)

== 해석

*정상 포즈 (Frame 200, 500, 800, 1400)*: 평균 5.87 cm — 목표에 근접

*이상 포즈 (Frame 1096)*: 최대 30 cm 오차 발생
- 원인: LEFT_HIP y ≈ 0 (골반 기준, 발레 다리 올리기 동작)
- 극단적 포즈에서 50 iteration으로 수렴 불가

*관절별 경향*:
- 말단 관절(손목, 발목)이 근위 관절(힙, 무릎)보다 오차 큼
- 이는 SMPL의 skinning weight 특성상 손목/발목은 joint regressor 정밀도가 낮기 때문

= 재투영 오차 시각화 (IMG_2633.MOV)

IMG_2633.MOV의 3개 프레임에서 MediaPipe를 실시간 추출하고 SMPL 피팅 후
원본 영상 위에 오버레이를 생성하였다.

- *초록 원*: MediaPipe 2D image landmark (ground truth)
- *파란 원*: SMPL world joint → 2D 재투영 (solvePnP 기반)
- *노란 선*: 두 점 사이 오차

== Frame 100

#figure(
  image("assets/reprojection_frame100.jpg", width: 90%),
  caption: [Frame 100: 재투영 오차 32.9px. 상체 관절 정렬 양호, 발목 오차 큼.]
)

== Frame 300

#figure(
  image("assets/reprojection_frame300.jpg", width: 90%),
  caption: [Frame 300: 재투영 오차 35.9px. 일반 보행 자세에서 중간 수준 정확도.]
)

== Frame 600

#figure(
  image("assets/reprojection_frame600.jpg", width: 90%),
  caption: [Frame 600: 재투영 오차 73.7px. 다리 벌림 포즈에서 피팅 품질 급격히 저하.]
)

= 가상 마커 오차

#figure(
  image("assets/marker_error_bar.png", width: 85%),
  caption: [가상 마커별 평균 위치 오차 (cm). MediaPipe 대응 관절과의 거리.]
)

말단 마커(손목, 발끝)의 오차가 큰 것은 SMPL 피팅 오차에서 유래한다.
근위 마커(GTROCHANTER, ACROMION)는 상대적으로 작은 오차를 보인다.

= 근본 문제 진단

이번 end-to-end 검증으로 확인된 파이프라인의 실제 문제:

#table(
  columns: (auto, 1fr, auto),
  inset: 8pt,
  align: (center, left, center),
  [*단계*], [*문제*], [*심각도*],
  [SMPL 피팅], [50 iteration으로 복잡한 발레 포즈 수렴 실패], [높음],
  [SMPL 피팅], [Pose prior(GMM) 미적용 → 물리 불가능한 포즈 발산], [높음],
  [가상 마커], [피팅 오차가 그대로 마커 오차로 전파], [중간],
  [VTP 카메라], [이전 세션에서 수정 완료], [해결됨],
  [버텍스 인덱스], [이전 세션에서 KDTree로 수정 완료], [해결됨],
)

= 개선 방향

== 단기: SMPL 피팅 품질 개선
1. *iteration 증가*: 50 → 150~200 (복잡한 포즈 수렴 보장)
2. *Pose Prior 적용*: `data/models/smpl/gmm_08.pkl` GMM prior 활성화
   - 현재 코드에서 `loss_pose = (body_pose**2).sum() * 0.01` 만 사용
   - GMM prior 추가 시 물리적으로 불가능한 포즈 억제
3. *2-stage 최적화* (SMPLify 표준):
   - Stage 1: global_orient + transl 만 최적화 (10 iter)
   - Stage 2: 전체 파라미터 최적화 (100 iter)

== 중기: 더 나은 피팅 기반
- 4D-Humans, HybrIK, HMR2.0 등 사전 학습된 모델 사용
- 이들은 단일 프레임에서 < 5cm MPJPE 달성

= 결론

기존 검증("대칭인가")은 올바른 질문이 아니었다.
올바른 질문은 "_SMPL이 MediaPipe 입력을 얼마나 잘 재현하는가_"이며,
이번 측정으로 평균 9.83 cm 오차로 목표(5cm) 미달임이 확인되었다.

SMPL 피팅 품질 개선(GMM Prior + 2-stage 최적화)이 다음 핵심 작업이다.

#line(length: 100%)
#text(size: 9pt, fill: gray)[검증 도구: tools/e2e_smpl_error.py, tools/e2e_reprojection_overlay.py | 생성일: 2026-03-01]
