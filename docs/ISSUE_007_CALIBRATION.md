# [Issue #7] 정적 캘리브레이션 및 피험자 맞춤형 모델 스케일링

## 1. 개요
표준(Generic) 모델과 피험자 신체 비율 간의 불일치를 해소하여, **Crouch Error(무릎 굽힘 오류)** 및 **Jittering(관절 떨림)** 현상을 근본적으로 해결함.

## 2. 발레 자세를 활용한 캘리브레이션 전략
사용자 제안에 따른 4단계 데이터 수집 및 보정:

| 단계 | 추천 포즈 | 주요 보정 항목 | 목적 |
| :--- | :--- | :--- | :--- |
| **Phase 1** | 평행 1번 (Parallel 1st) | 뼈 길이 (Scaling) | 대퇴골/경골/상완골의 실제 길이 측정 |
| **Phase 2** | 발레 1번/5번 (Turn-out) | 고관절 외회전 (Rotation) | 턴아웃 상태의 마커 오프셋 확정 |
| **Phase 3** | 플리에 (Plié) | 무릎 관절축 (Hinge Axis) | 굴곡 시 무릎 뒤틀림 방지 및 축 정렬 |
| **Phase 4** | 측면 뷰 (Side View) | 척추/골반 (Depth/Tilt) | 척추 전만/후만 및 골반 경사 보정 |

## 3. 기술적 구현 계획
1. **정적 프레임 추출 (Static Frame Extraction):**
   - 영상 시작 지점에서 피험자가 가만히 서 있는 10~20프레임을 자동으로 감지.
   - MediaPipe 3D World Landmarks의 분산을 체크하여 안정적인 데이터만 평균(Moving Average) 내어 사용.

2. **자동 스케일링 (Automated Scaling):**
   - OpenSim의 `Scale Tool`을 사용하거나, 직접 `.osim` XML의 `scale_factors`를 수정.
   - `pelvis_width`, `femur_length`, `tibia_length` 등을 계산하여 개별화된 모델 생성.

3. **마커 등록 (Marker Registration):**
   - MediaPipe의 점(피부 위)과 OpenSim의 Joint Center(뼈 중심) 사이의 오프셋(Offset)을 정적 자세에서 0으로 초기화.

## 4. 기대 효과
- **정확도:** F680 프레임에서 보인 "다리가 짧아 무릎을 굽히는 현상" 완전 제거.
- **안정성:** 모델과 마커가 일치하므로 IK 솔버의 수렴 속도가 빨라지고 떨림이 사라짐.
- **전문성:** 사용자별 맞춤형 바이오메카닉 모델링 가능.
