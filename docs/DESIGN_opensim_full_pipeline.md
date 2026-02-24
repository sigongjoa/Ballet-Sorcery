# OpenSim 전체 바이오메카닉스 파이프라인 설계
## 목표: 마커리스 영상 → 근육 활성화·관절 반력·뼈 접촉력 추정

**날짜**: 2026-02-23
**연구 근거**: `/sc:research` 결과 + Pose2Sim/pyopensim API 직접 탐색
**현재 상태**: NumPy 기하 각도 계산 → 실제 근골격 시뮬레이션으로 업그레이드 목표

---

## 1. 왜 OpenSim IK → ID → SO → JRA 가 필요한가

| 현재 (NumPy 기하) | 목표 (OpenSim 시뮬레이션) |
|---|---|
| 3점 각도 = 관절 굴곡 각도(°) | Inverse Kinematics → 40 DOF 정확한 관절 각도 |
| 없음 | Inverse Dynamics → 관절 모멘트(N·m) |
| 없음 | Static Optimization → 근육 활성화(0~1), 근육 힘(N) |
| 없음 | Joint Reaction Analysis → 관절 접촉력(N) |

사용자가 원하는 "실제 근육·인대·뼈 움직임 추정치"는 SO와 JRA 단계에서 나온다.

---

## 2. 전체 파이프라인 아키텍처

```
[입력] MediaPipe world_landmarks JSON
  ↓
[Step 1] TRC 내보내기 (수정 필요: BlazePose 마커 이름)
  - 25개 마커, lowercase 이름
  - Y축 반전 (MediaPipe Y-DOWN → OpenSim Y-UP)
  ↓
[Step 2] pyopensim Scaling
  - Scaling_Setup_Pose2Sim_Blazepose.xml 사용
  - 피험자 신체 비율을 기준 모델에 맞게 조정
  - 출력: scaled_model_{subject}.osim
  ↓
[Step 3] Inverse Kinematics (IK)
  - IK_Setup_Pose2Sim_Blazepose.xml 사용
  - 입력: scaled_model.osim + .trc 마커 궤적
  - 출력: {subject}_IK.mot (시간당 40 DOF 관절 각도)
  - 주요 DOF: hip_rotation_l/r (턴아웃), knee_angle_l/r, ankle_angle_l/r
  ↓
[Step 4] GRF 추정 (발레 플리에용 quasi-static 방법)
  - 이중 지지 단계 가정 (두 발이 땅에 닿아 있음)
  - COM(체질량 중심) 위치로 좌우 체중 분배 계산
  - 출력: {subject}_GRF.mot (지면반력 6축: Fx/Fy/Fz + COP)
  ↓
[Step 5] Inverse Dynamics (ID)
  - 입력: scaled_model.osim + IK.mot + GRF.mot
  - 출력: {subject}_ID.sto (시간당 관절 모멘트 N·m)
  - 주요 결과: hip_flexion_moment, knee_flexion_moment, ankle_moment
  ↓
[Step 6] Static Optimization (SO)
  - 입력: scaled_model.osim + IK.mot + ID.sto
  - 최소화: Σ(근육활성화²) s.t. 관절 모멘트 충족
  - 출력: {subject}_SO_StaticOptimization_activation.sto
    (발레 관련 근육: gluteus_max, rect_fem, gastroc, soleus, tib_ant 등)
  ↓
[Step 7] Joint Reaction Analysis (JRA)
  - 입력: scaled_model.osim + IK.mot + SO 결과
  - 출력: {subject}_JointReaction_ReactionLoads.sto
    (고관절, 무릎, 발목 접촉력 N·m, N)
  ↓
[Step 8] 비교 분석
  - DTW 정렬 후 ref vs comp 각 지표 비교
  - 시각화: 근육 활성화 곡선, 관절 모멘트 비교, 발레 스코어
```

---

## 3. pyopensim API 매핑

```python
# pip install pyopensim (설치 완료, libopenblas-base 필요)
from pyopensim import tools, analyses, common

# Step 2: Scaling
scale_tool = tools.ScaleTool("Scaling_Setup.xml")
scale_tool.run()

# Step 3: IK
ik_tool = tools.InverseKinematicsTool("IK_Setup.xml")
ik_tool.setMarkerDataFileName(trc_path)
ik_tool.setResultsDir(output_dir)
ik_tool.run()

# Step 5: ID
id_tool = tools.InverseDynamicsTool()
id_tool.setModel(scaled_model)
id_tool.setCoordinatesFileName(mot_path)     # IK 결과
id_tool.setExternalLoadsFileName(grf_path)   # GRF 추정값
id_tool.run()

# Step 6: Static Optimization
analyze_tool = tools.AnalyzeTool()
so = analyses.StaticOptimization()
analyze_tool.setAnalysis(so)
analyze_tool.run()

# Step 7: JRA
jra = analyses.JointReaction()
jra.setOnOff(True)
analyze_tool.setAnalysis(jra)
analyze_tool.run()
```

---

## 4. TRC 내보내기 수정 사항

**현재 문제**: `trc_exporter.py`가 OpenSim 약자 이름 사용 (LSHO, LHIP, LKNE 등)
**필요**: Pose2Sim BlazePose 설정이 기대하는 소문자 이름

### 마커 이름 변경표

| MediaPipe | 현재 (잘못됨) | 필요 (Pose2Sim BlazePose) |
|---|---|---|
| NOSE | Head | nose |
| LEFT_SHOULDER | LSHO | left_shoulder |
| RIGHT_SHOULDER | RSHO | right_shoulder |
| LEFT_ELBOW | LELB | left_elbow |
| RIGHT_ELBOW | RELB | right_elbow |
| LEFT_WRIST | LWRI | left_wrist |
| RIGHT_WRIST | RWRI | right_wrist |
| LEFT_HIP | LHIP | left_hip |
| RIGHT_HIP | RHIP | right_hip |
| LEFT_KNEE | LKNE | left_knee |
| RIGHT_KNEE | RKNE | right_knee |
| LEFT_ANKLE | LANK | left_ankle |
| RIGHT_ANKLE | RANK | right_ankle |
| LEFT_HEEL | LHEE | left_heel |
| RIGHT_HEEL | RHEE | right_heel |
| LEFT_FOOT_INDEX | LTOE | left_foot_index |
| RIGHT_FOOT_INDEX | RTOE | right_foot_index |
| (추가) LEFT_EYE | - | left_eye |
| (추가) RIGHT_EYE | - | right_eye |
| (추가) LEFT_INDEX | - | left_index |
| (추가) RIGHT_INDEX | - | right_index |
| (추가) LEFT_THUMB | - | left_thumb |
| (추가) RIGHT_THUMB | - | right_thumb |
| (추가) LEFT_PINKY | - | left_pinky |
| (추가) RIGHT_PINKY | - | right_pinky |

**참고**: PELV 가상 마커는 Pose2Sim에서 자동 생성 → 제거 가능

---

## 5. GRF 추정 알고리즘 (플리에 전용)

OpenSim ID 단계에서 GRF(지면반력)가 필수이지만 발레 스튜디오에는 포스 플레이트 없음.
**플리에는 이중 지지(double stance) 준정적 동작** → 뉴턴 역학으로 추정 가능.

```python
def estimate_grf_quasi_static(ik_mot: dict, body_mass_kg: float = 60.0) -> dict:
    """
    플리에 이중 지지 단계에서 지면반력 추정.

    원리:
    - 수직 GRF 합 = 체중 + 수직 가속도 보정 (준정적 → ≈ 체중)
    - 좌우 분배 = COM X 위치 비율
    - COP = 발 중심 위치

    Args:
        ik_mot: IK 결과 (COM 위치 포함)
        body_mass_kg: 피험자 체중

    Returns:
        GRF .mot 데이터 (Fx, Fy, Fz, COPx, COPy, COPz × 2발)
    """
    g = 9.81
    F_total = body_mass_kg * g  # 수직 전체 체중

    for frame in ik_mot['frames']:
        com_x = frame['pelvis_tx']  # pelvis X = 대략 COM X
        left_ankle_x = frame['left_ankle_x']
        right_ankle_x = frame['right_ankle_x']

        # 내분 비율로 좌우 분배
        span = right_ankle_x - left_ankle_x
        if abs(span) > 0.01:
            right_ratio = (com_x - left_ankle_x) / span
            right_ratio = max(0.0, min(1.0, right_ratio))
        else:
            right_ratio = 0.5

        left_Fy = F_total * (1.0 - right_ratio)
        right_Fy = F_total * right_ratio

        # COP = 발목 위치 (단순화)
        frame['GRF'] = {
            'left_Fx': 0.0, 'left_Fy': left_Fy, 'left_Fz': 0.0,
            'left_COPx': left_ankle_x, 'left_COPy': 0.0, 'left_COPz': ...,
            'right_Fx': 0.0, 'right_Fy': right_Fy, 'right_Fz': 0.0,
            'right_COPx': right_ankle_x, ...
        }
```

**한계**: 이 추정은 준정적 동작(플리에)에만 유효. 점프(allegro) 같은 동적 동작에는 오차 발생.

---

## 6. 발레 관련 근육 목록

Pose2Sim 단순 모델에 포함된 발레 관련 근육:

| 근육 | OpenSim 이름 | 발레 동작 | |
|---|---|---|---|
| 대둔근 | glut_max_l/r | 턴아웃, 아라베스크 | ⭐⭐⭐ |
| 중둔근 | glut_med_l/r | 균형, 측면 안정성 | ⭐⭐⭐ |
| 대퇴직근 | rect_fem_l/r | 플리에 제어 | ⭐⭐⭐ |
| 내측광근 | vas_med_l/r | 무릎 정렬 | ⭐⭐ |
| 외측광근 | vas_lat_l/r | 무릎 정렬 | ⭐⭐ |
| 비복근 | gastroc_l/r | 르레베, 포인트 | ⭐⭐⭐ |
| 가자미근 | soleus_l/r | 르레베 안정성 | ⭐⭐ |
| 전경골근 | tib_ant_l/r | 발목 조절 | ⭐⭐ |
| 내전근 | add_long_l/r | 턴아웃 보조 | ⭐⭐ |
| 햄스트링 | semimem_l/r 등 | 고관절 신전 | ⭐⭐ |

---

## 7. OpenSim 40 DOF → 발레 핵심 DOF

| DOF | OpenSim 이름 | 발레 의미 |
|---|---|---|
| 고관절 굴곡 | hip_flexion_l/r | 다리 들기 높이 |
| **고관절 외회전** | **hip_rotation_l/r** | **턴아웃 (발레 핵심)** |
| 고관절 외전 | hip_adduction_l/r | 옆으로 다리 벌림 |
| 무릎 굴곡 | knee_angle_l/r | 플리에 깊이 |
| 발목 굴곡 | ankle_angle_l/r | 르레베, 포인트 |
| **척추 측굴** | **L5_S1_Lat_Bending** | **척추 수직성** |
| 척추 전후굴 | L5_S1_Flex_Ext | 상체 전경 |
| 골반 경사 | pelvis_tilt | 전방 경사 |

---

## 8. Ref vs Comp 비교 지표

### 8.1 IK 관절 각도 비교 (40 DOF)
- DTW 정렬 후 평균 절대 차이
- 위상(timing) 차이 분석

### 8.2 근육 활성화 비교 (SO 결과)
```
발레 핵심 근육 10개 × 시간 시계열
- 겹쳐그리기: Ref(파랑) vs Comp(주황)
- RMS 차이 계산
- 비대칭 지수: |left - right| / max(left, right)
```

### 8.3 관절 모멘트 비교 (ID 결과)
```
- 피크 무릎 모멘트 (플리에 최저점)
- 플리에 내 무릎 모멘트 적분 (무릎 스트레스 지수)
- 발목 모멘트 피크 (르레베 순간)
```

### 8.4 관절 접촉력 비교 (JRA 결과)
```
- 무릎 관절 압축력 (N) - 부상 위험 지표
- 고관절 전단력 - 턴아웃 스트레스
```

---

## 9. 시각화 설계

### 탭 4: 🦴 바이오메카닉스 (업그레이드)

```
[섹션 A] 요약 카드
  - 🏆 종합 유사도 스코어
  - 💪 근육 사용 패턴 유사도
  - 🦴 관절 부하 비교 (Ref 대비 % )
  - ⚠️ 잠재 부상 위험 (무릎 부하 임계값 초과 여부)

[섹션 B] 근육 활성화 비교 (Plotly 시계열)
  - 턴아웃 근육: glut_max, add_long
  - 플리에 근육: rect_fem, vas_med, gastroc
  - 안정화 근육: glut_med, tib_ant
  각 그래프: Ref(파랑) vs Comp(주황), Y축 0~1 (활성화율)

[섹션 C] 관절 각도 시계열 (IK, 기존 개선)
  - hip_rotation_l/r (=실제 턴아웃 IK 계산)
  - knee_angle_l/r
  - ankle_angle_l/r
  - L5_S1_Lat_Bending (척추)

[섹션 D] 관절 모멘트 비교 (ID)
  - 막대 그래프: 피크 무릎 모멘트 Ref vs Comp
  - 시계열: 발목 모멘트 (르레베 타이밍 비교)

[섹션 E] 부상 위험 히트맵
  - 관절별 × 시간 × 접촉력 (JRA)
  - 임계값 초과 구간 빨간색 강조
```

---

## 10. 구현 계획

### Phase A: TRC 수정 (1일)
**파일**: `src/trc_exporter.py`
- MEDIAPIPE_TO_TRC 딕셔너리를 BlazePose 소문자 이름으로 변경
- 25개 마커 지원 (8개 추가: eyes, fingers)
- PELV 제거 (Pose2Sim 자동 생성)

### Phase B: pyopensim IK 통합 (2일)
**신규**: `src/opensim_ik.py`
- `ScaleTool` + `InverseKinematicsTool` 실행
- BlazePose 기존 XML 설정 파일 복사/활용
- .mot 파일 파싱 → 40 DOF 시계열 반환
- 실패 시 기존 NumPy fallback 유지

### Phase C: GRF 추정 + ID (2일)
**신규**: `src/grf_estimator.py`
- 플리에용 quasi-static GRF 추정
- GRF.mot 파일 생성

**신규**: `src/opensim_id.py`
- `InverseDynamicsTool` 실행
- .sto 파싱 → 관절 모멘트 dict

### Phase D: SO + JRA (2일)
**신규**: `src/opensim_so_jra.py`
- `AnalyzeTool` + `StaticOptimization` + `JointReaction`
- 근육 활성화 .sto, 접촉력 .sto 파싱
- 발레 핵심 근육 필터링 + 요약

### Phase E: 파이프라인 통합 (1일)
**수정**: `src/pose2sim_bridge.py`
- 4단계 파이프라인 순차 실행
- 각 단계 실패 시 fallback 전략

**수정**: `src/pipeline.py`
- Step 5에 전체 파이프라인 호출

### Phase F: 뷰어 업그레이드 (2일)
**수정**: `src/viewer_app.py`
- 섹션 A~E 시각화 구현
- Plotly 인터랙티브 그래프

---

## 11. 기술적 위험 및 대응

| 위험 | 확률 | 대응 |
|---|---|---|
| pyopensim SO 수렴 실패 | 중 | max iteration 증가, 초기값 조정 |
| 스케일링 오류 (키/몸무게 미지) | 중 | 기본값 1.7m/60kg, 어깨-골반 비율로 자동 추정 |
| GRF 추정 오차 (동적 구간) | 높음 | 플리에 페이즈 검출 후 quasi-static 구간만 적용 |
| 처리 시간 (SO는 느림) | 중 | 키 프레임 추출 후 처리, 전체 시퀀스 대신 샘플링 |
| 마커 결측 (손가락 등) | 낮음 | 결측 마커 가중치 0으로 IK 실행 |

---

## 12. 예상 출력 파일

```
data/e2e_output/
├── reference_poses.trc          (수정: 25 BlazePose 마커)
├── compare_poses.trc
├── reference_scaled.osim        (개인 신체 비율 반영 모델)
├── compare_scaled.osim
├── reference_IK.mot             (40 DOF 관절 각도)
├── compare_IK.mot
├── reference_GRF.mot            (추정 지면반력)
├── compare_GRF.mot
├── reference_ID.sto             (관절 모멘트 N·m)
├── compare_ID.sto
├── reference_SO_activation.sto  (근육 활성화 0~1)
├── compare_SO_activation.sto
├── reference_JR_loads.sto       (관절 접촉력 N)
├── compare_JR_loads.sto
└── full_comparison.json         (통합 비교 결과)
```

---

## 참고

- pyopensim: `pip install pyopensim` (v4.5.2.1, 29MB 자체 포함)
  - 의존성: `apt-get install libopenblas-base`
- Pose2Sim BlazePose 설정: `/usr/local/lib/python3.10/dist-packages/Pose2Sim/OpenSim_Setup/`
  - `Markers_BlazePose.xml`
  - `Scaling_Setup_Pose2Sim_Blazepose.xml`
  - `IK_Setup_Pose2Sim_Blazepose.xml`
  - `Model_Pose2Sim_simple.osim`
