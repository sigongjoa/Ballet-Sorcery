# 파뢰주법 (芭蕾呪法) - Project Overview

## 프로젝트 정의

발레 동작의 시계열 데이터와 음악의 주기성을 동기화하여,
사용자에게 "확정된 프레임"을 시각적으로 가이드하는 시스템.

- **Main Title**: 주법 (呪法 / JUBEOP)
- **Subtitle**: The Projection Ballet System
- **핵심 철학**: "신체는 확정적 알고리즘이다."

## 핵심 기능 (Technical Requirements)

| 기능 | 설명 | 기술적 목표 |
|------|------|-------------|
| BPM 클럭 동기화 | 오디오에서 비트 추출 → 시스템 메인 클럭 | Sync Drift < 10ms |
| Pose-to-Frame 매핑 | 관절 데이터를 24FPS 타임라인 프레임에 할당 | Inference Latency < 30ms |
| 궤적 보간 엔진 | 현재 포즈 → 목표 포즈 최적 경로 계산 | Cubic Spline Interpolation |
| 프레임 누락 감지 | 목표 궤적 이탈 시 에러 이벤트 발생 | Sensitivity 조절 가능 |

## 손실 함수 (Loss Function)

```
Loss_jubeop = λ1 * ||P_act - P_tar||^2 + λ2 * Δt_sync
```

- `P_act`: 실제 포즈 좌표 벡터
- `P_tar`: 해당 프레임의 목표 포즈 좌표 벡터
- `Δt_sync`: 음악 클럭과 실제 동작 완성 시점의 시간차

## 기술 스택

- **PoC**: Python, MediaPipe, OpenCV, librosa, NumPy
- **Production**: Swift, iOS SDK, ARKit, Vision Framework, Metal, Core Audio

## 로드맵

- v0.1 (PoC): PC GPU에서 두 영상 비교 → 포즈 유사도 검증
- v0.5 (Analysis): 실시간 카메라 + BPM 동기화 + Loss 점수화
- v1.0 (Release): iOS 앱, AR HUD, 파뢰주법 훈련 루틴
