#set document(title: "芭蕾呪法 - Phase 3 버그픽스 & 기능구현 테스트 리포트")
#set page(margin: 2cm, numbering: "1")
#set text(font: "Noto Sans CJK KR", size: 10pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 22pt, weight: "bold")[芭蕾呪法]
  #v(0.3em)
  #text(size: 15pt)[Phase 3 — 버그픽스 & 기능구현 테스트 리포트]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[2026-02-22 | /sc:duo 자동 생성]
]

#line(length: 100%)
#v(0.5em)

= 작업 요약

이번 세션에서는 기존 코드베이스의 버그 3건을 수정하고 신규 기능 5건을 구현하였다.
모든 작업은 Claude(설계/검증) + Gemini CLI(구현) 분업 워크플로우로 진행되었으며,
최종 pytest 실행 결과 *23 passed, 3 skipped, 0 failed* 를 달성하였다.

#v(0.5em)

= 버그픽스 결과

#table(
  columns: (0.5fr, 2.5fr, 1fr, 1.5fr),
  inset: 8pt,
  align: (center, left, center, center),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if calc.odd(row) { rgb("#f8f8ff") } else { white },
  table.header(
    text(fill: white, weight: "bold")[번호],
    text(fill: white, weight: "bold")[항목],
    text(fill: white, weight: "bold")[상태],
    text(fill: white, weight: "bold")[파일],
  ),

  [B-1], [test\_beat\_loss\_analyzer.py — \@patch 데코레이터 순서 버그], [✅ FIXED], [tests/test\_beat\_loss\_analyzer.py],
  [B-2], [beat\_loss\_analyzer.py — 하드코딩된 OUTPUT\_DIR 경로], [✅ FIXED], [src/beat\_loss\_analyzer.py],
  [B-3], [requirements.txt — 미사용 trimesh 의존성 제거], [✅ FIXED], [requirements.txt],
)

#v(0.5em)

== B-1: \@patch 데코레이터 순서 버그

*증상:* `test_main_function_generates_outputs` 에서 `mock_savefig.call_count >= 5` 어서션이 실패. 실제 호출 횟수가 0으로 나타남.

*원인:* Python의 `\@patch` 데코레이터는 아래에서 위로 적용되지만 함수 파라미터에는 위에서 아래 순서로 매핑된다. 데코레이터 순서와 파라미터 이름이 불일치하여 `mock_savefig`가 실제로는 `mock_addfont`를 참조하고 있었다.

*수정:* 데코레이터를 파라미터 순서(위→아래)와 일치하도록 재배치하여 각 mock 변수가 올바른 패치 대상을 참조하도록 수정.

```python
# 수정 전: 파라미터 매핑 불일치
# @patch('matplotlib.font_manager.FontManager.addfont')  # mock_savefig 에 잘못 매핑
# @patch('beat_loss_analyzer.plt.savefig')

# 수정 후: 파라미터 순서와 일치
# @patch('matplotlib.pyplot.close')         # 6번째 param = mock_close
# @patch('matplotlib.pyplot.figure')        # 5번째 param = mock_figure
# @patch('...FontManager.addfont')          # 4번째 param = mock_addfont
# @patch('beat_loss_analyzer.plt.savefig')  # 3번째 param = mock_savefig
# @patch('beat_loss_analyzer.plt.show')     # 2번째 param = mock_show
# @patch('builtins.open', ...)              # 1번째 param = mock_file_open
```

#v(0.5em)

== B-2: 하드코딩된 OUTPUT\_DIR 경로

*증상:* `beat_loss_analyzer.py` 및 `video_overlay_generator.py` 에 절대 경로가 하드코딩되어 다른 환경에서 실행 불가.

*수정:* 환경 변수 `JUBEOP_OUTPUT_DIR` 우선 참조, 없으면 프로젝트 루트 기반 상대 경로 자동 계산.

```python
OUTPUT_DIR = os.environ.get(
    "JUBEOP_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "reports", "2026-02-16_phase2_3d_comparison", "assets")
)
```

#v(0.5em)

== B-3: 미사용 trimesh 의존성 제거

`trimesh>=3.23.0` 은 `requirements.txt`에 등록되어 있었으나 코드베이스 어디에서도 import되지 않았음을 확인 후 삭제하여 설치 부담 감소.

#pagebreak()

= 신규 기능 구현 결과

#table(
  columns: (0.5fr, 2.5fr, 1fr, 1.5fr),
  inset: 8pt,
  align: (center, left, center, center),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if calc.odd(row) { rgb("#f0f8ff") } else { white },
  table.header(
    text(fill: white, weight: "bold")[번호],
    text(fill: white, weight: "bold")[기능],
    text(fill: white, weight: "bold")[상태],
    text(fill: white, weight: "bold")[파일],
  ),

  [F-1], [DTW 기반 자동 시간축 정렬], [✅ DONE], [src/comparator.py],
  [F-2], [통합 CLI 파이프라인], [✅ DONE], [src/pipeline.py],
  [F-3], [Lambda 자동 보정 — Loss\_jubeop 공식], [✅ DONE], [src/beat\_loss\_analyzer.py],
  [F-4], [SMPL 멀티프레임 시퀀스 처리], [✅ DONE], [src/smpl\_visualizer.py],
  [F-5], [실시간 카메라 포즈 비교], [✅ DONE], [src/realtime\_comparator.py],
)

#v(0.5em)

== F-1: DTW 기반 자동 시간축 정렬

기준 영상과 비교 영상의 시작 타이밍 차이를 자동으로 보정하는 DTW(Dynamic Time Warping) 정렬 기능을 `comparator.py`에 추가하였다.

*핵심 구현:*
- `scipy.spatial.distance.cdist`로 8개 주요 관절(힙, 어깨, 무릎, 발목)의 프레임 간 거리 행렬 계산
- DP 백트래킹으로 최적 정렬 경로 탐색
- 시작 오프셋 탐색 범위를 전체 시퀀스의 처음 30%로 제한하여 과도한 정렬 방지

*검증 결과 (샘플 데이터):*
- DTW 오프셋: ref=89 프레임, comp=0 프레임 (→ 약 3.7초 시차 자동 감지)
- 정렬 후 평균 손실: 0.2645
- 결과 메타데이터에 `dtw_aligned: true`, `dtw_offsets` 포함

#v(0.5em)

== F-2: 통합 CLI 파이프라인

포즈 추출 → 비교 → 시각화의 전체 분석 플로우를 단일 명령으로 실행하는 `src/pipeline.py`를 작성.

```bash
python3 src/pipeline.py \
  --ref reference.mp4 \
  --comp dancer.mp4 \
  --output results/
```

*생성 결과물:*
- `reference_poses.json`, `compare_poses.json`
- `comparison_result.json`
- `loss_graph.png`, `skeleton_max_loss.png`, `joint_heatmap.png`

#v(0.5em)

== F-3: Lambda 자동 보정

기존 `beat_loss_analyzer.py`의 Loss\_jubeop 공식에서 $lambda_1 = 1.0, lambda_2 = 1.0$ 이 하드코딩되어 있던 문제를 해결.

공식: $L_j = lambda_1 dot P_"err" + lambda_2 dot Delta t$

`calibrate_lambdas()` 함수가 실제 데이터 분포의 최댓값 기반으로 가중치를 자동 계산하여 두 항이 동등하게 기여하도록 정규화함.

*검증 결과:*
- $lambda_1 = 1.9896$ (beat\_loss 최댓값 0.5026 기반)
- $lambda_2 = 1.7082$ (delta\_t\_sync 최댓값 0.5854 기반)

#v(0.5em)

== F-4: SMPL 멀티프레임 시퀀스 처리

기존에 단일 프레임만 처리하던 SMPL 시각화를 시퀀스 전체로 확장.

- `generate_smpl_sequence()` 함수: 시퀀스에서 균등 샘플링, 프레임별 PNG 저장
- OpenCV 사용 가능 시 4fps mp4 자동 생성
- CLI: `python3 src/smpl_visualizer.py --seq --max-frames 10 ...`

#v(0.5em)

== F-5: 실시간 카메라 포즈 비교

카메라 피드에서 실시간으로 포즈를 감지하고 기준 포즈와 비교하는 `RealtimeComparator` 클래스를 구현.

*주요 특성:*
- MediaPipe `LIVE_STREAM` 비동기 모드로 고프레임율 유지
- 이동 평균 필터(configurable window)로 손실 값 안정화
- 화면 색상 피드백: 손실 ≤ 0.3 → 초록, > 0.3 → 빨강
- 헤드리스 환경(WSL/CI)에서도 import 가능, run() 호출 시에만 카메라 필요

```bash
python3 src/realtime_comparator.py \
  --ref data/sample/reference_poses_3d.json \
  --camera 0
```

#pagebreak()

= 전체 테스트 결과

== pytest 실행 결과

#raw(block: true, lang: "text", read("assets/pytest_output.txt"))

== 테스트 커버리지 요약

#table(
  columns: (2.5fr, 1fr, 1fr, 1fr),
  inset: 8pt,
  align: (left, center, center, center),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if calc.odd(row) { rgb("#f8fff8") } else { white },
  table.header(
    text(fill: white, weight: "bold")[테스트 파일],
    text(fill: white, weight: "bold")[통과],
    text(fill: white, weight: "bold")[스킵],
    text(fill: white, weight: "bold")[실패],
  ),

  [test\_beat\_loss\_analyzer.py], [5], [0], [0],
  [test\_phase0.py], [5], [0], [0],
  [test\_phase2.py], [6], [0], [0],
  [test\_pm\_agent\_integration.py], [7], [3], [0],
  [*합계*], [*23*], [*3*], [*0*],
)

#v(0.5em)
#text(fill: green.darken(20%))[*최종 결과: 23 passed, 3 skipped, 0 failed*]

#v(0.3em)
스킵된 3건은 실제 MediaPipe 모델 파일 또는 카메라 하드웨어가 필요한 통합 테스트로, 환경 제약상 정상 스킵 처리됨.

#pagebreak()

= 구현 아키텍처 변화

#table(
  columns: (1fr, 1fr),
  inset: 8pt,
  align: (left, left),
  fill: (col, row) => if row == 0 { rgb("#1a1a2e") } else if calc.odd(row) { rgb("#fff8f0") } else { white },
  table.header(
    text(fill: white, weight: "bold")[수정 전],
    text(fill: white, weight: "bold")[수정 후],
  ),

  [각 단계를 개별 스크립트로 수동 실행], [pipeline.py 단일 명령으로 전체 파이프라인 실행],
  [시간축 오프셋을 수동으로 설정], [DTW로 최적 오프셋 자동 탐색],
  [$lambda$ 가중치 하드코딩 (1.0, 1.0)], [calibrate\_lambdas()로 데이터 기반 자동 산출],
  [SMPL 단일 프레임만 시각화], [전체 시퀀스 배치 처리 + mp4 생성],
  [사후 분석만 가능], [realtime\_comparator.py로 실시간 피드백 가능],
  [OUTPUT\_DIR 하드코딩], [JUBEOP\_OUTPUT\_DIR 환경변수 또는 자동 경로 계산],
)

#v(1em)

= 결론 및 다음 단계

이번 Phase 3 작업으로 芭蕾呪法 PoC의 핵심 기능이 완성 단계에 진입하였다.

*완료된 항목:*
- 버그 3건 수정 (테스트 신뢰성 확보 — 0 failed 달성)
- DTW 자동 정렬로 실용성 향상
- 통합 파이프라인으로 사용성 개선
- Loss\_jubeop 공식 데이터 기반 보정 적용
- 실시간 비교 기반 마련

*다음 권장 단계:*
+ *실제 영상 E2E 테스트:* 실제 발레 영상으로 pipeline.py 전체 흐름 검증
+ *realtime\_comparator 통합 테스트:* 카메라 환경에서 실시간 응답성 측정
+ *SMPL 시퀀스 품질 평가:* 다양한 동작에서 메시 품질 확인
+ *UI/UX 레이어:* 비기술자도 사용 가능한 인터페이스 검토

#v(1em)
#line(length: 100%)
#align(center)[
  #text(size: 8pt, fill: gray)[
    /sc:duo 자동 생성 | Claude(설계/검증) + Gemini CLI(구현) | 2026-02-22
  ]
]
