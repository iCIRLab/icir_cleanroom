# icir_cleanroom

`cleanroom_empty` 가스 매핑 시뮬레이션을 실행하는 ROS 2 Humble 패키지입니다.

## 빌드

```bash
cd /home/gyu/icir_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select icir_cleanroom --symlink-install
source install/setup.bash
```

## 실행

```bash
ros2 launch icir_cleanroom cleanroom_empty.launch.py
```

이 launch는 Gazebo와 TurtleBot3, Nav2, 가스 지도 생성기, LRS 경로 계획기,
LRS/HRS 매핑 컨트롤러 및 RViz를 시작합니다.

## 코드 구조

세 `scripts/*.py` 파일은 기존 `ros2 run` 실행 이름을 보존하는 얇은
entrypoint입니다. 실제 구현은 `icir_cleanroom/gas_mapping` Python package에
역할별로 분리되어 있습니다.

```text
gas_mapping/
├── config.py, models.py, phase_machine.py
├── environment.py
├── mapping/       # GMRF, transactional update, field projection
├── planning/      # exact path, LRS/HRS planners and policies
├── history/       # in-memory model and JSON repository
├── application/   # state owners, measurement, async task orchestration
└── ros/           # nodes, Nav2/service adapters, workflows, publishers
```

`mapping`, `planning`, `history`의 계산 모듈은 ROS node를 import하지 않습니다.
Controller의 공개 토픽 23개, 파라미터 46개, 서비스와 Nav2 action 이름은
리팩터링 전과 동일합니다.

## 테스트

```bash
cd /home/gyu/icir_ws
source /opt/ros/humble/setup.bash
colcon test --packages-select icir_cleanroom
colcon test-result --verbose
```

테스트는 GMRF GaBP/CG 결과와 rollback, UCB/HRS stop, LRS/HRS 경로,
history/source JSON 호환성, phase 및 async generation, RViz point/color 대응,
ROS 인터페이스 manifest를 고정합니다.
