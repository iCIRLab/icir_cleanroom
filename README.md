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
