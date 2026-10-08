# cleanroom_amc 환경의 외부 자료 출처

이 환경의 월드, 지도, 로봇 형상은 `cleanroom_gas_sim` 패키지에서 가져왔다.

- 원본 위치: `cleanroom_ws/src/cleanroom_gas_sim`
- 원본 라이선스: MIT (`cleanroom_gas_sim_LICENSE.txt`)
- 원본 패키지의 `config/nav2.yaml`과 `config/nav2_amc.yaml`은
  turtlebot3_manipulation_navigation2 2.3.8 (ROBOTIS, Apache-2.0)에서 파생된
  파일이다. 해당 라이선스 전문은 `APACHE-2.0.txt`에 있다.
- 로봇 메시·URDF·컨트롤러·상위 launch 파일 중 ROBOTIS/ROS 패키지에서 참조만
  하는 자료는 원본과 마찬가지로 재배포하지 않는다.

## 가져온 파일과 변경 사항

| 이 패키지의 경로 | 원본 | 변경 |
|---|---|---|
| `worlds/cleanroom_amc/cleanroom.world` | `worlds/cleanroom.world` | 변경 없음 |
| `worlds/cleanroom_amc/cleanroom_enclosed.world` | `worlds/cleanroom_enclosed.world` | 변경 없음 |
| `maps/cleanroom_amc/cleanroom.pgm` | `maps/cleanroom.pgm` | 변경 없음 |
| `maps/cleanroom_amc/cleanroom.yaml` | `maps/cleanroom.yaml` | 변경 없음 |
| `config/cleanroom_amc/geometry.json` | `config/geometry.json` | 변경 없음 (참조용) |
| `config/cleanroom_amc/generate_world.py` | `scripts/generate_world.py` | 변경 없음 (형상 재생성용, 빌드에서 실행하지 않음) |
| `config/cleanroom_amc/generate_amc_robot.py` | `scripts/generate_amc_robot.py` | 변경 없음 (형상 재생성용, 빌드에서 실행하지 않음) |
| `urdf/mobile_amc.urdf` | `urdf/mobile_amc.urdf` | 아래 3가지 수정 |
| `urdf/mobile_amc.sdf` | 없음 | `mobile_amc.urdf`에서 `gz sdf -p`로 생성 |
| `config/navigation/cleanroom_amc_safe.yaml` | `config/nav2_amc.yaml` | 전체 복사가 아니라 이 프로젝트의 부분 프로필 형식으로 변환 |

원본의 `cleanroom_gas_sim/transport.py`, `gas_node.py`, `launch/cleanroom.launch.py`,
`config/gas.yaml`, `config/nav2.yaml`은 가져오지 않았다. 이 프로젝트는 자체
`gas_environment_node`와 `gas_mapping.launch.py`를 쓰며, 두 구조를 동시에 실행하면
Gazebo·로봇·Nav2·map_server·TF·가스 노드가 중복된다.

## `urdf/mobile_amc.urdf` 수정 내역

1. `gas_sensor_link`에 관성과 시각 형상을 추가하고, 해당 fixed joint에
   `preserveFixedJoint`와 `disableFixedJointLumping`을 설정했다.
   **이유:** Gazebo의 URDF→SDF 변환은 fixed joint의 자식 링크를 부모에 병합한다.
   원본 상태로 변환하면 남는 링크가 `base_footprint`, `wheel_left_link`,
   `wheel_right_link` 세 개뿐이고 `gas_sensor_link`가 사라진다. 그러면
   `libgas_sensor_plugin.so`의 `GetLink("gas_sensor_link")`가 실패해
   `/gas_sensor/sensor_pose`가 발행되지 않는다. TF에 프레임이 보이더라도
   Gazebo 링크가 존재한다는 뜻은 아니다.
2. `amc_diff_drive`의 `<odometry_source>`를 `0`(엔코더)에서 `1`(월드 기준)으로
   바꿨다. **이유:** 공용 launch는 `map → odom`을 항등 변환으로 발행한다.
   엔코더 오도메트리는 원점이 스폰 위치가 되므로 로봇이 map 좌표에서
   스폰 위치만큼 어긋난다. TurtleBot3 모델도 이 항목이 없어 기본값 1(월드)로 동작한다.
3. `libgas_sensor_plugin.so` 플러그인 블록을 추가했다. 원본 AMC 모델에는 이
   프로젝트의 가스 센서 플러그인이 없었다.

`gas_sensor_high_link`(높이 1.35 m)는 원본대로 질량 없는 링크로 두었다. TF
프레임으로는 존재하지만 Gazebo 링크로는 병합되어 사라지며, 현재 가스 모델이
2D(수평면)라 사용하지 않는다.

## `urdf/mobile_amc.sdf` 재생성 방법

```bash
cd urdf && gz sdf -p mobile_amc.urdf > mobile_amc.sdf
```

`mobile_amc.urdf`를 고치면 이 명령으로 SDF를 다시 만들어야 한다.
`test/test_cleanroom_amc_environment.py`가 두 파일의 일치를 검사한다.
