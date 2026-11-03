# MuJoCo 월드·로봇 모델의 출처

`worlds/mujoco/`와 `models/mujoco/`의 파일은 이 패키지 밖에서 만든 결과물이다.
변환·생성 스크립트는 가져오지 않았고, 빌드에서 다시 만들지 않는다.

## 월드

| 이 패키지의 경로 | 원본 | 변경 |
|---|---|---|
| `worlds/mujoco/cleanroom_amc/cleanroom.xml` | `worlds/gazebo/cleanroom_amc/cleanroom.world`를 `mujoco_cleanroom/gazebo_port/sdf_world_to_mjcf.py`로 변환 | 변경 없음 |
| `worlds/mujoco/cleanroom_v2/cleanroom_v2.xml`, `steel_grating.png` | `mujoco_cleanroom/cleanroom_v2/build_cleanroom.py`로 생성 | 변경 없음 |

## 로봇

### `models/mujoco/mobile_amc/mobile_amc.xml`

`urdf/mobile_amc.urdf`를 `mujoco_cleanroom/gazebo_port/urdf_amc_to_mjcf.py`로
변환했다. 원본 URDF의 출처는 `cleanroom_amc.md`를 따른다 (MIT).

### `models/mujoco/husky_franka/`

Clearpath Husky A200 위에 Franka Emika Panda(FER)와 Robotiq 2F-85를 올린 모델이다.
`icir_mujoco_nav2`(iCIRLab, 커밋 `2459253` 기준 작업 트리)의
`mobile_openarm_mujoco.model.build_model(base="husky", actors_file="none")`로
창고 월드를 포함한 MJCF를 만든 뒤, 로봇 부분(`base_footprint` 트리, 액추에이터,
contact exclude, 그리퍼 equality)만 잘라냈다. 메시 경로는 `meshes/` 상대경로로 바꿨다.
geom 그룹은 이 패키지 규칙(보이는 형상 1, 천장 2, 충돌 형상 3, 사이트 4)에 맞춰
원본의 보이는 형상 2 → 1, 충돌 형상 1 → 3으로 바꿨다. 이 밖에는 변경하지 않았다.

| 구성 | 원본 | 라이선스 |
|---|---|---|
| Husky 베이스 메시 | `husky/husky` `husky_description` (`icir_mujoco_nav2/src/husky_description` 경유) | BSD, `husky_description_LICENSE.txt` |
| Panda(FER) 메시·관절 | `ros-humble-franka-description` 1.0.1 | Apache-2.0, `APACHE-2.0.txt` |
| Robotiq 2F-85 메시·관절 | `ros-humble-robotiq-description` 0.0.1 | BSD |

`meshes/*.obj`는 원본 STL/DAE를 `build_model`이 OBJ로 변환한 파일이다.
