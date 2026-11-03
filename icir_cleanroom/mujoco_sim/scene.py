"""Compose a MuJoCo world and robot into one model."""
import math
import warnings
from pathlib import Path

import mujoco
import yaml

# The robot was tuned with these solver settings; attach() would otherwise
# keep the world's defaults for every one of them.
ROBOT_OPTIONS = (
    'timestep', 'integrator', 'iterations', 'noslip_iterations', 'cone')


def load_robot_config(path, name):
    with open(path, 'r', encoding='utf-8') as stream:
        robots = yaml.safe_load(stream)
    if name not in robots:
        raise ValueError(
            f'unknown robot {name!r}; choose from {", ".join(robots)}')
    return robots[name]


def world_file(package_dir, world):
    """Return the single MJCF scene in worlds/mujoco/<world>/."""
    scenes = sorted((Path(package_dir) / 'worlds' / 'mujoco' / world)
                    .glob('*.xml'))
    if len(scenes) != 1:
        raise ValueError(
            f'worlds/mujoco/{world} must hold exactly one .xml scene, '
            f'found {len(scenes)}')
    return scenes[0]


def compose(world_path, robot_path, base_body, spawn=(0.0, 0.0, 0.0)):
    """Attach the robot's base body to the world at spawn = (x, y, yaw)."""
    world = mujoco.MjSpec.from_file(str(world_path))
    robot = mujoco.MjSpec.from_file(str(robot_path))
    for option in ROBOT_OPTIONS:
        setattr(world.option, option, getattr(robot.option, option))
    x, y, yaw = spawn
    frame = world.worldbody.add_frame(
        pos=[x, y, 0.0], quat=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    # An empty prefix keeps the URDF joint and frame names; MuJoCo warns that
    # such a model may not save back to XML, which the bridge never does.
    previous = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda message: None)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Attaching a spec')
            frame.attach_body(robot.body(base_body), '', '')
            return world.compile()
    finally:
        mujoco.set_mju_user_warning(previous)
