"""MuJoCo worlds and robots compose, drive, and scan like the Gazebo setup."""
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from icir_cleanroom.mujoco_sim.robot import LIDAR_GEOM_GROUPS, MujocoRobot
from icir_cleanroom.mujoco_sim.scene import (
    compose, load_robot_config, world_file)

PACKAGE = Path(__file__).resolve().parents[1]
WORLDS = ('cleanroom_amc', 'cleanroom_v2')
ROBOTS = ('mobile_amc', 'husky_franka')


def make_robot(world, name, spawn=(0.0, 0.0, 0.0)):
    config = load_robot_config(
        PACKAGE / 'config' / 'mujoco' / 'robots.yaml', name)
    model = compose(world_file(PACKAGE, world), PACKAGE / config['model'],
                    config['base_body'], spawn)
    return MujocoRobot(model, config)


def drive(robot, linear, angular, seconds):
    for _ in range(round(seconds / 0.01)):
        robot.step(linear, angular, 0.01)


def yaw(quaternion):
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def test_unknown_robot_lists_the_choices():
    with pytest.raises(ValueError, match='mobile_amc'):
        load_robot_config(
            PACKAGE / 'config' / 'mujoco' / 'robots.yaml', 'turtlebot')


@pytest.mark.parametrize('world', WORLDS)
@pytest.mark.parametrize('name', ROBOTS)
def test_robot_drives_turns_and_scans_in_world(world, name):
    robot = make_robot(world, name)
    drive(robot, 0.0, 0.0, 1.0)
    start, _ = robot.pose()
    drive(robot, 0.3, 0.0, 4.0)
    position, _ = robot.pose()
    # 0.3 m/s for 4 s, less the acceleration ramp.
    assert position[0] - start[0] == pytest.approx(1.12, abs=0.05)
    assert abs(position[1] - start[1]) < 0.02
    assert robot.twist()[0] == pytest.approx(0.3, abs=0.01)
    drive(robot, 0.0, 0.4, math.pi / 2 / 0.4)
    drive(robot, 0.0, 0.0, 1.0)
    _, quaternion = robot.pose()
    assert math.degrees(yaw(quaternion)) == pytest.approx(90.0, abs=8.0)
    ranges = robot.scan()
    assert len(ranges) == robot.config['lidar']['samples']
    assert np.isfinite(ranges).mean() > 0.9
    assert ranges[np.isfinite(ranges)].min() > 0.5


def test_spawn_pose_places_the_robot():
    robot = make_robot('cleanroom_amc', 'mobile_amc', (-8.0, -5.0, 1.0))
    position, quaternion = robot.pose()
    assert position[:2] == pytest.approx([-8.0, -5.0], abs=1e-6)
    assert yaw(quaternion) == pytest.approx(1.0, abs=1e-6)


def cast(robot, origin, rays):
    geoms = np.empty(len(rays), dtype=np.int32)
    distances = np.empty(len(rays))
    mujoco.mj_multiRay(
        robot.model, robot.data, origin, np.ascontiguousarray(rays).ravel(),
        LIDAR_GEOM_GROUPS, 1, robot.base, geoms, distances, None, len(rays),
        30.0)
    return geoms


@pytest.mark.parametrize('name', ROBOTS)
def test_lidar_scan_plane_never_hits_the_robot(name):
    robot = make_robot('cleanroom_v2', name)
    site = robot.lidar_site
    geoms = cast(robot, robot.data.site_xpos[site].copy(),
                 robot.rays @ robot.data.site_xmat[site].reshape(3, 3).T)
    hits = geoms[geoms >= 0]
    assert not set(robot.model.geom_bodyid[hits]) & robot.robot_bodies


def test_husky_collision_meshes_are_invisible_to_the_lidar():
    # Rays aimed straight at each Husky collision geom, the arm included,
    # pass through: an arm swung into the scan plane never shadows it.
    robot = make_robot('cleanroom_v2', 'husky_franka')
    origin = robot.data.site_xpos[robot.lidar_site].copy()
    for geom in range(robot.model.ngeom):
        if (robot.model.geom_bodyid[geom] not in robot.robot_bodies
                or not robot.model.geom_contype[geom]):
            continue
        offset = robot.data.geom_xpos[geom] - origin
        hit = np.empty(1, dtype=np.int32)
        mujoco.mj_ray(robot.model, robot.data, origin,
                      offset / np.linalg.norm(offset), LIDAR_GEOM_GROUPS,
                      1, robot.base, hit)
        assert (hit[0] < 0
                or robot.model.geom_bodyid[hit[0]]
                not in robot.robot_bodies), robot.model.geom(geom).name


def test_held_arm_stays_in_the_transport_pose_while_driving():
    robot = make_robot('cleanroom_v2', 'husky_franka')
    drive(robot, 0.35, 0.0, 2.0)
    drive(robot, 0.0, 0.7, 2.0)
    names, positions, _ = robot.joint_states()
    joints = dict(zip(names, positions))
    for name, target in robot.config['arm_hold'].items():
        assert joints[name] == pytest.approx(target, abs=0.02), name


def test_lidar_frame_offset_matches_the_urdf_base_scan_joint():
    robot = make_robot('cleanroom_amc', 'mobile_amc')
    position, rotation = robot.site_offset(robot.lidar_site)
    # urdf/mobile_amc.urdf base_scan_fixed: xyz=".37 0 .36", rpy="0 0 0".
    assert position == pytest.approx([0.37, 0.0, 0.36], abs=1e-3)
    assert rotation == pytest.approx(np.eye(3), abs=1e-3)
