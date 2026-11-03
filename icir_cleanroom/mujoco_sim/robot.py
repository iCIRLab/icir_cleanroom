"""Differential/skid-steer drive, planar lidar, and pose of a MuJoCo robot."""
import math

import mujoco
import numpy as np

# Collision geoms: cleanroom_v2 leaves them in group 0, cleanroom_amc and the
# robots use group 3. Visual geoms (1), the ceiling (2), and sites stay out.
# The robot never shadows its own lidar: mj_multiRay excludes the base body
# (all of mobile_amc's collision boxes) and skips geoms with zero alpha (all
# of husky_franka's collision meshes, arm included).
LIDAR_GEOM_GROUPS = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)


class MujocoRobot:
    def __init__(self, model, config):
        self.model = model
        self.data = mujoco.MjData(model)
        self.config = config
        self.base = model.body(config['base_body']).id
        self.free_q = model.jnt_qposadr[model.body_jntadr[self.base]]
        self.free_v = model.jnt_dofadr[model.body_jntadr[self.base]]
        self.robot_bodies = {
            i for i in range(model.nbody) if model.body_rootid[i] == self.base}
        drive = config['drive']
        self.wheel_ctrl = [
            [model.actuator(name).id for name in drive[side]]
            for side in ('left', 'right')]
        self.track = drive['wheel_separation'] * drive['track_scale']
        self.command = np.zeros(2)
        self.joint_names = [
            model.joint(i).name for i in range(model.njnt)
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
            and model.jnt_bodyid[i] in self.robot_bodies]
        self.joint_q = [model.joint(n).qposadr[0] for n in self.joint_names]
        self.joint_v = [model.joint(n).dofadr[0] for n in self.joint_names]
        hold = config.get('arm_hold') or {}
        self.hold_v = [model.joint(n).dofadr[0] for n in hold]
        for name, value in hold.items():
            self.data.qpos[model.joint(name).qposadr[0]] = value
            self.data.ctrl[model.actuator(name + '_position').id] = value
        lidar = config['lidar']
        self.lidar_site = model.site(lidar['site']).id
        self.angles = np.linspace(
            -math.pi, math.pi, lidar['samples'], endpoint=False)
        self.rays = np.column_stack([
            np.cos(self.angles), np.sin(self.angles),
            np.zeros(lidar['samples'])])
        mujoco.mj_forward(model, self.data)

    def step(self, linear, angular, duration):
        """Advance by duration seconds toward the (linear, angular) command."""
        drive = self.config['drive']
        target = np.clip(
            [linear, angular],
            [-drive['max_linear'], -drive['max_angular']],
            [drive['max_linear'], drive['max_angular']])
        steps = max(1, round(duration / self.model.opt.timestep))
        dt = steps * self.model.opt.timestep
        change = np.array([
            drive['linear_acceleration'], drive['angular_acceleration']]) * dt
        self.command += np.clip(target - self.command, -change, change)
        v, w = self.command
        speeds = (np.array([v - w * self.track / 2, v + w * self.track / 2])
                  / drive['wheel_radius'])
        for ids, speed in zip(self.wheel_ctrl, speeds):
            self.data.ctrl[ids] = speed
        for _ in range(steps):
            # Gravity compensation, so the held arm does not sag on its gains.
            self.data.qfrc_applied[self.hold_v] = (
                self.data.qfrc_bias[self.hold_v])
            mujoco.mj_step(self.model, self.data)

    def pose(self):
        """World position (x, y, z) and quaternion (w, x, y, z) of the base."""
        q = self.data.qpos[self.free_q:self.free_q + 7]
        return q[:3].copy(), q[3:].copy()

    def twist(self):
        """Base-frame forward speed and yaw rate."""
        rotation = self.data.xmat[self.base].reshape(3, 3)
        velocity = self.data.qvel[self.free_v:self.free_v + 6]
        # Free joint: linear velocity in world axes, angular in body axes.
        return float((rotation.T @ velocity[:3])[0]), float(velocity[5])

    def joint_states(self):
        return (self.joint_names, self.data.qpos[self.joint_q].tolist(),
                self.data.qvel[self.joint_v].tolist())

    def site_position(self, name):
        return self.data.site_xpos[self.model.site(name).id].copy()

    def site_offset(self, site):
        """Site position and rotation matrix in the base body frame."""
        rotation = self.data.xmat[self.base].reshape(3, 3)
        position = rotation.T @ (
            self.data.site_xpos[site] - self.data.xpos[self.base])
        return position, rotation.T @ self.data.site_xmat[site].reshape(3, 3)

    def scan(self):
        """Ranges in the lidar site frame; inf outside the sensor range."""
        lidar = self.config['lidar']
        origin = self.data.site_xpos[self.lidar_site].copy()
        rays = np.ascontiguousarray(
            self.rays @ self.data.site_xmat[self.lidar_site].reshape(3, 3).T)
        count = len(rays)
        distances = np.empty(count)
        geoms = np.empty(count, dtype=np.int32)
        mujoco.mj_multiRay(
            self.model, self.data, origin, rays.ravel(), LIDAR_GEOM_GROUPS,
            1, self.base, geoms, distances, None, count, lidar['range_max'])
        distances[(distances < lidar['range_min'])
                  | (distances > lidar['range_max'])] = np.inf
        return distances
