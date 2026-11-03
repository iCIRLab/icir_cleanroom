"""ROS adapter that steps a MuJoCo world + robot and mirrors the Gazebo I/O.

Topics match the Gazebo plugins in urdf/mobile_amc.urdf: /cmd_vel in;
/clock, /odom with odom->base_footprint TF (ground truth, like
<odometry_source>1</odometry_source>), /scan, /joint_states, and
/gas_sensor/sensor_pose out.
"""
import math
import os
import time

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from ..robot import MujocoRobot
from ..scene import compose, load_robot_config, world_file

BRIDGE_RATE = 100.0
STATE_RATE = 50.0
GAS_SENSOR_RATE = 20.0


def stamp(seconds):
    nanoseconds = int(round(seconds * 1e9))
    return Time(sec=nanoseconds // 1_000_000_000,
                nanosec=nanoseconds % 1_000_000_000)


def quaternion_from_matrix(rotation):
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).ravel())
    return map(float, quaternion)


class Every:
    """True once per period of simulated time."""

    def __init__(self, rate):
        self.period = 1.0 / rate
        self.next = 0.0

    def due(self, now):
        if now + 1e-9 < self.next:
            return False
        self.next = max(self.next + self.period, now)
        return True


class MujocoBridgeNode(Node):
    def __init__(self):
        super().__init__('mujoco_bridge')
        world = self.declare_parameter('world', 'cleanroom_amc').value
        robot_name = self.declare_parameter('robot', 'mobile_amc').value
        spawn = (self.declare_parameter('x', 0.0).value,
                 self.declare_parameter('y', 0.0).value,
                 self.declare_parameter('yaw', 0.0).value)
        viewer = self.declare_parameter('viewer', False).value
        # 0 keeps the last command like libgazebo_ros_diff_drive.
        self.command_timeout = self.declare_parameter(
            'command_timeout', 0.0).value
        package_dir = get_package_share_directory('icir_cleanroom')
        self.config = load_robot_config(
            os.path.join(package_dir, 'config', 'mujoco', 'robots.yaml'),
            robot_name)
        model = compose(
            world_file(package_dir, world),
            os.path.join(package_dir, self.config['model']),
            self.config['base_body'], spawn)
        self.robot = MujocoRobot(model, self.config)
        self.command = (0.0, 0.0)
        self.command_time = time.monotonic()
        self.viewer = None
        if viewer:
            from mujoco import viewer as mujoco_viewer
            self.viewer = mujoco_viewer.launch_passive(
                model, self.robot.data)

        self.create_subscription(Twist, '/cmd_vel', self.receive_command, 10)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.joint_pub = self.create_publisher(
            JointState, '/joint_states', 10)
        self.gas_sensor_pub = None
        if self.config.get('gas_sensor_site'):
            self.gas_sensor_pub = self.create_publisher(
                PoseStamped, '/gas_sensor/sensor_pose', 10)
        self.tf = TransformBroadcaster(self)
        if not self.config.get('description'):
            self.static_tf = StaticTransformBroadcaster(self)
            self.publish_lidar_frame()
        self.state_timer = Every(STATE_RATE)
        self.scan_timer = Every(self.config['lidar']['rate'])
        self.gas_sensor_timer = Every(GAS_SENSOR_RATE)
        self.create_timer(1.0 / BRIDGE_RATE, self.advance)
        self.get_logger().info(
            f'{robot_name} in {world} at x={spawn[0]:.2f} y={spawn[1]:.2f} '
            f'yaw={spawn[2]:.2f}')

    def receive_command(self, msg):
        if not all(math.isfinite(v) for v in (msg.linear.x, msg.angular.z)):
            return
        self.command = (msg.linear.x, msg.angular.z)
        self.command_time = time.monotonic()

    def advance(self):
        command = self.command
        if (self.command_timeout > 0.0 and time.monotonic()
                - self.command_time > self.command_timeout):
            command = (0.0, 0.0)
        self.robot.step(*command, 1.0 / BRIDGE_RATE)
        now = self.robot.data.time
        header_stamp = stamp(now)
        self.clock_pub.publish(Clock(clock=header_stamp))
        if self.state_timer.due(now):
            self.publish_state(header_stamp)
        if self.scan_timer.due(now):
            self.publish_scan(header_stamp)
        if self.gas_sensor_pub and self.gas_sensor_timer.due(now):
            self.publish_gas_sensor(header_stamp)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def publish_state(self, header_stamp):
        position, quaternion = self.robot.pose()
        linear, angular = self.robot.twist()
        odom = Odometry()
        odom.header.stamp = header_stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = self.config['base_body']
        odom.pose.pose.position.x = float(position[0])
        odom.pose.pose.position.y = float(position[1])
        odom.pose.pose.position.z = float(position[2])
        (odom.pose.pose.orientation.w, odom.pose.pose.orientation.x,
         odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z) = map(float, quaternion)
        odom.twist.twist.linear.x = linear
        odom.twist.twist.angular.z = angular
        self.odom_pub.publish(odom)
        transform = TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = odom.child_frame_id
        transform.transform.translation.x = odom.pose.pose.position.x
        transform.transform.translation.y = odom.pose.pose.position.y
        transform.transform.translation.z = odom.pose.pose.position.z
        transform.transform.rotation = odom.pose.pose.orientation
        self.tf.sendTransform(transform)
        joints = JointState()
        joints.header.stamp = header_stamp
        joints.name, joints.position, joints.velocity = (
            self.robot.joint_states())
        self.joint_pub.publish(joints)

    def publish_scan(self, header_stamp):
        lidar = self.config['lidar']
        angles = self.robot.angles
        scan = LaserScan()
        scan.header.stamp = header_stamp
        scan.header.frame_id = lidar['frame']
        scan.angle_min = float(angles[0])
        scan.angle_max = float(angles[-1])
        scan.angle_increment = float(angles[1] - angles[0])
        scan.scan_time = 1.0 / lidar['rate']
        scan.range_min = float(lidar['range_min'])
        scan.range_max = float(lidar['range_max'])
        scan.ranges = self.robot.scan().astype('float32').tolist()
        self.scan_pub.publish(scan)

    def publish_gas_sensor(self, header_stamp):
        position = self.robot.site_position(self.config['gas_sensor_site'])
        pose = PoseStamped()
        pose.header.stamp = header_stamp
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.w = 1.0
        self.gas_sensor_pub.publish(pose)

    def publish_lidar_frame(self):
        position, rotation = self.robot.site_offset(self.robot.lidar_site)
        transform = TransformStamped()
        transform.header.frame_id = self.config['base_body']
        transform.child_frame_id = self.config['lidar']['frame']
        (transform.transform.translation.x,
         transform.transform.translation.y,
         transform.transform.translation.z) = map(float, position)
        (transform.transform.rotation.w, transform.transform.rotation.x,
         transform.transform.rotation.y,
         transform.transform.rotation.z) = quaternion_from_matrix(rotation)
        self.static_tf.sendTransform(transform)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
