"""ROS publisher registry and message conversion helpers."""

import copy
import math

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import (
    Bool, ColorRGBA, Float32MultiArray, Int32, String)
from visualization_msgs.msg import Marker

from ..mapping.field_projection import (
    logarithmic_display as transform_logarithmic_display, project_field)
from ..planning.hrs_policy import normalized_ucb


PUBLISHER_SPECS = (
    ('estimate_pub', OccupancyGrid, '/gas_mapping/estimate'),
    ('estimate_log_pub', OccupancyGrid, '/gas_mapping/estimate_log'),
    ('variance_pub', OccupancyGrid, '/gas_mapping/variance'),
    ('reward_pub', OccupancyGrid, '/gas_mapping/reward'),
    ('poses_pub', PoseArray, '/gas_mapping/measurements/poses'),
    ('values_pub', Float32MultiArray, '/gas_mapping/measurements/values'),
    ('lrs_status_pub', Marker, '/gas_mapping/lrs/status'),
    ('lrs_status_log_pub', Marker, '/gas_mapping/lrs/status_log'),
    ('lrs_active_route_pub', Path, '/gas_mapping/lrs/active_route'),
    ('lrs_reward_pub', OccupancyGrid, '/gas_mapping/lrs/reward'),
    ('lrs_priority_candidates_pub', Marker,
     '/gas_mapping/lrs/priority_candidates'),
    ('lrs_priority_route_pub', Path, '/gas_mapping/lrs/priority_route'),
    ('history_map_pub', OccupancyGrid,
     '/gas_mapping/history/concentration'),
    ('history_log_map_pub', OccupancyGrid,
     '/gas_mapping/history/concentration_log'),
    ('history_recent_log_map_pub', OccupancyGrid,
     '/gas_mapping/history/recent_log'),
    ('history_confirmed_peaks_pub', Marker,
     '/gas_mapping/history/confirmed_peaks'),
    ('history_revisit_pub', Marker, '/gas_mapping/history/revisit'),
    ('lrs_lap_pub', Int32, '/gas_mapping/lrs/lap'),
    ('hazard_pub', Bool, '/gas_mapping/hazard_state'),
    ('hrs_route_pub', Path, '/gas_mapping/hrs/route'),
    ('hrs_candidates_pub', Marker, '/gas_mapping/hrs/candidates'),
    ('hrs_status_pub', Marker, '/gas_mapping/hrs/status'),
    ('phase_pub', String, '/gas_mapping/phase'),
)


class MappingPublishers:
    """Create and own all retained controller outputs in one place."""

    def __init__(self, node, qos):
        self.handles = {}
        for attribute, message_type, topic in PUBLISHER_SPECS:
            publisher = node.create_publisher(message_type, topic, qos)
            setattr(self, attribute, publisher)
            self.handles[attribute] = publisher

    @staticmethod
    def occupancy_grid(template, values, free_mask, stamp):
        grid = copy.deepcopy(template)
        grid.header.stamp = stamp
        data = np.full(values.shape, -1, dtype=np.int16)
        clipped = np.clip(values, 0.0, 1.0)
        data[free_mask] = np.rint(
            100.0 * clipped[free_mask]).astype(np.int16)
        grid.data = data.astype(np.int8).ravel().tolist()
        return grid

    @staticmethod
    def value_color(value, log_concentration_min, logarithmic=False):
        if value is None:
            return ColorRGBA(r=0.55, g=0.55, b=0.55, a=1.0)
        normalized = min(1.0, max(0.0, float(value)))
        if logarithmic:
            log_min = float(log_concentration_min)
            clipped = max(np.exp(log_min), normalized)
            normalized = min(1.0, max(
                0.0, (np.log(clipped) - log_min) / -log_min))
        return ColorRGBA(
            r=normalized, g=0.0, b=1.0 - normalized, a=1.0)


class ControllerVisualization:
    """Build and publish all controller maps, paths, and markers."""

    METHODS = (
        'clear_hrs_candidates', 'display_field', 'occupancy_grid',
        'publish_maps', 'logarithmic_display', 'publish_measurements',
        'publish_lrs_active_route', 'publish_lrs_reward',
        'publish_lrs_priority_candidates', 'publish_lrs_priority_route',
        'publish_history', 'publish_empty_hrs_route', 'value_color',
        'publish_lrs_status', 'publish_candidates', 'publish_hrs_route',
        'publish_hrs_status',
    )

    def __init__(self, controller):
        self.controller = controller

    def clear_hrs_candidates(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.controller.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_hrs_candidates'
        marker.id = 0
        marker.action = Marker.DELETE
        self.controller.hrs_candidates_pub.publish(marker)

    def display_field(self, variable_values):
        projected = project_field(
            self.controller.gmrf, variable_values, self.controller.estimate_resolution)
        msg = copy.deepcopy(self.controller.map_msg)
        msg.info.resolution = projected.resolution
        msg.info.width = projected.width
        msg.info.height = projected.height
        msg.info.origin.position.x = projected.origin_x
        msg.info.origin.position.y = projected.origin_y
        return msg, projected.values, projected.free_mask

    def occupancy_grid(self, template, values, free_mask):
        return self.controller.mapping_publishers.occupancy_grid(
            template, values, free_mask, self.controller.get_clock().now().to_msg())

    def publish_maps(self):
        template, mean, free = self.controller.display_field(self.controller.gmrf.solution)
        self.controller.estimate_pub.publish(self.controller.occupancy_grid(template, mean, free))

        log_normalized = self.controller.logarithmic_display(mean)
        self.controller.estimate_log_pub.publish(
            self.controller.occupancy_grid(template, log_normalized, free))

        variance_template, variance, variance_free = self.controller.display_field(
            self.controller.gmrf.variance)
        self.controller.variance_pub.publish(self.controller.occupancy_grid(
            variance_template, variance, variance_free))
        reward = normalized_ucb(
            self.controller.gmrf.solution, self.controller.gmrf.variance,
            float(self.controller.hrs_ucb_k))
        reward_template, reward_values, reward_free = self.controller.display_field(reward)
        self.controller.reward_pub.publish(self.controller.occupancy_grid(
            reward_template, reward_values, reward_free))

    def logarithmic_display(self, values):
        return transform_logarithmic_display(
            values, self.controller.log_concentration_min)

    def publish_measurements(self):
        poses = PoseArray()
        poses.header.frame_id = 'map'
        poses.header.stamp = self.controller.get_clock().now().to_msg()
        poses.poses = []
        for measurement in self.controller.measurement_manager.measurements:
            pose = Pose()
            pose.position.x = measurement.pose.x
            pose.position.y = measurement.pose.y
            pose.orientation.z = math.sin(measurement.pose.yaw / 2.0)
            pose.orientation.w = math.cos(measurement.pose.yaw / 2.0)
            poses.poses.append(pose)
        self.controller.poses_pub.publish(poses)
        self.controller.values_pub.publish(Float32MultiArray(data=[
            measurement.value
            for measurement in self.controller.measurement_manager.measurements]))

    def publish_lrs_active_route(self):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.controller.get_clock().now().to_msg()
        for goal in self.controller.lrs_goals + self.controller.lrs_goals[:1]:
            pose = copy.deepcopy(goal)
            pose.header = path.header
            path.poses.append(pose)
        self.controller.lrs_active_route_pub.publish(path)

    def publish_lrs_reward(self):
        if self.controller.gmrf is None or self.controller.lrs_reward_values is None:
            return
        template, values, free = self.controller.display_field(self.controller.lrs_reward_values)
        self.controller.lrs_reward_pub.publish(
            self.controller.occupancy_grid(template, values, free))

    def publish_lrs_priority_candidates(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.controller.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_lrs_priority_candidates'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.48
        for point in self.controller.lrs_priority_candidates:
            marker.points.append(Point(x=point.x, y=point.y, z=0.20))
            if point.mandatory:
                marker.colors.append(ColorRGBA(
                    r=0.0, g=1.0, b=0.55, a=1.0))
            else:
                marker.colors.append(ColorRGBA(
                    r=1.0, g=0.25 + 0.55 * point.reward,
                    b=0.0, a=1.0))
        self.controller.lrs_priority_candidates_pub.publish(marker)

    def publish_lrs_priority_route(self):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.controller.get_clock().now().to_msg()
        if self.controller.latest_pose is not None and self.controller.lrs_priority_points:
            start = copy.deepcopy(self.controller.latest_pose)
            start.header = path.header
            path.poses.append(start)
        for point in self.controller.lrs_priority_points:
            pose = self.controller.make_pose(point.x, point.y)
            pose.header = path.header
            path.poses.append(pose)
        self.controller.lrs_priority_route_pub.publish(path)

    def publish_history(self):
        if self.controller.history is None or self.controller.gmrf is None:
            return
        grid = copy.deepcopy(self.controller.map_msg)
        grid.header.stamp = self.controller.get_clock().now().to_msg()
        history_values = self.controller.history.history_mean[self.controller.gmrf.free]
        data = np.full(
            (self.controller.gmrf.height, self.controller.gmrf.width), -1, dtype=np.int16)
        data[self.controller.gmrf.free] = np.rint(
            100.0 * np.clip(history_values, 0.0, 1.0)).astype(np.int16)
        grid.data = data.astype(np.int8).ravel().tolist()
        self.controller.history_map_pub.publish(grid)

        log_template, interpolated, log_free = self.controller.display_field(
            history_values)
        log_values = self.controller.logarithmic_display(interpolated)
        self.controller.history_log_map_pub.publish(self.controller.occupancy_grid(
            log_template, log_values, log_free))

        recent_values = self.controller.history.history_recent[self.controller.gmrf.free]
        recent_template, recent_interpolated, recent_free = (
            self.controller.display_field(recent_values))
        recent_log_values = self.controller.logarithmic_display(recent_interpolated)
        self.controller.history_recent_log_map_pub.publish(self.controller.occupancy_grid(
            recent_template, recent_log_values, recent_free))

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.controller.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_history_revisit'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.28
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.points = [
            Point(x=record['x'], y=record['y'], z=0.18)
            for record in self.controller.active_history_regions]
        self.controller.history_revisit_pub.publish(marker)

        peaks = Marker()
        peaks.header.frame_id = 'map'
        peaks.header.stamp = self.controller.get_clock().now().to_msg()
        peaks.ns = 'gas_mapping_history_confirmed_peaks'
        peaks.id = 0
        peaks.type = Marker.POINTS
        peaks.action = Marker.ADD
        peaks.scale.x = peaks.scale.y = 0.36
        latest_sequence = self.controller.history.confirmed_event_sequence
        events = sorted(
            self.controller.history.confirmed_peak_events.values(),
            key=lambda event: (
                int(event['sequence']), str(event['event_id'])))
        for event in events:
            x, y = self.controller.history.cell_center(event['row'], event['col'])
            peaks.points.append(Point(x=x, y=y, z=0.24))
            if int(event['sequence']) == latest_sequence:
                peaks.colors.append(ColorRGBA(
                    r=0.0, g=1.0, b=0.55, a=1.0))
            else:
                peaks.colors.append(ColorRGBA(
                    r=1.0, g=0.35, b=0.0, a=0.9))
        self.controller.history_confirmed_peaks_pub.publish(peaks)

    def publish_empty_hrs_route(self):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.controller.get_clock().now().to_msg()
        self.controller.hrs_route_pub.publish(path)

    def value_color(self, value, logarithmic=False):
        return self.controller.mapping_publishers.value_color(
            value, self.controller.log_concentration_min, logarithmic)

    def publish_lrs_status(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.controller.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_lrs_status'
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.30
        marker.points = [
            Point(x=goal.pose.position.x, y=goal.pose.position.y, z=0.15)
            for goal in self.controller.lrs_goals]
        marker.colors = [self.controller.value_color(value)
                         for value in self.controller.lrs_status_values]
        self.controller.lrs_status_pub.publish(marker)
        log_marker = copy.deepcopy(marker)
        log_marker.ns = 'gas_mapping_lrs_status_log'
        log_marker.colors = [self.controller.value_color(value, logarithmic=True)
                             for value in self.controller.lrs_status_values]
        self.controller.lrs_status_log_pub.publish(log_marker)

    def publish_candidates(self, candidates):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.controller.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_hrs_candidates'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.35
        marker.points = [Point(x=cell.x, y=cell.y, z=0.14)
                         for cell in candidates]
        marker.colors = [ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)
                         for _ in candidates]
        self.controller.hrs_candidates_pub.publish(marker)

    def publish_hrs_route(self, plan):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.controller.get_clock().now().to_msg()
        if self.controller.latest_pose is not None:
            start = copy.deepcopy(self.controller.latest_pose)
            start.header = path.header
            path.poses.append(start)
        for cell in plan.cells:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = cell.x
            pose.pose.position.y = cell.y
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.controller.hrs_route_pub.publish(path)

    def publish_hrs_status(self):
        if (self.controller.gmrf is None or
                self.controller.navigation_goal_variables is None):
            return
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.controller.get_clock().now().to_msg()
        marker.ns = 'gas_mapping_hrs_status'
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = 0.12
        marker.scale.z = 0.02
        for variable in sorted(self.controller.navigation_goal_variables):
            x, y = self.controller.gmrf.cell_center(variable)
            marker.points.append(Point(x=x, y=y, z=0.10))
            if variable in self.controller.unreachable_variables:
                marker.colors.append(ColorRGBA(
                    r=0.0, g=0.0, b=0.0, a=1.0))
            else:
                marker.colors.append(self.controller.value_color(
                    self.controller.hrs_values.get(variable)))
        self.controller.hrs_status_pub.publish(marker)


__all__ = [
    'ControllerVisualization', 'MappingPublishers', 'PUBLISHER_SPECS']
