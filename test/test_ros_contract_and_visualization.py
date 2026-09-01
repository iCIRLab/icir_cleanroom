"""Verify the preserved ROS surface and point/color alignment."""

from pathlib import Path
from types import SimpleNamespace

import yaml
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker

from icir_cleanroom.gas_mapping.config import DEFAULT_CONTROLLER_PARAMETERS
from icir_cleanroom.gas_mapping.mapping.kmeans_partition import (
    ClusterCell, KMeansPartition)
from icir_cleanroom.gas_mapping.ros.lrs_planner_node import (
    LrsPathPlannerNode)
from icir_cleanroom.gas_mapping.ros.publishers import (
    ControllerVisualization, MappingPublishers, PUBLISHER_SPECS)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class CapturePublisher:
    def __init__(self):
        self.message = None

    def publish(self, message):
        self.message = message


def test_manifest_matches_controller_publishers_and_parameter_count():
    manifest = yaml.safe_load(
        (PACKAGE_ROOT / 'test' / 'ros_interface_contract.yaml').read_text(
            encoding='utf-8'))
    expected_topics = set(manifest['topics']['controller']['publishers'])
    actual_topics = {topic for _, _, topic in PUBLISHER_SPECS}
    assert actual_topics == expected_topics
    assert len(PUBLISHER_SPECS) == 23
    assert len(DEFAULT_CONTROLLER_PARAMETERS) == 47


def test_scripts_are_logic_free_compatible_entrypoints():
    for executable in (
            'gas_empty_map_node.py', 'lrs_path_planner_node.py',
            'gas_mapping_controller_node.py'):
        source = (PACKAGE_ROOT / 'scripts' / executable).read_text(
            encoding='utf-8')
        assert 'def ' not in source
        assert 'from icir_cleanroom.gas_mapping.ros.' in source


def test_hrs_marker_keeps_each_point_aligned_with_one_color():
    captured = CapturePublisher()
    gmrf = SimpleNamespace(
        resolution=0.5,
        var_cells=[(0, 0), (0, 1), (1, 0)],
        cell_center=lambda variable: (float(variable), float(variable + 1)))
    clock = SimpleNamespace(now=lambda: SimpleNamespace(
        to_msg=lambda: Time()))
    controller = SimpleNamespace(
        gmrf=gmrf, navigation_goal_variables={0, 1},
        unreachable_variables={1}, hrs_values={0: 0.2},
        hrs_status_pub=captured, get_clock=lambda: clock,
        log_concentration_min=-4.0)
    controller.mapping_publishers = SimpleNamespace(
        value_color=MappingPublishers.value_color)
    visualization = ControllerVisualization(controller)
    controller.value_color = visualization.value_color

    visualization.publish_hrs_status()

    assert captured.message.type == Marker.CUBE_LIST
    assert captured.message.scale.x == 0.12
    assert captured.message.scale.y == 0.12
    assert captured.message.pose.orientation.w == 1.0
    assert len(captured.message.points) == 2
    assert len(captured.message.colors) == 2
    assert captured.message.colors[1].r == 0.0
    assert captured.message.colors[1].g == 0.0
    assert captured.message.colors[1].b == 0.0


def test_lrs_cluster_marker_uses_gmrf_cell_scale_and_cluster_colors():
    captured = CapturePublisher()
    clock = SimpleNamespace(now=lambda: SimpleNamespace(
        to_msg=lambda: Time()))
    planner = object.__new__(LrsPathPlannerNode)
    planner.clusters_pub = captured
    planner.gmrf_map = SimpleNamespace(
        info=SimpleNamespace(resolution=0.5))
    planner.get_clock = lambda: clock
    partition = KMeansPartition(
        cells=(
            ClusterCell(0, 0, 0.0, 0.0, 0),
            ClusterCell(0, 1, 0.5, 0.0, 0),
            ClusterCell(1, 0, 0.0, 0.5, 1)),
        centroids=((0.25, 0.0), (0.0, 0.5)), inertia=0.125)

    planner.publish_clusters(partition)

    assert captured.message.type == Marker.CUBE_LIST
    assert captured.message.scale.x == 0.45
    assert captured.message.scale.y == 0.45
    assert captured.message.pose.orientation.w == 1.0
    assert len(captured.message.points) == 3
    assert len(captured.message.colors) == 3
    assert captured.message.colors[0] == captured.message.colors[1]
    assert captured.message.colors[0] != captured.message.colors[2]
