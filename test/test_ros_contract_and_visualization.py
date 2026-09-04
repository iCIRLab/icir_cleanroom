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
    assert len(PUBLISHER_SPECS) == 30
    assert len(DEFAULT_CONTROLLER_PARAMETERS) == 44


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


def test_field_value_labels_show_one_numeric_value_per_gmrf_cell():
    gmrf = SimpleNamespace(
        resolution=0.5,
        var_cells=[(0, 0), (0, 1)],
        cell_center=lambda variable: (float(variable), float(variable + 1)))
    clock = SimpleNamespace(now=lambda: SimpleNamespace(
        to_msg=lambda: Time()))
    visualization = ControllerVisualization(SimpleNamespace(
        gmrf=gmrf, get_clock=lambda: clock))

    labels = visualization.field_value_labels(
        [0.1236, 0.9], 'test_ucb_values', (1.0, 1.0, 0.0))

    assert len(labels.markers) == 2
    assert [marker.text for marker in labels.markers] == ['0.124', '0.900']
    assert all(marker.type == Marker.TEXT_VIEW_FACING
               for marker in labels.markers)
    assert all(marker.ns == 'test_ucb_values' for marker in labels.markers)

    sparse_labels = visualization.field_value_labels(
        [float('nan'), 0.42], 'test_sparse_values', (1.0, 0.45, 0.0))
    assert len(sparse_labels.markers) == 1
    assert sparse_labels.markers[0].id == 1
    assert sparse_labels.markers[0].text == '0.420'


def test_ucb_rviz_maps_and_value_labels_are_disabled_by_default():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'rviz' / 'cleanroom_empty.rviz').read_text(
            encoding='utf-8'))
    displays = {
        display.get('Name'): display
        for display in config['Visualization Manager']['Displays']}

    expected = {
        'GMRFEstimateLinearValues': '/gas_mapping/estimate_labels',
        'GMRFVarianceValues': '/gas_mapping/variance_labels',
        'MeasuredGasMap': '/gas_mapping/measurements/grid',
        'MeasuredGasValues': '/gas_mapping/measurements/value_labels',
        'HRSPotentialUCB': '/gas_mapping/hrs/ucb',
        'HRSPotentialUCBValues': '/gas_mapping/hrs/ucb_labels',
        'HRSPotentialDDUCB': '/gas_mapping/hrs/dd_ucb',
        'HRSPotentialDDUCBValues': '/gas_mapping/hrs/dd_ucb_labels',
    }
    for name, topic in expected.items():
        assert displays[name]['Enabled'] is False
        assert displays[name]['Topic']['Value'] == topic


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
