"""Verify the preserved ROS surface and point/color alignment."""

from pathlib import Path
from types import SimpleNamespace

import yaml
from builtin_interfaces.msg import Time

from icir_cleanroom.gas_mapping.config import DEFAULT_CONTROLLER_PARAMETERS
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
    assert len(DEFAULT_CONTROLLER_PARAMETERS) == 46


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
        var_cells=[(0, 0), (0, 1), (1, 0)],
        cell_center=lambda variable: (float(variable), float(variable + 1)))
    clock = SimpleNamespace(now=lambda: SimpleNamespace(
        to_msg=lambda: Time()))
    controller = SimpleNamespace(
        gmrf=gmrf, unreachable_variables={1}, hrs_values={0: 0.2},
        hrs_status_pub=captured, get_clock=lambda: clock,
        log_concentration_min=-4.0)
    controller.mapping_publishers = SimpleNamespace(
        value_color=MappingPublishers.value_color)
    visualization = ControllerVisualization(controller)
    controller.value_color = visualization.value_color

    visualization.publish_hrs_status()

    assert len(captured.message.points) == 3
    assert len(captured.message.colors) == 3
    assert captured.message.colors[1].r == 0.0
    assert captured.message.colors[1].g == 0.0
    assert captured.message.colors[1].b == 0.0
