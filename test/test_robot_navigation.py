"""Robot-centred navigation and observation attribution, independent of the sensor mount."""
from concurrent.futures import Future
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformException

from icir_cleanroom.gas_mapping.application.hrs import HrsManager
from icir_cleanroom.gas_mapping.application.measurement import MeasurementManager
from icir_cleanroom.gas_mapping.application.navigation import NavigationManager
from icir_cleanroom.gas_mapping.mapping.gmrf import GmrfGrid
from icir_cleanroom.gas_mapping.navigation_profile import load_navigation_settings
from icir_cleanroom.gas_mapping.ros import nav2_client
from icir_cleanroom.gas_mapping.ros.controller_node import GasMappingControllerNode
from icir_cleanroom.gas_mapping.ros.navigation_workflow import NavigationWorkflow


def pose(x, y, yaw=0.0):
    result = PoseStamped()
    result.header.frame_id = 'map'
    result.header.stamp.sec = 10
    result.pose.position.x, result.pose.position.y = float(x), float(y)
    result.pose.orientation.z = math.sin(yaw/2)
    result.pose.orientation.w = math.cos(yaw/2)
    return result


@pytest.fixture
def client(monkeypatch):
    transform = TransformStamped()
    transform.header.frame_id = 'map'
    transform.header.stamp.sec = 10
    transform.child_frame_id = 'base_link'
    transform.transform.translation.x = .25
    transform.transform.translation.y = .25
    transform.transform.rotation.w = 1.
    requests, lookups = [], []

    def send_goal(goal):
        future = Future()
        requests.append((goal, future))
        return future

    def lookup(target, source, stamp):
        lookups.append((target, source))
        assert (target, source) == ('map', 'base_link')
        return transform

    monkeypatch.setattr(nav2_client, 'ActionClient', lambda *args: NS(
        server_is_ready=lambda: True, send_goal_async=send_goal))
    monkeypatch.setattr(nav2_client, 'Buffer', lambda **kwargs: NS(lookup_transform=lookup))
    monkeypatch.setattr(nav2_client, 'TransformListener', lambda *args: None)
    node = NS(latest_pose=None,
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=10*10**9)))
    node.pose_callback = lambda p: GasMappingControllerNode.pose_callback(node, p)
    manager = NavigationManager()
    adapter = nav2_client.Nav2Client(node, manager)
    node.nav2 = adapter
    return NS(adapter=adapter, manager=manager, requests=requests,
              transform=transform, lookups=lookups, node=node)


def start(client, target=None, arrived=None):
    success, failures = [], []
    client.adapter.send(target or pose(.25, .25), client.manager.issue_goal(),
                        arrived or (lambda: success.append(True)), failures.append)
    return success, failures


def nav2_result(client, status=GoalStatus.STATUS_SUCCEEDED):
    result = Future()
    handle = NS(accepted=True, get_result_async=lambda: result,
                cancel_goal_async=lambda: None)
    client.requests[-1][1].set_result(handle)
    result.set_result(NS(status=status))


@pytest.mark.parametrize('yaw', [0., math.pi/2, math.pi, -math.pi/2])
def test_goal_is_robot_centre_without_rotated_sensor_offset(client, yaw):
    target = pose(-3., 1.5, yaw)
    start(client, target)
    assert client.requests[0][0].pose == target
    assert not client.lookups  # No sensor transform is needed to send a goal.


def test_robot_pose_is_map_base_link_and_is_copied(client):
    actual = client.adapter.robot_pose()
    assert actual == pose(.25, .25)
    actual.pose.orientation.w = 0.
    assert client.transform.transform.rotation.w == 1.
    assert client.adapter.server_is_ready()


@pytest.mark.parametrize('seconds', [9, 11])
def test_stale_or_future_robot_pose_cannot_start_or_complete_measurement(client, seconds):
    client.transform.header.stamp.sec = seconds
    assert not client.adapter.server_is_ready()
    success, failures = start(client)
    nav2_result(client)
    assert not success and failures == ['robot pose unavailable after Nav2 arrival']


def test_missing_robot_tf_does_not_fall_back_to_sensor_pose(client):
    def unavailable(*args):
        raise TransformException('missing base transform')
    client.adapter._tf.lookup_transform = unavailable
    client.node.latest_pose = pose(.70, .41)  # A previous position must not survive a TF gap.
    GasMappingControllerNode.poll_robot_pose(client.node)
    assert client.node.latest_pose is None
    assert not client.adapter.server_is_ready()


def test_invalid_robot_position_is_not_recorded(client):
    client.transform.transform.translation.x = float('nan')
    assert client.adapter.robot_pose() is None


def test_nav2_success_refreshes_robot_pose_without_waiting_for_sensor(client):
    success, failures = start(client)
    nav2_result(client)
    assert success == [True] and not failures
    assert client.node.latest_pose == pose(.25, .25)


@pytest.mark.parametrize('lead', [.001, .096, .15, .2])
def test_gazebo_tf_ahead_of_last_clock_tick_still_allows_measurement(client, lead):
    client.transform.header.stamp.nanosec = round(lead*1e9)
    assert client.adapter.server_is_ready()
    success, failures = start(client)
    nav2_result(client)
    assert success == [True] and not failures
    assert client.node.latest_pose.pose.position.x == .25


def test_pose_too_far_ahead_is_still_rejected(client):
    client.transform.header.stamp.nanosec = 201_000_000
    assert client.adapter.robot_pose() is None


def test_cancelled_action_cannot_update_robot_pose_or_complete(client):
    success, failures = start(client)
    client.manager.invalidate_goal()
    nav2_result(client)
    assert not success and not failures and client.node.latest_pose is None


def test_nav2_failure_does_not_begin_measurement(client):
    success, failures = start(client)
    nav2_result(client, GoalStatus.STATUS_ABORTED)
    assert not success and failures == ['status=6']


def test_hrs_sensor_value_is_assigned_to_robot_cell_without_false_failure(client, map_message_factory):
    c = client.node
    g = GmrfGrid(map_message_factory(width=5, height=5, resolution=.5))
    measured = MeasurementManager()
    manager = HrsManager()
    target = next(cell for cell in manager.build_candidates(g, set(), .2)
                  if cell.variable == g.nearest_variable(.25, .25))
    # On AMC, the physical sensor at body + (.45, .16) lies in a different cell.
    sensor_variable = g.nearest_variable(.70, .41)
    assert sensor_variable != target.variable
    history, visited, failures = [], [], []
    c.__dict__.update(
        phase='HRS_NAVIGATION', returning=False, retry=0,
        gmrf=g, active_hrs_target=target, hrs_manager=manager,
        measurement_manager=measured, hrs_dwell_seconds=2.,
        create_timer=lambda *args: NS(cancel=lambda: None), destroy_timer=lambda timer: None,
        history=NS(update=lambda *args: history.append(args)),
        publish_history=lambda: None, record_event_measurement=lambda *args: visited.append(args),
        publish_measurements=lambda: None, update_gmrf=lambda *args, **kwargs: True,
        record_hrs_failure=lambda: failures.append(True), advance_after_target=lambda: None,
        get_logger=lambda: NS(info=lambda msg: None, warning=lambda msg: None))
    workflow = NavigationWorkflow(c)
    c.finish_dwell = workflow.finish_dwell
    start(client, pose(.25, .25), workflow.navigation_succeeded)
    nav2_result(client)
    # Keep sensor readings unchanged; only their collection coordinates use the body.
    measured.add_sample(.8)
    measured.add_sample(.84)
    workflow.finish_dwell()
    observation = measured.measurements[0]
    assert (observation.pose.x, observation.pose.y) == (.25, .25)
    assert observation.value == pytest.approx(.82)
    assert observation.variable == target.variable
    assert g.obs_weight[target.variable] > 0 and g.obs_weight[sensor_variable] == 0
    assert history[0][:2] == (0, 0)
    assert visited[0][0] == target.variable and not failures


@pytest.mark.parametrize('profile', ['empty_fast', 'aws_warehouse_safe', 'cleanroom_amc_safe'])
def test_profiles_use_robot_centre_and_do_not_require_sensor_heading(profile):
    package = Path(__file__).resolve().parents[1]
    config, _ = load_navigation_settings(
        package/'config/nav2_params.yaml', package/f'config/navigation/{profile}.yaml')
    checker = config['controller_server']['ros__parameters']['general_goal_checker']
    assert checker['xy_goal_tolerance'] == .15
    assert checker['yaw_goal_tolerance'] == math.pi
    for name in ('local_costmap', 'global_costmap'):
        assert config[name][name]['ros__parameters']['robot_base_frame'] == nav2_client.Nav2Client.BASE_FRAME
