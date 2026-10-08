"""Optional instrumentation hooks. Source position never enters HRS policy."""
import json
import math
import time

from ..application.benchmark import EvaluationDomain


def evaluation_domain(controller):
    gmrf = getattr(controller, "gmrf", None)
    return EvaluationDomain.snapshot(
        getattr(controller, "navigation_goal_geometry", None),
        getattr(controller, "navigation_goal_mask", None),
        getattr(gmrf, "geometry", None))


def source_position(controller, marker):
    previous = getattr(controller, 'hrs_log_source', None)
    xy = (float(marker.pose.position.x), float(marker.pose.position.y))
    controller.hrs_log_source = (xy if marker.action == 0 and marker.header.frame_id == 'map'
                                 and all(math.isfinite(v) for v in xy) else None)
    if controller.hrs_log_source != previous:
        publish = getattr(controller, 'publish_evaluation_reference', None)
        if publish is not None:
            publish()  # A new/missing source must not retain the preceding reference.


def source_truth(controller, message):
    """Record-only ground truth; malformed payloads leave the fields blank."""
    try:
        payload = json.loads(message.data)
    except (TypeError, ValueError):
        payload = None
    controller.hrs_log_source_truth = payload if isinstance(payload, dict) else None


def publish_evaluation(controller, recorder, to_cell_center=None, row=None):
    """Use the same frozen reference and estimate coordinates as the CSV writer."""
    publish = getattr(controller, 'publish_evaluation_reference', None)
    if publish is None:
        return
    if row is None and recorder.active is not None:
        estimate = recorder.current_estimate(to_cell_center)
        xy = (None if estimate['estimated_source_x'] is None else
              (estimate['estimated_source_x'], estimate['estimated_source_y']))
        row = dict(estimate, **recorder.active['evaluation_reference'].evaluate(
            xy, controller.hrs_log_source))
    publish(row)


def record_lrs(controller, kind, **details):
    recorder = getattr(controller, 'hrs_run_log', None)
    if recorder is None:
        return
    try:
        ros = controller.get_clock().now().nanoseconds * 1.e-9
        wall = time.monotonic()
        pose = getattr(controller, 'latest_pose', None)
        xy = None if pose is None else (pose.pose.position.x, pose.pose.position.y)
        if kind == 'start':
            recorder.start_lrs(ros, wall, event_id=controller.current_event_id,
                               lrs_lap=controller.lrs_lap,
                               method_id=getattr(controller, 'method', 'M3'), robot_xy=xy,
                               evaluation_domain=evaluation_domain(controller))
            publish = getattr(controller, 'publish_evaluation_reference', None)
            if publish is not None:
                publish()
            controller.get_logger().info(f'LRS results directory: {recorder.directory}')
        elif kind == 'measurement':
            gmrf = getattr(controller, 'gmrf', None)
            to_cell_center = None if gmrf is None else (
                lambda x, y: gmrf.geometry.cell_center(
                    *gmrf.geometry.world_to_cell(x, y)))
            return recorder.lrs_measurement(
                ros, wall, to_cell_center=to_cell_center, **details)
        elif kind == 'finish':
            row = recorder.finish_lrs(
                ros, wall, robot_xy=xy,
                hazard_detected=controller.lap_hazard_detected, **details)
            if row is not None:
                controller.get_logger().info(
                    f'LRS result: lap={row["lrs_lap"]}, '
                    f'lrs_distance_m={row["lrs_distance_m"]}, outcome={row["outcome"]}; '
                    f'file={recorder.directory / "lrs_runs.csv"}')
            return row
    except (OSError, ValueError, RuntimeError) as error:
        controller.get_logger().error(f'LRS result logging failed: {error}')
    return None


def record_hrs(controller, kind, **details):
    recorder = getattr(controller, 'hrs_run_log', None)
    if recorder is None:
        return
    try:
        ros = controller.get_clock().now().nanoseconds * 1.e-9
        wall = time.monotonic()
        if kind == 'start':
            pose = getattr(controller, 'latest_pose', None)
            xy = None if pose is None else (pose.pose.position.x, pose.pose.position.y)
            recorder.start(ros, wall, event_id=controller.current_event_id,
                           lrs_lap=controller.lrs_lap,
                           source=controller.hrs_log_source,
                           robot_xy=xy,
                           parameters=dict(controller.config.flat_values(),
                                           measurement_position_basis='robot_base_link'),
                           evaluation_domain=evaluation_domain(controller))
            publish_evaluation(controller, recorder)
            controller.get_logger().info(f'HRS results directory: {recorder.directory}')
        elif kind == 'initial_complete':
            recorder.initial_complete(ros, wall)
        elif kind in ('measurement', 'estimate', 'finish'):
            pose = controller.latest_pose
            xy = None if pose is None else (pose.pose.position.x, pose.pose.position.y)
            gmrf = getattr(controller, 'gmrf', None)
            to_cell_center = None if gmrf is None else (
                lambda x, y: gmrf.geometry.cell_center(*gmrf.geometry.world_to_cell(x, y)))
            if kind == 'measurement':
                recorder.measurement(ros, wall, source=controller.hrs_log_source,
                                     to_cell_center=to_cell_center, **details)
                publish_evaluation(controller, recorder, to_cell_center)
                return None
            if kind == 'estimate':
                estimate = recorder.final_estimate(to_cell_center)
                recorder.event('final_estimate_selected', ros, wall, **(estimate or {}))
                recorder.record_estimate(ros, wall, event='final_estimate_selected',
                                         source=controller.hrs_log_source)
                publish_evaluation(controller, recorder, to_cell_center)
                return estimate
            row = recorder.finish(ros, wall, robot_xy=xy,
                                  source_end=controller.hrs_log_source,
                                  source_truth=getattr(
                                      controller, 'hrs_log_source_truth', None),
                                  to_cell_center=to_cell_center, **details)
            if row is not None:
                publish_evaluation(controller, recorder, row=row)
                controller.get_logger().info(
                    f'HRS result: initial={row["initial_seconds"]}, '
                    f'search={row["search_seconds"]}, total={row["total_seconds"]} '
                    f'({row["time_basis"]}), '
                    f'total_distance_m={row["total_distance_m"]}, '
                    f'lrs_distance_m={row["lrs_distance_m"]}, '
                    f'hrs_distance_m={row["hrs_distance_m"]}, '
                    f'source=({row["source_x"]},{row["source_y"]}), '
                    f'estimated=({row["estimated_source_x"]},{row["estimated_source_y"]}), '
                    f'source_position_error={row["source_position_error"]}, '
                    f'reason={row["termination_reason"]}; file={recorder.directory / "runs.csv"}')
            return row
        else:
            recorder.event(kind, ros, wall, **details)
    except (OSError, ValueError, RuntimeError) as error:
        controller.get_logger().error(f'HRS result logging failed: {error}')
    return None
