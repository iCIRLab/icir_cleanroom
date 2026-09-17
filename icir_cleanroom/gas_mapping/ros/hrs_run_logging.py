"""Optional instrumentation hooks. Source position never enters HRS policy."""
import math
import time


def source_position(controller, marker):
    if marker.action != 0 or marker.header.frame_id != 'map':
        controller.hrs_log_source = None
        return
    xy = (float(marker.pose.position.x), float(marker.pose.position.y))
    controller.hrs_log_source = xy if all(math.isfinite(v) for v in xy) else None


def record_hrs(controller, kind, **details):
    recorder = getattr(controller, 'hrs_run_log', None)
    if recorder is None:
        return
    try:
        ros = controller.get_clock().now().nanoseconds * 1.e-9
        wall = time.monotonic()
        if kind == 'start':
            recorder.start(ros, wall, event_id=controller.current_event_id,
                           lrs_lap=controller.lrs_lap,
                           source=controller.hrs_log_source,
                           parameters=controller.config.flat_values())
            controller.get_logger().info(f'HRS results directory: {recorder.directory}')
        elif kind == 'initial_complete':
            recorder.initial_complete(ros, wall)
        elif kind == 'measurement':
            recorder.measurement(ros, wall, **details)
        elif kind == 'finish':
            pose = controller.latest_pose
            xy = None if pose is None else (pose.pose.position.x, pose.pose.position.y)
            row = recorder.finish(ros, wall, robot_xy=xy,
                                  source_end=controller.hrs_log_source, **details)
            if row is not None:
                controller.get_logger().info(
                    f'HRS result: initial={row["initial_seconds"]}, '
                    f'search={row["search_seconds"]}, total={row["total_seconds"]} '
                    f'({row["time_basis"]}), outcome={row["outcome"]}, '
                    f'source=({row["source_x"]},{row["source_y"]}), '
                    f'estimated=({row["estimated_source_x"]},{row["estimated_source_y"]}), '
                    f'reason={row["termination_reason"]}; file={recorder.directory / "runs.csv"}')
        else:
            recorder.event(kind, ros, wall, **details)
    except (OSError, ValueError, RuntimeError) as error:
        controller.get_logger().error(f'HRS result logging failed: {error}')
