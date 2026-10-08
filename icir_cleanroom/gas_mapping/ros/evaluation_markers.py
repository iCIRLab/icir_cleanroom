"""Display the logged benchmark reference without feeding navigation policy."""
import json
import math

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from ..application.benchmark import valid_xy


def evaluation_markers(row, stamp):
    """A retained message is a complete snapshot, including stale-marker removal."""
    markers = []

    def marker(namespace, kind, color, identifier=0):
        item = Marker()
        item.header.frame_id = 'map'
        item.header.stamp = stamp
        item.ns, item.id = namespace, identifier
        item.type, item.action = kind, Marker.ADD
        item.pose.orientation.w = 1.0
        item.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
        markers.append(item)
        return item

    clear = marker('evaluation_clear', Marker.CUBE_LIST, (1.0, 1.0, 1.0))
    clear.action = Marker.DELETEALL
    result = MarkerArray()
    result.markers = markers
    if (not row or str(row.get('evaluation_reference_adjusted')).lower() != 'true'
            or row.get('evaluation_reason') not in ('available', 'missing_estimate')):
        return result
    candidates = json.loads(row['evaluation_reference_candidates'])
    if not candidates or any(not valid_xy(xy) for xy in candidates):
        return result

    pink, cyan = (1.0, 0.2, 0.85), (0.0, 0.85, 1.0)
    cells = marker('evaluation_candidates', Marker.CUBE_LIST, pink)
    cells.scale.x = cells.scale.y = 0.46
    cells.scale.z = 0.04
    cells.color.a = 0.75
    cells.points = [Point(x=float(x), y=float(y), z=0.28) for x, y in candidates]
    if str(row.get('evaluation_available')).lower() == 'true':
        estimate = (float(row['estimated_source_x']), float(row['estimated_source_y']))
        nearest = min(candidates, key=lambda xy: math.dist(estimate, xy))
        line = marker('evaluation_error', Marker.LINE_LIST, cyan)
        line.scale.x = 0.045
        line.points = [Point(x=float(x), y=float(y), z=0.36) for x, y in (estimate, nearest)]
        target = marker('evaluation_used_reference', Marker.SPHERE, cyan)
        target.pose.position = Point(x=float(nearest[0]), y=float(nearest[1]), z=0.38)
        target.scale.x = target.scale.y = target.scale.z = 0.14
    # ROS sequence assignment may copy its input; assign after all additions.
    result.markers = markers
    return result
