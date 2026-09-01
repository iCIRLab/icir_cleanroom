"""Navigation Workflow adapter."""

import copy
import math
import time
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


class NavigationWorkflow:
    METHODS = ['active_target','send_current_goal','goal_orientation','navigation_succeeded','finish_dwell','navigation_failed','record_hrs_failure','advance_after_target']

    def __init__(self, controller):
        self.controller = controller

    def active_target(self):
        if self.controller.phase == 'LRS':
            if self.controller.returning:
                return self.controller.lrs_goals[0]
            if self.controller.current_index < len(self.controller.lrs_goals):
                return self.controller.lrs_goals[self.controller.current_index]
        if (self.controller.phase == 'HRS_NAVIGATION' and
                self.controller.active_hrs_route is not None and
                self.controller.current_index < len(self.controller.active_hrs_route.cells)):
            cell = self.controller.active_hrs_route.cells[self.controller.current_index]
            target = PoseStamped()
            target.header.frame_id = 'map'
            target.pose.position.x = cell.x
            target.pose.position.y = cell.y
            target.pose.orientation.w = 1.0
            return target
        return None

    def send_current_goal(self):
        target = self.controller.active_target()
        if target is None:
            if self.controller.phase == 'LRS':
                self.controller.finish_lrs_navigation()
            elif self.controller.phase == 'HRS_NAVIGATION':
                self.controller.finish_hrs_cycle()
            return

        goal_pose = copy.deepcopy(target)
        goal_pose.header.stamp = self.controller.get_clock().now().to_msg()
        goal_pose.pose.orientation = self.controller.goal_orientation(target)
        if self.controller.phase == 'LRS':
            label = ('시작 LRS 포인트 복귀' if self.controller.returning else
                     f'LRS [{self.controller.current_index + 1}/{len(self.controller.lrs_goals)}]')
        else:
            label = (
                f'HRS cycle {self.controller.hrs_cycles + 1} '
                f'[{self.controller.current_index + 1}/'
                f'{len(self.controller.active_hrs_route.cells)}]')
        self.controller.get_logger().info(
            f'{label} -> ({target.pose.position.x:.2f}, '
            f'{target.pose.position.y:.2f})')
        goal_generation = self.controller.navigation_manager.issue_goal()
        self.controller.nav2.send(
            goal_pose, goal_generation, self.controller.navigation_succeeded,
            self.controller.navigation_failed)

    def goal_orientation(self, target):
        following = None
        if self.controller.phase == 'LRS' and not self.controller.returning and self.controller.lrs_goals:
            following = self.controller.lrs_goals[
                (self.controller.current_index + 1) % len(self.controller.lrs_goals)]
        elif (self.controller.phase == 'HRS_NAVIGATION' and
              self.controller.active_hrs_route and
              self.controller.current_index + 1 < len(self.controller.active_hrs_route.cells)):
            cell = self.controller.active_hrs_route.cells[self.controller.current_index + 1]
            following = PoseStamped()
            following.pose.position.x = cell.x
            following.pose.position.y = cell.y
        pose = copy.deepcopy(target.pose.orientation)
        if following is None:
            pose.w = 1.0
            return pose
        yaw = math.atan2(
            following.pose.position.y - target.pose.position.y,
            following.pose.position.x - target.pose.position.x)
        pose.x = pose.y = 0.0
        pose.z = math.sin(yaw / 2.0)
        pose.w = math.cos(yaw / 2.0)
        return pose

    def navigation_succeeded(self):
        self.controller.retry = 0
        if self.controller.phase == 'LRS' and self.controller.returning:
            self.controller.finish_lrs_navigation()
            return
        target_variable = None
        if (self.controller.phase == 'HRS_NAVIGATION' and
                self.controller.active_hrs_route is not None):
            target_variable = int(
                self.controller.active_hrs_route.cells[self.controller.current_index].variable)
        dwell_generation = self.controller.measurement_manager.start(
            copy.deepcopy(self.controller.latest_pose), self.controller.phase, target_variable)
        dwell = (self.controller.lrs_dwell_seconds if self.controller.phase == 'LRS'
                 else self.controller.hrs_dwell_seconds)
        self.controller.dwell_timer = self.controller.create_timer(
            float(dwell),
            lambda generation=dwell_generation: self.controller.finish_dwell(generation))

    def finish_dwell(self, generation=None):
        if generation is None and self.controller.measurement_manager.active is not None:
            generation = self.controller.measurement_manager.active.generation
        result = self.controller.measurement_manager.finish(generation)
        if result is None:
            return
        timer = self.controller.dwell_timer
        if timer is not None:
            timer.cancel()
            self.controller.destroy_timer(timer)
        self.controller.dwell_timer = None
        if result.phase != self.controller.phase:
            return
        if result.mean is None:
            self.controller.get_logger().warning(
                '농도 표본이 없어 이 포인트를 미측정으로 남깁니다')
            if self.controller.phase == 'HRS_NAVIGATION':
                self.controller.record_hrs_failure()
        else:
            value = result.mean
            pose = result.pose.pose
            row, col = self.controller.gmrf.set_observation(
                pose.position.x, pose.position.y, value)
            variable = self.controller.gmrf.variable_at(int(row), int(col))
            measurement = self.controller.measurement_manager.commit(
                result, variable, time.time())
            self.controller.history.update(
                int(row), int(col), value, self.controller.phase,
                measurement.timestamp)
            self.controller.publish_history()
            self.controller.record_event_measurement(variable, value)
            if self.controller.phase == 'LRS':
                self.controller.lrs_status_values[self.controller.current_index] = value
                first_hazard = self.controller.lrs_manager.record_measurement(
                    value, self.controller.hazard_threshold)
                if first_hazard:
                    self.controller.get_logger().warning(
                        f'위험 가스 감지: value={value:.4f} >= '
                        f'{float(self.controller.hazard_threshold):.4f}; '
                        '현재 LRS 순찰 완료 후 HRS로 전환합니다')
                if self.controller.lap_hazard_detected:
                    self.controller.hazard_pub.publish(Bool(data=True))
                self.controller.update_gmrf('LRS measurement')
            else:
                planned = self.controller.active_hrs_route.cells[self.controller.current_index]
                if variable == planned.variable:
                    self.controller.hrs_manager.record_success(planned.variable)
                else:
                    self.controller.get_logger().warning(
                        f'목표 셀 ({planned.row},{planned.col})과 실제 측정 '
                        f'셀 ({int(row)},{int(col)})이 달라 목표를 '
                        '미측정으로 유지합니다')
                    self.controller.record_hrs_failure()
                self.controller.hrs_gmrf_dirty = True
                if self.controller.update_gmrf(
                        f'{self.controller.phase} measurement '
                        f'[{self.controller.current_index + 1}/'
                        f'{len(self.controller.active_hrs_route.cells)}]',
                        compare_with_cg=False):
                    self.controller.hrs_gmrf_dirty = False
                else:
                    self.controller.get_logger().warning(
                        'HRS 지점별 GaBP 갱신을 다음 측정 또는 '
                        '배치 종료 시 재시도합니다')
            self.controller.publish_measurements()
            self.controller.get_logger().info(
                f'측정 완료: phase={self.controller.phase}, '
                f'pose=({pose.position.x:.3f},{pose.position.y:.3f}), '
                f'mean={value:.4f}, samples={result.sample_count}, '
                f'cell=({row},{col})')
            if (self.controller.phase == 'HRS_NAVIGATION' and
                    self.controller.confirm_hrs_response(
                        variable, value, measurement.timestamp)):
                return
        self.controller.advance_after_target()

    def navigation_failed(self, reason):
        self.controller.retry += 1
        if self.controller.retry <= int(self.controller.max_retries):
            self.controller.get_logger().warning(
                f'이동 실패({reason}), 재시도 '
                f'{self.controller.retry}/{self.controller.max_retries}')
            self.controller.send_current_goal()
            return
        self.controller.retry = 0
        self.controller.get_logger().warning(
            f'이동 실패({reason}), 미측정으로 남기고 다음 포인트 진행')
        if self.controller.phase == 'LRS' and self.controller.returning:
            self.controller.finish_lrs_navigation()
            return
        if self.controller.phase == 'HRS_NAVIGATION':
            self.controller.record_hrs_failure()
        self.controller.advance_after_target()

    def record_hrs_failure(self):
        cell = self.controller.active_hrs_route.cells[self.controller.current_index]
        count, unreachable = self.controller.hrs_manager.record_failure(
            cell.variable, self.controller.max_cell_failures)
        if unreachable:
            self.controller.get_logger().error(
                f'HRS cell ({cell.row},{cell.col})가 {count}회 실패하여 '
                '도달 불가로 기록되었습니다')

    def advance_after_target(self):
        self.controller.current_index += 1
        if self.controller.phase == 'LRS':
            self.controller.publish_lrs_status()
            if self.controller.current_index >= len(self.controller.lrs_goals):
                self.controller.returning = True
            self.controller.send_current_goal()
        elif self.controller.phase == 'HRS_NAVIGATION':
            self.controller.publish_hrs_status()
            self.controller.send_current_goal()


__all__ = ['NavigationWorkflow']
