"""Source Transition Workflow adapter."""

from std_srvs.srv import Trigger


class SourceTransitionWorkflow:
    METHODS = ['start_source_transition','source_advance_response','source_advance_timed_out','cancel_source_advance_timeout','finish_source_transition','return_to_lrs']

    def __init__(self, controller):
        self.controller = controller

    def start_source_transition(self, reason):
        if (self.controller.phase == 'SOURCE_TRANSITION' or
                self.controller.source_advance_future is not None):
            self.controller.get_logger().warning(
                '가스원 전환 요청이 이미 진행 중이므로 중복 요청을 '
                f'무시합니다: {reason}')
            return

        self.controller.get_logger().info(
            f'HRS 대응 임계값 검출 후 가스원 전환: {reason}')
        self.controller.commit_history_snapshot(
            f'confirmed hazard event ending after LRS lap {self.controller.lrs_lap}')
        self.controller.active_hrs_route = None
        self.controller.publish_empty_hrs_route()
        self.controller.clear_hrs_candidates()
        transition_generation = self.controller.orchestrator.begin_source_transition(
            reason)
        self.controller.publish_phase('SOURCE_TRANSITION')

        if not self.controller.source_advance_client.service_is_ready():
            self.controller.get_logger().warning(
                '/gas_mapping/source/advance 서비스를 사용할 수 없어 '
                '기존 가스원으로 LRS를 계속합니다')
            self.controller.finish_source_transition(False, 'service unavailable')
            return

        try:
            future = self.controller.source_advance_client.call_async(Trigger.Request())
        except Exception as error:  # keep patrol alive on middleware errors
            self.controller.get_logger().warning(
                f'가스원 전환 요청 실패({error!r}); '
                '기존 가스원으로 LRS를 계속합니다')
            self.controller.finish_source_transition(False, 'request failed')
            return
        self.controller.source_advance_future = future
        self.controller.source_advance_timeout_timer = self.controller.create_timer(
            float(self.controller.source_advance_timeout_seconds),
            lambda generation=transition_generation:
            self.controller.source_advance_timed_out(generation),
            clock=self.controller.source_transition_clock)
        future.add_done_callback(
            lambda completed, generation=transition_generation:
            self.controller.source_advance_response(completed, generation))

    def source_advance_response(self, future, generation=None):
        if (generation is not None and generation !=
                self.controller.source_transition_state.generation):
            self.controller.get_logger().warning(
                '완료된 이전 가스원 전환 generation을 무시합니다')
            return
        if future is not self.controller.source_advance_future:
            self.controller.get_logger().warning(
                '완료된 이전 가스원 전환 응답을 무시합니다')
            return
        self.controller.source_advance_future = None
        self.controller.cancel_source_advance_timeout()
        try:
            response = future.result()
        except Exception as error:  # keep patrol alive on service errors
            self.controller.get_logger().warning(
                f'가스원 전환 서비스 실패({error!r}); '
                '기존 가스원으로 LRS를 계속합니다')
            self.controller.finish_source_transition(False, 'service call failed')
            return
        if response is None or not response.success:
            message = ('empty response' if response is None else
                       str(response.message))
            self.controller.get_logger().warning(
                f'가스원 전환 거부({message}); '
                '기존 가스원으로 LRS를 계속합니다')
            self.controller.finish_source_transition(False, 'service rejected request')
            return
        message = str(response.message)
        if message.startswith('changed=true;'):
            source_changed = True
        elif message.startswith('changed=false;'):
            source_changed = False
        else:
            self.controller.get_logger().warning(
                '가스원 전환 응답에 changed 상태가 없어 기존 가스원 '
                f'이벤트로 보수적으로 처리합니다: {message!r}')
            source_changed = False
        self.controller.get_logger().info(
            f'가스원 전환 서비스 완료: changed={source_changed}; '
            '새 LRS 순찰을 시작합니다')
        self.controller.finish_source_transition(
            source_changed, 'source service acknowledged')

    def source_advance_timed_out(self, generation=None):
        if (generation is not None and generation !=
                self.controller.source_transition_state.generation):
            return
        future = self.controller.source_advance_future
        if future is None:
            self.controller.cancel_source_advance_timeout()
            return
        self.controller.source_advance_future = None
        self.controller.cancel_source_advance_timeout()
        future.cancel()
        self.controller.get_logger().warning(
            f'가스원 전환 서비스가 '
            f'{float(self.controller.source_advance_timeout_seconds):.1f}초 안에 '
            '응답하지 않아 기존 가스원으로 LRS를 '
            '계속합니다')
        self.controller.finish_source_transition(False, 'service timeout')

    def cancel_source_advance_timeout(self):
        timer = self.controller.source_advance_timeout_timer
        self.controller.source_advance_timeout_timer = None
        if timer is None:
            return
        timer.cancel()
        self.controller.destroy_timer(timer)

    def finish_source_transition(self, source_changed, outcome):
        self.controller.cancel_source_advance_timeout()
        reason = self.controller.source_transition_reason
        self.controller.source_transition_reason = ''
        self.controller.orchestrator.finish_source_transition(
            source_changed, timed_out=(outcome == 'service timeout'))
        status = ('new source event' if source_changed else
                  'existing source event')
        self.controller.start_lrs_lap(
            f'{reason}; source_transition={outcome}; using {status}')

    def return_to_lrs(self, reason):
        self.controller.get_logger().info(f'HRS 종료 후 LRS 복귀: {reason}')
        self.controller.commit_history_snapshot(
            f'hazard event ending after LRS lap {self.controller.lrs_lap}')
        self.controller.active_hrs_route = None
        self.controller.publish_empty_hrs_route()
        self.controller.clear_hrs_candidates()

        self.controller.start_lrs_lap(reason)


__all__ = ['SourceTransitionWorkflow']
