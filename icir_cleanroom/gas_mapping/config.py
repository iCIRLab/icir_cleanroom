"""Typed, immutable configuration for the gas-mapping controller."""

from dataclasses import asdict, dataclass
import math


HRS_PLANNER_MODES = ('reward_ordered_exact', 'paper_exhaustive_milp')


@dataclass(frozen=True)
class NavigationConfig:
    lrs_dwell_seconds: float = 1.0
    hrs_dwell_seconds: float = 2.0
    max_retries: int = 2
    max_cell_failures: int = 3


@dataclass(frozen=True)
class GmrfConfig:
    observation_variance_scale: float = 0.01
    observation_variance_floor: float = 0.001
    smoothness_precision: float = 1.0
    cardinal_weight: float = 0.75
    diagonal_weight: float = 0.25
    background_precision: float = 1.0
    background_mean: float = 0.0
    gabp_max_iterations: int = 500
    gabp_tolerance: float = 1.0e-6
    gabp_damping: float = 0.5
    gabp_retry_damping: float = 0.25
    gabp_cg_warning_tolerance: float = 1.0e-4


@dataclass(frozen=True)
class DisplayConfig:
    estimate_resolution: float = 0.1
    log_concentration_min: float = -4.0


@dataclass(frozen=True)
class HrsConfig:
    hrs_ucb_k: float = 1.0
    hrs_distance_weight: float = 0.01
    hrs_response_threshold: float = 0.50
    hrs_max_cycles_per_alert: int = 10
    hrs_candidate_count: int = 1
    hrs_visit_count: int = 1
    hrs_speed: float = 5.0
    hrs_update_seconds: float = 50.0
    hrs_combination_time_limit: float = 5.0
    hrs_planner_mode: str = 'reward_ordered_exact'
    hazard_threshold: float = 0.2


@dataclass(frozen=True)
class LrsConfig:
    lrs_history_replace_radius: float = 5.0
    lrs_priority_candidate_count: int = 15
    lrs_priority_count: int = 6
    lrs_route_length_ratio: float = 1.10
    lrs_reward_recurrence_weight: float = 0.40
    lrs_reward_severity_weight: float = 0.20
    lrs_reward_staleness_weight: float = 0.25
    lrs_reward_uncertainty_weight: float = 0.15


@dataclass(frozen=True)
class HistoryConfig:
    history_top_k: int = 3
    history_merge_radius: float = 5.0
    history_recent_alpha: float = 0.5
    history_event_half_life: float = 10.0
    history_event_kernel_sigma: float = 5.0
    history_file: str = '~/.ros/icir_cleanroom/gas_history.json'


@dataclass(frozen=True)
class ControllerConfig:
    navigation: NavigationConfig
    gmrf: GmrfConfig
    display: DisplayConfig
    hrs: HrsConfig
    lrs: LrsConfig
    history: HistoryConfig
    source_advance_timeout_seconds: float = 5.0

    @classmethod
    def defaults(cls):
        return cls(
            navigation=NavigationConfig(), gmrf=GmrfConfig(),
            display=DisplayConfig(), hrs=HrsConfig(), lrs=LrsConfig(),
            history=HistoryConfig())

    @classmethod
    def from_mapping(cls, values):
        def build(model):
            defaults = asdict(model())
            return model(**{
                name: values.get(name, default)
                for name, default in defaults.items()})

        config = cls(
            navigation=build(NavigationConfig),
            gmrf=build(GmrfConfig),
            display=build(DisplayConfig),
            hrs=build(HrsConfig),
            lrs=build(LrsConfig),
            history=build(HistoryConfig),
            source_advance_timeout_seconds=values.get(
                'source_advance_timeout_seconds', 5.0))
        config.validate()
        return config

    def flat_values(self):
        values = {}
        for group in (
                self.navigation, self.gmrf, self.display, self.hrs,
                self.lrs, self.history):
            values.update(asdict(group))
        values['source_advance_timeout_seconds'] = (
            self.source_advance_timeout_seconds)
        return values

    def validate(self):
        nav, gmrf, hrs = self.navigation, self.gmrf, self.hrs
        lrs, history = self.lrs, self.history
        if min(float(nav.lrs_dwell_seconds),
               float(nav.hrs_dwell_seconds)) < 0.0:
            raise ValueError('dwell time must be non-negative')
        if int(hrs.hrs_candidate_count) <= 0 or int(hrs.hrs_visit_count) <= 0:
            raise ValueError('HRS candidate and visit counts must be positive')
        if not 0.0 <= float(hrs.hazard_threshold) <= 1.0:
            raise ValueError('hazard_threshold must be in [0, 1]')
        if int(history.history_top_k) <= 0:
            raise ValueError('history_top_k must be positive')
        if (int(lrs.lrs_priority_candidate_count) <= 0 or
                int(lrs.lrs_priority_count) <= 0):
            raise ValueError('LRS priority candidate/count must be positive')
        if (int(lrs.lrs_priority_count) >
                int(lrs.lrs_priority_candidate_count)):
            raise ValueError(
                'lrs_priority_count cannot exceed candidate count')
        if (not math.isfinite(float(lrs.lrs_route_length_ratio)) or
                float(lrs.lrs_route_length_ratio) < 1.0):
            raise ValueError(
                'lrs_route_length_ratio must be finite and at least 1')
        reward_weights = [
            float(lrs.lrs_reward_recurrence_weight),
            float(lrs.lrs_reward_severity_weight),
            float(lrs.lrs_reward_staleness_weight),
            float(lrs.lrs_reward_uncertainty_weight),
        ]
        if (not all(math.isfinite(value) and value >= 0.0
                    for value in reward_weights) or
                sum(reward_weights) <= 0.0):
            raise ValueError(
                'LRS reward weights must be finite, non-negative, and '
                'have a positive sum')
        if (not 0.0 < float(history.history_recent_alpha) <= 1.0 or
                not math.isfinite(float(history.history_recent_alpha))):
            raise ValueError('history_recent_alpha must be in (0, 1]')
        if (not math.isfinite(float(history.history_event_half_life)) or
                float(history.history_event_half_life) <= 0.0 or
                not math.isfinite(float(history.history_event_kernel_sigma)) or
                float(history.history_event_kernel_sigma) <= 0.0):
            raise ValueError(
                'history event half-life and kernel sigma must be positive')
        if float(hrs.hrs_ucb_k) < 0.0:
            raise ValueError('hrs_ucb_k must be non-negative')

        if (not math.isfinite(float(hrs.hrs_distance_weight)) or
                float(hrs.hrs_distance_weight) < 0.0):
            raise ValueError(
                'hrs_distance_weight must be finite and non-negative')

        if not 0.0 <= float(hrs.hrs_response_threshold) <= 1.0:
            raise ValueError('hrs_response_threshold must be in [0, 1]')
        if int(hrs.hrs_max_cycles_per_alert) <= 0:
            raise ValueError('hrs_max_cycles_per_alert must be positive')
        if (float(history.history_merge_radius) < 0.0 or
                float(lrs.lrs_history_replace_radius) < 0.0):
            raise ValueError('history radii must be non-negative')
        if (float(hrs.hrs_speed) <= 0.0 or
                float(hrs.hrs_update_seconds) <= 0.0):
            raise ValueError('HRS speed and update period must be positive')
        if not 0.0 < float(gmrf.gabp_damping) <= 1.0:
            raise ValueError('GaBP damping must be in (0, 1]')
        if not 0.0 < float(gmrf.gabp_retry_damping) <= 1.0:
            raise ValueError('GaBP retry damping must be in (0, 1]')
        if str(hrs.hrs_planner_mode) not in HRS_PLANNER_MODES:
            raise ValueError(
                f'hrs_planner_mode must be one of {HRS_PLANNER_MODES}')
        timeout = float(self.source_advance_timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError('source_advance_timeout_seconds must be positive')


DEFAULT_CONTROLLER_PARAMETERS = ControllerConfig.defaults().flat_values()


def declare_controller_config(node):
    """Declare all public ROS parameters and return their typed snapshot."""
    values = {}
    for name, default in DEFAULT_CONTROLLER_PARAMETERS.items():
        node.declare_parameter(name, default)
        values[name] = node.get_parameter(name).value
    return ControllerConfig.from_mapping(values)


__all__ = [
    'ControllerConfig', 'DisplayConfig', 'GmrfConfig', 'HistoryConfig',
    'HrsConfig', 'LrsConfig', 'NavigationConfig',
    'DEFAULT_CONTROLLER_PARAMETERS', 'declare_controller_config',
]
