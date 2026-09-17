"""Typed, immutable configuration for the gas-mapping controller."""

from dataclasses import asdict, dataclass
import math


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
    method: str = 'M3'
    spiral_step_cells: int = 1
    spiral_recenter_epsilon: float = 0.0
    hrs_search_patience: int = 3
    hrs_search_min_relative_improvement: float = 0.05
    repeat_after_hrs: bool = True
    hrs_results_directory: str = '~/.ros/icir_cleanroom/hrs_results'
    hrs_ucb_k: float = 0.2
    hrs_distance_weight: float = 0.03
    hrs_max_cycles_per_alert: int = 10
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
    save_history: bool = True
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
            values.update({
                name: value for name, value in asdict(group).items()
                if value is not None})
        values['source_advance_timeout_seconds'] = (
            self.source_advance_timeout_seconds)
        return values

    def validate(self):
        nav, gmrf, hrs = self.navigation, self.gmrf, self.hrs
        lrs, history = self.lrs, self.history
        from .application.hrs_methods import METHODS
        if hrs.method not in METHODS:
            raise ValueError('method must be M1 through M7')
        if (isinstance(hrs.spiral_step_cells, bool) or
                not isinstance(hrs.spiral_step_cells, int) or hrs.spiral_step_cells < 1):
            raise ValueError('spiral_step_cells must be a positive integer')
        if (not math.isfinite(hrs.spiral_recenter_epsilon) or
                hrs.spiral_recenter_epsilon < 0):
            raise ValueError('spiral_recenter_epsilon must be finite and nonnegative')
        if (isinstance(hrs.hrs_search_patience, bool) or
                not isinstance(hrs.hrs_search_patience, int) or hrs.hrs_search_patience < 1):
            raise ValueError('hrs_search_patience must be a positive integer')
        if (not math.isfinite(hrs.hrs_search_min_relative_improvement) or
                hrs.hrs_search_min_relative_improvement < 0):
            raise ValueError(
                'hrs_search_min_relative_improvement must be finite and nonnegative')
        if not isinstance(hrs.repeat_after_hrs, bool):
            raise ValueError('repeat_after_hrs must be a boolean')
        if not isinstance(hrs.hrs_results_directory, str) or not hrs.hrs_results_directory.strip():
            raise ValueError('hrs_results_directory must be a nonempty path')
        if min(float(nav.lrs_dwell_seconds),
               float(nav.hrs_dwell_seconds)) < 0.0:
            raise ValueError('dwell time must be non-negative')
        if not 0.0 <= float(hrs.hazard_threshold) <= 1.0:
            raise ValueError('hazard_threshold must be in [0, 1]')
        if not isinstance(history.save_history, bool):
            raise ValueError('save_history must be a boolean')
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
        if (not math.isfinite(float(hrs.hrs_ucb_k)) or
                float(hrs.hrs_ucb_k) < 0.0):
            raise ValueError('hrs_ucb_k must be finite and non-negative')

        if (not math.isfinite(float(hrs.hrs_distance_weight)) or
                float(hrs.hrs_distance_weight) < 0.0):
            raise ValueError(
                'hrs_distance_weight must be finite and non-negative')

        if int(hrs.hrs_max_cycles_per_alert) <= 0:
            raise ValueError('hrs_max_cycles_per_alert must be positive')
        if (float(history.history_merge_radius) < 0.0 or
                float(lrs.lrs_history_replace_radius) < 0.0):
            raise ValueError('history radii must be non-negative')
        if not 0.0 < float(gmrf.gabp_damping) <= 1.0:
            raise ValueError('GaBP damping must be in (0, 1]')
        if not 0.0 < float(gmrf.gabp_retry_damping) <= 1.0:
            raise ValueError('GaBP retry damping must be in (0, 1]')
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
