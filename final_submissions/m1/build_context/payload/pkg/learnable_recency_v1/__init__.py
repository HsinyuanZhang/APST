"""Learnable recency-bias family (CABLE / FoX essence) for RIFT temporal stacks."""

from .config import LearnableRecencyConfig, dataset_config, scaled_half_lives
from .cpu_temporal import CpuLearnableRecencyRuntime
from .temporal import LearnableRecencyState, LearnableRecencyTemporal
from .wrap import (
    LearnableJointM1ConcatDecoder,
    LearnableRiftConcatDecoder,
    LearnableRiftDecoder,
    LearnableRiftStreamDecoder,
    attach_learnable_recency,
    install_temporal,
    new_parameter_names,
    trainable_new_parameter_count,
)

__all__ = [
    "CpuLearnableRecencyRuntime",
    "LearnableJointM1ConcatDecoder",
    "LearnableRecencyConfig",
    "LearnableRecencyState",
    "LearnableRecencyTemporal",
    "LearnableRiftConcatDecoder",
    "LearnableRiftDecoder",
    "LearnableRiftStreamDecoder",
    "attach_learnable_recency",
    "dataset_config",
    "install_temporal",
    "new_parameter_names",
    "scaled_half_lives",
    "trainable_new_parameter_count",
]
