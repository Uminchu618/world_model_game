"""Configuration dataclasses for the cooperative signaling world model experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class GameConfig:
    """Configuration for the tag-based iterated prisoner's dilemma environment."""

    num_tags: int = 4
    max_steps: int = 20
    seed: int = 0
    payoff_matrix: Dict[Tuple[int, int], Tuple[float, float]] = field(
        default_factory=lambda: {
            (0, 0): (3.0, 3.0),  # mutual cooperation
            (0, 1): (0.0, 5.0),  # agent cooperates, opponent defects
            (1, 0): (5.0, 0.0),  # agent defects, opponent cooperates
            (1, 1): (1.0, 1.0),  # mutual defection
        }
    )


@dataclass
class TrainingConfig:
    """Configuration parameters for self-play training."""

    episodes: int = 4000
    gamma: float = 0.96
    learning_rate: float = 3e-4
    hidden_dim: int = 128
    attention_dim: int = 64
    attention_window: int = 10
    entropy_coeff_action: float = 0.01
    entropy_coeff_tag: float = 0.01
    value_coeff: float = 0.5
    grad_clip: float = 1.0
    log_interval: int = 100
    device: str = "cpu"
    seed: int = 42


@dataclass
class AnalysisConfig:
    """Configuration for analysis utilities such as probing and interventions."""

    probe_epochs: int = 200
    probe_learning_rate: float = 1e-2
    evaluation_episodes: int = 50
    intervention_episodes: int = 20
    intervention_step: int = 5
    intervention_tag: int = 0
