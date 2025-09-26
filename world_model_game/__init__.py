"""協調的世界モデル・シグナリングゲームの中核パッケージ。"""

from .config import GameConfig, TrainingConfig, AnalysisConfig
from .environment import TagPrisonersDilemma, Observation
from .models import WorldModelPolicy, AgentState, ObservationSummary
from .training import SelfPlayTrainer

__all__ = [
    "AgentState",
    "AnalysisConfig",
    "GameConfig",
    "Observation",
    "ObservationSummary",
    "SelfPlayTrainer",
    "TagPrisonersDilemma",
    "TrainingConfig",
    "WorldModelPolicy",
]
