"""協調的なシグナリング世界モデル実験で利用する設定群。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class GameConfig:
    """タグ付き反復囚人のジレンマ環境の設定をまとめたデータクラス。"""

    num_tags: int = 4
    max_steps: int = 20
    num_agents: int = 2
    seed: int = 0
    payoff_matrix: Dict[Tuple[int, int], Tuple[float, float]] = field(
        default_factory=lambda: {
            (0, 0): (3.0, 3.0),  # 相互協調
            (0, 1): (0.0, 5.0),  # 自分が協調・相手が裏切り
            (1, 0): (5.0, 0.0),  # 自分が裏切り・相手が協調
            (1, 1): (1.0, 1.0),  # 相互裏切り
        }
    )


@dataclass
class TrainingConfig:
    """自己対戦学習に用いるハイパーパラメータ群。"""

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
    """プロービングや介入解析に関する設定値。"""

    probe_epochs: int = 200
    probe_learning_rate: float = 1e-2
    evaluation_episodes: int = 50
    intervention_episodes: int = 20
    intervention_step: int = 5
    intervention_tag: int = 0
