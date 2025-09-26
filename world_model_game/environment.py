"""タグ付き反復囚人のジレンマ環境の定義。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .config import GameConfig


@dataclass
class Observation:
    """エージェントに返される観測を保持するデータ構造。"""

    vector: torch.Tensor
    opponent_action: Optional[int]
    opponent_tag: Optional[int]
    step: int


class TagPrisonersDilemma:
    """タグ選択を伴う反復囚人のジレンマ環境。"""

    def __init__(self, config: GameConfig):
        """設定を受け取り、環境の内部状態を初期化する。"""
        self.config = config
        self.device = torch.device("cpu")
        self.payoff_matrix = config.payoff_matrix
        self.max_steps = config.max_steps
        self.num_tags = config.num_tags
        self.num_agents = config.num_agents

        if self.num_agents % 2 != 0:
            raise ValueError("Number of agents must be even for random pairing.")

        # 観測ベクトルにおける各情報のスライス位置
        self._action_slice = slice(0, 2)
        self._tag_slice = slice(2, 2 + self.num_tags)
        self._time_fraction_index = 2 + self.num_tags
        self._bias_index = 3 + self.num_tags

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(config.seed)

        self.reset()

    @property
    def observation_size(self) -> int:
        """観測ベクトルの全長を返す。"""
        return 4 + self.num_tags

    def _build_observation(
        self, opponent_action: Optional[int], opponent_tag: Optional[int], step: int
    ) -> Observation:
        """相手の行動・タグ情報から観測ベクトルを構成する。"""
        vec = torch.zeros(self.observation_size, dtype=torch.float32, device=self.device)

        if opponent_action is not None:
            vec[self._action_slice.start + opponent_action] = 1.0

        if opponent_tag is not None:
            vec[self._tag_slice.start + opponent_tag] = 1.0

        if self.max_steps > 0:
            vec[self._time_fraction_index] = float(step) / float(self.max_steps)
        vec[self._bias_index] = 1.0

        return Observation(vec, opponent_action, opponent_tag, step)

    def override_observation_tag(self, observation: Observation, new_tag: Optional[int]) -> Observation:
        """観測のタグ成分だけを書き換えた複製を返し、介入実験に用いる。"""

        vector = observation.vector.clone()
        vector[self._tag_slice] = 0.0
        if new_tag is not None:
            vector[self._tag_slice.start + new_tag] = 1.0

        return Observation(vector, observation.opponent_action, new_tag, observation.step)

    def reset(self) -> List[Observation]:
        """環境を初期状態に戻し、各エージェントへの初期観測を生成する。"""
        self.step_index = 0
        self._history: List[dict] = []
        self._last_actions: List[Optional[int]] = [None] * self.num_agents
        self._last_tags: List[Optional[int]] = [None] * self.num_agents

        return [self._build_observation(None, None, 0) for _ in range(self.num_agents)]

    def _compute_pair_rewards(self, action_a: int, action_b: int) -> Tuple[float, float]:
        """行動ペアに応じた報酬を利得表から取得する。"""
        payoff = self.payoff_matrix[(action_a, action_b)]
        return float(payoff[0]), float(payoff[1])

    def _sample_pairs(self) -> List[Tuple[int, int]]:
        """ラウンド毎にエージェントをランダムにペアリングする。"""
        ordering = torch.randperm(self.num_agents, generator=self._rng).tolist()
        return [(ordering[i], ordering[i + 1]) for i in range(0, self.num_agents, 2)]

    def step(
        self, actions: Sequence[int], tags: Sequence[int]
    ) -> Tuple[List[Observation], Tuple[float, ...], bool, dict]:
        """全エージェントの行動とタグを受け取り、次状態と報酬を計算する。"""
        if len(actions) != self.num_agents or len(tags) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions and tags.")

        pairs = self._sample_pairs()
        rewards: List[float] = [0.0 for _ in range(self.num_agents)]
        opponents: List[Optional[int]] = [None for _ in range(self.num_agents)]
        next_observations: List[Observation] = [
            self._build_observation(None, None, self.step_index + 1)
            for _ in range(self.num_agents)
        ]

        pair_history: List[dict] = []

        self.step_index += 1
        self._last_actions = list(actions)
        self._last_tags = list(tags)

        for agent_a, agent_b in pairs:
            reward_a, reward_b = self._compute_pair_rewards(actions[agent_a], actions[agent_b])
            rewards[agent_a] = reward_a
            rewards[agent_b] = reward_b

            opponents[agent_a] = agent_b
            opponents[agent_b] = agent_a

            next_observations[agent_a] = self._build_observation(
                actions[agent_b], tags[agent_b], self.step_index
            )
            next_observations[agent_b] = self._build_observation(
                actions[agent_a], tags[agent_a], self.step_index
            )

            pair_history.append(
                {
                    "agents": (agent_a, agent_b),
                    "actions": (actions[agent_a], actions[agent_b]),
                    "tags": (tags[agent_a], tags[agent_b]),
                    "rewards": (reward_a, reward_b),
                }
            )

        history_entry = {"step": self.step_index, "pairs": pair_history}
        self._history.append(history_entry)

        done = self.step_index >= self.max_steps

        cooperation_count = 0
        for entry in self._history:
            for pair in entry["pairs"]:
                cooperation_count += sum(1 for action in pair["actions"] if action == 0)
        total_actions = max(1, len(self._history) * self.num_agents)
        info = {
            "history": list(self._history),
            "cooperation_rate": cooperation_count / total_actions,
            "pairings": pairs,
            "opponents": tuple(opponents),
        }

        return next_observations, tuple(rewards), done, info

    def get_history(self) -> List[dict]:
        """これまでのペア情報と行動履歴をコピーして返す。"""
        return list(self._history)
