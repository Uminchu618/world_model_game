"""Environment definition for the tag-based iterated prisoner's dilemma."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .config import GameConfig


@dataclass
class Observation:
    """Observation returned to an agent."""

    vector: torch.Tensor
    opponent_action: Optional[int]
    opponent_tag: Optional[int]
    step: int


class TagPrisonersDilemma:
    """Iterated prisoner's dilemma with dynamically selected tags."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.device = torch.device("cpu")
        self.payoff_matrix = config.payoff_matrix
        self.max_steps = config.max_steps
        self.num_tags = config.num_tags

        # Observation layout slices
        self._action_slice = slice(0, 2)
        self._tag_slice = slice(2, 2 + self.num_tags)
        self._time_fraction_index = 2 + self.num_tags
        self._bias_index = 3 + self.num_tags

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(config.seed)

        self.reset()

    @property
    def observation_size(self) -> int:
        return 4 + self.num_tags

    def _build_observation(
        self, opponent_action: Optional[int], opponent_tag: Optional[int], step: int
    ) -> Observation:
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
        """Return a copy of the observation with a different perceived opponent tag."""

        vector = observation.vector.clone()
        vector[self._tag_slice] = 0.0
        if new_tag is not None:
            vector[self._tag_slice.start + new_tag] = 1.0

        return Observation(vector, observation.opponent_action, new_tag, observation.step)

    def reset(self) -> List[Observation]:
        self.step_index = 0
        self._history: List[dict] = []
        self._last_actions: List[Optional[int]] = [None, None]
        self._last_tags: List[Optional[int]] = [None, None]

        return [self._build_observation(None, None, 0) for _ in range(2)]

    def _compute_rewards(self, actions: Sequence[int]) -> Tuple[float, float]:
        payoff = self.payoff_matrix[(actions[0], actions[1])]
        return float(payoff[0]), float(payoff[1])

    def step(
        self, actions: Sequence[int], tags: Sequence[int]
    ) -> Tuple[List[Observation], Tuple[float, float], bool, dict]:
        if len(actions) != 2 or len(tags) != 2:
            raise ValueError("Two actions and two tags must be provided.")

        rewards = self._compute_rewards(actions)

        self.step_index += 1
        self._last_actions = list(actions)
        self._last_tags = list(tags)

        next_observations = [
            self._build_observation(actions[1], tags[1], self.step_index),
            self._build_observation(actions[0], tags[0], self.step_index),
        ]

        history_entry = {
            "step": self.step_index,
            "actions": tuple(actions),
            "tags": tuple(tags),
            "rewards": rewards,
        }
        self._history.append(history_entry)

        done = self.step_index >= self.max_steps

        cooperation_count = sum(1 for entry in self._history if entry["actions"][0] == 0)
        info = {
            "history": list(self._history),
            "cooperation_rate": cooperation_count / max(1, len(self._history)),
        }

        return next_observations, rewards, done, info

    def get_history(self) -> List[dict]:
        return list(self._history)
