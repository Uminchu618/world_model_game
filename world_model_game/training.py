"""Training utilities for world-model agents in the signaling game."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch

from .config import TrainingConfig
from .environment import Observation, TagPrisonersDilemma
from .models import ActionOutput, ObservationSummary, WorldModelPolicy


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class StepLog:
    observation: Observation
    action_output: ActionOutput
    reward: float
    opponent_action: int
    opponent_tag: int


@dataclass
class EpisodeLog:
    steps_agent0: List[StepLog]
    steps_agent1: List[StepLog]
    total_rewards: List[float]
    history: List[dict]


def _build_payload(observation: Observation) -> ObservationSummary:
    return ObservationSummary(
        step=observation.step,
        opponent_action=observation.opponent_action,
        opponent_tag=observation.opponent_tag,
    )


class SelfPlayTrainer:
    """Train two agents sharing parameters in self-play."""

    def __init__(self, env: TagPrisonersDilemma, policy: WorldModelPolicy, config: TrainingConfig):
        self.env = env
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        set_seed(config.seed)

    def _compute_returns(self, rewards: List[float]) -> List[float]:
        returns = []
        cumulative = 0.0
        for reward in reversed(rewards):
            cumulative = reward + self.config.gamma * cumulative
            returns.append(cumulative)
        return list(reversed(returns))

    def _episode_loss(self, trajectory: List[StepLog], returns: List[float]) -> Dict[str, torch.Tensor]:
        policy_loss = torch.tensor(0.0, device=self.policy.device)
        value_loss = torch.tensor(0.0, device=self.policy.device)
        entropy = torch.tensor(0.0, device=self.policy.device)

        for step_log, ret in zip(trajectory, returns):
            value = step_log.action_output.value
            advantage = torch.tensor(ret, device=self.policy.device) - value
            policy_loss = policy_loss - (step_log.action_output.logp_action + step_log.action_output.logp_tag) * advantage.detach()
            value_loss = value_loss + advantage.pow(2)
            entropy = entropy + step_log.action_output.entropy_action + step_log.action_output.entropy_tag

        return {
            "policy": policy_loss,
            "value": value_loss,
            "entropy": entropy,
        }

    def run_episode(self, sample: bool = True) -> EpisodeLog:
        observations = self.env.reset()
        states = [self.policy.initial_state(), self.policy.initial_state()]

        step_logs = [[], []]
        cumulative_rewards = [0.0, 0.0]

        for step in range(self.env.max_steps):
            payloads = [_build_payload(observations[0]), _build_payload(observations[1])]

            outputs = [
                self.policy.step(observations[i].vector, states[i], step, payloads[i], sample=sample)
                for i in range(2)
            ]

            actions = [int(outputs[0].action.item()), int(outputs[1].action.item())]
            tags = [int(outputs[0].tag.item()), int(outputs[1].tag.item())]

            next_observations, rewards, done, info = self.env.step(actions, tags)

            for i in range(2):
                opponent_index = 1 - i
                step_logs[i].append(
                    StepLog(
                        observation=observations[i],
                        action_output=outputs[i],
                        reward=rewards[i],
                        opponent_action=actions[opponent_index],
                        opponent_tag=tags[opponent_index],
                    )
                )
                states[i] = outputs[i].new_state
                cumulative_rewards[i] += rewards[i]

            observations = next_observations

            if done:
                break

        return EpisodeLog(
            steps_agent0=step_logs[0],
            steps_agent1=step_logs[1],
            total_rewards=cumulative_rewards,
            history=self.env.get_history(),
        )

    def train(self) -> List[Dict[str, float]]:
        logs: List[Dict[str, float]] = []

        for episode in range(1, self.config.episodes + 1):
            episode_log = self.run_episode(sample=True)

            returns_agent0 = self._compute_returns([step.reward for step in episode_log.steps_agent0])
            returns_agent1 = self._compute_returns([step.reward for step in episode_log.steps_agent1])

            losses0 = self._episode_loss(episode_log.steps_agent0, returns_agent0)
            losses1 = self._episode_loss(episode_log.steps_agent1, returns_agent1)

            policy_loss = losses0["policy"] + losses1["policy"]
            value_loss = losses0["value"] + losses1["value"]
            entropy = losses0["entropy"] + losses1["entropy"]

            loss = policy_loss + self.config.value_coeff * value_loss - (
                self.config.entropy_coeff_action + self.config.entropy_coeff_tag
            ) * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
            self.optimizer.step()

            avg_reward = sum(episode_log.total_rewards) / 2.0
            cooperation_rate = sum(1 for entry in episode_log.history if entry["actions"][0] == 0) / max(
                1, len(episode_log.history)
            )

            logs.append({
                "episode": episode,
                "avg_reward": avg_reward,
                "total_reward_agent0": episode_log.total_rewards[0],
                "total_reward_agent1": episode_log.total_rewards[1],
                "cooperation_rate_agent0": cooperation_rate,
                "loss": float(loss.item()),
            })

            if episode % self.config.log_interval == 0:
                print(
                    f"Episode {episode}: avg_reward={avg_reward:.3f} "
                    f"loss={float(loss.item()):.4f} coop_rate={cooperation_rate:.3f}"
                )

        return logs
