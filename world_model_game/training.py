"""Training utilities for world-model agents in the signaling game."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    opponent_index: Optional[int]
    opponent_action: Optional[int]
    opponent_tag: Optional[int]


@dataclass
class EpisodeLog:
    step_logs: List[List[StepLog]]
    total_rewards: List[float]
    history: List[dict]


def _build_payload(observation: Observation) -> ObservationSummary:
    return ObservationSummary(
        step=observation.step,
        opponent_action=observation.opponent_action,
        opponent_tag=observation.opponent_tag,
    )


class SelfPlayTrainer:
    """Train a population of agents sharing parameters in self-play."""

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
        policy_loss = torch.zeros((), device=self.policy.device)
        value_loss = torch.zeros((), device=self.policy.device)
        entropy = torch.zeros((), device=self.policy.device)

        for step_log, ret in zip(trajectory, returns):
            value = step_log.action_output.value
            target = torch.tensor(ret, device=self.policy.device)
            advantage = target - value
            policy_loss = policy_loss - (
                step_log.action_output.logp_action + step_log.action_output.logp_tag
            ) * advantage.detach()
            value_loss = value_loss + advantage.pow(2)
            entropy = entropy + step_log.action_output.entropy_action + step_log.action_output.entropy_tag

        return {
            "policy": policy_loss,
            "value": value_loss,
            "entropy": entropy,
        }

    def run_episode(self, sample: bool = True) -> EpisodeLog:
        observations = self.env.reset()
        num_agents = self.env.num_agents
        states = [self.policy.initial_state() for _ in range(num_agents)]

        step_logs: List[List[StepLog]] = [[] for _ in range(num_agents)]
        cumulative_rewards = [0.0 for _ in range(num_agents)]

        for step in range(self.env.max_steps):
            payloads = [_build_payload(observation) for observation in observations]

            outputs = [
                self.policy.step(observations[i].vector, states[i], step, payloads[i], sample=sample)
                for i in range(num_agents)
            ]

            actions = [int(output.action.item()) for output in outputs]
            tags = [int(output.tag.item()) for output in outputs]

            next_observations, rewards, done, info = self.env.step(actions, tags)
            opponents = info.get("opponents", (None,) * num_agents)

            for i in range(num_agents):
                opponent_index = opponents[i]
                opponent_action = actions[opponent_index] if opponent_index is not None else None
                opponent_tag = tags[opponent_index] if opponent_index is not None else None

                step_logs[i].append(
                    StepLog(
                        observation=observations[i],
                        action_output=outputs[i],
                        reward=rewards[i],
                        opponent_index=opponent_index,
                        opponent_action=opponent_action,
                        opponent_tag=opponent_tag,
                    )
                )
                states[i] = outputs[i].new_state
                cumulative_rewards[i] += rewards[i]

            observations = next_observations

            if done:
                break

        return EpisodeLog(step_logs=step_logs, total_rewards=cumulative_rewards, history=self.env.get_history())

    def train(self) -> List[Dict[str, float]]:
        logs: List[Dict[str, float]] = []

        for episode in range(1, self.config.episodes + 1):
            episode_log = self.run_episode(sample=True)

            agent_returns = [self._compute_returns([step.reward for step in steps]) for steps in episode_log.step_logs]
            agent_losses = [self._episode_loss(steps, returns) for steps, returns in zip(episode_log.step_logs, agent_returns)]

            policy_loss = sum(loss["policy"] for loss in agent_losses)
            value_loss = sum(loss["value"] for loss in agent_losses)
            entropy = sum(loss["entropy"] for loss in agent_losses)

            loss = policy_loss + self.config.value_coeff * value_loss - (
                self.config.entropy_coeff_action + self.config.entropy_coeff_tag
            ) * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
            self.optimizer.step()

            avg_reward = sum(episode_log.total_rewards) / max(1, self.env.num_agents)

            total_steps = max(1, len(episode_log.history) * self.env.num_agents)
            cooperation_actions = 0
            for entry in episode_log.history:
                for pair in entry.get("pairs", []):
                    cooperation_actions += sum(1 for action in pair["actions"] if action == 0)
            cooperation_rate = cooperation_actions / total_steps

            logs.append({
                "episode": episode,
                "avg_reward": avg_reward,
                "total_rewards": list(episode_log.total_rewards),
                "cooperation_rate": cooperation_rate,
                "loss": float(loss.item()),
            })

            if episode % self.config.log_interval == 0:
                print(
                    f"Episode {episode}: avg_reward={avg_reward:.3f} "
                    f"loss={float(loss.item()):.4f} coop_rate={cooperation_rate:.3f}"
                )

        return logs
