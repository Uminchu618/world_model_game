"""Analysis utilities for probing the learned world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import AnalysisConfig
from .environment import TagPrisonersDilemma
from .models import ObservationSummary, WorldModelPolicy


@dataclass
class AgentStepRecord:
    step_index: int
    hidden: torch.Tensor
    action: int
    tag: int
    reward: float
    opponent_action: int
    opponent_tag: int
    observed_opponent_action: Optional[int]
    observed_opponent_tag: Optional[int]
    attention_weights: Optional[torch.Tensor]
    attention_steps: Optional[torch.Tensor]
    attention_payloads: Optional[List[ObservationSummary]]


@dataclass
class EpisodeRecord:
    agent_steps: List[List[AgentStepRecord]]
    history: List[dict]


def collect_rollout(
    env: TagPrisonersDilemma,
    policy: WorldModelPolicy,
    max_steps: Optional[int] = None,
    sample: bool = False,
) -> EpisodeRecord:
    observations = env.reset()
    states = [policy.initial_state(), policy.initial_state()]

    step_records = [[], []]
    total_steps = max_steps or env.max_steps

    with torch.no_grad():
        for step in range(total_steps):
            payloads = [
                ObservationSummary(
                    step=obs.step,
                    opponent_action=obs.opponent_action,
                    opponent_tag=obs.opponent_tag,
                )
                for obs in observations
            ]

            outputs = [
                policy.step(observations[i].vector, states[i], step, payloads[i], sample=sample)
                for i in range(2)
            ]

            actions = [int(outputs[0].action.item()), int(outputs[1].action.item())]
            tags = [int(outputs[0].tag.item()), int(outputs[1].tag.item())]

            next_obs, rewards, done, info = env.step(actions, tags)

            for i in range(2):
                opponent_index = 1 - i
                step_records[i].append(
                    AgentStepRecord(
                        step_index=step,
                        hidden=outputs[i].hidden.detach().cpu(),
                        action=actions[i],
                        tag=tags[i],
                        reward=rewards[i],
                        opponent_action=actions[opponent_index],
                        opponent_tag=tags[opponent_index],
                        observed_opponent_action=observations[i].opponent_action,
                        observed_opponent_tag=observations[i].opponent_tag,
                        attention_weights=outputs[i].attention_weights.detach().cpu()
                        if outputs[i].attention_weights is not None
                        else None,
                        attention_steps=outputs[i].attention_steps.detach().cpu()
                        if outputs[i].attention_steps is not None
                        else None,
                        attention_payloads=outputs[i].attention_payloads,
                    )
                )
                states[i] = outputs[i].new_state

            observations = next_obs
            if done:
                break

    return EpisodeRecord(agent_steps=step_records, history=env.get_history())


def collect_rollouts(env: TagPrisonersDilemma, policy: WorldModelPolicy, episodes: int) -> List[EpisodeRecord]:
    records = []
    for _ in range(episodes):
        record = collect_rollout(env, policy, sample=False)
        records.append(record)
    return records


def _prepare_probe_dataset(
    records: Sequence[EpisodeRecord], agent_index: int, target: str
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    features: List[torch.Tensor] = []
    labels: List[int] = []

    for record in records:
        for step in record.agent_steps[agent_index]:
            features.append(step.hidden)
            if target == "self_tag":
                labels.append(step.tag)
            elif target == "self_action":
                labels.append(step.action)
            elif target == "opponent_action":
                labels.append(step.opponent_action)
            else:
                raise ValueError(f"Unknown probe target: {target}")

    feature_tensor = torch.stack(features)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    if labels.numel() == 0:
        raise ValueError("No data collected for probe")

    if target in {"self_action", "opponent_action"}:
        num_classes = 2
    else:
        num_classes = int(labels.max().item() + 1)

    return feature_tensor, label_tensor, num_classes


def train_linear_probe(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int, config: AnalysisConfig
) -> Dict[str, float]:
    dataset_size = features.shape[0]
    indices = torch.randperm(dataset_size)
    split = max(1, int(0.8 * dataset_size))
    train_idx = indices[:split]
    test_idx = indices[split:]

    x_train = features[train_idx]
    y_train = labels[train_idx]
    x_test = features[test_idx]
    y_test = labels[test_idx]

    model = nn.Linear(features.shape[1], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.probe_learning_rate)

    for _ in range(config.probe_epochs):
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(x_test)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == y_test).float().mean().item() if len(y_test) > 0 else 0.0

    return {"loss": float(loss.item()), "accuracy": accuracy}


def probe_hidden_states(
    records: Sequence[EpisodeRecord], config: AnalysisConfig, agent_index: int = 0
) -> Dict[str, Dict[str, float]]:
    results = {}
    for target in ["self_tag", "self_action", "opponent_action"]:
        features, labels, num_classes = _prepare_probe_dataset(records, agent_index, target)
        results[target] = train_linear_probe(features, labels, num_classes, config)
    return results


def summarize_attention_focus(records: Sequence[EpisodeRecord], agent_index: int = 0) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, List[float]] = {
        "after_cooperation_same_tag": [],
        "after_cooperation_defect_history": [],
        "after_defection_same_tag": [],
        "after_defection_defect_history": [],
    }

    for record in records:
        for step in record.agent_steps[agent_index]:
            if step.observed_opponent_action is None or step.attention_weights is None:
                continue

            weights = step.attention_weights.numpy()
            payloads = step.attention_payloads or []

            same_tag_weight = 0.0
            defect_history_weight = 0.0

            for weight, payload in zip(weights, payloads):
                if payload.opponent_tag is not None and payload.opponent_tag == step.observed_opponent_tag:
                    same_tag_weight += float(weight)
                if payload.opponent_action == 1:
                    defect_history_weight += float(weight)

            if step.observed_opponent_action == 1:
                stats["after_defection_same_tag"].append(same_tag_weight)
                stats["after_defection_defect_history"].append(defect_history_weight)
            else:
                stats["after_cooperation_same_tag"].append(same_tag_weight)
                stats["after_cooperation_defect_history"].append(defect_history_weight)

    return {k: {"mean": float(np.mean(v)) if v else 0.0, "count": len(v)} for k, v in stats.items()}


def intervene_on_tags(
    env: TagPrisonersDilemma,
    policy: WorldModelPolicy,
    intervention_step: int,
    forced_tag: int,
    episodes: int,
) -> Dict[str, float]:
    baseline_cooperation: List[float] = []
    intervention_cooperation: List[float] = []

    for _ in range(episodes):
        base_env = TagPrisonersDilemma(env.config)
        int_env = TagPrisonersDilemma(env.config)

        base_record = collect_rollout(base_env, policy, sample=False)

        observations = int_env.reset()
        states = [policy.initial_state(), policy.initial_state()]
        coop_counts = [0, 0]

        with torch.no_grad():
            for step in range(int_env.max_steps):
                payloads = []
                mod_observations = []
                for agent_index in range(2):
                    obs = observations[agent_index]
                    if step == intervention_step:
                        obs = int_env.override_observation_tag(obs, forced_tag)
                    payloads.append(
                        ObservationSummary(
                            step=obs.step,
                            opponent_action=obs.opponent_action,
                            opponent_tag=obs.opponent_tag,
                        )
                    )
                    mod_observations.append(obs)

                outputs = [
                    policy.step(mod_observations[i].vector, states[i], step, payloads[i], sample=False)
                    for i in range(2)
                ]

                actions = [int(outputs[0].action.item()), int(outputs[1].action.item())]
                tags = [int(outputs[0].tag.item()), int(outputs[1].tag.item())]

                next_obs, rewards, done, info = int_env.step(actions, tags)

                for i in range(2):
                    coop_counts[i] += 1 if actions[i] == 0 else 0
                    states[i] = outputs[i].new_state

                observations = next_obs
                if done:
                    break

        base_coop = sum(1 for entry in base_record.history if entry["actions"][0] == 0) / max(
            1, len(base_record.history)
        )
        intervention_rate = coop_counts[0] / max(1, int_env.max_steps)

        baseline_cooperation.append(base_coop)
        intervention_cooperation.append(intervention_rate)

    return {
        "baseline_mean": float(np.mean(baseline_cooperation)) if baseline_cooperation else 0.0,
        "intervention_mean": float(np.mean(intervention_cooperation)) if intervention_cooperation else 0.0,
        "difference": float(np.mean(intervention_cooperation) - np.mean(baseline_cooperation))
        if baseline_cooperation and intervention_cooperation
        else 0.0,
    }
