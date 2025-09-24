"""Neural modules implementing the world model agent."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from typing import List, Optional

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class ObservationSummary:
    """Lightweight summary of an observed opponent signal for analysis."""

    step: int
    opponent_action: Optional[int]
    opponent_tag: Optional[int]


@dataclass
class AgentState:
    """Hidden state of the agent's world model."""

    hidden: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    memory_steps: torch.Tensor
    payloads: List[ObservationSummary]

    def detach(self) -> "AgentState":
        return AgentState(
            hidden=self.hidden.detach(),
            keys=self.keys.detach(),
            values=self.values.detach(),
            memory_steps=self.memory_steps.detach(),
            payloads=list(self.payloads),
        )


@dataclass
class ActionOutput:
    action: torch.Tensor
    tag: torch.Tensor
    logp_action: torch.Tensor
    logp_tag: torch.Tensor
    entropy_action: torch.Tensor
    entropy_tag: torch.Tensor
    value: torch.Tensor
    new_state: AgentState
    hidden: torch.Tensor
    attention_weights: Optional[torch.Tensor]
    attention_steps: Optional[torch.Tensor]
    attention_payloads: Optional[List[ObservationSummary]]


class AttentiveWorldModel(nn.Module):
    """RNN-based world model with a content-based attention mechanism."""

    def __init__(self, obs_dim: int, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        if attention_dim > 0:
            self.key_layer = nn.Linear(hidden_dim, attention_dim)
            self.value_layer = nn.Linear(hidden_dim, attention_dim)
            self.query_layer = nn.Linear(hidden_dim, attention_dim)
            gru_input = hidden_dim + attention_dim
        else:
            self.key_layer = None
            self.value_layer = None
            self.query_layer = None
            gru_input = hidden_dim

        self.gru_cell = nn.GRUCell(gru_input, hidden_dim)

    def forward(
        self, observation: torch.Tensor, state: AgentState
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        embed = self.encoder(observation)

        context = None
        weights = None
        key = None
        value = None

        if self.attention_dim > 0 and state.keys.numel() > 0:
            key = self.key_layer(embed)
            value = self.value_layer(embed)
            query = self.query_layer(state.hidden)
            scaled = torch.matmul(state.keys, query) / sqrt(float(self.attention_dim))
            weights = torch.softmax(scaled, dim=0)
            context = torch.matmul(weights.unsqueeze(0), state.values).squeeze(0)
            gru_input = torch.cat([embed, context], dim=-1)
        elif self.attention_dim > 0:
            key = self.key_layer(embed)
            value = self.value_layer(embed)
            context = torch.zeros(self.attention_dim, device=embed.device, dtype=embed.dtype)
            gru_input = torch.cat([embed, context], dim=-1)
        else:
            gru_input = embed

        new_hidden = self.gru_cell(gru_input, state.hidden)

        return new_hidden, key, value, weights


class WorldModelController(nn.Module):
    def __init__(self, hidden_dim: int, num_tags: int):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_dim, 2)
        self.tag_head = nn.Linear(hidden_dim, num_tags)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.policy_head(hidden)
        return self.action_head(features), self.tag_head(features), self.value_head(features)


class WorldModelPolicy(nn.Module):
    """Full agent consisting of an attentive world model and a controller."""

    def __init__(
        self,
        observation_dim: int,
        num_tags: int,
        hidden_dim: int = 128,
        attention_dim: int = 64,
        attention_window: int = 10,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.attention_window = attention_window
        self.num_tags = num_tags

        self.world_model = AttentiveWorldModel(observation_dim, hidden_dim, attention_dim)
        self.controller = WorldModelController(hidden_dim, num_tags)

        self.to(self.device)

    def initial_state(self) -> AgentState:
        hidden = torch.zeros(self.hidden_dim, device=self.device)
        if self.attention_dim > 0:
            keys = torch.zeros((0, self.attention_dim), device=self.device)
            values = torch.zeros((0, self.attention_dim), device=self.device)
        else:
            keys = torch.zeros((0, 1), device=self.device)
            values = torch.zeros((0, 1), device=self.device)
        memory_steps = torch.zeros(0, device=self.device, dtype=torch.long)
        payloads: List[ObservationSummary] = []
        return AgentState(hidden=hidden, keys=keys, values=values, memory_steps=memory_steps, payloads=payloads)

    def _update_memory(
        self,
        state: AgentState,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        step: int,
        payload: ObservationSummary,
    ) -> AgentState:
        if self.attention_dim == 0:
            return replace(state, hidden=state.hidden, payloads=(state.payloads + [payload])[-self.attention_window :])

        keys = state.keys
        values = state.values
        memory_steps = state.memory_steps
        payloads = list(state.payloads)

        if key is not None and value is not None:
            keys = torch.cat([keys, key.unsqueeze(0)], dim=0) if keys.numel() > 0 else key.unsqueeze(0)
            values = torch.cat([values, value.unsqueeze(0)], dim=0) if values.numel() > 0 else value.unsqueeze(0)
            memory_steps = torch.cat(
                [memory_steps, torch.tensor([step], dtype=torch.long, device=self.device)], dim=0
            )
            payloads.append(payload)

            if keys.shape[0] > self.attention_window:
                keys = keys[-self.attention_window :]
                values = values[-self.attention_window :]
                memory_steps = memory_steps[-self.attention_window :]
                payloads = payloads[-self.attention_window :]

        return AgentState(hidden=state.hidden, keys=keys, values=values, memory_steps=memory_steps, payloads=payloads)

    def step(
        self,
        observation: torch.Tensor,
        state: AgentState,
        time_step: int,
        payload: ObservationSummary,
        sample: bool = True,
    ) -> ActionOutput:
        observation = observation.to(self.device)
        state = AgentState(
            hidden=state.hidden.to(self.device),
            keys=state.keys.to(self.device),
            values=state.values.to(self.device),
            memory_steps=state.memory_steps.to(self.device),
            payloads=list(state.payloads),
        )

        new_hidden, key, value, weights = self.world_model(observation, state)

        action_logits, tag_logits, value_head = self.controller(new_hidden)

        action_dist = Categorical(logits=action_logits)
        tag_dist = Categorical(logits=tag_logits)

        if sample:
            action = action_dist.sample()
            tag = tag_dist.sample()
        else:
            action = torch.argmax(action_logits, dim=-1)
            tag = torch.argmax(tag_logits, dim=-1)

        logp_action = action_dist.log_prob(action)
        logp_tag = tag_dist.log_prob(tag)
        entropy_action = action_dist.entropy()
        entropy_tag = tag_dist.entropy()

        updated_state = self._update_memory(
            AgentState(new_hidden, state.keys, state.values, state.memory_steps, state.payloads),
            key,
            value,
            time_step,
            payload,
        )

        return ActionOutput(
            action=action,
            tag=tag,
            logp_action=logp_action,
            logp_tag=logp_tag,
            entropy_action=entropy_action,
            entropy_tag=entropy_tag,
            value=value_head.squeeze(-1),
            new_state=AgentState(
                hidden=new_hidden,
                keys=updated_state.keys,
                values=updated_state.values,
                memory_steps=updated_state.memory_steps,
                payloads=updated_state.payloads,
            ),
            hidden=new_hidden,
            attention_weights=weights,
            attention_steps=state.memory_steps if weights is not None else None,
            attention_payloads=state.payloads if weights is not None else None,
        )
