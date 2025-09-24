"""世界モデルシグナリング実験を一括実行するCLIスクリプト。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model_game import (
    AnalysisConfig,
    GameConfig,
    SelfPlayTrainer,
    TagPrisonersDilemma,
    TrainingConfig,
    WorldModelPolicy,
)
from world_model_game import analysis


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数を定義したパーサを構築する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    parser.add_argument("--max-steps", type=int, default=GameConfig.max_steps)
    parser.add_argument("--num-tags", type=int, default=GameConfig.num_tags)
    parser.add_argument("--num-agents", type=int, default=GameConfig.num_agents)
    parser.add_argument("--hidden-dim", type=int, default=TrainingConfig.hidden_dim)
    parser.add_argument("--attention-dim", type=int, default=TrainingConfig.attention_dim)
    parser.add_argument("--attention-window", type=int, default=TrainingConfig.attention_window)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--device", type=str, default=TrainingConfig.device)
    parser.add_argument("--log-interval", type=int, default=TrainingConfig.log_interval)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--output", type=Path, default=Path("experiment_results.json"))
    return parser


def main() -> None:
    """学習・評価・解析を順に実行し、結果をJSONにまとめる。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.num_agents % 2 != 0:
        raise SystemExit("--num-agents must be an even number")

    game_config = GameConfig(
        num_tags=args.num_tags,
        max_steps=args.max_steps,
        num_agents=args.num_agents,
        seed=args.seed,
    )
    training_config = TrainingConfig(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        attention_window=args.attention_window,
        device=args.device,
        log_interval=args.log_interval,
        seed=args.seed,
    )

    env = TagPrisonersDilemma(game_config)
    policy = WorldModelPolicy(
        observation_dim=env.observation_size,
        num_tags=game_config.num_tags,
        hidden_dim=training_config.hidden_dim,
        attention_dim=training_config.attention_dim,
        attention_window=training_config.attention_window,
        device=training_config.device,
    )

    trainer = SelfPlayTrainer(env, policy, training_config)
    training_logs = trainer.train()

    analysis_config = AnalysisConfig()

    evaluation_env = TagPrisonersDilemma(game_config)
    rollout_records = analysis.collect_rollouts(evaluation_env, policy, analysis_config.evaluation_episodes)
    probe_results = analysis.probe_hidden_states(rollout_records, analysis_config)
    attention_stats = analysis.summarize_attention_focus(rollout_records)
    intervention_stats = analysis.intervene_on_tags(
        TagPrisonersDilemma(game_config),
        policy,
        analysis_config.intervention_step,
        analysis_config.intervention_tag,
        analysis_config.intervention_episodes,
    )

    avg_training_reward = sum(log["avg_reward"] for log in training_logs) / max(1, len(training_logs))
    total_actions = sum(len(agent_steps) for record in rollout_records for agent_steps in record.agent_steps)
    cooperative_actions = sum(
        1 if entry.action == 0 else 0
        for record in rollout_records
        for agent_steps in record.agent_steps
        for entry in agent_steps
    )
    avg_cooperation = cooperative_actions / max(1, total_actions)

    results = {
        "training": {
            "episodes": args.episodes,
            "average_reward": avg_training_reward,
        },
        "evaluation": {
            "average_cooperation_rate": avg_cooperation,
            "probe_results": probe_results,
            "attention": attention_stats,
            "tag_intervention": intervention_stats,
        },
    }

    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
