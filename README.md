# Multi-Agent Foraging RL Coach

A multi-agent reinforcement learning system for grid-world foraging environment. Agents learn to collaboratively collect food items while avoiding obstacles to achieve the highest possible score.

## Features

- **Q-Learning with Experience Replay**: Agents use optimized Q-learning with experience replay for stable learning
- **Multi-Agent Coordination**: Multiple agents work together in a shared environment
- **Optimized Hyperparameters**: Pre-tuned parameters for achieving high scores quickly
- **Configurable Environment**: Customizable grid size, number of agents, food, and obstacles

## Requirements

- Java 11 or higher

## Quick Start

```bash
# Run with default (optimized) settings
./run.sh

# Quick test run (fewer episodes)
./quick_run.sh

# Run with demonstration
./run.sh --demo
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--agents` | 3 | Number of agents |
| `--width` | 10 | Grid width |
| `--height` | 10 | Grid height |
| `--food` | 15 | Number of food items |
| `--obstacles` | 5 | Number of obstacles |
| `--max-steps` | 200 | Maximum steps per episode |
| `--train-episodes` | 2000 | Training episodes |
| `--eval-episodes` | 10 | Evaluation episodes |
| `--demo` | false | Show demonstration after training |
| `--quick` | false | Quick run with fewer episodes |

## Example Usage

```bash
# Train with more agents
./run.sh --agents 5 --food 25

# Larger grid
./run.sh --width 15 --height 15 --food 30

# Quick training with demo
./run.sh --quick --demo
```

## Score System

- **100+**: 🏆 Excellent - All food collected efficiently
- **80-99**: ⭐ Great - High performance
- **60-79**: ✓ Good - Decent performance
- **Below 60**: Needs more training

## Architecture

- `GridWorld.java`: Grid-world foraging environment
- `QLearningAgent.java`: Q-learning agent with experience replay
- `MultiAgentCoach.java`: Coordinator for multi-agent training
- `Main.java`: Entry point with CLI argument parsing

## Algorithm Details

The agents use epsilon-greedy Q-learning with:
- Experience replay buffer (size: 10,000)
- Batch learning (batch size: 32)
- Epsilon decay (0.997 per episode)
- Learning rate: 0.15
- Discount factor: 0.95