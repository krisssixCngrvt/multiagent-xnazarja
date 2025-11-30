# multiagent-xnazarja

A simple multiagent simulation system that demonstrates agent interactions and scoring.

## Overview

This project implements a grid-based environment where multiple agents with different strategies compete for resources and interact with each other. Agents accumulate scores based on their actions and interactions.

## Features

- **Multiple Agent Strategies**:
  - `random`: Agents move randomly across the grid
  - `greedy`: Agents move towards the center where resources tend to be
  - `cooperative`: Agents move in structured patterns and cooperate with others

- **Resource Collection**: Agents can collect resources scattered across the environment
- **Agent Interactions**: When agents meet, they interact based on their strategies
- **Scoring System**: Agents earn points through resource collection and interactions

## Installation

No external dependencies required - uses Python standard library only.

```bash
# Clone the repository
git clone https://github.com/krisssixCngrvt/multiagent-xnazarja.git
cd multiagent-xnazarja
```

## Usage

Run the simulation:

```bash
python main.py
```

## Example Output

```
============================================================
                    MULTIAGENT SCOREBOARD
============================================================

Rank  Agent ID  Strategy       Score     Resources Interactions
------------------------------------------------------------
1     2         cooperative    250       1         90          
2     3         random         246       25        16          
3     5         cooperative    246       0         90          
4     0         random         109       22        2           
5     1         greedy         -96       1         98          
6     4         greedy         -96       0         98          
------------------------------------------------------------

Simulation Statistics:
  Steps completed: 100
  Total score: 659
  Average score: 109.83
```

## Project Structure

```
multiagent-xnazarja/
├── main.py              # Entry point - runs the simulation
├── multiagent/          # Package directory
│   ├── __init__.py      # Package initialization
│   ├── agent.py         # Agent class with behaviors and scoring
│   ├── environment.py   # Environment with resources
│   └── simulation.py    # Simulation orchestration
└── README.md            # This file
```

## How Scoring Works

- **Resource Collection**: +1 to +10 points per resource
- **Cooperative + Cooperative Interaction**: Both gain +3 points
- **Greedy + Greedy Interaction**: Both lose -1 point
- **Cooperative + Greedy Interaction**: Greedy gains +5, Cooperative gains +0
- **Random Interactions**: Variable results (-1 to +2 points)