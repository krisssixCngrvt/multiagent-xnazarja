"""
Agent module for the multiagent system.
Defines individual agents with behaviors and scoring.
"""

import random
from typing import Optional, Tuple


class Agent:
    """
    Represents an individual agent in the multiagent system.
    
    Each agent has:
    - A unique identifier
    - A position in the environment
    - A score that accumulates based on actions
    - A strategy for decision-making
    """
    
    def __init__(self, agent_id: int, strategy: str = "random"):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            strategy: The decision-making strategy ("random", "greedy", "cooperative")
        """
        self.agent_id = agent_id
        self.strategy = strategy
        self.score = 0
        self.position: Tuple[int, int] = (0, 0)
        self.resources_collected = 0
        self.interactions = 0
    
    def move(self, grid_size: int) -> Tuple[int, int]:
        """
        Decide on a movement direction based on strategy.
        
        Args:
            grid_size: The size of the environment grid
            
        Returns:
            New position as (x, y) tuple
        """
        x, y = self.position
        
        if self.strategy == "random":
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
        elif self.strategy == "greedy":
            # Move towards center where resources tend to be
            center = grid_size // 2
            dx = 1 if x < center else (-1 if x > center else 0)
            dy = 1 if y < center else (-1 if y > center else 0)
        elif self.strategy == "cooperative":
            # Move with bias towards positive directions (predictable pattern)
            dx = 1 if random.random() > 0.5 else 0
            dy = 1 if random.random() > 0.5 else 0
        else:
            dx, dy = 0, 0
        
        # Apply boundary constraints
        new_x = max(0, min(grid_size - 1, x + dx))
        new_y = max(0, min(grid_size - 1, y + dy))
        
        self.position = (new_x, new_y)
        return self.position
    
    def collect_resource(self, value: int) -> int:
        """
        Collect a resource and add to score.
        
        Args:
            value: The value of the resource
            
        Returns:
            The agent's new score
        """
        self.score += value
        self.resources_collected += 1
        return self.score
    
    def interact(self, other: "Agent") -> Tuple[int, int]:
        """
        Interact with another agent based on strategies.
        
        Args:
            other: The other agent to interact with
            
        Returns:
            Score changes for (self, other)
        """
        self.interactions += 1
        other.interactions += 1
        
        # Cooperation vs competition based on strategies
        if self.strategy == "cooperative" and other.strategy == "cooperative":
            # Both cooperate - mutual benefit
            self_gain = 3
            other_gain = 3
        elif self.strategy == "greedy" and other.strategy == "greedy":
            # Both compete - mutual loss
            self_gain = -1
            other_gain = -1
        elif self.strategy == "cooperative":
            # Self cooperates, other exploits
            self_gain = 0
            other_gain = 5
        elif other.strategy == "cooperative":
            # Self exploits, other cooperates
            self_gain = 5
            other_gain = 0
        else:
            # Random or mixed interaction
            self_gain = random.randint(-1, 2)
            other_gain = random.randint(-1, 2)
        
        self.score += self_gain
        other.score += other_gain
        
        return self_gain, other_gain
    
    def get_stats(self) -> dict:
        """
        Get the agent's current statistics.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            "agent_id": self.agent_id,
            "strategy": self.strategy,
            "score": self.score,
            "position": self.position,
            "resources_collected": self.resources_collected,
            "interactions": self.interactions
        }
    
    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, strategy={self.strategy}, score={self.score})"
