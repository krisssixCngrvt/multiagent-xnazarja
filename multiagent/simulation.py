"""
Simulation module for running multiagent experiments.
Orchestrates agents and environment.
"""

import random
from typing import Dict, List, Optional

from .agent import Agent
from .environment import Environment


class Simulation:
    """
    Manages the simulation of multiple agents in an environment.
    
    Features:
    - Configurable number of agents with different strategies
    - Step-by-step or continuous simulation
    - Score tracking and reporting
    """
    
    def __init__(
        self,
        num_agents: int = 5,
        grid_size: int = 10,
        strategies: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the simulation.
        
        Args:
            num_agents: Number of agents to create
            grid_size: Size of the environment grid
            strategies: List of strategies to distribute among agents
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        
        self.environment = Environment(grid_size=grid_size)
        self.agents: List[Agent] = []
        
        # Default strategies if none provided
        if strategies is None:
            strategies = ["random", "greedy", "cooperative"]
        
        # Create agents with distributed strategies
        for i in range(num_agents):
            strategy = strategies[i % len(strategies)]
            agent = Agent(agent_id=i, strategy=strategy)
            # Random starting position
            agent.position = (
                random.randint(0, grid_size - 1),
                random.randint(0, grid_size - 1)
            )
            self.agents.append(agent)
        
        self.step_count = 0
        self.history: List[Dict] = []
    
    def step(self) -> Dict:
        """
        Execute one simulation step.
        
        Returns:
            Dictionary with step results
        """
        step_results = {
            "step": self.step_count,
            "resources_collected": 0,
            "interactions": 0,
            "agent_scores": {}
        }
        
        # Each agent takes an action
        for agent in self.agents:
            # Move the agent
            agent.move(self.environment.grid_size)
            
            # Try to collect resource at new position
            resource = self.environment.collect_resource_at(agent.position)
            if resource > 0:
                agent.collect_resource(resource)
                step_results["resources_collected"] += 1
        
        # Check for agent interactions (agents at same position)
        positions: Dict[tuple, List[Agent]] = {}
        for agent in self.agents:
            if agent.position not in positions:
                positions[agent.position] = []
            positions[agent.position].append(agent)
        
        # Process interactions
        for position, agents_at_pos in positions.items():
            if len(agents_at_pos) > 1:
                # All pairs interact
                for i in range(len(agents_at_pos)):
                    for j in range(i + 1, len(agents_at_pos)):
                        agents_at_pos[i].interact(agents_at_pos[j])
                        step_results["interactions"] += 1
        
        # Update environment
        self.environment.step()
        
        # Record scores
        for agent in self.agents:
            step_results["agent_scores"][agent.agent_id] = agent.score
        
        self.step_count += 1
        self.history.append(step_results)
        
        return step_results
    
    def run(self, num_steps: int = 100) -> List[Dict]:
        """
        Run the simulation for a number of steps.
        
        Args:
            num_steps: Number of steps to simulate
            
        Returns:
            List of results for each step
        """
        results = []
        for _ in range(num_steps):
            results.append(self.step())
        return results
    
    def get_scores(self) -> Dict[int, int]:
        """
        Get current scores for all agents.
        
        Returns:
            Dictionary mapping agent_id to score
        """
        return {agent.agent_id: agent.score for agent in self.agents}
    
    def get_rankings(self) -> List[Agent]:
        """
        Get agents ranked by score (highest first).
        
        Returns:
            List of agents sorted by score
        """
        return sorted(self.agents, key=lambda a: a.score, reverse=True)
    
    def get_stats(self) -> dict:
        """
        Get comprehensive simulation statistics.
        
        Returns:
            Dictionary with simulation statistics
        """
        scores = [agent.score for agent in self.agents]
        
        return {
            "steps_completed": self.step_count,
            "num_agents": len(self.agents),
            "environment": self.environment.get_stats(),
            "total_score": sum(scores),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "agent_stats": [agent.get_stats() for agent in self.agents]
        }
    
    def print_scoreboard(self) -> None:
        """Print a formatted scoreboard."""
        print("\n" + "=" * 60)
        print("                    MULTIAGENT SCOREBOARD")
        print("=" * 60)
        
        rankings = self.get_rankings()
        
        print(f"\n{'Rank':<6}{'Agent ID':<10}{'Strategy':<15}{'Score':<10}{'Resources':<10}{'Interactions':<12}")
        print("-" * 60)
        
        for rank, agent in enumerate(rankings, 1):
            stats = agent.get_stats()
            print(f"{rank:<6}{stats['agent_id']:<10}{stats['strategy']:<15}{stats['score']:<10}"
                  f"{stats['resources_collected']:<10}{stats['interactions']:<12}")
        
        print("-" * 60)
        
        sim_stats = self.get_stats()
        print(f"\nSimulation Statistics:")
        print(f"  Steps completed: {sim_stats['steps_completed']}")
        print(f"  Total score: {sim_stats['total_score']}")
        print(f"  Average score: {sim_stats['average_score']:.2f}")
        print(f"  Remaining resources: {sim_stats['environment']['resource_count']}")
        print("=" * 60 + "\n")
    
    def __repr__(self) -> str:
        return f"Simulation(agents={len(self.agents)}, steps={self.step_count})"
