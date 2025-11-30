"""
Environment module for the multiagent system.
Defines the world where agents operate.
"""

import random
from typing import Dict, List, Optional, Tuple


class Environment:
    """
    Represents the environment where agents operate.
    
    The environment is a grid-based world with:
    - Resources that agents can collect
    - Positions for multiple agents
    - Rules for interactions
    """
    
    def __init__(self, grid_size: int = 10, resource_density: float = 0.2):
        """
        Initialize the environment.
        
        Args:
            grid_size: The size of the square grid
            resource_density: Probability of a cell having a resource
        """
        self.grid_size = grid_size
        self.resource_density = resource_density
        self.resources: Dict[Tuple[int, int], int] = {}
        self.step_count = 0
        
        self._generate_resources()
    
    def _generate_resources(self) -> None:
        """Generate resources randomly across the grid."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if random.random() < self.resource_density:
                    # Resources have random values between 1 and 10
                    self.resources[(x, y)] = random.randint(1, 10)
    
    def get_resource_at(self, position: Tuple[int, int]) -> Optional[int]:
        """
        Get the resource value at a position, if any.
        
        Args:
            position: The (x, y) position to check
            
        Returns:
            Resource value or None if no resource
        """
        return self.resources.get(position)
    
    def collect_resource_at(self, position: Tuple[int, int]) -> int:
        """
        Collect and remove the resource at a position.
        
        Args:
            position: The (x, y) position to collect from
            
        Returns:
            Resource value or 0 if no resource
        """
        return self.resources.pop(position, 0)
    
    def respawn_resources(self, rate: float = 0.05) -> int:
        """
        Randomly spawn new resources.
        
        Args:
            rate: Probability of spawning at each empty cell
            
        Returns:
            Number of new resources spawned
        """
        new_count = 0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.resources and random.random() < rate:
                    self.resources[(x, y)] = random.randint(1, 10)
                    new_count += 1
        return new_count
    
    def get_total_resources(self) -> int:
        """Get the total value of all resources in the environment."""
        return sum(self.resources.values())
    
    def get_resource_count(self) -> int:
        """Get the number of resource locations."""
        return len(self.resources)
    
    def step(self) -> None:
        """Advance the environment by one time step."""
        self.step_count += 1
        # Periodically respawn resources
        if self.step_count % 5 == 0:
            self.respawn_resources()
    
    def get_stats(self) -> dict:
        """
        Get environment statistics.
        
        Returns:
            Dictionary with environment statistics
        """
        return {
            "grid_size": self.grid_size,
            "step_count": self.step_count,
            "resource_count": self.get_resource_count(),
            "total_resource_value": self.get_total_resources()
        }
    
    def __repr__(self) -> str:
        return f"Environment(size={self.grid_size}x{self.grid_size}, resources={self.get_resource_count()})"
