"""
Multiagent System Package
A simple multiagent simulation with scoring functionality.
"""

from .agent import Agent
from .environment import Environment
from .simulation import Simulation

__version__ = "0.1.0"
__all__ = ["Agent", "Environment", "Simulation"]
