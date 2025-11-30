#!/usr/bin/env python3
"""
Main entry point for running the multiagent simulation.
Demonstrates the multiagent system and displays scores.
"""

from multiagent import Simulation


def main():
    """Run the multiagent simulation and display results."""
    print("\n" + "=" * 60)
    print("       MULTIAGENT SIMULATION - xnazarja")
    print("=" * 60)
    print("\nInitializing simulation...")
    
    # Create simulation with 6 agents using different strategies
    simulation = Simulation(
        num_agents=6,
        grid_size=15,
        strategies=["random", "greedy", "cooperative"],
        seed=42  # Fixed seed for reproducible results
    )
    
    print(f"Created {len(simulation.agents)} agents with different strategies:")
    for agent in simulation.agents:
        print(f"  - Agent {agent.agent_id}: {agent.strategy} strategy")
    
    print(f"\nEnvironment: {simulation.environment.grid_size}x{simulation.environment.grid_size} grid")
    print(f"Initial resources: {simulation.environment.get_resource_count()}")
    
    # Run the simulation
    print("\nRunning simulation for 100 steps...")
    print("-" * 40)
    
    # Show intermediate progress
    for step in range(100):
        result = simulation.step()
        if (step + 1) % 25 == 0:
            print(f"Step {step + 1}: Resources collected this step: {result['resources_collected']}, "
                  f"Interactions: {result['interactions']}")
    
    # Display final scoreboard
    simulation.print_scoreboard()
    
    # Additional analysis by strategy
    print("\nScore Analysis by Strategy:")
    print("-" * 40)
    
    strategy_scores = {}
    for agent in simulation.agents:
        strategy = agent.strategy
        if strategy not in strategy_scores:
            strategy_scores[strategy] = []
        strategy_scores[strategy].append(agent.score)
    
    for strategy, scores in sorted(strategy_scores.items()):
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"  {strategy.capitalize():<12}: Avg Score = {avg_score:.2f}, "
              f"Total = {sum(scores)}, Agents = {len(scores)}")
    
    print("\n" + "=" * 60)
    print("                 SIMULATION COMPLETE")
    print("=" * 60 + "\n")
    
    return simulation


if __name__ == "__main__":
    main()
