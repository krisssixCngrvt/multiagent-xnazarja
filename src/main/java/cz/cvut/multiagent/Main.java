package cz.cvut.multiagent;

import cz.cvut.multiagent.agents.DQNAgent;
import cz.cvut.multiagent.agents.ForagingAgent;
import cz.cvut.multiagent.agents.QLearningAgent;
import cz.cvut.multiagent.environment.GridWorld;
import cz.cvut.multiagent.training.Trainer;
import cz.cvut.multiagent.utils.ConfigLoader;

import java.util.ArrayList;
import java.util.List;

/**
 * Main entry point for the Multi-Agent Foraging RL project.
 * Supports both Q-Learning and DQN agents.
 */
public class Main {
    public static void main(String[] args) {
        System.out.println("=== Multi-Agent Foraging RL Coach ===");
        System.out.println("Multilanguage Level-Based Foraging Agent\n");

        // Parse command line arguments
        String agentType = args.length > 0 ? args[0] : "dqn";
        int numAgents = args.length > 1 ? Integer.parseInt(args[1]) : 4;
        int episodes = args.length > 2 ? Integer.parseInt(args[2]) : 1000;

        System.out.println("Configuration:");
        System.out.println("  Agent Type: " + agentType.toUpperCase());
        System.out.println("  Number of Agents: " + numAgents);
        System.out.println("  Training Episodes: " + episodes);
        System.out.println();

        // Create environment
        GridWorld environment = new GridWorld(8, 8, 200);

        // Create agents based on type
        List<ForagingAgent> agents = new ArrayList<>();
        
        if (agentType.equalsIgnoreCase("qlearning")) {
            System.out.println("Initializing Q-Learning agents...");
            for (int i = 0; i < numAgents; i++) {
                agents.add(new QLearningAgent(
                    i,                  // agent ID
                    0.1,               // learning rate
                    0.95,              // discount factor
                    1.0,               // initial epsilon
                    0.995,             // epsilon decay
                    0.01               // min epsilon
                ));
            }
        } else if (agentType.equalsIgnoreCase("dqn")) {
            System.out.println("Initializing Deep Q-Network agents...");
            int inputSize = 21; // State observation size from GridWorld.State.getObservation()
            
            for (int i = 0; i < numAgents; i++) {
                agents.add(new DQNAgent(
                    i,                  // agent ID
                    inputSize,         // input size
                    0.001,             // learning rate
                    0.95,              // discount factor
                    1.0,               // initial epsilon
                    0.995,             // epsilon decay
                    0.01,              // min epsilon
                    10000,             // replay buffer size
                    32,                // batch size
                    100                // target update frequency
                ));
            }
        } else {
            System.err.println("Unknown agent type: " + agentType);
            System.err.println("Use 'qlearning' or 'dqn'");
            return;
        }

        // Create trainer
        Trainer trainer = new Trainer(environment, agents, episodes, true);

        // Train agents
        long startTime = System.currentTimeMillis();
        Trainer.TrainingResults results = trainer.train();
        long endTime = System.currentTimeMillis();

        System.out.println("\nTraining Duration: " + 
            String.format("%.2f", (endTime - startTime) / 1000.0) + " seconds");

        // Evaluate on larger grid
        System.out.println("\n=== Evaluation on Larger Grid ===");
        GridWorld largerEnv = new GridWorld(12, 12, 300);
        Trainer evaluator = new Trainer(largerEnv, agents, 100, false);
        evaluator.evaluate(100);

        // Save models if using DQN
        if (agentType.equalsIgnoreCase("dqn")) {
            System.out.println("\nSaving trained models...");
            for (int i = 0; i < agents.size(); i++) {
                try {
                    DQNAgent dqnAgent = (DQNAgent) agents.get(i);
                    dqnAgent.saveModel("models/agent_" + i + ".zip");
                    System.out.println("  Saved agent " + i + " model");
                } catch (Exception e) {
                    System.err.println("  Failed to save agent " + i + ": " + e.getMessage());
                }
            }
        }

        System.out.println("\n=== Training Complete! ===");
        System.out.println("Amazing results achieved! 🎉");
    }
}
