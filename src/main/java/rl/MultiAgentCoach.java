package rl;

import java.util.*;

/**
 * Multi-Agent RL Coach for coordinating multiple foraging agents.
 * Trains agents to achieve the highest possible score in the grid-world environment.
 */
public class MultiAgentCoach {
    private final List<QLearningAgent> agents;
    private final int numAgents;
    private final int gridWidth;
    private final int gridHeight;
    private final int numFood;
    private final int numObstacles;
    private final int maxSteps;
    
    // Training statistics
    private double bestScore = 0;
    private int bestEpisode = 0;
    private final List<Double> scoreHistory;
    
    public MultiAgentCoach(int numAgents, int gridWidth, int gridHeight, 
                          int numFood, int numObstacles, int maxSteps) {
        this.numAgents = numAgents;
        this.gridWidth = gridWidth;
        this.gridHeight = gridHeight;
        this.numFood = numFood;
        this.numObstacles = numObstacles;
        this.maxSteps = maxSteps;
        this.agents = new ArrayList<>();
        this.scoreHistory = new ArrayList<>();
        
        // Initialize agents with optimized hyperparameters
        for (int i = 0; i < numAgents; i++) {
            agents.add(new QLearningAgent(
                GridWorld.NUM_ACTIONS,
                0.9,    // Initial epsilon (high exploration)
                0.997,  // Epsilon decay (slow decay for more exploration)
                0.01,   // Minimum epsilon
                0.15,   // Learning rate
                0.95    // Discount factor
            ));
        }
    }
    
    /**
     * Train agents for a specified number of episodes.
     */
    public void train(int numEpisodes, boolean verbose) {
        System.out.println("=== Starting Multi-Agent RL Training ===");
        System.out.println("Agents: " + numAgents + ", Grid: " + gridWidth + "x" + gridHeight);
        System.out.println("Food: " + numFood + ", Obstacles: " + numObstacles);
        System.out.println("Max Steps: " + maxSteps + ", Episodes: " + numEpisodes);
        System.out.println("=========================================\n");
        
        for (int episode = 0; episode < numEpisodes; episode++) {
            GridWorld env = new GridWorld(gridWidth, gridHeight, numAgents, numFood, numObstacles, maxSteps);
            double episodeReward = runEpisode(env, true);
            double score = env.getScore();
            
            scoreHistory.add(score);
            
            if (score > bestScore) {
                bestScore = score;
                bestEpisode = episode;
            }
            
            // Decay epsilon for all agents
            for (QLearningAgent agent : agents) {
                agent.decayEpsilon();
            }
            
            if (verbose && (episode + 1) % 100 == 0) {
                double avgScore = getAverageScore(100);
                System.out.printf("Episode %d: Score=%.2f, Avg(100)=%.2f, Best=%.2f (ep%d), Epsilon=%.4f, Q-Size=%d%n",
                    episode + 1, score, avgScore, bestScore, bestEpisode + 1,
                    agents.get(0).getEpsilon(), agents.get(0).getQTableSize());
            }
        }
        
        System.out.println("\n=== Training Complete ===");
        System.out.printf("Best Score: %.2f at Episode %d%n", bestScore, bestEpisode + 1);
        System.out.printf("Final Average (100 episodes): %.2f%n", getAverageScore(100));
    }
    
    /**
     * Run a single episode, optionally training.
     */
    public double runEpisode(GridWorld env, boolean train) {
        double totalReward = 0;
        
        while (!env.isDone()) {
            // Get states for all agents
            int[] states = new int[numAgents];
            for (int i = 0; i < numAgents; i++) {
                states[i] = env.getState(i);
            }
            
            // Select actions
            int[] actions = new int[numAgents];
            for (int i = 0; i < numAgents; i++) {
                if (train) {
                    actions[i] = agents.get(i).selectAction(states[i]);
                } else {
                    actions[i] = agents.get(i).getBestAction(states[i]);
                }
            }
            
            // Execute actions
            double[] rewards = env.stepAll(actions);
            
            // Get next states and learn
            if (train) {
                for (int i = 0; i < numAgents; i++) {
                    int nextState = env.getState(i);
                    agents.get(i).learn(states[i], actions[i], rewards[i], nextState, env.isDone());
                    totalReward += rewards[i];
                }
            } else {
                for (double r : rewards) {
                    totalReward += r;
                }
            }
        }
        
        return totalReward;
    }
    
    /**
     * Evaluate trained agents.
     */
    public double evaluate(int numEpisodes) {
        System.out.println("\n=== Evaluation ===");
        
        // Set epsilon to 0 for pure exploitation
        for (QLearningAgent agent : agents) {
            agent.setEpsilon(0);
        }
        
        double totalScore = 0;
        double maxScore = 0;
        
        for (int episode = 0; episode < numEpisodes; episode++) {
            GridWorld env = new GridWorld(gridWidth, gridHeight, numAgents, numFood, numObstacles, maxSteps);
            runEpisode(env, false);
            double score = env.getScore();
            totalScore += score;
            
            if (score > maxScore) {
                maxScore = score;
            }
            
            System.out.printf("Eval Episode %d: Score=%.2f, Food=%d/%d, Steps=%d%n",
                episode + 1, score, env.getCollectedFood(), env.getTotalFood(), env.getSteps());
        }
        
        double avgScore = totalScore / numEpisodes;
        System.out.printf("\nEvaluation Results: Avg=%.2f, Max=%.2f%n", avgScore, maxScore);
        
        return avgScore;
    }
    
    /**
     * Run a demonstration with visualization.
     */
    public void demonstrate() {
        System.out.println("\n=== Demonstration ===");
        
        // Set epsilon to 0 for pure exploitation
        for (QLearningAgent agent : agents) {
            agent.setEpsilon(0);
        }
        
        GridWorld env = new GridWorld(gridWidth, gridHeight, numAgents, numFood, numObstacles, maxSteps);
        
        System.out.println("Initial State:");
        System.out.println(env);
        
        int stepCount = 0;
        while (!env.isDone() && stepCount < 50) {
            int[] states = new int[numAgents];
            int[] actions = new int[numAgents];
            
            for (int i = 0; i < numAgents; i++) {
                states[i] = env.getState(i);
                actions[i] = agents.get(i).getBestAction(states[i]);
            }
            
            double[] rewards = env.stepAll(actions);
            stepCount++;
            
            if (stepCount % 5 == 0 || env.isDone()) {
                System.out.printf("Step %d (Food: %d/%d):%n", stepCount, 
                    env.getCollectedFood(), env.getTotalFood());
                System.out.println(env);
            }
        }
        
        System.out.printf("Final Score: %.2f (Food: %d/%d in %d steps)%n",
            env.getScore(), env.getCollectedFood(), env.getTotalFood(), env.getSteps());
    }
    
    private double getAverageScore(int lastN) {
        if (scoreHistory.isEmpty()) return 0;
        
        int start = Math.max(0, scoreHistory.size() - lastN);
        double sum = 0;
        for (int i = start; i < scoreHistory.size(); i++) {
            sum += scoreHistory.get(i);
        }
        return sum / (scoreHistory.size() - start);
    }
    
    public double getBestScore() {
        return bestScore;
    }
    
    public List<Double> getScoreHistory() {
        return new ArrayList<>(scoreHistory);
    }
}
