package cz.cvut.multiagent.training;

import cz.cvut.multiagent.agents.ForagingAgent;
import cz.cvut.multiagent.environment.GridWorld;
import cz.cvut.multiagent.environment.GridWorld.Action;
import cz.cvut.multiagent.environment.GridWorld.Position;
import cz.cvut.multiagent.environment.GridWorld.State;

import java.util.*;

/**
 * Training coordinator for multi-agent foraging.
 * Manages environment setup, agent training loops, and evaluation.
 */
public class Trainer {
    private final GridWorld environment;
    private final List<ForagingAgent> agents;
    private final int episodeCount;
    private final boolean verbose;
    
    // Statistics
    private final List<Double> episodeRewards;
    private final List<Integer> episodeLengths;
    private final List<Integer> foodCollected;

    public Trainer(GridWorld environment, List<ForagingAgent> agents, 
                   int episodeCount, boolean verbose) {
        this.environment = environment;
        this.agents = agents;
        this.episodeCount = episodeCount;
        this.verbose = verbose;
        this.episodeRewards = new ArrayList<>();
        this.episodeLengths = new ArrayList<>();
        this.foodCollected = new ArrayList<>();
    }

    /**
     * Train agents over multiple episodes
     */
    public TrainingResults train() {
        System.out.println("Starting training for " + episodeCount + " episodes...");
        
        for (int episode = 0; episode < episodeCount; episode++) {
            double episodeReward = runEpisode(true);
            
            if (verbose && episode % 100 == 0) {
                printProgress(episode);
            }
        }
        
        System.out.println("\nTraining completed!");
        printFinalStatistics();
        
        return new TrainingResults(episodeRewards, episodeLengths, foodCollected);
    }

    /**
     * Run a single episode
     */
    private double runEpisode(boolean training) {
        // Reset environment and agents
        environment.reset();
        initializeEnvironment();
        
        for (ForagingAgent agent : agents) {
            agent.reset();
        }
        
        State state = environment.getState();
        double totalReward = 0.0;
        int steps = 0;
        int initialFoodCount = environment.getFoods().size();
        
        while (!environment.isDone()) {
            // Get actions from all agents
            Map<Integer, Action> actions = new HashMap<>();
            for (ForagingAgent agent : agents) {
                actions.put(agent.getAgentId(), agent.selectAction(state));
            }
            
            // Execute actions
            Map<Integer, Double> rewards = environment.step(actions);
            State nextState = environment.getState();
            boolean done = environment.isDone();
            
            // Update agents
            if (training) {
                for (ForagingAgent agent : agents) {
                    int agentId = agent.getAgentId();
                    agent.learn(state, actions.get(agentId), 
                               rewards.get(agentId), nextState, done);
                }
            }
            
            // Accumulate rewards
            totalReward += rewards.values().stream().mapToDouble(Double::doubleValue).sum();
            state = nextState;
            steps++;
        }
        
        int finalFoodCount = environment.getFoods().size();
        int collected = initialFoodCount - finalFoodCount;
        
        episodeRewards.add(totalReward);
        episodeLengths.add(steps);
        foodCollected.add(collected);
        
        return totalReward;
    }

    /**
     * Evaluate agents without training
     */
    public EvaluationResults evaluate(int numEpisodes) {
        System.out.println("\nEvaluating agents over " + numEpisodes + " episodes...");
        
        List<Double> evalRewards = new ArrayList<>();
        List<Integer> evalLengths = new ArrayList<>();
        List<Integer> evalFood = new ArrayList<>();
        
        for (int i = 0; i < numEpisodes; i++) {
            environment.reset();
            initializeEnvironment();
            
            State state = environment.getState();
            double totalReward = 0.0;
            int steps = 0;
            int initialFood = environment.getFoods().size();
            
            while (!environment.isDone()) {
                Map<Integer, Action> actions = new HashMap<>();
                for (ForagingAgent agent : agents) {
                    actions.put(agent.getAgentId(), agent.selectAction(state));
                }
                
                Map<Integer, Double> rewards = environment.step(actions);
                state = environment.getState();
                
                totalReward += rewards.values().stream().mapToDouble(Double::doubleValue).sum();
                steps++;
            }
            
            int collected = initialFood - environment.getFoods().size();
            evalRewards.add(totalReward);
            evalLengths.add(steps);
            evalFood.add(collected);
        }
        
        double avgReward = evalRewards.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double avgLength = evalLengths.stream().mapToInt(Integer::intValue).average().orElse(0);
        double avgFood = evalFood.stream().mapToInt(Integer::intValue).average().orElse(0);
        double successRate = evalFood.stream().filter(f -> f >= 3).count() / (double) numEpisodes;
        
        System.out.println("\nEvaluation Results:");
        System.out.println("  Average Reward: " + String.format("%.2f", avgReward));
        System.out.println("  Average Episode Length: " + String.format("%.1f", avgLength));
        System.out.println("  Average Food Collected: " + String.format("%.1f", avgFood));
        System.out.println("  Success Rate: " + String.format("%.1f%%", successRate * 100));
        
        return new EvaluationResults(avgReward, avgLength, avgFood, successRate);
    }

    /**
     * Initialize environment with agents and food
     */
    private void initializeEnvironment() {
        Random rand = new Random();
        int width = environment.getWidth();
        int height = environment.getHeight();
        
        // Add agents at random positions
        for (ForagingAgent agent : agents) {
            int x = rand.nextInt(width);
            int y = rand.nextInt(height);
            environment.addAgent(agent.getAgentId(), 1 + rand.nextInt(2), new Position(x, y));
        }
        
        // Add food items
        int foodCount = 3 + rand.nextInt(3); // 3-5 food items
        for (int i = 0; i < foodCount; i++) {
            int x = rand.nextInt(width);
            int y = rand.nextInt(height);
            int level = 1 + rand.nextInt(3); // Level 1-3
            environment.addFood(level, new Position(x, y));
        }
    }

    private void printProgress(int episode) {
        int window = Math.min(100, episodeRewards.size());
        double avgReward = episodeRewards.subList(
            episodeRewards.size() - window, episodeRewards.size())
            .stream().mapToDouble(Double::doubleValue).average().orElse(0);
        
        double avgFood = foodCollected.subList(
            foodCollected.size() - window, foodCollected.size())
            .stream().mapToInt(Integer::intValue).average().orElse(0);
        
        System.out.println(String.format(
            "Episode %d/%d - Avg Reward (last %d): %.2f, Avg Food: %.1f",
            episode, episodeCount, window, avgReward, avgFood));
    }

    private void printFinalStatistics() {
        double finalAvgReward = episodeRewards.subList(
            Math.max(0, episodeRewards.size() - 100), episodeRewards.size())
            .stream().mapToDouble(Double::doubleValue).average().orElse(0);
        
        double finalAvgFood = foodCollected.subList(
            Math.max(0, foodCollected.size() - 100), foodCollected.size())
            .stream().mapToInt(Integer::intValue).average().orElse(0);
        
        System.out.println("\nFinal Training Statistics (last 100 episodes):");
        System.out.println("  Average Reward: " + String.format("%.2f", finalAvgReward));
        System.out.println("  Average Food Collected: " + String.format("%.1f", finalAvgFood));
    }

    public List<Double> getEpisodeRewards() {
        return new ArrayList<>(episodeRewards);
    }

    public List<Integer> getFoodCollected() {
        return new ArrayList<>(foodCollected);
    }

    /**
     * Training results data class
     */
    public static class TrainingResults {
        public final List<Double> rewards;
        public final List<Integer> lengths;
        public final List<Integer> foodCollected;

        public TrainingResults(List<Double> rewards, List<Integer> lengths, List<Integer> foodCollected) {
            this.rewards = new ArrayList<>(rewards);
            this.lengths = new ArrayList<>(lengths);
            this.foodCollected = new ArrayList<>(foodCollected);
        }
    }

    /**
     * Evaluation results data class
     */
    public static class EvaluationResults {
        public final double avgReward;
        public final double avgLength;
        public final double avgFoodCollected;
        public final double successRate;

        public EvaluationResults(double avgReward, double avgLength, 
                                double avgFoodCollected, double successRate) {
            this.avgReward = avgReward;
            this.avgLength = avgLength;
            this.avgFoodCollected = avgFoodCollected;
            this.successRate = successRate;
        }
    }
}
