package rl;

import java.util.*;

/**
 * Q-Learning Agent with optimized hyperparameters for multi-agent foraging.
 * Uses epsilon-greedy exploration and experience replay for stable learning.
 */
public class QLearningAgent {
    private final Map<Integer, double[]> qTable;
    private final int numActions;
    private double epsilon;
    private final double epsilonDecay;
    private final double epsilonMin;
    private final double learningRate;
    private final double discountFactor;
    private final Random random;
    
    // Experience replay
    private final List<Experience> experienceBuffer;
    private final int bufferSize;
    private final int batchSize;
    
    public QLearningAgent(int numActions) {
        this(numActions, 0.9, 0.995, 0.01, 0.1, 0.95);
    }
    
    public QLearningAgent(int numActions, double epsilon, double epsilonDecay, 
                         double epsilonMin, double learningRate, double discountFactor) {
        this.numActions = numActions;
        this.epsilon = epsilon;
        this.epsilonDecay = epsilonDecay;
        this.epsilonMin = epsilonMin;
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.qTable = new HashMap<>();
        this.random = new Random();
        
        // Experience replay settings
        this.experienceBuffer = new ArrayList<>();
        this.bufferSize = 10000;
        this.batchSize = 32;
    }
    
    /**
     * Get Q-values for a state, initializing if needed.
     */
    private double[] getQValues(int state) {
        return qTable.computeIfAbsent(state, k -> {
            double[] values = new double[numActions];
            // Initialize with small random values for exploration
            for (int i = 0; i < numActions; i++) {
                values[i] = random.nextDouble() * 0.01;
            }
            return values;
        });
    }
    
    /**
     * Select an action using epsilon-greedy policy.
     */
    public int selectAction(int state) {
        if (random.nextDouble() < epsilon) {
            return random.nextInt(numActions);
        }
        return getBestAction(state);
    }
    
    /**
     * Select the best action (for evaluation).
     */
    public int getBestAction(int state) {
        double[] qValues = getQValues(state);
        int bestAction = 0;
        double bestValue = qValues[0];
        
        for (int i = 1; i < numActions; i++) {
            if (qValues[i] > bestValue) {
                bestValue = qValues[i];
                bestAction = i;
            }
        }
        return bestAction;
    }
    
    /**
     * Update Q-value using the Q-learning update rule.
     */
    public void learn(int state, int action, double reward, int nextState, boolean done) {
        // Store experience
        storeExperience(state, action, reward, nextState, done);
        
        // Online update
        double[] qValues = getQValues(state);
        double[] nextQValues = getQValues(nextState);
        
        double maxNextQ = done ? 0 : getMaxQ(nextQValues);
        double target = reward + discountFactor * maxNextQ;
        
        // Q-learning update
        qValues[action] += learningRate * (target - qValues[action]);
        
        // Experience replay (batch learning)
        if (experienceBuffer.size() >= batchSize) {
            replayExperience();
        }
    }
    
    private double getMaxQ(double[] qValues) {
        double maxQ = qValues[0];
        for (int i = 1; i < qValues.length; i++) {
            if (qValues[i] > maxQ) {
                maxQ = qValues[i];
            }
        }
        return maxQ;
    }
    
    private void storeExperience(int state, int action, double reward, int nextState, boolean done) {
        if (experienceBuffer.size() >= bufferSize) {
            experienceBuffer.remove(0);
        }
        experienceBuffer.add(new Experience(state, action, reward, nextState, done));
    }
    
    private void replayExperience() {
        if (experienceBuffer.size() < batchSize) return;
        
        // Sample random batch
        List<Experience> batch = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            batch.add(experienceBuffer.get(random.nextInt(experienceBuffer.size())));
        }
        
        // Learn from batch
        for (Experience exp : batch) {
            double[] qValues = getQValues(exp.state);
            double[] nextQValues = getQValues(exp.nextState);
            
            double maxNextQ = exp.done ? 0 : getMaxQ(nextQValues);
            double target = exp.reward + discountFactor * maxNextQ;
            
            qValues[exp.action] += learningRate * 0.5 * (target - qValues[exp.action]);
        }
    }
    
    /**
     * Decay epsilon for exploration-exploitation balance.
     */
    public void decayEpsilon() {
        epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);
    }
    
    public double getEpsilon() {
        return epsilon;
    }
    
    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }
    
    public int getQTableSize() {
        return qTable.size();
    }
    
    /**
     * Experience tuple for replay buffer.
     */
    private static class Experience {
        final int state;
        final int action;
        final double reward;
        final int nextState;
        final boolean done;
        
        Experience(int state, int action, double reward, int nextState, boolean done) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
        }
    }
}
