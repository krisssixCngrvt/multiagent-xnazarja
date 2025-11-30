package cz.cvut.multiagent.agents;

import cz.cvut.multiagent.environment.GridWorld;
import cz.cvut.multiagent.environment.GridWorld.Action;
import cz.cvut.multiagent.environment.GridWorld.State;

import java.util.*;

/**
 * Tabular Q-Learning agent for the foraging environment.
 * Uses epsilon-greedy exploration and Q-table updates.
 */
public class QLearningAgent implements ForagingAgent {
    private final int agentId;
    private final double learningRate;
    private final double discountFactor;
    private double epsilon;
    private final double epsilonDecay;
    private final double epsilonMin;
    
    // Q-table: state hash -> action -> Q-value
    private final Map<String, Map<Action, Double>> qTable;
    private final Random random;
    
    // Statistics
    private int totalSteps;
    private double totalReward;

    public QLearningAgent(int agentId, double learningRate, double discountFactor, 
                          double epsilon, double epsilonDecay, double epsilonMin) {
        this.agentId = agentId;
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.epsilon = epsilon;
        this.epsilonDecay = epsilonDecay;
        this.epsilonMin = epsilonMin;
        this.qTable = new HashMap<>();
        this.random = new Random();
        this.totalSteps = 0;
        this.totalReward = 0.0;
    }

    @Override
    public Action selectAction(State state) {
        String stateKey = getStateKey(state);
        
        // Initialize Q-values for this state if not present
        initializeQValues(stateKey);
        
        // Epsilon-greedy exploration
        if (random.nextDouble() < epsilon) {
            return Action.values()[random.nextInt(Action.values().length)];
        }
        
        return getBestAction(stateKey);
    }

    @Override
    public void learn(State state, Action action, double reward, State nextState, boolean done) {
        totalSteps++;
        totalReward += reward;
        
        String stateKey = getStateKey(state);
        String nextStateKey = getStateKey(nextState);
        
        // Initialize Q-values if needed
        initializeQValues(stateKey);
        initializeQValues(nextStateKey);
        
        // Current Q-value
        double currentQ = qTable.get(stateKey).get(action);
        
        // Max Q-value for next state
        double maxNextQ = done ? 0.0 : getMaxQValue(nextStateKey);
        
        // Q-learning update: Q(s,a) = Q(s,a) + α * (r + γ * max Q(s',a') - Q(s,a))
        double newQ = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
        qTable.get(stateKey).put(action, newQ);
        
        // Decay epsilon
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }
    }

    @Override
    public int getAgentId() {
        return agentId;
    }

    @Override
    public void reset() {
        totalSteps = 0;
        totalReward = 0.0;
    }

    /**
     * Create a simplified state representation for Q-table indexing.
     * Discretizes positions to reduce state space.
     */
    private String getStateKey(State state) {
        StringBuilder key = new StringBuilder();
        
        // Find this agent - return default key if agent not found (episode over)
        var agentOpt = state.agents.stream()
            .filter(a -> a.id == agentId)
            .findFirst();
        
        if (agentOpt.isEmpty()) {
            return "terminal";
        }
        
        var agent = agentOpt.get();
        
        // Agent position (discretized to 2x2 grid cells)
        int gridX = agent.position.x / 2;
        int gridY = agent.position.y / 2;
        key.append(gridX).append(",").append(gridY).append(";");
        
        // Nearest food position (relative)
        if (!state.foods.isEmpty()) {
            var nearestFood = state.foods.stream()
                .min(Comparator.comparingDouble(f -> 
                    Math.abs(f.position.x - agent.position.x) + 
                    Math.abs(f.position.y - agent.position.y)))
                .get();
            
            int relX = Integer.signum(nearestFood.position.x - agent.position.x);
            int relY = Integer.signum(nearestFood.position.y - agent.position.y);
            key.append(relX).append(",").append(relY).append(",").append(nearestFood.level);
        } else {
            key.append("done");
        }
        
        return key.toString();
    }

    private void initializeQValues(String stateKey) {
        if (!qTable.containsKey(stateKey)) {
            Map<Action, Double> actions = new HashMap<>();
            for (Action action : Action.values()) {
                actions.put(action, 0.0);
            }
            qTable.put(stateKey, actions);
        }
    }

    private Action getBestAction(String stateKey) {
        Map<Action, Double> actions = qTable.get(stateKey);
        return actions.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(Action.STAY);
    }

    private double getMaxQValue(String stateKey) {
        Map<Action, Double> actions = qTable.get(stateKey);
        return actions.values().stream()
            .max(Double::compare)
            .orElse(0.0);
    }

    public double getEpsilon() {
        return epsilon;
    }

    public int getQTableSize() {
        return qTable.size();
    }

    public double getTotalReward() {
        return totalReward;
    }

    public int getTotalSteps() {
        return totalSteps;
    }
}
