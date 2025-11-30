package cz.cvut.multiagent.agents;

import cz.cvut.multiagent.environment.GridWorld.Action;
import cz.cvut.multiagent.environment.GridWorld.State;

/**
 * Interface for foraging agents in the multi-agent environment.
 */
public interface ForagingAgent {
    /**
     * Select an action based on the current state.
     */
    Action selectAction(State state);
    
    /**
     * Update the agent's policy based on experience.
     */
    void learn(State state, Action action, double reward, State nextState, boolean done);
    
    /**
     * Get the agent's ID.
     */
    int getAgentId();
    
    /**
     * Reset the agent's episode statistics.
     */
    void reset();
}
