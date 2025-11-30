package rl;

import java.util.*;

/**
 * Grid-world foraging environment for multi-agent reinforcement learning.
 * Agents must collect food items scattered across the grid to maximize score.
 */
public class GridWorld {
    private static final int DEFAULT_SEED = 42;
    
    private final int width;
    private final int height;
    private final int[][] grid;
    private final List<int[]> agentPositions;
    private final Random random;
    private int totalFood;
    private int collectedFood;
    private int steps;
    private final int maxSteps;
    
    // Grid cell types
    public static final int EMPTY = 0;
    public static final int FOOD = 1;
    public static final int OBSTACLE = 2;
    
    // Actions
    public static final int UP = 0;
    public static final int DOWN = 1;
    public static final int LEFT = 2;
    public static final int RIGHT = 3;
    public static final int STAY = 4;
    public static final int NUM_ACTIONS = 5;
    
    public GridWorld(int width, int height, int numAgents, int numFood, int numObstacles, int maxSteps) {
        this.width = width;
        this.height = height;
        this.maxSteps = maxSteps;
        this.grid = new int[height][width];
        this.agentPositions = new ArrayList<>();
        this.random = new Random(DEFAULT_SEED);
        
        initialize(numAgents, numFood, numObstacles);
    }
    
    private void initialize(int numAgents, int numFood, int numObstacles) {
        // Clear grid
        for (int y = 0; y < height; y++) {
            Arrays.fill(grid[y], EMPTY);
        }
        
        agentPositions.clear();
        collectedFood = 0;
        steps = 0;
        
        // Place food items
        totalFood = numFood;
        int placed = 0;
        while (placed < numFood) {
            int x = random.nextInt(width);
            int y = random.nextInt(height);
            if (grid[y][x] == EMPTY) {
                grid[y][x] = FOOD;
                placed++;
            }
        }
        
        // Place obstacles
        placed = 0;
        while (placed < numObstacles) {
            int x = random.nextInt(width);
            int y = random.nextInt(height);
            if (grid[y][x] == EMPTY) {
                grid[y][x] = OBSTACLE;
                placed++;
            }
        }
        
        // Place agents at random empty positions
        for (int i = 0; i < numAgents; i++) {
            boolean agentPlaced = false;
            while (!agentPlaced) {
                int x = random.nextInt(width);
                int y = random.nextInt(height);
                if (grid[y][x] == EMPTY && !isAgentAt(x, y)) {
                    agentPositions.add(new int[]{x, y});
                    agentPlaced = true;
                }
            }
        }
    }
    
    private boolean isAgentAt(int x, int y) {
        for (int[] pos : agentPositions) {
            if (pos[0] == x && pos[1] == y) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Get the state representation for an agent.
     * State includes: relative positions of nearby food and obstacles.
     */
    public int getState(int agentIndex) {
        int[] pos = agentPositions.get(agentIndex);
        int x = pos[0];
        int y = pos[1];
        
        // Create state based on 3x3 local view + distance to nearest food
        int state = 0;
        int multiplier = 1;
        
        // Encode 3x3 neighborhood
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                int cellValue = 0;
                
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    cellValue = 3; // Wall/boundary
                } else {
                    cellValue = grid[ny][nx];
                }
                
                state += cellValue * multiplier;
                multiplier *= 4;
            }
        }
        
        // Add direction to nearest food
        int nearestFoodDir = getNearestFoodDirection(x, y);
        state += nearestFoodDir * multiplier;
        
        return state;
    }
    
    private int getNearestFoodDirection(int x, int y) {
        int nearestDist = Integer.MAX_VALUE;
        int nearestDir = 4; // No food
        
        for (int gy = 0; gy < height; gy++) {
            for (int gx = 0; gx < width; gx++) {
                if (grid[gy][gx] == FOOD) {
                    int dist = Math.abs(gx - x) + Math.abs(gy - y);
                    if (dist < nearestDist) {
                        nearestDist = dist;
                        // Determine direction
                        if (gy < y) nearestDir = UP;
                        else if (gy > y) nearestDir = DOWN;
                        else if (gx < x) nearestDir = LEFT;
                        else if (gx > x) nearestDir = RIGHT;
                        else nearestDir = STAY;
                    }
                }
            }
        }
        
        return nearestDir;
    }
    
    /**
     * Execute an action for an agent and return the reward.
     */
    public double step(int agentIndex, int action) {
        int[] pos = agentPositions.get(agentIndex);
        int x = pos[0];
        int y = pos[1];
        
        int newX = x;
        int newY = y;
        
        switch (action) {
            case UP: newY--; break;
            case DOWN: newY++; break;
            case LEFT: newX--; break;
            case RIGHT: newX++; break;
            case STAY: break;
        }
        
        double reward = -0.01; // Small step penalty to encourage efficiency
        
        // Check bounds
        if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
            // Check for obstacles
            if (grid[newY][newX] != OBSTACLE && !isAgentAt(newX, newY)) {
                pos[0] = newX;
                pos[1] = newY;
                
                // Check for food collection
                if (grid[newY][newX] == FOOD) {
                    grid[newY][newX] = EMPTY;
                    collectedFood++;
                    reward = 10.0; // Large reward for food
                }
            } else {
                reward = -0.5; // Penalty for hitting obstacle
            }
        } else {
            reward = -0.5; // Penalty for hitting wall
        }
        
        steps++;
        return reward;
    }
    
    /**
     * Execute actions for all agents simultaneously.
     */
    public double[] stepAll(int[] actions) {
        double[] rewards = new double[actions.length];
        for (int i = 0; i < actions.length; i++) {
            rewards[i] = step(i, actions[i]);
        }
        return rewards;
    }
    
    public boolean isDone() {
        return collectedFood >= totalFood || steps >= maxSteps;
    }
    
    public void reset(int numAgents, int numFood, int numObstacles) {
        // Use time-based seed for variety during training episodes
        // This ensures agents learn to generalize across different layouts
        random.setSeed(System.currentTimeMillis());
        initialize(numAgents, numFood, numObstacles);
    }
    
    public int getCollectedFood() {
        return collectedFood;
    }
    
    public int getTotalFood() {
        return totalFood;
    }
    
    public int getSteps() {
        return steps;
    }
    
    public int getNumAgents() {
        return agentPositions.size();
    }
    
    public int getWidth() {
        return width;
    }
    
    public int getHeight() {
        return height;
    }
    
    public int[][] getGrid() {
        return grid;
    }
    
    public List<int[]> getAgentPositions() {
        return agentPositions;
    }
    
    /**
     * Calculate score based on food collected and efficiency.
     */
    public double getScore() {
        double foodScore = (double) collectedFood / totalFood * 100;
        double efficiencyBonus = collectedFood > 0 ? Math.max(0, 50 - steps / (double) collectedFood) : 0;
        return foodScore + efficiencyBonus;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                boolean hasAgent = false;
                for (int i = 0; i < agentPositions.size(); i++) {
                    if (agentPositions.get(i)[0] == x && agentPositions.get(i)[1] == y) {
                        sb.append((char)('A' + i));
                        hasAgent = true;
                        break;
                    }
                }
                if (!hasAgent) {
                    switch (grid[y][x]) {
                        case EMPTY: sb.append('.'); break;
                        case FOOD: sb.append('F'); break;
                        case OBSTACLE: sb.append('#'); break;
                    }
                }
            }
            sb.append('\n');
        }
        return sb.toString();
    }
}
