package cz.cvut.multiagent.environment;

import java.util.*;

/**
 * Grid-world environment for level-based foraging.
 * Multiple agents navigate a grid to collect food items.
 * Each food item and agent has a level, requiring cooperation.
 */
public class GridWorld {
    private final int width;
    private final int height;
    private final List<Agent> agents;
    private final List<Food> foods;
    private final boolean[][] walls;
    private final Random random;
    private int stepCount;
    private final int maxSteps;

    public GridWorld(int width, int height, int maxSteps) {
        this.width = width;
        this.height = height;
        this.maxSteps = maxSteps;
        this.agents = new ArrayList<>();
        this.foods = new ArrayList<>();
        this.walls = new boolean[height][width];
        this.random = new Random();
        this.stepCount = 0;
    }

    public void addAgent(int id, int level, Position position) {
        if (isValidPosition(position) && !isOccupied(position)) {
            agents.add(new Agent(id, level, position));
        }
    }

    public void addFood(int level, Position position) {
        if (isValidPosition(position)) {
            foods.add(new Food(level, position));
        }
    }

    public void addWall(Position position) {
        if (isValidPosition(position)) {
            walls[position.y][position.x] = true;
        }
    }

    /**
     * Execute a step in the environment with all agent actions.
     * @param actions Map of agent ID to action
     * @return Map of agent ID to reward
     */
    public Map<Integer, Double> step(Map<Integer, Action> actions) {
        stepCount++;
        Map<Integer, Double> rewards = new HashMap<>();
        
        // Initialize all rewards to 0
        for (Agent agent : agents) {
            rewards.put(agent.id, 0.0);
        }

        // Execute movement for all agents
        Map<Integer, Position> newPositions = new HashMap<>();
        for (Agent agent : agents) {
            Action action = actions.getOrDefault(agent.id, Action.STAY);
            Position newPos = getNewPosition(agent.position, action);
            
            if (isValidMove(newPos, agent.id, newPositions)) {
                newPositions.put(agent.id, newPos);
            } else {
                newPositions.put(agent.id, agent.position);
                // Small penalty for invalid moves
                rewards.put(agent.id, rewards.get(agent.id) - 0.01);
            }
        }

        // Update agent positions
        for (Agent agent : agents) {
            agent.position = newPositions.get(agent.id);
        }

        // Check for food collection
        List<Food> collectedFood = new ArrayList<>();
        for (Food food : foods) {
            List<Agent> adjacentAgents = getAdjacentAgents(food.position);
            int totalLevel = adjacentAgents.stream().mapToInt(a -> a.level).sum();
            
            if (totalLevel >= food.level) {
                // Food collected! Distribute reward
                double reward = food.level * 1.0;
                for (Agent agent : adjacentAgents) {
                    rewards.put(agent.id, rewards.get(agent.id) + reward / adjacentAgents.size());
                }
                collectedFood.add(food);
            }
        }

        foods.removeAll(collectedFood);

        // Small time penalty to encourage efficiency
        for (Agent agent : agents) {
            rewards.put(agent.id, rewards.get(agent.id) - 0.001);
        }

        return rewards;
    }

    private Position getNewPosition(Position current, Action action) {
        return switch (action) {
            case UP -> new Position(current.x, current.y - 1);
            case DOWN -> new Position(current.x, current.y + 1);
            case LEFT -> new Position(current.x - 1, current.y);
            case RIGHT -> new Position(current.x + 1, current.y);
            case STAY -> current;
        };
    }

    private boolean isValidMove(Position pos, int agentId, Map<Integer, Position> newPositions) {
        if (!isValidPosition(pos) || walls[pos.y][pos.x]) {
            return false;
        }
        
        // Check collision with other agents (in their new positions)
        for (Map.Entry<Integer, Position> entry : newPositions.entrySet()) {
            if (entry.getKey() != agentId && entry.getValue().equals(pos)) {
                return false;
            }
        }
        
        return true;
    }

    private boolean isValidPosition(Position pos) {
        return pos.x >= 0 && pos.x < width && pos.y >= 0 && pos.y < height;
    }

    private boolean isOccupied(Position pos) {
        return agents.stream().anyMatch(a -> a.position.equals(pos));
    }

    private List<Agent> getAdjacentAgents(Position foodPos) {
        List<Agent> adjacent = new ArrayList<>();
        for (Agent agent : agents) {
            if (isAdjacent(agent.position, foodPos)) {
                adjacent.add(agent);
            }
        }
        return adjacent;
    }

    private boolean isAdjacent(Position p1, Position p2) {
        return Math.abs(p1.x - p2.x) <= 1 && Math.abs(p1.y - p2.y) <= 1;
    }

    public boolean isDone() {
        return foods.isEmpty() || stepCount >= maxSteps;
    }

    public void reset() {
        stepCount = 0;
        foods.clear();
        agents.clear();
    }

    public State getState() {
        return new State(
            new ArrayList<>(agents),
            new ArrayList<>(foods),
            width,
            height,
            stepCount
        );
    }

    // Getters
    public int getWidth() { return width; }
    public int getHeight() { return height; }
    public List<Agent> getAgents() { return new ArrayList<>(agents); }
    public List<Food> getFoods() { return new ArrayList<>(foods); }
    public int getStepCount() { return stepCount; }
    public int getMaxSteps() { return maxSteps; }

    // Inner classes
    public static class Agent {
        public final int id;
        public final int level;
        public Position position;

        public Agent(int id, int level, Position position) {
            this.id = id;
            this.level = level;
            this.position = position;
        }

        public Agent copy() {
            return new Agent(id, level, new Position(position.x, position.y));
        }
    }

    public static class Food {
        public final int level;
        public final Position position;

        public Food(int level, Position position) {
            this.level = level;
            this.position = position;
        }

        public Food copy() {
            return new Food(level, new Position(position.x, position.y));
        }
    }

    public static class Position {
        public final int x;
        public final int y;

        public Position(int x, int y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Position position = (Position) o;
            return x == position.x && y == position.y;
        }

        @Override
        public int hashCode() {
            return Objects.hash(x, y);
        }

        @Override
        public String toString() {
            return "(" + x + "," + y + ")";
        }
    }

    public enum Action {
        UP, DOWN, LEFT, RIGHT, STAY
    }

    public static class State {
        public final List<Agent> agents;
        public final List<Food> foods;
        public final int width;
        public final int height;
        public final int stepCount;

        public State(List<Agent> agents, List<Food> foods, int width, int height, int stepCount) {
            this.agents = agents;
            this.foods = foods;
            this.width = width;
            this.height = height;
            this.stepCount = stepCount;
        }

        /**
         * Get observation for a specific agent (partial observability can be added here)
         */
        public double[] getObservation(int agentId) {
            Agent agent = agents.stream()
                .filter(a -> a.id == agentId)
                .findFirst()
                .orElseThrow();

            List<Double> obs = new ArrayList<>();
            
            // Agent's own position (normalized)
            obs.add(agent.position.x / (double) width);
            obs.add(agent.position.y / (double) height);
            obs.add(agent.level / 10.0);

            // Other agents' positions and levels (up to 3 nearest)
            List<Agent> others = agents.stream()
                .filter(a -> a.id != agentId)
                .sorted(Comparator.comparingDouble(a -> 
                    distance(agent.position, a.position)))
                .limit(3)
                .toList();

            for (int i = 0; i < 3; i++) {
                if (i < others.size()) {
                    Agent other = others.get(i);
                    obs.add((other.position.x - agent.position.x) / (double) width);
                    obs.add((other.position.y - agent.position.y) / (double) height);
                    obs.add(other.level / 10.0);
                } else {
                    obs.add(0.0);
                    obs.add(0.0);
                    obs.add(0.0);
                }
            }

            // Food positions and levels (up to 3 nearest)
            List<Food> nearestFoods = foods.stream()
                .sorted(Comparator.comparingDouble(f -> 
                    distance(agent.position, f.position)))
                .limit(3)
                .toList();

            for (int i = 0; i < 3; i++) {
                if (i < nearestFoods.size()) {
                    Food food = nearestFoods.get(i);
                    obs.add((food.position.x - agent.position.x) / (double) width);
                    obs.add((food.position.y - agent.position.y) / (double) height);
                    obs.add(food.level / 10.0);
                } else {
                    obs.add(0.0);
                    obs.add(0.0);
                    obs.add(0.0);
                }
            }

            return obs.stream().mapToDouble(Double::doubleValue).toArray();
        }

        private double distance(Position p1, Position p2) {
            return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
        }
    }
}
