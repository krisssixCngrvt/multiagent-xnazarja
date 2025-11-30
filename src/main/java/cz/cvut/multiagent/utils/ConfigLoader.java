package cz.cvut.multiagent.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Configuration loader for hyperparameters and environment settings.
 */
public class ConfigLoader {
    private static final Gson gson = new GsonBuilder().setPrettyPrinting().create();

    public static class Config {
        // Environment settings
        public int gridWidth = 8;
        public int gridHeight = 8;
        public int maxSteps = 200;
        public int numAgents = 4;
        public int minFood = 3;
        public int maxFood = 5;

        // Training settings
        public int trainingEpisodes = 1000;
        public int evaluationEpisodes = 100;
        public boolean verbose = true;

        // Q-Learning hyperparameters
        public double qLearningRate = 0.1;
        public double qDiscountFactor = 0.95;
        public double qInitialEpsilon = 1.0;
        public double qEpsilonDecay = 0.995;
        public double qMinEpsilon = 0.01;

        // DQN hyperparameters
        public double dqnLearningRate = 0.001;
        public double dqnDiscountFactor = 0.95;
        public double dqnInitialEpsilon = 1.0;
        public double dqnEpsilonDecay = 0.995;
        public double dqnMinEpsilon = 0.01;
        public int dqnReplayBufferSize = 10000;
        public int dqnBatchSize = 32;
        public int dqnTargetUpdateFreq = 100;

        // Neural network architecture
        public int[] hiddenLayers = {128, 128, 64};
        public String activation = "relu";
    }

    /**
     * Load configuration from JSON file
     */
    public static Config loadConfig(String filepath) throws IOException {
        try (FileReader reader = new FileReader(filepath)) {
            return gson.fromJson(reader, Config.class);
        }
    }

    /**
     * Save configuration to JSON file
     */
    public static void saveConfig(Config config, String filepath) throws IOException {
        try (FileWriter writer = new FileWriter(filepath)) {
            gson.toJson(config, writer);
        }
    }

    /**
     * Get default configuration
     */
    public static Config getDefaultConfig() {
        return new Config();
    }
}
