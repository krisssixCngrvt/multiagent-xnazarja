package cz.cvut.multiagent.agents;

import cz.cvut.multiagent.environment.GridWorld.Action;
import cz.cvut.multiagent.environment.GridWorld.State;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;

/**
 * Deep Q-Network (DQN) agent with experience replay and target network.
 * Uses neural networks to approximate Q-values for continuous state spaces.
 */
public class DQNAgent implements ForagingAgent {
    private final int agentId;
    private final int inputSize;
    private final int outputSize;
    
    // Neural networks
    private final MultiLayerNetwork qNetwork;
    private final MultiLayerNetwork targetNetwork;
    
    // Hyperparameters
    private final double learningRate;
    private final double discountFactor;
    private double epsilon;
    private final double epsilonDecay;
    private final double epsilonMin;
    private final int targetUpdateFrequency;
    
    // Experience replay
    private final ExperienceReplay replayBuffer;
    private final int batchSize;
    
    // Statistics
    private int totalSteps;
    private double totalReward;
    private int updateCounter;
    private final Random random;

    public DQNAgent(int agentId, int inputSize, double learningRate, double discountFactor,
                    double epsilon, double epsilonDecay, double epsilonMin,
                    int replayBufferSize, int batchSize, int targetUpdateFrequency) {
        this.agentId = agentId;
        this.inputSize = inputSize;
        this.outputSize = Action.values().length;
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.epsilon = epsilon;
        this.epsilonDecay = epsilonDecay;
        this.epsilonMin = epsilonMin;
        this.batchSize = batchSize;
        this.targetUpdateFrequency = targetUpdateFrequency;
        this.replayBuffer = new ExperienceReplay(replayBufferSize);
        this.random = new Random();
        this.totalSteps = 0;
        this.totalReward = 0.0;
        this.updateCounter = 0;
        
        // Initialize networks
        this.qNetwork = createNetwork();
        this.targetNetwork = createNetwork();
        updateTargetNetwork();
    }

    private MultiLayerNetwork createNetwork() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(learningRate))
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(inputSize)
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(1, new DenseLayer.Builder()
                .nIn(128)
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(2, new DenseLayer.Builder()
                .nIn(128)
                .nOut(64)
                .activation(Activation.RELU)
                .build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(64)
                .nOut(outputSize)
                .activation(Activation.IDENTITY)
                .build())
            .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        return network;
    }

    @Override
    public Action selectAction(State state) {
        // Epsilon-greedy exploration
        if (random.nextDouble() < epsilon) {
            return Action.values()[random.nextInt(Action.values().length)];
        }
        
        // Get observation and predict Q-values
        double[] observation = state.getObservation(agentId);
        INDArray input = Nd4j.create(observation).reshape(1, inputSize);
        INDArray qValues = qNetwork.output(input);
        
        // Select action with highest Q-value
        int actionIndex = Nd4j.argMax(qValues, 1).getInt(0);
        return Action.values()[actionIndex];
    }

    @Override
    public void learn(State state, Action action, double reward, State nextState, boolean done) {
        totalSteps++;
        totalReward += reward;
        
        // Store experience in replay buffer
        double[] obs = state.getObservation(agentId);
        double[] nextObs = nextState.getObservation(agentId);
        replayBuffer.add(new Experience(obs, action, reward, nextObs, done));
        
        // Train if we have enough experiences
        if (replayBuffer.size() >= batchSize) {
            trainOnBatch();
        }
        
        // Update target network periodically
        updateCounter++;
        if (updateCounter % targetUpdateFrequency == 0) {
            updateTargetNetwork();
        }
        
        // Decay epsilon
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }
    }

    private void trainOnBatch() {
        List<Experience> batch = replayBuffer.sample(batchSize);
        
        INDArray states = Nd4j.create(batchSize, inputSize);
        INDArray targets = Nd4j.create(batchSize, outputSize);
        
        for (int i = 0; i < batch.size(); i++) {
            Experience exp = batch.get(i);
            
            // Set state
            states.putRow(i, Nd4j.create(exp.state));
            
            // Calculate target Q-values
            INDArray currentQ = qNetwork.output(Nd4j.create(exp.state).reshape(1, inputSize));
            INDArray targetQ = currentQ.dup();
            
            if (exp.done) {
                targetQ.putScalar(exp.action.ordinal(), exp.reward);
            } else {
                // Use target network for stability
                INDArray nextQ = targetNetwork.output(Nd4j.create(exp.nextState).reshape(1, inputSize));
                double maxNextQ = nextQ.maxNumber().doubleValue();
                double target = exp.reward + discountFactor * maxNextQ;
                targetQ.putScalar(exp.action.ordinal(), target);
            }
            
            targets.putRow(i, targetQ);
        }
        
        // Train network
        qNetwork.fit(states, targets);
    }

    private void updateTargetNetwork() {
        targetNetwork.setParams(qNetwork.params().dup());
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

    public double getEpsilon() {
        return epsilon;
    }

    public double getTotalReward() {
        return totalReward;
    }

    public int getTotalSteps() {
        return totalSteps;
    }

    public MultiLayerNetwork getNetwork() {
        return qNetwork;
    }

    /**
     * Save model to file
     */
    public void saveModel(String path) throws Exception {
        qNetwork.save(new java.io.File(path));
    }

    /**
     * Load model from file
     */
    public void loadModel(String path) throws Exception {
        MultiLayerNetwork loaded = MultiLayerNetwork.load(new java.io.File(path), true);
        qNetwork.setParams(loaded.params());
        updateTargetNetwork();
    }

    /**
     * Experience replay buffer for storing and sampling transitions
     */
    private static class ExperienceReplay {
        private final int capacity;
        private final List<Experience> buffer;
        private final Random random;

        public ExperienceReplay(int capacity) {
            this.capacity = capacity;
            this.buffer = new ArrayList<>();
            this.random = new Random();
        }

        public void add(Experience experience) {
            if (buffer.size() >= capacity) {
                buffer.remove(0);
            }
            buffer.add(experience);
        }

        public List<Experience> sample(int batchSize) {
            List<Experience> batch = new ArrayList<>();
            for (int i = 0; i < batchSize; i++) {
                batch.add(buffer.get(random.nextInt(buffer.size())));
            }
            return batch;
        }

        public int size() {
            return buffer.size();
        }
    }

    /**
     * Experience tuple for replay buffer
     */
    private static class Experience {
        final double[] state;
        final Action action;
        final double reward;
        final double[] nextState;
        final boolean done;

        Experience(double[] state, Action action, double reward, double[] nextState, boolean done) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
        }
    }
}
