package com.neuralga.neural;

import com.neuralga.config.CascadeConfig;
import com.neuralga.config.TrainingConfig;
import com.neuralga.data.SlidingWindowDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class CascadeCorrelationNetwork extends Network {

    private static final Logger log = LoggerFactory.getLogger(CascadeCorrelationNetwork.class);

    private final TrainingConfig trainingConfig;
    private final CascadeConfig cascadeConfig;
    private final List<Layer> hiddenLayers = new ArrayList<>();
    private Layer outputLayer;

    public CascadeCorrelationNetwork(TrainingConfig trainingConfig, CascadeConfig cascadeConfig) {
        super();
        this.trainingConfig = trainingConfig;
        this.cascadeConfig = cascadeConfig;

        // Create initial output layer (no hidden layers yet)
        int inputSize = trainingConfig.getInputWindow();
        int outputSize = trainingConfig.getPredictionHorizon();
        outputLayer = new Layer(outputSize, inputSize, ActivationFunction.fromString(trainingConfig.getOutputActivation()));
        layers.add(outputLayer);
        setLearningRate(trainingConfig.getLearningRate());
    }

    public void addHiddenLayer(double[][] weights, double[] biases) {
        int inputSize = getInputSize();
        int hiddenSize = trainingConfig.getNeuronsPerHiddenLayer();
        ActivationFunction activation = ActivationFunction.fromString(trainingConfig.getActivation());

        Layer newHidden = new Layer(hiddenSize, inputSize, activation);
        newHidden.setWeightsMatrix(weights);
        newHidden.setBiases(biases);

        // Insert before output layer
        layers.add(layers.size() - 1, newHidden);
        hiddenLayers.add(newHidden);

        // Update output layer to accept hidden layer outputs
        rebuildOutputLayer();

        log.info("Added hidden layer: {} neurons, total layers: {}", hiddenSize, layers.size());
    }

    private void rebuildOutputLayer() {
        int newInputSize = hiddenLayers.isEmpty() ?
                trainingConfig.getInputWindow() :
                trainingConfig.getNeuronsPerHiddenLayer() * hiddenLayers.size() + trainingConfig.getInputWindow();

        // Preserve output layer weights if possible
        double[][] oldWeights = outputLayer.getWeightsMatrix();
        double[] oldBiases = outputLayer.getBiases();

        ActivationFunction outActivation = ActivationFunction.fromString(trainingConfig.getOutputActivation());
        int outputSize = trainingConfig.getPredictionHorizon();

        outputLayer = new Layer(outputSize, newInputSize, outActivation);

        // Initialize new weights, copy old ones where applicable
        double[][] newWeights = outputLayer.getWeightsMatrix();
        double[] newBiases = outputLayer.getBiases();

        int oldInputSize = oldWeights[0].length;
        for (int i = 0; i < outputSize; i++) {
            int copyLen = Math.min(oldInputSize, newInputSize);
            System.arraycopy(oldWeights[i], 0, newWeights[i], 0, copyLen);
            if (i < oldBiases.length) {
                newBiases[i] = oldBiases[i];
            }
        }

        outputLayer.setWeightsMatrix(newWeights);
        outputLayer.setBiases(newBiases);

        // Replace last layer
        layers.set(layers.size() - 1, outputLayer);
    }

    @Override
    public double[] forward(double[] input) {
        // For cascade, we need to concatenate original input with hidden layer outputs
        double[] current = input;

        // Pass through hidden layers
        double[] hiddenOutputs = new double[0];
        for (Layer hidden : hiddenLayers) {
            double[] hiddenOut = hidden.forward(current);
            // Concatenate hidden output to current
            double[] newCurrent = new double[current.length + hiddenOut.length];
            System.arraycopy(current, 0, newCurrent, 0, current.length);
            System.arraycopy(hiddenOut, 0, newCurrent, current.length, hiddenOut.length);
            current = newCurrent;
            hiddenOutputs = hiddenOut;
        }

        // Pass through output layer
        return outputLayer.forward(current);
    }

    public int getHiddenLayerCount() {
        return hiddenLayers.size();
    }

    public boolean canAddMoreLayers() {
        return hiddenLayers.size() < trainingConfig.getMaxHiddenLayers();
    }

    public List<Layer> getHiddenLayers() {
        return List.copyOf(hiddenLayers);
    }

    public Layer getOutputLayer() {
        return outputLayer;
    }

    public static class CandidateHiddenLayer {
        public final double[][] weights;
        public final double[] biases;
        public final double validationScore;

        public CandidateHiddenLayer(double[][] weights, double[] biases, double validationScore) {
            this.weights = weights;
            this.biases = biases;
            this.validationScore = validationScore;
        }
    }
}