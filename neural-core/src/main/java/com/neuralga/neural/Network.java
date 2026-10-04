package com.neuralga.neural;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

public class Network {

    protected final List<Layer> layers = new ArrayList<>();
    protected double learningRate = 0.01;

    @JsonCreator
    public Network(
            @JsonProperty("layers") List<Layer> layers,
            @JsonProperty("learningRate") Double learningRate
    ) {
        if (layers != null) {
            this.layers.addAll(layers);
        }
        if (learningRate != null) {
            this.learningRate = learningRate;
        }
    }

    public Network() {}

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public void addLayer(int numNeurons, int inputSize, ActivationFunction activation) {
        addLayer(new Layer(numNeurons, inputSize, activation));
    }

    public double[] forward(double[] input) {
        double[] current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }

    public void backward(double[] target) {
        // Output layer
        Layer outputLayer = layers.get(layers.size() - 1);
        double[] deltas = outputLayer.backwardOutput(target);

        // Hidden layers (backwards)
        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer current = layers.get(i);
            Layer next = layers.get(i + 1);

            // Build weight matrix from next layer (transposed for backward)
            double[][] nextWeights = next.getWeightsMatrix();
            double[] flatWeights = new double[nextWeights.length * nextWeights[0].length];
            for (int r = 0; r < nextWeights.length; r++) {
                for (int c = 0; c < nextWeights[r].length; c++) {
                    flatWeights[r * nextWeights[r].length + c] = nextWeights[r][c];
                }
            }

            deltas = current.backward(deltas, flatWeights);
        }
    }

    public void updateWeights() {
        for (Layer layer : layers) {
            layer.updateWeights(learningRate);
        }
    }

    public void train(double[] input, double[] target) {
        forward(input);
        backward(target);
        updateWeights();
    }

    public double calculateLoss(double[] input, double[] target, LossFunction lossFunction) {
        double[] output = forward(input);
        return lossFunction.compute(output, target);
    }

    public double[] getOutput() {
        if (layers.isEmpty()) return new double[0];
        return layers.get(layers.size() - 1).getOutputs();
    }

    public List<Layer> getLayers() { return List.copyOf(layers); }
    public int getNumLayers() { return layers.size(); }
    public int getInputSize() { return layers.isEmpty() ? 0 : layers.get(0).getInputSize(); }
    public int getOutputSize() { return layers.isEmpty() ? 0 : layers.get(layers.size() - 1).getSize(); }

    public void setLearningRate(double learningRate) { this.learningRate = learningRate; }
    public double getLearningRate() { return learningRate; }

    public NetworkParams getParameters() {
        List<double[][]> allWeights = new ArrayList<>();
        List<double[]> allBiases = new ArrayList<>();

        for (Layer layer : layers) {
            allWeights.add(layer.getWeightsMatrix());
            allBiases.add(layer.getBiases());
        }

        return new NetworkParams(allWeights, allBiases, learningRate);
    }

    public void setParameters(NetworkParams params) {
        if (params.weights.size() != layers.size()) {
            throw new IllegalArgumentException("Parameter layer count mismatch");
        }
        for (int i = 0; i < layers.size(); i++) {
            layers.get(i).setWeightsMatrix(params.weights.get(i));
            layers.get(i).setBiases(params.biases.get(i));
        }
        this.learningRate = params.learningRate;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Network{\n");
        for (int i = 0; i < layers.size(); i++) {
            sb.append("  Layer ").append(i).append(": ").append(layers.get(i)).append("\n");
        }
        sb.append("  learningRate=").append(learningRate).append("\n}");
        return sb.toString();
    }

    public static class NetworkParams {
        public final List<double[][]> weights;
        public final List<double[]> biases;
        public final double learningRate;

        @JsonCreator
        public NetworkParams(
                @JsonProperty("weights") List<double[][]> weights,
                @JsonProperty("biases") List<double[]> biases,
                @JsonProperty("learningRate") double learningRate
        ) {
            this.weights = weights;
            this.biases = biases;
            this.learningRate = learningRate;
        }
    }
}