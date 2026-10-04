package com.neuralga.neural;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.Random;

public class Layer {

    protected final Neuron[] neurons;
    protected final ActivationFunction activation;
    protected final Random random;
    protected double[] lastInputs;
    protected double[] outputs;

    @JsonCreator
    public Layer(
            @JsonProperty("neurons") Neuron[] neurons,
            @JsonProperty("activation") String activationName
    ) {
        this.neurons = neurons != null ? neurons : new Neuron[0];
        this.activation = ActivationFunction.fromString(activationName);
        this.random = new Random();
        this.outputs = new double[this.neurons.length];
    }

    public Layer(int numNeurons, int inputSize, ActivationFunction activation, long seed) {
        this.neurons = new Neuron[numNeurons];
        this.activation = activation;
        this.random = new Random(seed);
        this.outputs = new double[numNeurons];

        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(inputSize, activation, seed + i);
        }
    }

    public Layer(int numNeurons, int inputSize, ActivationFunction activation) {
        this(numNeurons, inputSize, activation, System.nanoTime());
    }

    public double[] forward(double[] inputs) {
        if (inputs.length != neurons[0].getInputSize()) {
            throw new IllegalArgumentException(
                    "Input size " + inputs.length + " != expected " + neurons[0].getInputSize());
        }

        lastInputs = inputs.clone();
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].forward(inputs);
        }
        return outputs.clone();
    }

    public double[] backward(double[] nextLayerDeltas, double[] nextLayerWeights) {
        // For hidden layers: compute deltas based on next layer
        double[] deltas = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            double error = 0.0;
            for (int j = 0; j < nextLayerDeltas.length; j++) {
                error += nextLayerDeltas[j] * nextLayerWeights[j * neurons.length + i];
            }
            neurons[i].backward(lastInputs, error);
            deltas[i] = neurons[i].getDelta();
        }
        return deltas;
    }

    public double[] backwardOutput(double[] targets) {
        // For output layer: compute deltas directly from targets
        double[] deltas = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            double error = targets[i] - neurons[i].getOutput();
            neurons[i].backward(lastInputs, error);
            deltas[i] = neurons[i].getDelta();
        }
        return deltas;
    }

    public void updateWeights(double learningRate) {
        for (Neuron neuron : neurons) {
            for (int i = 0; i < neuron.weights.length; i++) {
                neuron.weights[i] += learningRate * neuron.getDelta() * lastInputs[i];
            }
            neuron.bias += learningRate * neuron.getDelta();
        }
    }

    public double[] getOutputs() { return outputs.clone(); }
    public Neuron[] getNeurons() { return neurons; }
    public int getSize() { return neurons.length; }
    public int getInputSize() { return neurons.length > 0 ? neurons[0].getInputSize() : 0; }
    public ActivationFunction getActivation() { return activation; }

    public double[][] getWeightsMatrix() {
        double[][] matrix = new double[neurons.length][];
        for (int i = 0; i < neurons.length; i++) {
            matrix[i] = neurons[i].getWeights();
        }
        return matrix;
    }

    public double[] getBiases() {
        double[] biases = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            biases[i] = neurons[i].getBias();
        }
        return biases;
    }

    public void setWeightsMatrix(double[][] weights) {
        for (int i = 0; i < neurons.length && i < weights.length; i++) {
            neurons[i].setWeights(weights[i]);
        }
    }

    public void setBiases(double[] biases) {
        for (int i = 0; i < neurons.length && i < biases.length; i++) {
            neurons[i].setBias(biases[i]);
        }
    }

    @Override
    public String toString() {
        return String.format("Layer{neurons=%d, inputSize=%d, activation=%s}",
                neurons.length, getInputSize(), activation.name());
    }
}