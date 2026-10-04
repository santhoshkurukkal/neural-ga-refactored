package com.neuralga.neural;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Random;

public class Neuron {

    protected double[] weights;
    protected double bias;
    protected double output;
    protected double delta;
    protected final ActivationFunction activation;
    protected final Random random;

    @JsonCreator
    public Neuron(
            @JsonProperty("weights") double[] weights,
            @JsonProperty("bias") double bias,
            @JsonProperty("activation") String activationName,
            @JsonProperty("inputSize") int inputSize
    ) {
        this.weights = weights != null ? weights.clone() : new double[0];
        this.bias = bias;
        this.activation = ActivationFunction.fromString(activationName);
        this.random = new Random();
        this.output = 0.0;
        this.delta = 0.0;
        // If weights array is empty but inputSize is provided, initialize
        if (this.weights.length == 0 && inputSize > 0) {
            this.weights = new double[inputSize];
            initializeWeights();
        }
    }

    public Neuron(int inputSize, ActivationFunction activation, long seed) {
        this.weights = new double[inputSize];
        this.activation = activation;
        this.random = new Random(seed);
        this.output = 0.0;
        this.delta = 0.0;
        initializeWeights();
    }

    public Neuron(int inputSize, ActivationFunction activation) {
        this(inputSize, activation, System.nanoTime());
    }

    protected void initializeWeights() {
        // Xavier/Glorot initialization
        double limit = Math.sqrt(6.0 / (weights.length + 1));
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextDouble() * 2 * limit - limit;
        }
        bias = random.nextDouble() * 2 * limit - limit;
    }

    public double forward(double[] inputs) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException(
                    "Input size " + inputs.length + " != weight size " + weights.length);
        }

        double sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        output = activation.activate(sum);
        return output;
    }

    public void backward(double[] inputs, double error) {
        double derivative = activation.derivative(output);
        delta = error * derivative;

        for (int i = 0; i < weights.length; i++) {
            weights[i] += inputs[i] * delta;
        }
        bias += delta;
    }

    public double getOutput() { return output; }
    public double getDelta() { return delta; }
    public double[] getWeights() { return weights.clone(); }
    public double getBias() { return bias; }
    public ActivationFunction getActivation() { return activation; }
    public int getInputSize() { return weights.length; }

    @com.fasterxml.jackson.annotation.JsonIgnore
    public void setInputSize(int inputSize) {
        // Only used for deserialization
    }

    public void setWeights(double[] weights) {
        this.weights = weights != null ? weights.clone() : new double[0];
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public void setWeightsAndBias(double[] weights, double bias) {
        setWeights(weights);
        setBias(bias);
    }

    @Override
    public String toString() {
        return String.format("Neuron{inputs=%d, activation=%s, bias=%.4f}",
                weights.length, activation.name(), bias);
    }
}