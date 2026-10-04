package com.neuralga.neural;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public abstract class Optimizer {
    protected double learningRate;

    @JsonCreator
    public Optimizer(@JsonProperty("learningRate") Double learningRate) {
        this.learningRate = learningRate != null ? learningRate : 0.01;
    }

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public abstract void update(Layer layer, double learningRate);

    public void setLearningRate(double lr) { this.learningRate = lr; }
    public double getLearningRate() { return learningRate; }

    public static Optimizer create(String name, double learningRate) {
        if (name == null) return new SGD(learningRate);
        return switch (name.toUpperCase()) {
            case "SGD" -> new SGD(learningRate);
            case "ADAM" -> new Adam(learningRate);
            case "RMSPROP" -> new RMSprop(learningRate);
            default -> new SGD(learningRate);
        };
    }

    public static class SGD extends Optimizer {
        public SGD(double learningRate) { super(learningRate); }
        @Override
        public void update(Layer layer, double learningRate) {
            double lr = learningRate > 0 ? learningRate : this.learningRate;
            for (Neuron neuron : layer.getNeurons()) {
                for (int i = 0; i < neuron.weights.length; i++) {
                    neuron.weights[i] += lr * neuron.getDelta() * layer.lastInputs[i];
                }
                neuron.bias += lr * neuron.getDelta();
            }
        }
    }

    public static class Adam extends Optimizer {
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private double epsilon = 1e-8;
        private int t = 0;

        // Per-neuron moment estimates
        private double[][] mWeights;
        private double[][] vWeights;
        private double[] mBias;
        private double[] vBias;

        public Adam(double learningRate) { super(learningRate); }

        @Override
        public void update(Layer layer, double learningRate) {
            double lr = learningRate > 0 ? learningRate : this.learningRate;
            t++;

            Neuron[] neurons = layer.getNeurons();
            int numNeurons = neurons.length;
            int inputSize = neurons[0].getInputSize();

            if (mWeights == null) {
                mWeights = new double[numNeurons][inputSize];
                vWeights = new double[numNeurons][inputSize];
                mBias = new double[numNeurons];
                vBias = new double[numNeurons];
            }

            for (int n = 0; n < numNeurons; n++) {
                Neuron neuron = neurons[n];
                double delta = neuron.getDelta();

                for (int i = 0; i < inputSize; i++) {
                    double grad = delta * layer.lastInputs[i];
                    mWeights[n][i] = beta1 * mWeights[n][i] + (1 - beta1) * grad;
                    vWeights[n][i] = beta2 * vWeights[n][i] + (1 - beta2) * grad * grad;

                    double mHat = mWeights[n][i] / (1 - Math.pow(beta1, t));
                    double vHat = vWeights[n][i] / (1 - Math.pow(beta2, t));

                    neuron.weights[i] += lr * mHat / (Math.sqrt(vHat) + epsilon);
                }

                double biasGrad = delta;
                mBias[n] = beta1 * mBias[n] + (1 - beta1) * biasGrad;
                vBias[n] = beta2 * vBias[n] + (1 - beta2) * biasGrad * biasGrad;

                double mHatBias = mBias[n] / (1 - Math.pow(beta1, t));
                double vHatBias = vBias[n] / (1 - Math.pow(beta2, t));

                neuron.bias += lr * mHatBias / (Math.sqrt(vHatBias) + epsilon);
            }
        }

        public void reset() { t = 0; }
    }

    public static class RMSprop extends Optimizer {
        private double decayRate = 0.99;
        private double epsilon = 1e-8;
        private double[][] cache;

        public RMSprop(double learningRate) { super(learningRate); }

        @Override
        public void update(Layer layer, double learningRate) {
            double lr = learningRate > 0 ? learningRate : this.learningRate;
            Neuron[] neurons = layer.getNeurons();
            int numNeurons = neurons.length;
            int inputSize = neurons[0].getInputSize();

            if (cache == null) {
                cache = new double[numNeurons][inputSize];
            }

            for (int n = 0; n < numNeurons; n++) {
                Neuron neuron = neurons[n];
                double delta = neuron.getDelta();

                for (int i = 0; i < inputSize; i++) {
                    double grad = delta * layer.lastInputs[i];
                    cache[n][i] = decayRate * cache[n][i] + (1 - decayRate) * grad * grad;
                    neuron.weights[i] += lr * grad / (Math.sqrt(cache[n][i]) + epsilon);
                }
                neuron.bias += lr * delta;
            }
        }
    }
}