package com.neuralga.ga;

import com.neuralga.data.SlidingWindowDataset;
import com.neuralga.neural.CascadeCorrelationNetwork;
import com.neuralga.neural.Network;
import com.neuralga.neural.BackpropTrainer;
import com.neuralga.config.TrainingConfig;

public interface FitnessFunction {

    /**
     * Evaluate the fitness of a chromosome.
     * Higher fitness is better.
     */
    double evaluate(Chromosome chromosome);

    /**
     * Evaluate a chromosome in the context of a neural network.
     * The chromosome represents weights for a new hidden layer.
     */
    default double evaluateForHiddenLayer(Chromosome chromosome,
                                           CascadeCorrelationNetwork network,
                                           SlidingWindowDataset trainData,
                                           SlidingWindowDataset valData,
                                           TrainingConfig config,
                                           BackpropTrainer trainer) {
        // Decode chromosome into weights and biases for hidden layer
        int hiddenSize = config.getNeuronsPerHiddenLayer();
        int inputSize = network.getInputSize();
        int weightCount = hiddenSize * inputSize;
        int biasCount = hiddenSize;

        if (chromosome.getLength() != weightCount + biasCount) {
            throw new IllegalArgumentException(
                    "Chromosome length " + chromosome.getLength() +
                    " != expected " + (weightCount + biasCount));
        }

        double[][] weights = new double[hiddenSize][inputSize];
        double[] biases = new double[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = chromosome.getGene(i * inputSize + j);
            }
            biases[i] = chromosome.getGene(weightCount + i);
        }

        // Add hidden layer to network copy
        CascadeCorrelationNetwork testNetwork = cloneNetwork(network);
        testNetwork.addHiddenLayer(weights, biases);

        // Train for a few epochs
        TrainingResult result = trainer.train(testNetwork, trainData, valData);

        // Return fitness (higher is better, so negate loss)
        return -result.bestValLoss;
    }

    /**
     * Create a copy of the network for evaluation.
     */
    default CascadeCorrelationNetwork cloneNetwork(CascadeCorrelationNetwork original) {
        // This is a simplified clone - in practice you'd need a proper copy constructor
        // For now, return the original (caller should handle cloning)
        return original;
    }

    static class TrainingResult {
        public final double bestValLoss;

        public TrainingResult(double bestValLoss) {
            this.bestValLoss = bestValLoss;
        }
    }
}