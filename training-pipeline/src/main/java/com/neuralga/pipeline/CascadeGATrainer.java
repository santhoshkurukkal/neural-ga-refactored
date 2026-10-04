package com.neuralga.pipeline;

import com.neuralga.config.CascadeConfig;
import com.neuralga.config.GAConfig;
import com.neuralga.config.TrainingConfig;
import com.neuralga.data.DataPipeline;
import com.neuralga.data.SlidingWindowDataset;
import com.neuralga.ga.Chromosome;
import com.neuralga.ga.FitnessFunction;
import com.neuralga.ga.GeneticAlgorithm;
import com.neuralga.ga.Population;
import com.neuralga.neural.BackpropTrainer;
import com.neuralga.neural.CascadeCorrelationNetwork;
import com.neuralga.neural.Network;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

public class CascadeGATrainer {

    private static final Logger log = LoggerFactory.getLogger(CascadeGATrainer.class);

    private final TrainingConfig trainingConfig;
    private final CascadeConfig cascadeConfig;
    private final GAConfig gaConfig;
    private final BackpropTrainer backpropTrainer;

    private CascadeCorrelationNetwork bestNetwork;
    private double bestValLoss = Double.POSITIVE_INFINITY;
    private int totalCascadeIterations = 0;

    public CascadeGATrainer(TrainingConfig trainingConfig, CascadeConfig cascadeConfig, GAConfig gaConfig) {
        this.trainingConfig = trainingConfig;
        this.cascadeConfig = cascadeConfig;
        this.gaConfig = gaConfig;
        this.backpropTrainer = new BackpropTrainer(trainingConfig);
    }

    public CascadeCorrelationNetwork train(DataPipeline.PreparedData data) {
        SlidingWindowDataset trainData = data.getTrainDataset();
        SlidingWindowDataset valData = data.getValDataset();

        // Phase 1: Train initial network (output layer only)
        log.info("=== Phase 1: Training initial output layer ===");
        CascadeCorrelationNetwork network = new CascadeCorrelationNetwork(trainingConfig, cascadeConfig);
        BackpropTrainer.TrainingResult initialResult = backpropTrainer.train(network, trainData, valData);

        bestNetwork = network;
        bestValLoss = initialResult.bestValLoss;
        totalCascadeIterations = 0;

        log.info("Initial network: valLoss={:.6f}, epochs={}", bestValLoss, initialResult.epochsTrained);

        // Phase 2: Cascade correlation with GA
        int cascadeIteration = 0;
        while (shouldContinueCascade(bestValLoss, cascadeIteration)) {
            cascadeIteration++;
            totalCascadeIterations++;

            log.info("=== Cascade iteration {}/{} ===", cascadeIteration, cascadeConfig.getMaxCascadeIterations());

            double newValLoss = addHiddenLayerWithGA(network, trainData, valData);

            if (newValLoss < bestValLoss - cascadeConfig.getMinImprovementForGrowth()) {
                bestValLoss = newValLoss;
                bestNetwork = network;
                log.info("Accepted new hidden layer. New best valLoss: {:.6f}", bestValLoss);
            } else {
                log.info("Rejected new layer (improvement {:.6f} < threshold {:.6f}). Stopping cascade.",
                        bestValLoss - newValLoss, cascadeConfig.getMinImprovementForGrowth());
                break;
            }
        }

        log.info("=== Training complete ===");
        log.info("Best validation loss: {:.6f}", bestValLoss);
        log.info("Hidden layers added: {}", bestNetwork.getHiddenLayerCount());
        log.info("Total cascade iterations: {}", totalCascadeIterations);

        return bestNetwork;
    }

    private boolean shouldContinueCascade(double currentValLoss, int iteration) {
        if (iteration >= cascadeConfig.getMaxCascadeIterations()) {
            log.info("Max cascade iterations reached");
            return false;
        }
        if (!bestNetwork.canAddMoreLayers()) {
            log.info("Max hidden layers reached");
            return false;
        }
        if (currentValLoss <= cascadeConfig.getErrorThreshold()) {
            log.info("Error threshold reached: {:.6f} <= {:.6f}", currentValLoss, cascadeConfig.getErrorThreshold());
            return false;
        }
        return true;
    }

    private double addHiddenLayerWithGA(CascadeCorrelationNetwork network,
                                         SlidingWindowDataset trainData,
                                         SlidingWindowDataset valData) {
        int hiddenSize = trainingConfig.getNeuronsPerHiddenLayer();
        int inputSize = network.getInputSize();
        int chromosomeLength = hiddenSize * inputSize + hiddenSize;

        // Create fitness function for GA
        FitnessFunction fitnessFunction = chromosome -> {
            // Decode chromosome
            double[][] weights = new double[hiddenSize][inputSize];
            double[] biases = new double[hiddenSize];
            int weightCount = hiddenSize * inputSize;

            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < inputSize; j++) {
                    weights[i][j] = chromosome.getGene(i * inputSize + j);
                }
                biases[i] = chromosome.getGene(weightCount + i);
            }

            // Create test network with this hidden layer
            CascadeCorrelationNetwork testNet = cloneNetwork(network);
            testNet.addHiddenLayer(weights, biases);

            // Fine-tune with reduced learning rate
            double originalLR = backpropTrainer.getOptimizer().getLearningRate();
            backpropTrainer.getOptimizer().setLearningRate(
                    originalLR * cascadeConfig.getFineTuneLearningRateFactor());

            // Train for retrainEpochsAfterGrowth
            BackpropTrainer fineTuner = new BackpropTrainer(trainingConfig);
            fineTuner.getOptimizer().setLearningRate(
                    originalLR * cascadeConfig.getFineTuneLearningRateFactor());

            BackpropTrainer.TrainingResult result = fineTuner.train(testNet, trainData, valData);

            // Restore learning rate
            backpropTrainer.getOptimizer().setLearningRate(originalLR);

            return -result.bestValLoss; // Negative because GA maximizes
        };

        // Run GA
        GeneticAlgorithm ga = new GeneticAlgorithm(gaConfig, fitnessFunction);
        ga.setChromosomeLength(chromosomeLength);
        Chromosome bestChromosome = ga.run();

        if (bestChromosome == null) {
            log.warn("GA failed to find solution");
            return Double.POSITIVE_INFINITY;
        }

        // Decode best chromosome and add to actual network
        double[][] bestWeights = new double[hiddenSize][inputSize];
        double[] bestBiases = new double[hiddenSize];
        int weightCount = hiddenSize * inputSize;

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                bestWeights[i][j] = bestChromosome.getGene(i * inputSize + j);
            }
            bestBiases[i] = bestChromosome.getGene(weightCount + i);
        }

        network.addHiddenLayer(bestWeights, bestBiases);

        // Fine-tune the full network
        double originalLR = backpropTrainer.getOptimizer().getLearningRate();
        backpropTrainer.getOptimizer().setLearningRate(
                originalLR * cascadeConfig.getFineTuneLearningRateFactor());

        BackpropTrainer fineTuner = new BackpropTrainer(trainingConfig);
        fineTuner.getOptimizer().setLearningRate(
                originalLR * cascadeConfig.getFineTuneLearningRateFactor());

        BackpropTrainer.TrainingResult result = fineTuner.train(network, trainData, valData);

        backpropTrainer.getOptimizer().setLearningRate(originalLR);

        return result.bestValLoss;
    }

    private CascadeCorrelationNetwork cloneNetwork(CascadeCorrelationNetwork original) {
        CascadeCorrelationNetwork clone = new CascadeCorrelationNetwork(trainingConfig, cascadeConfig);

        // Copy existing hidden layers
        for (int i = 0; i < original.getHiddenLayerCount(); i++) {
            var hidden = original.getHiddenLayers().get(i);
            clone.addHiddenLayer(hidden.getWeightsMatrix(), hidden.getBiases());
        }

        return clone;
    }

    public CascadeCorrelationNetwork getBestNetwork() { return bestNetwork; }
    public double getBestValLoss() { return bestValLoss; }
    public int getTotalCascadeIterations() { return totalCascadeIterations; }

    public Map<String, Object> getTrainingMetrics(SlidingWindowDataset testData) {
        Map<String, Object> metrics = new HashMap<>();
        if (bestNetwork != null && testData != null) {
            double testLoss = backpropTrainer.evaluate(bestNetwork, testData);
            metrics.put("testMSE", testLoss);
            metrics.put("valMSE", bestValLoss);
            metrics.put("hiddenLayers", bestNetwork.getHiddenLayerCount());
            metrics.put("cascadeIterations", totalCascadeIterations);
        }
        return metrics;
    }
}