package com.neuralga.neural;

import com.neuralga.config.TrainingConfig;
import com.neuralga.data.SlidingWindowDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

public class BackpropTrainer {

    private static final Logger log = LoggerFactory.getLogger(BackpropTrainer.class);

    private final TrainingConfig config;
    private final LossFunction lossFunction;
    private final Optimizer optimizer;
    private final EarlyStopping earlyStopping;
    private final Random random;

    private double lastTrainLoss = Double.NaN;
    private double lastValLoss = Double.NaN;
    private int epochsTrained = 0;

    public BackpropTrainer(TrainingConfig config) {
        this.config = config;
        this.lossFunction = LossFunction.fromString(config.getLoss());
        this.optimizer = Optimizer.create(config.getOptimizer(), config.getLearningRate());
        this.earlyStopping = new EarlyStopping(
                config.getEarlyStopping().getPatience(),
                config.getEarlyStopping().getMinDelta()
        );
        this.random = new Random(config.getInputWindow()); // seed from input window as default
    }

    public TrainingResult train(Network network, SlidingWindowDataset trainData, SlidingWindowDataset valData) {
        earlyStopping.reset();
        epochsTrained = 0;

        log.info("Starting training: epochs={}, batchSize={}, lr={}, optimizer={}",
                config.getEpochs(), config.getBatchSize(), config.getLearningRate(), config.getOptimizer());

        for (int epoch = 1; epoch <= config.getEpochs(); epoch++) {
            double trainLoss = trainEpoch(network, trainData);
            double valLoss = evaluate(network, valData);

            lastTrainLoss = trainLoss;
            lastValLoss = valLoss;
            epochsTrained = epoch;

            log.debug("Epoch {}/{}: trainLoss={:.6f}, valLoss={:.6f}",
                    epoch, config.getEpochs(), trainLoss, valLoss);

            if (earlyStopping.shouldStop(valLoss, network)) {
                log.info("Early stopping at epoch {} (best valLoss={:.6f})", epoch, earlyStopping.getBestScore());
                break;
            }
        }

        return new TrainingResult(epochsTrained, lastTrainLoss, lastValLoss, earlyStopping.getBestScore());
    }

    public double trainEpoch(Network network, SlidingWindowDataset data) {
        int n = data.size();
        int batchSize = Math.min(config.getBatchSize(), n);
        double totalLoss = 0.0;
        int batches = 0;

        // Shuffle indices for mini-batch training
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        shuffle(indices);

        for (int start = 0; start < n; start += batchSize) {
            int end = Math.min(start + batchSize, n);
            double batchLoss = 0.0;

            for (int i = start; i < end; i++) {
                int idx = indices[i];
                double[] input = data.getInput(idx);
                double[] target = data.getTarget(idx);

                network.train(input, target);
                batchLoss += lossFunction.compute(network.getOutput(), target);
            }

            totalLoss += batchLoss / (end - start);
            batches++;
        }

        return batches > 0 ? totalLoss / batches : 0.0;
    }

    public double evaluate(Network network, SlidingWindowDataset data) {
        if (data.size() == 0) return Double.NaN;

        double totalLoss = 0.0;
        for (int i = 0; i < data.size(); i++) {
            double[] input = data.getInput(i);
            double[] target = data.getTarget(i);
            totalLoss += lossFunction.compute(network.forward(input), target);
        }
        return totalLoss / data.size();
    }

    public double[] predict(Network network, double[] input) {
        return network.forward(input);
    }

    private void shuffle(int[] array) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    public double getLastTrainLoss() { return lastTrainLoss; }
    public double getLastValLoss() { return lastValLoss; }
    public int getEpochsTrained() { return epochsTrained; }
    public LossFunction getLossFunction() { return lossFunction; }
    public Optimizer getOptimizer() { return optimizer; }

    public static class TrainingResult {
        public final int epochsTrained;
        public final double finalTrainLoss;
        public final double finalValLoss;
        public final double bestValLoss;

        public TrainingResult(int epochsTrained, double finalTrainLoss, double finalValLoss, double bestValLoss) {
            this.epochsTrained = epochsTrained;
            this.finalTrainLoss = finalTrainLoss;
            this.finalValLoss = finalValLoss;
            this.bestValLoss = bestValLoss;
        }

        @Override
        public String toString() {
            return String.format("TrainingResult{epochs=%d, trainLoss=%.6f, valLoss=%.6f, bestValLoss=%.6f}",
                    epochsTrained, finalTrainLoss, finalValLoss, bestValLoss);
        }
    }
}