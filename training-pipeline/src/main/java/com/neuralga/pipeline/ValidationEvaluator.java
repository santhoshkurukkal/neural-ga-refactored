package com.neuralga.pipeline;

import com.neuralga.data.SlidingWindowDataset;
import com.neuralga.neural.BackpropTrainer;
import com.neuralga.neural.CascadeCorrelationNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

public class ValidationEvaluator {

    private static final Logger log = LoggerFactory.getLogger(ValidationEvaluator.class);

    private final BackpropTrainer trainer;

    public ValidationEvaluator(BackpropTrainer trainer) {
        this.trainer = trainer;
    }

    public EvaluationResult evaluate(CascadeCorrelationNetwork network,
                                      SlidingWindowDataset trainData,
                                      SlidingWindowDataset valData,
                                      SlidingWindowDataset testData) {

        double trainLoss = evaluateDataset(network, trainData, "train");
        double valLoss = evaluateDataset(network, valData, "val");
        double testLoss = evaluateDataset(network, testData, "test");

        double r2Train = calculateR2(network, trainData);
        double r2Val = calculateR2(network, valData);
        double r2Test = calculateR2(network, testData);

        Map<String, Double> losses = new HashMap<>();
        losses.put("trainMSE", trainLoss);
        losses.put("valMSE", valLoss);
        losses.put("testMSE", testLoss);

        Map<String, Double> r2Scores = new HashMap<>();
        r2Scores.put("trainR2", r2Train);
        r2Scores.put("valR2", r2Val);
        r2Scores.put("testR2", r2Test);

        log.info("Evaluation: trainMSE={:.6f}, valMSE={:.6f}, testMSE={:.6f}", trainLoss, valLoss, testLoss);
        log.info("R² scores: train={:.4f}, val={:.4f}, test={:.4f}", r2Train, r2Val, r2Test);

        return new EvaluationResult(losses, r2Scores);
    }

    private double evaluateDataset(CascadeCorrelationNetwork network,
                                    SlidingWindowDataset data,
                                    String name) {
        if (data == null || data.size() == 0) {
            log.warn("{} dataset is empty", name);
            return Double.NaN;
        }
        return trainer.evaluate(network, data);
    }

    private double calculateR2(CascadeCorrelationNetwork network, SlidingWindowDataset data) {
        if (data == null || data.size() == 0) return Double.NaN;

        double ssRes = 0.0;
        double ssTot = 0.0;
        double meanTarget = 0.0;

        // Calculate mean
        for (int i = 0; i < data.size(); i++) {
            meanTarget += data.getTarget(i)[0];
        }
        meanTarget /= data.size();

        // Calculate SS
        for (int i = 0; i < data.size(); i++) {
            double[] input = data.getInput(i);
            double[] target = data.getTarget(i);
            double pred = network.forward(input)[0];

            double residual = target[0] - pred;
            ssRes += residual * residual;
            ssTot += (target[0] - meanTarget) * (target[0] - meanTarget);
        }

        if (ssTot == 0) return 1.0;
        return 1.0 - (ssRes / ssTot);
    }

    public static class EvaluationResult {
        public final Map<String, Double> losses;
        public final Map<String, Double> r2Scores;

        public EvaluationResult(Map<String, Double> losses, Map<String, Double> r2Scores) {
            this.losses = losses;
            this.r2Scores = r2Scores;
        }

        public double getTrainMSE() { return losses.get("trainMSE"); }
        public double getValMSE() { return losses.get("valMSE"); }
        public double getTestMSE() { return losses.get("testMSE"); }
        public double getTrainR2() { return r2Scores.get("trainR2"); }
        public double getValR2() { return r2Scores.get("valR2"); }
        public double getTestR2() { return r2Scores.get("testR2"); }

        @Override
        public String toString() {
            return String.format("EvaluationResult{MSE: train=%.6f, val=%.6f, test=%.6f | R²: train=%.4f, val=%.4f, test=%.4f}",
                    getTrainMSE(), getValMSE(), getTestMSE(),
                    getTrainR2(), getValR2(), getTestR2());
        }
    }
}