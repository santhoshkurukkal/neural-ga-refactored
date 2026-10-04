package com.neuralga.service;

import com.neuralga.data.SlidingWindowDataset;
import com.neuralga.neural.CascadeCorrelationNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class PredictionService {

    private static final Logger log = LoggerFactory.getLogger(PredictionService.class);

    private final ModelLoader.Model model;
    private final int inputWindow;
    private final int predictionHorizon;

    private final List<Double> history = new ArrayList<>();

    public PredictionService(ModelLoader.Model model) {
        this.model = model;
        this.inputWindow = model.getInputWindow();
        this.predictionHorizon = model.getPredictionHorizon();
    }

    public PredictionService(CascadeCorrelationNetwork network,
                              com.neuralga.data.DataNormalizer normalizer,
                              int inputWindow, int predictionHorizon) {
        this.model = new ModelLoader.Model(network, normalizer,
                new com.neuralga.config.TrainingConfig(), new java.util.HashMap<>());
        this.inputWindow = inputWindow;
        this.predictionHorizon = predictionHorizon;
    }

    public double[] predict(double[] input) {
        if (input.length != inputWindow) {
            throw new IllegalArgumentException(
                    "Input size " + input.length + " != expected " + inputWindow);
        }

        // Normalize input
        double[] normalized = model.getNormalizer().normalize(input);

        // Forward pass
        double[] normalizedOutput = model.getNetwork().forward(normalized);

        // Denormalize output
        double[] output = model.getNormalizer().denormalize(normalizedOutput);

        return output;
    }

    public double predictNext(double value) {
        history.add(value);

        // Keep only last inputWindow values
        while (history.size() > inputWindow) {
            history.remove(0);
        }

        if (history.size() < inputWindow) {
            throw new IllegalStateException(
                    "Need " + inputWindow + " values for prediction, have " + history.size());
        }

        double[] input = new double[inputWindow];
        for (int i = 0; i < inputWindow; i++) {
            input[i] = history.get(i);
        }

        double[] prediction = predict(input);
        return prediction[0]; // Single step prediction
    }

    public List<Double> predictSequence(double[] initialInput, int steps) {
        if (initialInput.length != inputWindow) {
            throw new IllegalArgumentException(
                    "Initial input size " + initialInput.length + " != expected " + inputWindow);
        }

        List<Double> predictions = new ArrayList<>();
        double[] currentInput = initialInput.clone();

        for (int step = 0; step < steps; step++) {
            double[] pred = predict(currentInput);
            predictions.add(pred[0]);

            // Shift window and add prediction
            if (predictionHorizon == 1) {
                System.arraycopy(currentInput, 1, currentInput, 0, inputWindow - 1);
                currentInput[inputWindow - 1] = pred[0];
            } else {
                // Multi-step: just use last prediction
                System.arraycopy(currentInput, predictionHorizon, currentInput, 0,
                        inputWindow - predictionHorizon);
                System.arraycopy(pred, 0, currentInput, inputWindow - predictionHorizon, predictionHorizon);
            }
        }

        return predictions;
    }

    public void resetHistory() {
        history.clear();
    }

    public void addHistory(double value) {
        history.add(value);
        while (history.size() > inputWindow) {
            history.remove(0);
        }
    }

    public ModelLoader.Model getModel() { return model; }
    public int getInputWindow() { return inputWindow; }
    public int getPredictionHorizon() { return predictionHorizon; }

    public static class BatchPredictor {
        private final PredictionService service;

        public BatchPredictor(PredictionService service) {
            this.service = service;
        }

        public double[][] predict(SlidingWindowDataset dataset) {
            int n = dataset.size();
            double[][] predictions = new double[n][];

            for (int i = 0; i < n; i++) {
                predictions[i] = service.predict(dataset.getInput(i));
            }

            return predictions;
        }

        public List<double[]> predictToList(SlidingWindowDataset dataset) {
            List<double[]> predictions = new ArrayList<>();
            for (int i = 0; i < dataset.size(); i++) {
                predictions.add(service.predict(dataset.getInput(i)));
            }
            return predictions;
        }
    }
}