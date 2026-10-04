package com.neuralga.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class SlidingWindowDataset {

    private static final Logger log = LoggerFactory.getLogger(SlidingWindowDataset.class);

    private final int inputWindow;
    private final int predictionHorizon;
    private final int stride;

    private List<double[]> inputs;
    private List<double[]> targets;

    public SlidingWindowDataset(int inputWindow, int predictionHorizon) {
        this(inputWindow, predictionHorizon, 1);
    }

    public SlidingWindowDataset(int inputWindow, int predictionHorizon, int stride) {
        if (inputWindow <= 0) throw new IllegalArgumentException("inputWindow must be > 0");
        if (predictionHorizon <= 0) throw new IllegalArgumentException("predictionHorizon must be > 0");
        if (stride <= 0) throw new IllegalArgumentException("stride must be > 0");

        this.inputWindow = inputWindow;
        this.predictionHorizon = predictionHorizon;
        this.stride = stride;
    }

    public void build(List<Double> data) {
        if (data == null || data.size() < inputWindow + predictionHorizon) {
            throw new IllegalArgumentException(
                    "Data size (" + (data != null ? data.size() : 0) +
                    ") must be >= inputWindow + predictionHorizon (" + (inputWindow + predictionHorizon) + ")");
        }

        int maxStart = data.size() - inputWindow - predictionHorizon + 1;
        int numSamples = (maxStart + stride - 1) / stride;

        inputs = new ArrayList<>(numSamples);
        targets = new ArrayList<>(numSamples);

        for (int start = 0; start < maxStart; start += stride) {
            double[] input = new double[inputWindow];
            double[] target = new double[predictionHorizon];

            for (int i = 0; i < inputWindow; i++) {
                input[i] = data.get(start + i);
            }
            for (int h = 0; h < predictionHorizon; h++) {
                target[h] = data.get(start + inputWindow + h);
            }

            inputs.add(input);
            targets.add(target);
        }

        log.info("Built sliding window dataset: samples={}, inputWindow={}, horizon={}, stride={}",
                inputs.size(), inputWindow, predictionHorizon, stride);
    }

    public int size() {
        return inputs != null ? inputs.size() : 0;
    }

    public double[] getInput(int index) {
        return inputs.get(index);
    }

    public double[] getTarget(int index) {
        return targets.get(index);
    }

    public List<double[]> getInputs() {
        return inputs != null ? List.copyOf(inputs) : List.of();
    }

    public List<double[]> getTargets() {
        return targets != null ? List.copyOf(targets) : List.of();
    }

    public int getInputWindow() { return inputWindow; }
    public int getPredictionHorizon() { return predictionHorizon; }
    public int getStride() { return stride; }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int inputWindow = 5;
        private int predictionHorizon = 1;
        private int stride = 1;

        public Builder inputWindow(int inputWindow) {
            this.inputWindow = inputWindow;
            return this;
        }

        public Builder predictionHorizon(int predictionHorizon) {
            this.predictionHorizon = predictionHorizon;
            return this;
        }

        public Builder stride(int stride) {
            this.stride = stride;
            return this;
        }

        public SlidingWindowDataset build() {
            return new SlidingWindowDataset(inputWindow, predictionHorizon, stride);
        }
    }
}