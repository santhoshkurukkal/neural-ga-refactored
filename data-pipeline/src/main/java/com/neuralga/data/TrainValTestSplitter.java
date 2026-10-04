package com.neuralga.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class TrainValTestSplitter {

    private static final Logger log = LoggerFactory.getLogger(TrainValTestSplitter.class);

    private final double trainRatio;
    private final double valRatio;
    private final double testRatio;
    private final boolean shuffle;
    private final long seed;

    public TrainValTestSplitter(double trainRatio, double valRatio, double testRatio) {
        this(trainRatio, valRatio, testRatio, false, System.currentTimeMillis());
    }

    public TrainValTestSplitter(double trainRatio, double valRatio, double testRatio, boolean shuffle, long seed) {
        double sum = trainRatio + valRatio + testRatio;
        if (Math.abs(sum - 1.0) > 1e-10) {
            throw new IllegalArgumentException("Ratios must sum to 1.0, got: " + sum);
        }
        this.trainRatio = trainRatio;
        this.valRatio = valRatio;
        this.testRatio = testRatio;
        this.shuffle = shuffle;
        this.seed = seed;
    }

    public SplitResult split(List<Double> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("Data cannot be null or empty");
        }

        List<Integer> indices = IntStream.range(0, data.size()).boxed().toList();

        if (shuffle) {
            List<Integer> shuffled = new ArrayList<>(indices);
            Collections.shuffle(shuffled, new Random(seed));
            indices = shuffled;
        }

        int n = data.size();
        int trainEnd = (int) Math.round(n * trainRatio);
        int valEnd = trainEnd + (int) Math.round(n * valRatio);

        List<Double> train = new ArrayList<>(trainEnd);
        List<Double> val = new ArrayList<>(valEnd - trainEnd);
        List<Double> test = new ArrayList<>(n - valEnd);

        for (int i = 0; i < trainEnd; i++) {
            train.add(data.get(indices.get(i)));
        }
        for (int i = trainEnd; i < valEnd; i++) {
            val.add(data.get(indices.get(i)));
        }
        for (int i = valEnd; i < n; i++) {
            test.add(data.get(indices.get(i)));
        }

        log.info("Split data: train={}, val={}, test={} (shuffle={})", train.size(), val.size(), test.size(), shuffle);

        return new SplitResult(train, val, test);
    }

    public static SplitResult chronologicalSplit(List<Double> data, double trainRatio, double valRatio, double testRatio) {
        return new TrainValTestSplitter(trainRatio, valRatio, testRatio, false, 0).split(data);
    }

    public static class SplitResult {
        private final List<Double> train;
        private final List<Double> val;
        private final List<Double> test;

        public SplitResult(List<Double> train, List<Double> val, List<Double> test) {
            this.train = List.copyOf(train);
            this.val = List.copyOf(val);
            this.test = List.copyOf(test);
        }

        public List<Double> getTrain() { return train; }
        public List<Double> getVal() { return val; }
        public List<Double> getTest() { return test; }

        public int getTrainSize() { return train.size(); }
        public int getValSize() { return val.size(); }
        public int getTestSize() { return test.size(); }
        public int getTotalSize() { return train.size() + val.size() + test.size(); }
    }
}