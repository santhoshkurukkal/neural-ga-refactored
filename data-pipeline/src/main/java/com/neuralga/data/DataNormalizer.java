package com.neuralga.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

public class DataNormalizer {

    private static final Logger log = LoggerFactory.getLogger(DataNormalizer.class);

    public enum Method {
        MIN_MAX,
        Z_SCORE,
        ROBUST,
        NONE
    }

    private final Method method;
    private double min;
    private double max;
    private double mean;
    private double std;
    private double median;
    private double q1;
    private double q3;
    private boolean fitted = false;

    public DataNormalizer(Method method) {
        this.method = method;
    }

    public void fit(List<Double> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("Data cannot be null or empty");
        }

        double[] values = data.stream().mapToDouble(Double::doubleValue).toArray();
        Arrays.sort(values);

        switch (method) {
            case MIN_MAX -> {
                min = values[0];
                max = values[values.length - 1];
                if (max - min < 1e-12) {
                    max = min + 1.0; // Avoid division by zero
                }
            }
            case Z_SCORE -> {
                mean = Arrays.stream(values).average().orElse(0.0);
                std = Math.sqrt(Arrays.stream(values).map(v -> (v - mean) * (v - mean)).average().orElse(1.0));
                if (std < 1e-12) {
                    std = 1.0;
                }
            }
            case ROBUST -> {
                median = percentile(values, 50);
                q1 = percentile(values, 25);
                q3 = percentile(values, 75);
                double iqr = q3 - q1;
                if (iqr < 1e-12) {
                    iqr = 1.0;
                }
            }
            case NONE -> {}
        }

        fitted = true;
        log.debug("Fitted {} normalizer: min={}, max={}, mean={}, std={}", method, min, max, mean, std);
    }

    public double normalize(double value) {
        if (!fitted) {
            throw new IllegalStateException("Normalizer must be fitted before use");
        }

        return switch (method) {
            case MIN_MAX -> (value - min) / (max - min);
            case Z_SCORE -> (value - mean) / std;
            case ROBUST -> (value - median) / (q3 - q1);
            case NONE -> value;
        };
    }

    public double denormalize(double normalizedValue) {
        if (!fitted) {
            throw new IllegalStateException("Normalizer must be fitted before use");
        }

        return switch (method) {
            case MIN_MAX -> normalizedValue * (max - min) + min;
            case Z_SCORE -> normalizedValue * std + mean;
            case ROBUST -> normalizedValue * (q3 - q1) + median;
            case NONE -> normalizedValue;
        };
    }

    public List<Double> normalize(List<Double> data) {
        return data.stream().map(this::normalize).toList();
    }

    public List<Double> denormalize(List<Double> normalizedData) {
        return normalizedData.stream().map(this::denormalize).toList();
    }

    public double[] normalize(double[] data) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = normalize(data[i]);
        }
        return result;
    }

    public double[] denormalize(double[] normalizedData) {
        double[] result = new double[normalizedData.length];
        for (int i = 0; i < normalizedData.length; i++) {
            result[i] = denormalize(normalizedData[i]);
        }
        return result;
    }

    public boolean isFitted() {
        return fitted;
    }

    public Method getMethod() {
        return method;
    }

    public double getMin() { return min; }
    public double getMax() { return max; }
    public double getMean() { return mean; }
    public double getStd() { return std; }
    public double getMedian() { return median; }
    public double getQ1() { return q1; }
    public double getQ3() { return q3; }

    private double percentile(double[] sortedValues, double percentile) {
        int n = sortedValues.length;
        double index = (percentile / 100.0) * (n - 1);
        int lower = (int) Math.floor(index);
        int upper = (int) Math.ceil(index);
        if (lower == upper) {
            return sortedValues[lower];
        }
        double weight = index - lower;
        return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
    }

    public static DataNormalizer createAuto(List<Double> data) {
        // Heuristic: use Z_SCORE for roughly normal data, MIN_MAX for bounded
        DataNormalizer zscore = new DataNormalizer(Method.Z_SCORE);
        zscore.fit(data);
        double skewness = calculateSkewness(data, zscore.getMean(), zscore.getStd());

        Method method = Math.abs(skewness) > 1.0 ? Method.MIN_MAX : Method.Z_SCORE;
        DataNormalizer normalizer = new DataNormalizer(method);
        normalizer.fit(data);
        return normalizer;
    }

    private static double calculateSkewness(List<Double> data, double mean, double std) {
        if (std == 0) return 0;
        double sum = data.stream().mapToDouble(v -> Math.pow((v - mean) / std, 3)).sum();
        return sum / data.size();
    }
}