package com.neuralga.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SyntheticDataGenerator {

    private static final Logger log = LoggerFactory.getLogger(SyntheticDataGenerator.class);

    public enum DataType {
        SINE,
        SINE_WITH_NOISE,
        ARMA,
        MACKEY_GLASS,
        RANDOM_WALK,
        COSINE,
        SQUARE_WAVE
    }

    private final Random random;
    private final DataType type;
    private final int length;
    private final double noiseLevel;
    private final double frequency;
    private final double amplitude;

    private SyntheticDataGenerator(Builder builder) {
        this.random = new Random(builder.seed);
        this.type = builder.type;
        this.length = builder.length;
        this.noiseLevel = builder.noiseLevel;
        this.frequency = builder.frequency;
        this.amplitude = builder.amplitude;
    }

    public List<Double> generate() {
        return switch (type) {
            case SINE -> generateSine();
            case SINE_WITH_NOISE -> generateSineWithNoise();
            case COSINE -> generateCosine();
            case SQUARE_WAVE -> generateSquareWave();
            case ARMA -> generateARMA();
            case MACKEY_GLASS -> generateMackeyGlass();
            case RANDOM_WALK -> generateRandomWalk();
        };
    }

    public void saveToCsv(Path path) throws IOException {
        List<Double> data = generate();
        String csv = data.stream()
                .map(Object::toString)
                .collect(Collectors.joining("\n"));
        Files.writeString(path, csv);
        log.info("Saved {} synthetic {} data points to {}", data.size(), type, path);
    }

    private List<Double> generateSine() {
        return IntStream.range(0, length)
                .mapToObj(i -> amplitude * Math.sin(2 * Math.PI * frequency * i / length))
                .collect(Collectors.toList());
    }

    private List<Double> generateSineWithNoise() {
        return IntStream.range(0, length)
                .mapToObj(i -> {
                    double signal = amplitude * Math.sin(2 * Math.PI * frequency * i / length);
                    double noise = random.nextGaussian() * noiseLevel * amplitude;
                    return signal + noise;
                })
                .collect(Collectors.toList());
    }

    private List<Double> generateCosine() {
        return IntStream.range(0, length)
                .mapToObj(i -> amplitude * Math.cos(2 * Math.PI * frequency * i / length))
                .collect(Collectors.toList());
    }

    private List<Double> generateSquareWave() {
        return IntStream.range(0, length)
                .mapToObj(i -> {
                    double phase = 2 * Math.PI * frequency * i / length;
                    return amplitude * (Math.sin(phase) >= 0 ? 1.0 : -1.0);
                })
                .collect(Collectors.toList());
    }

    private List<Double> generateARMA() {
        // ARMA(2,1) process: x_t = 0.7*x_{t-1} - 0.2*x_{t-2} + e_t + 0.5*e_{t-1}
        double ar1 = 0.7, ar2 = -0.2, ma1 = 0.5;
        List<Double> data = new ArrayList<>(length);
        List<Double> errors = new ArrayList<>(length);

        for (int i = 0; i < length; i++) {
            double error = random.nextGaussian() * noiseLevel;
            errors.add(error);

            double value = error;
            if (i >= 1) value += ar1 * data.get(i - 1) + ma1 * errors.get(i - 1);
            if (i >= 2) value += ar2 * data.get(i - 2);

            data.add(value);
        }
        return data;
    }

    private List<Double> generateMackeyGlass() {
        // Mackey-Glass chaotic time series
        // dx/dt = (a * x(t-tau)) / (1 + x(t-tau)^c) - b * x(t)
        // Discretized with Euler method
        int tau = 17;
        double a = 0.2, b = 0.1, c = 10.0;
        double dt = 1.0;

        List<Double> data = new ArrayList<>(length + tau);
        // Initialize with small random values
        for (int i = 0; i < tau; i++) {
            data.add(0.1 + random.nextDouble() * 0.1);
        }

        for (int i = tau; i < length + tau; i++) {
            double x_tau = data.get(i - tau);
            double x = data.get(i - 1);
            double dx = (a * x_tau) / (1 + Math.pow(x_tau, c)) - b * x;
            double next = x + dx * dt + random.nextGaussian() * noiseLevel;
            data.add(next);
        }

        // Return only the generated part (skip initialization)
        return data.subList(tau, data.size());
    }

    private List<Double> generateRandomWalk() {
        List<Double> data = new ArrayList<>(length);
        double value = 0.0;
        for (int i = 0; i < length; i++) {
            value += random.nextGaussian() * noiseLevel;
            data.add(value);
        }
        return data;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private long seed = System.currentTimeMillis();
        private DataType type = DataType.SINE_WITH_NOISE;
        private int length = 1000;
        private double noiseLevel = 0.05;
        private double frequency = 1.0;
        private double amplitude = 1.0;

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder type(DataType type) {
            this.type = type;
            return this;
        }

        public Builder length(int length) {
            this.length = length;
            return this;
        }

        public Builder noiseLevel(double noiseLevel) {
            this.noiseLevel = noiseLevel;
            return this;
        }

        public Builder frequency(double frequency) {
            this.frequency = frequency;
            return this;
        }

        public Builder amplitude(double amplitude) {
            this.amplitude = amplitude;
            return this;
        }

        public SyntheticDataGenerator build() {
            return new SyntheticDataGenerator(this);
        }
    }
}