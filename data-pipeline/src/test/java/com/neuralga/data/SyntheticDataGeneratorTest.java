package com.neuralga.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

class SyntheticDataGeneratorTest {

    @ParameterizedTest
    @EnumSource(SyntheticDataGenerator.DataType.class)
    void shouldGenerateDataOfCorrectLength(SyntheticDataGenerator.DataType type) {
        SyntheticDataGenerator gen = SyntheticDataGenerator.builder()
                .type(type)
                .length(100)
                .seed(42)
                .build();

        List<Double> data = gen.generate();

        assertThat(data).hasSize(100);
        assertThat(data).allMatch(v -> !v.isNaN() && !v.isInfinite());
    }

    @Test
    void shouldGenerateReproducibleDataWithSameSeed() {
        SyntheticDataGenerator gen1 = SyntheticDataGenerator.builder()
                .type(SyntheticDataGenerator.DataType.SINE_WITH_NOISE)
                .length(50)
                .seed(12345)
                .build();

        SyntheticDataGenerator gen2 = SyntheticDataGenerator.builder()
                .type(SyntheticDataGenerator.DataType.SINE_WITH_NOISE)
                .length(50)
                .seed(12345)
                .build();

        List<Double> data1 = gen1.generate();
        List<Double> data2 = gen2.generate();

        assertThat(data1).isEqualTo(data2);
    }

    @Test
    void shouldGenerateDifferentDataWithDifferentSeeds() {
        SyntheticDataGenerator gen1 = SyntheticDataGenerator.builder()
                .type(SyntheticDataGenerator.DataType.RANDOM_WALK)
                .length(50)
                .seed(1)
                .build();

        SyntheticDataGenerator gen2 = SyntheticDataGenerator.builder()
                .type(SyntheticDataGenerator.DataType.RANDOM_WALK)
                .length(50)
                .seed(2)
                .build();

        List<Double> data1 = gen1.generate();
        List<Double> data2 = gen2.generate();

        assertThat(data1).isNotEqualTo(data2);
    }

    @Test
    void shouldGenerateSineWaveWithCorrectProperties() {
        SyntheticDataGenerator gen = SyntheticDataGenerator.builder()
                .type(SyntheticDataGenerator.DataType.SINE)
                .length(1000)
                .frequency(2.0)
                .amplitude(5.0)
                .seed(42)
                .build();

        List<Double> data = gen.generate();

        // Check amplitude bounds
        assertThat(data).allMatch(v -> v >= -5.0 && v <= 5.0);

        // Check periodicity (roughly)
        int period = 500; // 1000 samples / 2 cycles
        double correlation = calculateCorrelation(data, period);
        assertThat(correlation).isGreaterThan(0.9);
    }

    @Test
    void shouldGenerateSineWithNoise() {
        SyntheticDataGenerator gen = SyntheticDataGenerator.builder()
                .type(SyntheticDataGenerator.DataType.SINE_WITH_NOISE)
                .length(1000)
                .amplitude(1.0)
                .noiseLevel(0.1)
                .seed(42)
                .build();

        List<Double> data = gen.generate();

        // Values should exceed pure sine bounds due to noise
        boolean hasOutOfBounds = data.stream().anyMatch(v -> v > 1.0 || v < -1.0);
        assertThat(hasOutOfBounds).isTrue();
    }

    @Test
    void shouldGenerateSquareWave() {
        SyntheticDataGenerator gen = SyntheticDataGenerator.builder()
                .type(SyntheticDataGenerator.DataType.SQUARE_WAVE)
                .length(100)
                .amplitude(2.0)
                .seed(42)
                .build();

        List<Double> data = gen.generate();

        // Should only have two values: +2.0 and -2.0
        assertThat(data).allMatch(v -> v == 2.0 || v == -2.0);
        assertThat(data).contains(2.0, -2.0);
    }

    @Test
    void shouldGenerateARMA() {
        SyntheticDataGenerator gen = SyntheticDataGenerator.builder()
                .type(SyntheticDataGenerator.DataType.ARMA)
                .length(200)
                .seed(42)
                .build();

        List<Double> data = gen.generate();

        assertThat(data).hasSize(200);
        // ARMA should have autocorrelation
        double acf1 = autocorrelation(data, 1);
        assertThat(Math.abs(acf1)).isGreaterThan(0.1);
    }

    @Test
    void shouldGenerateMackeyGlass() {
        SyntheticDataGenerator gen = SyntheticDataGenerator.builder()
                .type(SyntheticDataGenerator.DataType.MACKEY_GLASS)
                .length(500)
                .seed(42)
                .build();

        List<Double> data = gen.generate();

        assertThat(data).hasSize(500);
        // Mackey-Glass is chaotic but bounded
        assertThat(data).allMatch(v -> v > 0 && v < 2.0);
    }

    @Test
    void shouldGenerateRandomWalk() {
        SyntheticDataGenerator gen = SyntheticDataGenerator.builder()
                .type(SyntheticDataGenerator.DataType.RANDOM_WALK)
                .length(100)
                .noiseLevel(0.1)
                .seed(42)
                .build();

        List<Double> data = gen.generate();

        assertThat(data).hasSize(100);
        // Random walk should have high autocorrelation
        double acf1 = autocorrelation(data, 1);
        assertThat(acf1).isGreaterThan(0.9);
    }

    @Test
    void shouldSaveToCsv(@TempDir Path tempDir) throws Exception {
        Path file = tempDir.resolve("synthetic.csv");
        SyntheticDataGenerator gen = SyntheticDataGenerator.builder()
                .type(SyntheticDataGenerator.DataType.SINE)
                .length(10)
                .seed(42)
                .build();

        gen.saveToCsv(file);

        assertThat(file).exists();
        String content = java.nio.file.Files.readString(file);
        String[] lines = content.split("\n");
        assertThat(lines).hasSize(10);
    }

    private double calculateCorrelation(List<Double> data, int lag) {
        if (data.size() <= lag) return 0;
        double mean = data.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double num = 0, den1 = 0, den2 = 0;
        for (int i = 0; i < data.size() - lag; i++) {
            double x = data.get(i) - mean;
            double y = data.get(i + lag) - mean;
            num += x * y;
            den1 += x * x;
            den2 += y * y;
        }
        return den1 == 0 || den2 == 0 ? 0 : num / Math.sqrt(den1 * den2);
    }

    private double autocorrelation(List<Double> data, int lag) {
        return calculateCorrelation(data, lag);
    }
}