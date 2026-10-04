package com.neuralga.data;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.DoubleStream;

import static org.assertj.core.api.Assertions.*;

class DataNormalizerTest {

    @Test
    void shouldNormalizeWithMinMax() {
        List<Double> data = List.of(0.0, 5.0, 10.0, 15.0, 20.0);
        DataNormalizer normalizer = new DataNormalizer(DataNormalizer.Method.MIN_MAX);
        normalizer.fit(data);

        assertThat(normalizer.normalize(0.0)).isEqualTo(0.0);
        assertThat(normalizer.normalize(20.0)).isEqualTo(1.0);
        assertThat(normalizer.normalize(10.0)).isEqualTo(0.5);

        // Denormalize
        assertThat(normalizer.denormalize(0.0)).isEqualTo(0.0);
        assertThat(normalizer.denormalize(1.0)).isEqualTo(20.0);
        assertThat(normalizer.denormalize(0.5)).isEqualTo(10.0);
    }

    @Test
    void shouldNormalizeWithZScore() {
        List<Double> data = List.of(10.0, 20.0, 30.0, 40.0, 50.0); // mean=30, std≈14.14
        DataNormalizer normalizer = new DataNormalizer(DataNormalizer.Method.Z_SCORE);
        normalizer.fit(data);

        assertThat(normalizer.getMean()).isEqualTo(30.0);
        assertThat(normalizer.getStd()).isCloseTo(14.14, within(0.01));

        assertThat(normalizer.normalize(30.0)).isEqualTo(0.0);
        assertThat(normalizer.normalize(44.14)).isCloseTo(1.0, within(0.01));
        assertThat(normalizer.normalize(15.86)).isCloseTo(-1.0, within(0.01));

        // Denormalize
        assertThat(normalizer.denormalize(0.0)).isEqualTo(30.0);
        assertThat(normalizer.denormalize(1.0)).isCloseTo(44.14, within(0.01));
    }

    @Test
    void shouldNormalizeWithRobust() {
        // Data with outliers
        // Sorted: [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        // median = (3+4)/2 = 3.5
        // Q1 (25th percentile): index = 0.25 * 5 = 1.25, between 2.0 and 3.0 = 2.25
        // Q3 (75th percentile): index = 0.75 * 5 = 3.75, between 4.0 and 5.0 = 4.75
        List<Double> data = List.of(1.0, 2.0, 3.0, 4.0, 5.0, 100.0);
        DataNormalizer normalizer = new DataNormalizer(DataNormalizer.Method.ROBUST);
        normalizer.fit(data);

        assertThat(normalizer.getMedian()).isEqualTo(3.5);
        assertThat(normalizer.getQ1()).isEqualTo(2.25);
        assertThat(normalizer.getQ3()).isEqualTo(4.75);

        // Outlier should be compressed
        double normalizedOutlier = normalizer.normalize(100.0);
        assertThat(normalizedOutlier).isLessThan(50.0); // Much less than min-max would give
    }

    @Test
    void shouldHandleConstantData() {
        List<Double> data = List.of(5.0, 5.0, 5.0, 5.0);
        DataNormalizer normalizer = new DataNormalizer(DataNormalizer.Method.MIN_MAX);
        normalizer.fit(data);

        // All values should normalize to 0 (or 0.5 depending on implementation)
        assertThat(normalizer.normalize(5.0)).isEqualTo(0.0);
    }

    @Test
    void shouldNormalizeList() {
        List<Double> data = List.of(0.0, 10.0, 20.0);
        DataNormalizer normalizer = new DataNormalizer(DataNormalizer.Method.MIN_MAX);
        normalizer.fit(data);

        List<Double> normalized = normalizer.normalize(data);
        assertThat(normalized).containsExactly(0.0, 0.5, 1.0);
    }

    @Test
    void shouldNormalizeArray() {
        double[] data = {0.0, 10.0, 20.0};
        DataNormalizer normalizer = new DataNormalizer(DataNormalizer.Method.MIN_MAX);
        normalizer.fit(List.of(0.0, 10.0, 20.0));

        double[] normalized = normalizer.normalize(data);
        assertThat(normalized).containsExactly(0.0, 0.5, 1.0);
    }

    @Test
    void shouldThrowWhenNotFitted() {
        DataNormalizer normalizer = new DataNormalizer(DataNormalizer.Method.MIN_MAX);
        assertThatThrownBy(() -> normalizer.normalize(1.0))
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void shouldCreateAutoNormalizer() {
        // Normal-like data -> Z_SCORE
        List<Double> normalData = DoubleStream.iterate(0, x -> x + 1).limit(100).boxed().toList();
        DataNormalizer auto1 = DataNormalizer.createAuto(normalData);
        assertThat(auto1.getMethod()).isEqualTo(DataNormalizer.Method.Z_SCORE);

        // Skewed data -> MIN_MAX
        List<Double> skewedData = List.of(1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0);
        DataNormalizer auto2 = DataNormalizer.createAuto(skewedData);
        assertThat(auto2.getMethod()).isEqualTo(DataNormalizer.Method.MIN_MAX);
    }

    @Test
    void shouldRoundTripNormalizeDenormalize() {
        List<Double> original = List.of(1.5, 2.5, 3.5, 4.5, 5.5, 10.0, 20.0, 30.0);

        for (DataNormalizer.Method method : DataNormalizer.Method.values()) {
            DataNormalizer normalizer = new DataNormalizer(method);
            normalizer.fit(original);

            List<Double> normalized = normalizer.normalize(original);
            List<Double> denormalized = normalizer.denormalize(normalized);

            for (int i = 0; i < original.size(); i++) {
                assertThat(denormalized.get(i)).isCloseTo(original.get(i), within(1e-10));
            }
        }
    }
}