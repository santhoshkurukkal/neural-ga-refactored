package com.neuralga.data;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.*;

class SlidingWindowDatasetTest {

    @Test
    void shouldBuildDatasetWithDefaultStride() {
        List<Double> data = IntStream.range(0, 20).mapToObj(i -> (double) i).toList();

        SlidingWindowDataset dataset = SlidingWindowDataset.builder()
                .inputWindow(5)
                .predictionHorizon(1)
                .build();
        dataset.build(data);

        assertThat(dataset.size()).isEqualTo(15); // 20 - 5 - 1 + 1 = 15

        // First sample: input [0,1,2,3,4], target [5]
        assertThat(dataset.getInput(0)).containsExactly(0.0, 1.0, 2.0, 3.0, 4.0);
        assertThat(dataset.getTarget(0)).containsExactly(5.0);

        // Last sample: input [14,15,16,17,18], target [19]
        assertThat(dataset.getInput(14)).containsExactly(14.0, 15.0, 16.0, 17.0, 18.0);
        assertThat(dataset.getTarget(14)).containsExactly(19.0);
    }

    @Test
    void shouldBuildDatasetWithStride() {
        List<Double> data = IntStream.range(0, 20).mapToObj(i -> (double) i).toList();

        SlidingWindowDataset dataset = SlidingWindowDataset.builder()
                .inputWindow(5)
                .predictionHorizon(1)
                .stride(2)
                .build();
        dataset.build(data);

        assertThat(dataset.size()).isEqualTo(8); // ceil(15/2) = 8

        // Samples at indices 0, 2, 4, 6, 8, 10, 12, 14
        assertThat(dataset.getInput(0)).containsExactly(0.0, 1.0, 2.0, 3.0, 4.0);
        assertThat(dataset.getInput(1)).containsExactly(2.0, 3.0, 4.0, 5.0, 6.0);
    }

    @Test
    void shouldBuildDatasetWithMultiStepHorizon() {
        List<Double> data = IntStream.range(0, 20).mapToObj(i -> (double) i).toList();

        SlidingWindowDataset dataset = SlidingWindowDataset.builder()
                .inputWindow(5)
                .predictionHorizon(3)
                .build();
        dataset.build(data);

        assertThat(dataset.size()).isEqualTo(13); // 20 - 5 - 3 + 1 = 13

        // First sample: input [0,1,2,3,4], target [5,6,7]
        assertThat(dataset.getInput(0)).containsExactly(0.0, 1.0, 2.0, 3.0, 4.0);
        assertThat(dataset.getTarget(0)).containsExactly(5.0, 6.0, 7.0);

        // Last sample: input [12,13,14,15,16], target [17,18,19]
        assertThat(dataset.getInput(12)).containsExactly(12.0, 13.0, 14.0, 15.0, 16.0);
        assertThat(dataset.getTarget(12)).containsExactly(17.0, 18.0, 19.0);
    }

    @Test
    void shouldThrowWhenDataTooSmall() {
        List<Double> data = List.of(1.0, 2.0, 3.0);

        SlidingWindowDataset dataset = SlidingWindowDataset.builder()
                .inputWindow(5)
                .predictionHorizon(1)
                .build();

        assertThatThrownBy(() -> dataset.build(data))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must be >=");
    }

    @Test
    void shouldThrowOnInvalidParameters() {
        assertThatThrownBy(() -> new SlidingWindowDataset(0, 1))
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> new SlidingWindowDataset(5, 0))
                .isInstanceOf(IllegalArgumentException.class);

        assertThatThrownBy(() -> new SlidingWindowDataset(5, 1, 0))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void shouldReturnImmutableLists() {
        List<Double> data = IntStream.range(0, 10).mapToObj(i -> (double) i).toList();

        SlidingWindowDataset dataset = SlidingWindowDataset.builder()
                .inputWindow(3)
                .predictionHorizon(1)
                .build();
        dataset.build(data);

        List<double[]> inputs = dataset.getInputs();
        List<double[]> targets = dataset.getTargets();

        assertThat(inputs).hasSize(6);
        assertThat(targets).hasSize(6);

        // Verify immutability
        assertThatThrownBy(() -> inputs.add(new double[3]))
                .isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void builderShouldCreateIndependentInstances() {
        SlidingWindowDataset d1 = SlidingWindowDataset.builder().inputWindow(5).build();
        SlidingWindowDataset d2 = SlidingWindowDataset.builder().inputWindow(10).build();

        assertThat(d1.getInputWindow()).isEqualTo(5);
        assertThat(d2.getInputWindow()).isEqualTo(10);
    }
}