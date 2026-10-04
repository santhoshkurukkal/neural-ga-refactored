package com.neuralga.data;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.*;

class TrainValTestSplitterTest {

    @Test
    void shouldSplitChronologically() {
        List<Double> data = IntStream.range(0, 100).mapToObj(i -> (double) i).toList();

        TrainValTestSplitter splitter = new TrainValTestSplitter(0.7, 0.2, 0.1);
        TrainValTestSplitter.SplitResult result = splitter.split(data);

        assertThat(result.getTrainSize()).isEqualTo(70);
        assertThat(result.getValSize()).isEqualTo(20);
        assertThat(result.getTestSize()).isEqualTo(10);

        // Check chronological order preserved
        assertThat(result.getTrain()).containsExactlyElementsOf(data.subList(0, 70));
        assertThat(result.getVal()).containsExactlyElementsOf(data.subList(70, 90));
        assertThat(result.getTest()).containsExactlyElementsOf(data.subList(90, 100));
    }

    @Test
    void shouldSplitWithShuffle() {
        List<Double> data = IntStream.range(0, 100).mapToObj(i -> (double) i).toList();

        TrainValTestSplitter splitter = new TrainValTestSplitter(0.7, 0.2, 0.1, true, 42);
        TrainValTestSplitter.SplitResult result = splitter.split(data);

        assertThat(result.getTrainSize()).isEqualTo(70);
        assertThat(result.getValSize()).isEqualTo(20);
        assertThat(result.getTestSize()).isEqualTo(10);

        // With shuffle, order should be different (very high probability)
        boolean sameOrder = true;
        for (int i = 0; i < 70; i++) {
            if (!result.getTrain().get(i).equals(data.get(i))) {
                sameOrder = false;
                break;
            }
        }
        assertThat(sameOrder).isFalse();
    }

    @Test
    void shouldThrowWhenRatiosDontSumToOne() {
        assertThatThrownBy(() -> new TrainValTestSplitter(0.5, 0.5, 0.5))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("sum to 1.0");
    }

    @Test
    void shouldThrowOnEmptyData() {
        TrainValTestSplitter splitter = new TrainValTestSplitter(0.7, 0.2, 0.1);
        assertThatThrownBy(() -> splitter.split(List.of()))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void shouldThrowOnNullData() {
        TrainValTestSplitter splitter = new TrainValTestSplitter(0.7, 0.2, 0.1);
        assertThatThrownBy(() -> splitter.split(null))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void shouldReproduceSameSplitWithSameSeed() {
        List<Double> data = IntStream.range(0, 100).mapToObj(i -> (double) i).toList();

        TrainValTestSplitter splitter1 = new TrainValTestSplitter(0.7, 0.2, 0.1, true, 42);
        TrainValTestSplitter splitter2 = new TrainValTestSplitter(0.7, 0.2, 0.1, true, 42);

        var result1 = splitter1.split(data);
        var result2 = splitter2.split(data);

        assertThat(result1.getTrain()).isEqualTo(result2.getTrain());
        assertThat(result1.getVal()).isEqualTo(result2.getVal());
        assertThat(result1.getTest()).isEqualTo(result2.getTest());
    }

    @Test
    void staticChronologicalSplitShouldWork() {
        List<Double> data = IntStream.range(0, 100).mapToObj(i -> (double) i).toList();

        var result = TrainValTestSplitter.chronologicalSplit(data, 0.6, 0.2, 0.2);

        assertThat(result.getTrainSize()).isEqualTo(60);
        assertThat(result.getValSize()).isEqualTo(20);
        assertThat(result.getTestSize()).isEqualTo(20);
    }
}