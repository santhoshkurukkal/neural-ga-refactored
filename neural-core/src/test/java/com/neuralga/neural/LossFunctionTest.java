package com.neuralga.neural;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class LossFunctionTest {

    @Test
    void mseShouldComputeCorrectly() {
        double[] pred = {1.0, 2.0, 3.0};
        double[] target = {1.0, 2.0, 3.0};
        assertThat(LossFunction.MSE.compute(pred, target)).isEqualTo(0.0);

        double[] pred2 = {0.0, 0.0, 0.0};
        double[] target2 = {1.0, 1.0, 1.0};
        assertThat(LossFunction.MSE.compute(pred2, target2)).isEqualTo(1.0);
    }

    @Test
    void mseGradientShouldBeCorrect() {
        double[] pred = {1.0, 2.0, 3.0};
        double[] target = {0.0, 0.0, 0.0};
        double[] grad = LossFunction.MSE.gradient(pred, target);

        // 2 * (pred - target) / n = 2 * pred / 3
        assertThat(grad[0]).isCloseTo(2.0/3.0, within(1e-10));
        assertThat(grad[1]).isCloseTo(4.0/3.0, within(1e-10));
        assertThat(grad[2]).isCloseTo(6.0/3.0, within(1e-10));
    }

    @Test
    void maeShouldComputeCorrectly() {
        double[] pred = {1.0, 2.0, 3.0};
        double[] target = {1.0, 2.0, 3.0};
        assertThat(LossFunction.MAE.compute(pred, target)).isEqualTo(0.0);

        double[] pred2 = {0.0, 0.0, 0.0};
        double[] target2 = {1.0, 1.0, 1.0};
        assertThat(LossFunction.MAE.compute(pred2, target2)).isEqualTo(1.0);
    }

    @Test
    void huberShouldComputeCorrectly() {
        double[] pred = {1.0, 2.0};
        double[] target = {1.0, 2.0};
        assertThat(LossFunction.HUBER.compute(pred, target)).isEqualTo(0.0);

        // Small diff -> quadratic
        double[] pred2 = {1.5, 2.0};
        double[] target2 = {1.0, 2.0};
        double loss = LossFunction.HUBER.compute(pred2, target2);
        // 0.5 * 0.5^2 = 0.125, avg = 0.0625
        assertThat(loss).isCloseTo(0.0625, within(1e-4));
    }

    @Test
    void fromStringShouldBeCaseInsensitive() {
        assertThat(LossFunction.fromString("mse")).isEqualTo(LossFunction.MSE);
        assertThat(LossFunction.fromString("MAE")).isEqualTo(LossFunction.MAE);
        assertThat(LossFunction.fromString("unknown")).isEqualTo(LossFunction.MSE);
    }
}