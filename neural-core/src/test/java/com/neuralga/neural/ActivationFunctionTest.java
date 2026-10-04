package com.neuralga.neural;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class ActivationFunctionTest {

    @Test
    void tanhShouldActivateCorrectly() {
        assertThat(ActivationFunction.TANH.activate(0)).isEqualTo(0.0);
        assertThat(ActivationFunction.TANH.activate(1)).isCloseTo(0.76159, within(1e-4));
        assertThat(ActivationFunction.TANH.activate(-1)).isCloseTo(-0.76159, within(1e-4));
        assertThat(ActivationFunction.TANH.activate(100)).isCloseTo(1.0, within(1e-4));
        assertThat(ActivationFunction.TANH.activate(-100)).isCloseTo(-1.0, within(1e-4));
    }

    @Test
    void tanhDerivativeShouldBeCorrect() {
        // derivative = 1 - tanh^2
        double x = 0.5;
        double act = ActivationFunction.TANH.activate(x);
        double deriv = ActivationFunction.TANH.derivative(x);
        assertThat(deriv).isCloseTo(1.0 - act * act, within(1e-10));
    }

    @Test
    void sigmoidShouldActivateCorrectly() {
        assertThat(ActivationFunction.SIGMOID.activate(0)).isEqualTo(0.5);
        assertThat(ActivationFunction.SIGMOID.activate(10)).isCloseTo(1.0, within(1e-4));
        assertThat(ActivationFunction.SIGMOID.activate(-10)).isCloseTo(0.0, within(1e-4));
    }

    @Test
    void reluShouldActivateCorrectly() {
        assertThat(ActivationFunction.RELU.activate(0)).isEqualTo(0.0);
        assertThat(ActivationFunction.RELU.activate(5)).isEqualTo(5.0);
        assertThat(ActivationFunction.RELU.activate(-5)).isEqualTo(0.0);
    }

    @Test
    void reluDerivativeShouldBeCorrect() {
        assertThat(ActivationFunction.RELU.derivative(5)).isEqualTo(1.0);
        assertThat(ActivationFunction.RELU.derivative(0)).isEqualTo(0.0);
        assertThat(ActivationFunction.RELU.derivative(-5)).isEqualTo(0.0);
    }

    @Test
    void linearShouldActivateCorrectly() {
        assertThat(ActivationFunction.LINEAR.activate(0)).isEqualTo(0.0);
        assertThat(ActivationFunction.LINEAR.activate(5)).isEqualTo(5.0);
        assertThat(ActivationFunction.LINEAR.activate(-5)).isEqualTo(-5.0);
    }

    @Test
    void linearDerivativeShouldBeOne() {
        assertThat(ActivationFunction.LINEAR.derivative(0)).isEqualTo(1.0);
        assertThat(ActivationFunction.LINEAR.derivative(100)).isEqualTo(1.0);
    }

    @Test
    void fromStringShouldBeCaseInsensitive() {
        assertThat(ActivationFunction.fromString("tanh")).isEqualTo(ActivationFunction.TANH);
        assertThat(ActivationFunction.fromString("TANH")).isEqualTo(ActivationFunction.TANH);
        assertThat(ActivationFunction.fromString("relu")).isEqualTo(ActivationFunction.RELU);
        assertThat(ActivationFunction.fromString("unknown")).isEqualTo(ActivationFunction.LINEAR);
    }
}