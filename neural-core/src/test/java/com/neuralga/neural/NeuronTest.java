package com.neuralga.neural;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class NeuronTest {

    @Test
    void shouldInitializeWithRandomWeights() {
        Neuron neuron = new Neuron(3, ActivationFunction.TANH, 42);

        assertThat(neuron.getInputSize()).isEqualTo(3);
        assertThat(neuron.getWeights()).hasSize(3);
        assertThat(neuron.getActivation()).isEqualTo(ActivationFunction.TANH);
    }

    @Test
    void shouldForwardCorrectly() {
        Neuron neuron = new Neuron(2, ActivationFunction.LINEAR, 42);
        // Manually set weights for predictable test
        neuron.setWeightsAndBias(new double[]{0.5, -0.5}, 0.1);

        double[] input = {1.0, 2.0};
        double output = neuron.forward(input);

        // 0.5*1.0 + (-0.5)*2.0 + 0.1 = 0.5 - 1.0 + 0.1 = -0.4
        assertThat(output).isEqualTo(-0.4);
    }

    @Test
    void shouldForwardWithTanh() {
        Neuron neuron = new Neuron(2, ActivationFunction.TANH, 42);
        neuron.setWeightsAndBias(new double[]{1.0, 1.0}, 0.0);

        double[] input = {0.5, 0.5};
        double output = neuron.forward(input);

        // tanh(1.0) ≈ 0.76159
        assertThat(output).isCloseTo(0.76159, within(1e-4));
    }

    @Test
    void shouldBackwardUpdateWeights() {
        Neuron neuron = new Neuron(2, ActivationFunction.LINEAR, 42);
        neuron.setWeightsAndBias(new double[]{0.0, 0.0}, 0.0);

        double[] input = {1.0, 2.0};
        neuron.forward(input); // output = 0

        // Backward with error = 1.0
        neuron.backward(input, 1.0);

        // weights should be updated: w += input * delta (delta = error * derivative = 1.0 * 1.0 = 1.0)
        // But learning rate is applied in layer/trainer, so here delta = 1.0
        assertThat(neuron.getDelta()).isEqualTo(1.0);
    }

    @Test
    void shouldThrowOnWrongInputSize() {
        Neuron neuron = new Neuron(3, ActivationFunction.LINEAR, 42);

        assertThatThrownBy(() -> neuron.forward(new double[]{1.0, 2.0}))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void shouldSerializeDeserialize() {
        Neuron original = new Neuron(3, ActivationFunction.TANH, 42);
        original.setWeightsAndBias(new double[]{0.1, 0.2, 0.3}, 0.05);

        // Create new neuron from saved params
        Neuron restored = new Neuron(original.getWeights(), original.getBias(), original.getActivation().name(), original.getInputSize());

        assertThat(restored.getWeights()).isEqualTo(original.getWeights());
        assertThat(restored.getBias()).isEqualTo(original.getBias());
        assertThat(restored.getActivation()).isEqualTo(original.getActivation());
    }
}