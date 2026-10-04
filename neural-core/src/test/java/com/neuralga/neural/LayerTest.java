package com.neuralga.neural;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class LayerTest {

    @Test
    void shouldCreateLayerWithNeurons() {
        Layer layer = new Layer(3, 2, ActivationFunction.TANH, 42);

        assertThat(layer.getSize()).isEqualTo(3);
        assertThat(layer.getInputSize()).isEqualTo(2);
        assertThat(layer.getActivation()).isEqualTo(ActivationFunction.TANH);
        assertThat(layer.getNeurons()).hasSize(3);
    }

    @Test
    void shouldForwardCorrectly() {
        Layer layer = new Layer(2, 2, ActivationFunction.LINEAR, 42);
        // Set known weights
        double[][] weights = {{1.0, 0.0}, {0.0, 1.0}};
        double[] biases = {0.1, 0.2};
        layer.setWeightsMatrix(weights);
        layer.setBiases(biases);

        double[] input = {2.0, 3.0};
        double[] output = layer.forward(input);

        // Neuron 0: 1*2 + 0*3 + 0.1 = 2.1
        // Neuron 1: 0*2 + 1*3 + 0.2 = 3.2
        assertThat(output).containsExactly(2.1, 3.2);
    }

    @Test
    void shouldBackwardOutputLayer() {
        Layer layer = new Layer(2, 2, ActivationFunction.LINEAR, 42);
        double[][] weights = {{1.0, 0.0}, {0.0, 1.0}};
        double[] biases = {0.0, 0.0};
        layer.setWeightsMatrix(weights);
        layer.setBiases(biases);

        double[] input = {1.0, 1.0};
        layer.forward(input); // output = [1, 1]

        double[] target = {2.0, 3.0};
        double[] deltas = layer.backwardOutput(target);

        // delta = target - output = [1, 2]
        assertThat(deltas).containsExactly(1.0, 2.0);
    }

    @Test
    void shouldUpdateWeights() {
        Layer layer = new Layer(2, 2, ActivationFunction.LINEAR, 42);
        double[][] weights = {{0.0, 0.0}, {0.0, 0.0}};
        double[] biases = {0.0, 0.0};
        layer.setWeightsMatrix(weights);
        layer.setBiases(biases);

        double[] input = {1.0, 2.0};
        layer.forward(input);

        double[] target = {1.0, 1.0};
        layer.backwardOutput(target);

        layer.updateWeights(0.1);

        // Check weights were updated
        double[][] newWeights = layer.getWeightsMatrix();
        assertThat(newWeights[0][0]).isNotEqualTo(0.0);
        assertThat(newWeights[1][1]).isNotEqualTo(0.0);
    }

    @Test
    void shouldGetWeightsMatrix() {
        Layer layer = new Layer(2, 3, ActivationFunction.TANH, 42);
        double[][] weights = {{1,2,3}, {4,5,6}};
        layer.setWeightsMatrix(weights);

        double[][] retrieved = layer.getWeightsMatrix();
        assertThat(retrieved).isEqualTo(weights);
    }

    @Test
    void shouldThrowOnWrongInputSize() {
        Layer layer = new Layer(2, 3, ActivationFunction.LINEAR, 42);

        assertThatThrownBy(() -> layer.forward(new double[]{1.0, 2.0}))
                .isInstanceOf(IllegalArgumentException.class);
    }
}