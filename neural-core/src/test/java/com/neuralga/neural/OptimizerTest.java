package com.neuralga.neural;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class OptimizerTest {

    @Test
    void sgdShouldUpdateWeights() {
        Layer layer = new Layer(2, 2, ActivationFunction.LINEAR, 42);
        double[][] weights = {{0.0, 0.0}, {0.0, 0.0}};
        double[] biases = {0.0, 0.0};
        layer.setWeightsMatrix(weights);
        layer.setBiases(biases);

        double[] input = {1.0, 1.0};
        layer.forward(input);

        double[] target = {1.0, 1.0};
        layer.backwardOutput(target);

        Optimizer.SGD sgd = new Optimizer.SGD(0.1);
        sgd.update(layer, 0.1);

        double[][] newWeights = layer.getWeightsMatrix();
        assertThat(newWeights[0][0]).isNotEqualTo(0.0);
        assertThat(newWeights[1][1]).isNotEqualTo(0.0);
    }

    @Test
    void adamShouldUpdateWeights() {
        Layer layer = new Layer(2, 2, ActivationFunction.LINEAR, 42);
        double[][] weights = {{0.0, 0.0}, {0.0, 0.0}};
        layer.setWeightsMatrix(weights);

        double[] input = {1.0, 1.0};
        layer.forward(input);

        double[] target = {1.0, 1.0};
        layer.backwardOutput(target);

        Optimizer.Adam adam = new Optimizer.Adam(0.1);
        adam.update(layer, 0.1);

        double[][] newWeights = layer.getWeightsMatrix();
        assertThat(newWeights[0][0]).isNotEqualTo(0.0);
    }

    @Test
    void rmspropShouldUpdateWeights() {
        Layer layer = new Layer(2, 2, ActivationFunction.LINEAR, 42);
        double[][] weights = {{0.0, 0.0}, {0.0, 0.0}};
        layer.setWeightsMatrix(weights);

        double[] input = {1.0, 1.0};
        layer.forward(input);

        double[] target = {1.0, 1.0};
        layer.backwardOutput(target);

        Optimizer.RMSprop rms = new Optimizer.RMSprop(0.1);
        rms.update(layer, 0.1);

        double[][] newWeights = layer.getWeightsMatrix();
        assertThat(newWeights[0][0]).isNotEqualTo(0.0);
    }

    @Test
    void createShouldReturnCorrectOptimizer() {
        assertThat(Optimizer.create("SGD", 0.01)).isInstanceOf(Optimizer.SGD.class);
        assertThat(Optimizer.create("ADAM", 0.01)).isInstanceOf(Optimizer.Adam.class);
        assertThat(Optimizer.create("RMSPROP", 0.01)).isInstanceOf(Optimizer.RMSprop.class);
        assertThat(Optimizer.create("unknown", 0.01)).isInstanceOf(Optimizer.SGD.class);
    }

    @Test
    void adamShouldReset() {
        Optimizer.Adam adam = new Optimizer.Adam(0.01);
        adam.reset();
        // Just verify no exception
    }
}