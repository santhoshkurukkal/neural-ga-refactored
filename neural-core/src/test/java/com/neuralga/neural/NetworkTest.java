package com.neuralga.neural;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class NetworkTest {

    @Test
    void shouldCreateEmptyNetwork() {
        Network network = new Network();
        assertThat(network.getNumLayers()).isEqualTo(0);
    }

    @Test
    void shouldAddLayers() {
        Network network = new Network();
        network.addLayer(3, 2, ActivationFunction.TANH);
        network.addLayer(1, 3, ActivationFunction.LINEAR);

        assertThat(network.getNumLayers()).isEqualTo(2);
        assertThat(network.getInputSize()).isEqualTo(2);
        assertThat(network.getOutputSize()).isEqualTo(1);
    }

    @Test
    void shouldForwardThroughNetwork() {
        Network network = new Network();
        // 2 inputs -> 3 hidden (tanh) -> 1 output (linear)
        network.addLayer(3, 2, ActivationFunction.TANH);
        network.addLayer(1, 3, ActivationFunction.LINEAR);

        // Set simple weights for predictable output
        Layer hidden = network.getLayers().get(0);
        hidden.setWeightsMatrix(new double[][]{{1,0}, {0,1}, {1,1}});
        hidden.setBiases(new double[]{0, 0, 0});

        Layer output = network.getLayers().get(1);
        output.setWeightsMatrix(new double[][]{{1, 1, 1}});
        output.setBiases(new double[]{0});

        double[] input = {1.0, 2.0};
        double[] outputArr = network.forward(input);

        // Hidden: [tanh(1), tanh(2), tanh(3)] ≈ [0.76, 0.96, 1.0]
        // Output: 0.76 + 0.96 + 1.0 ≈ 2.72
        assertThat(outputArr).hasSize(1);
        assertThat(outputArr[0]).isCloseTo(2.72, within(0.05));
    }

    @Test
    void shouldTrainSingleStep() {
        Network network = new Network();
        network.addLayer(2, 2, ActivationFunction.LINEAR);
        network.setLearningRate(0.1);

        double[] input = {1.0, 0.0};
        double[] target = {1.0, 0.0};

        double[] outputBefore = network.forward(input);
        network.train(input, target);
        double[] outputAfter = network.getOutput();

        // After one training step, loss should not increase (may stay same or decrease)
        double lossBefore = LossFunction.MSE.compute(outputBefore, new double[]{1.0, 0.0});
        double lossAfter = LossFunction.MSE.compute(outputAfter, new double[]{1.0, 0.0});
        assertThat(lossAfter).isLessThanOrEqualTo(lossBefore);
    }

    @Test
    void shouldGetAndSetParameters() {
        Network network = new Network();
        network.addLayer(2, 2, ActivationFunction.TANH);
        network.addLayer(1, 2, ActivationFunction.LINEAR);

        Network.NetworkParams params = network.getParameters();
        assertThat(params.weights).hasSize(2);
        assertThat(params.biases).hasSize(2);
        assertThat(params.learningRate).isEqualTo(0.01);

        // Modify and restore
        double[][] newWeights = params.weights.get(0);
        newWeights[0][0] = 999.0;
        network.setParameters(params);

        double[][] checkWeights = network.getLayers().get(0).getWeightsMatrix();
        assertThat(checkWeights[0][0]).isEqualTo(999.0);
    }

    @Test
    void shouldCalculateLoss() {
        Network network = new Network();
        network.addLayer(1, 1, ActivationFunction.LINEAR);

        double[] input = {1.0};
        double[] target = {0.0};

        double loss = network.calculateLoss(input, target, LossFunction.MSE);
        // With random weights, loss will vary but should be >= 0
        assertThat(loss).isGreaterThanOrEqualTo(0.0);
    }
}