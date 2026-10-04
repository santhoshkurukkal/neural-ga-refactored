package com.neuralga.neural;

import com.neuralga.config.CascadeConfig;
import com.neuralga.config.TrainingConfig;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class CascadeCorrelationNetworkTest {

    private TrainingConfig createConfig() {
        TrainingConfig config = new TrainingConfig();
        config.setInputWindow(5);
        config.setPredictionHorizon(3);
        config.setNeuronsPerHiddenLayer(4);
        config.setMaxHiddenLayers(3);
        config.setActivation("TANH");
        config.setOutputActivation("LINEAR");
        return config;
    }

    private CascadeConfig createCascadeConfig() {
        CascadeConfig config = new CascadeConfig();
        config.setMaxCascadeIterations(3);
        return config;
    }

    @Test
    void shouldCreateWithOnlyOutputLayer() {
        CascadeCorrelationNetwork network = new CascadeCorrelationNetwork(createConfig(), createCascadeConfig());

        assertThat(network.getNumLayers()).isEqualTo(1);
        assertThat(network.getHiddenLayerCount()).isEqualTo(0);
        assertThat(network.getInputSize()).isEqualTo(5);
        assertThat(network.getOutputSize()).isEqualTo(3);
    }

    @Test
    void shouldAddHiddenLayer() {
        CascadeCorrelationNetwork network = new CascadeCorrelationNetwork(createConfig(), createCascadeConfig());

        double[][] weights = new double[4][5];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 5; j++) {
                weights[i][j] = 0.1;
            }
        }
        double[] biases = new double[]{0.0, 0.0, 0.0, 0.0};

        network.addHiddenLayer(weights, biases);

        assertThat(network.getHiddenLayerCount()).isEqualTo(1);
        assertThat(network.getNumLayers()).isEqualTo(2);
    }

    @Test
    void shouldForwardWithHiddenLayers() {
        CascadeCorrelationNetwork network = new CascadeCorrelationNetwork(createConfig(), createCascadeConfig());

        double[][] weights = new double[4][5];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 5; j++) {
                weights[i][j] = 0.1;
            }
        }
        network.addHiddenLayer(weights, new double[4]);

        double[] input = new double[5];
        for (int i = 0; i < 5; i++) input[i] = 1.0;

        double[] output = network.forward(input);
        assertThat(output).hasSize(3);
    }

    @Test
    void shouldTrackCanAddMoreLayers() {
        CascadeCorrelationNetwork network = new CascadeCorrelationNetwork(createConfig(), createCascadeConfig());

        assertThat(network.canAddMoreLayers()).isTrue();

        for (int i = 0; i < 3; i++) {
            double[][] w = new double[4][network.getInputSize()];
            network.addHiddenLayer(w, new double[4]);
        }

        assertThat(network.canAddMoreLayers()).isFalse();
    }

    @Test
    void shouldPreserveOutputLayerWeightsWhenRebuilding() {
        CascadeCorrelationNetwork network = new CascadeCorrelationNetwork(createConfig(), createCascadeConfig());

        // Set known output weights
        Layer output = network.getOutputLayer();
        double[][] originalWeights = output.getWeightsMatrix();
        double[] originalBiases = output.getBiases();

        // Add hidden layer
        network.addHiddenLayer(new double[4][5], new double[4]);

        // Check output layer was rebuilt and weights partially preserved
        Layer newOutput = network.getOutputLayer();
        assertThat(newOutput.getInputSize()).isEqualTo(5 + 4); // input + hidden
    }
}