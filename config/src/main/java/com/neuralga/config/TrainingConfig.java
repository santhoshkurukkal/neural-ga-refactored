package com.neuralga.config;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

public class TrainingConfig {

    @JsonProperty("input_window")
    private int inputWindow = 5;

    @JsonProperty("prediction_horizon")
    private int predictionHorizon = 1;

    @JsonProperty("initial_output_neurons")
    private int initialOutputNeurons = 5;

    @JsonProperty("max_hidden_layers")
    private int maxHiddenLayers = 10;

    @JsonProperty("neurons_per_hidden_layer")
    private int neuronsPerHiddenLayer = 5;

    @JsonProperty("activation")
    private String activation = "TANH";

    @JsonProperty("output_activation")
    private String outputActivation = "LINEAR";

    @JsonProperty("epochs")
    private int epochs = 100;

    @JsonProperty("batch_size")
    private int batchSize = 32;

    @JsonProperty("learning_rate")
    private double learningRate = 0.01;

    @JsonProperty("optimizer")
    private String optimizer = "ADAM";

    @JsonProperty("loss")
    private String loss = "MSE";

    private EarlyStoppingConfig earlyStopping = new EarlyStoppingConfig();

    @JsonProperty("validation_split")
    private double validationSplit = 0.2;

    @JsonProperty("test_split")
    private double testSplit = 0.1;

    public TrainingConfig() {}

    public int getInputWindow() { return inputWindow; }
    public void setInputWindow(int inputWindow) { this.inputWindow = inputWindow; }

    public int getPredictionHorizon() { return predictionHorizon; }
    public void setPredictionHorizon(int predictionHorizon) { this.predictionHorizon = predictionHorizon; }

    public int getInitialOutputNeurons() { return initialOutputNeurons; }
    public void setInitialOutputNeurons(int initialOutputNeurons) { this.initialOutputNeurons = initialOutputNeurons; }

    public int getMaxHiddenLayers() { return maxHiddenLayers; }
    public void setMaxHiddenLayers(int maxHiddenLayers) { this.maxHiddenLayers = maxHiddenLayers; }

    public int getNeuronsPerHiddenLayer() { return neuronsPerHiddenLayer; }
    public void setNeuronsPerHiddenLayer(int neuronsPerHiddenLayer) { this.neuronsPerHiddenLayer = neuronsPerHiddenLayer; }

    public String getActivation() { return activation; }
    public void setActivation(String activation) { this.activation = activation; }

    public String getOutputActivation() { return outputActivation; }
    public void setOutputActivation(String outputActivation) { this.outputActivation = outputActivation; }

    public int getEpochs() { return epochs; }
    public void setEpochs(int epochs) { this.epochs = epochs; }

    public int getBatchSize() { return batchSize; }
    public void setBatchSize(int batchSize) { this.batchSize = batchSize; }

    public double getLearningRate() { return learningRate; }
    public void setLearningRate(double learningRate) { this.learningRate = learningRate; }

    public String getOptimizer() { return optimizer; }
    public void setOptimizer(String optimizer) { this.optimizer = optimizer; }

    public String getLoss() { return loss; }
    public void setLoss(String loss) { this.loss = loss; }

    public EarlyStoppingConfig getEarlyStopping() { return earlyStopping; }
    public void setEarlyStopping(EarlyStoppingConfig earlyStopping) { this.earlyStopping = earlyStopping; }

    public double getValidationSplit() { return validationSplit; }
    public void setValidationSplit(double validationSplit) { this.validationSplit = validationSplit; }

    public double getTestSplit() { return testSplit; }
    public void setTestSplit(double testSplit) { this.testSplit = testSplit; }

    public static class EarlyStoppingConfig {
        @JsonProperty("patience")
        private int patience = 10;

        @JsonProperty("min_delta")
        private double minDelta = 1e-4;

        public EarlyStoppingConfig() {}

        public int getPatience() { return patience; }
        public void setPatience(int patience) { this.patience = patience; }

        public double getMinDelta() { return minDelta; }
        public void setMinDelta(double minDelta) { this.minDelta = minDelta; }
    }
}