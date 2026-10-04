package com.neuralga.pipeline;

import com.neuralga.config.*;
import com.neuralga.data.DataPipeline;
import com.neuralga.data.SlidingWindowDataset;
import com.neuralga.neural.CascadeCorrelationNetwork;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class TrainingPipelineTest {

    private TrainingConfig createTrainingConfig() {
        TrainingConfig c = new TrainingConfig();
        c.setInputWindow(3);
        c.setPredictionHorizon(1);
        c.setInitialOutputNeurons(2);
        c.setNeuronsPerHiddenLayer(3);
        c.setMaxHiddenLayers(2);
        c.setEpochs(10);
        c.setBatchSize(5);
        c.setLearningRate(0.1);
        c.getEarlyStopping().setPatience(3);
        return c;
    }

    private CascadeConfig createCascadeConfig() {
        CascadeConfig c = new CascadeConfig();
        c.setErrorThreshold(1.0);
        c.setMaxCascadeIterations(2);
        c.setRetrainEpochsAfterGrowth(5);
        c.setFineTuneLearningRateFactor(0.1);
        return c;
    }

    private GAConfig createGAConfig() {
        GAConfig c = new GAConfig();
        c.setPopulationSize(20);
        c.setGenerations(10);
        c.setElitismCount(1);
        c.setTournamentSize(2);
        c.setCrossoverRate(0.8);
        c.setMutationRate(0.1);
        c.getTermination().setMaxGenerations(10);
        return c;
    }

    private AppConfig createAppConfig() {
        AppConfig c = new AppConfig();
        c.setDataDir("data/sample");
        return c;
    }

    @Test
    void shouldCreatePipeline() {
        TrainingPipeline pipeline = new TrainingPipeline(
                createTrainingConfig(), createCascadeConfig(), createGAConfig(), createAppConfig());

        assertThat(pipeline).isNotNull();
    }

    @Test
    void shouldTrainOnSyntheticData() {
        TrainingPipeline pipeline = new TrainingPipeline(
                createTrainingConfig(), createCascadeConfig(), createGAConfig(), createAppConfig());

        CascadeCorrelationNetwork network = pipeline
                .generateData(com.neuralga.data.SyntheticDataGenerator.DataType.SINE, 100, 0.05)
                .prepareData()
                .train();

        assertThat(network).isNotNull();
        assertThat(network.getHiddenLayerCount()).isGreaterThanOrEqualTo(0);
    }
}