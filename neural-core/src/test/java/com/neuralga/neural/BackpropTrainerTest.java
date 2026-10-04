package com.neuralga.neural;

import com.neuralga.config.TrainingConfig;
import com.neuralga.data.SlidingWindowDataset;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class BackpropTrainerTest {

    private TrainingConfig createConfig() {
        TrainingConfig config = new TrainingConfig();
        config.setInputWindow(3);
        config.setPredictionHorizon(1);
        config.setEpochs(50);
        config.setBatchSize(10);
        config.setLearningRate(0.1);
        config.setOptimizer("SGD");
        config.setLoss("MSE");
        config.getEarlyStopping().setPatience(10);
        config.getEarlyStopping().setMinDelta(1e-4);
        return config;
    }

    @Test
    void shouldTrainOnSimplePattern() {
        // Create data: y = x (identity)
        SlidingWindowDataset trainData = SlidingWindowDataset.builder()
                .inputWindow(3)
                .predictionHorizon(1)
                .build();
        trainData.build(java.util.stream.DoubleStream.iterate(0, x -> x + 0.1).limit(50).boxed().toList());

        SlidingWindowDataset valData = SlidingWindowDataset.builder()
                .inputWindow(3)
                .predictionHorizon(1)
                .build();
        valData.build(java.util.stream.DoubleStream.iterate(5, x -> x + 0.1).limit(20).boxed().toList());

        Network network = new Network();
        network.addLayer(5, 3, ActivationFunction.TANH);
        network.addLayer(1, 5, ActivationFunction.LINEAR);

        BackpropTrainer trainer = new BackpropTrainer(createConfig());
        BackpropTrainer.TrainingResult result = trainer.train(network, trainData, valData);

        assertThat(result.epochsTrained).isGreaterThan(0);
        assertThat(result.finalValLoss).isLessThan(1.0); // Should learn something
    }

    @Test
    void shouldEarlyStop() {
        TrainingConfig config = createConfig();
        config.setEpochs(1000);
        config.getEarlyStopping().setPatience(3);
        config.getEarlyStopping().setMinDelta(1e-3);

        SlidingWindowDataset trainData = SlidingWindowDataset.builder()
                .inputWindow(3)
                .predictionHorizon(1)
                .build();
        trainData.build(java.util.List.of(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0));

        SlidingWindowDataset valData = SlidingWindowDataset.builder()
                .inputWindow(3)
                .predictionHorizon(1)
                .build();
        valData.build(java.util.List.of(1.0, 2.0, 3.0, 4.0, 5.0));

        Network network = new Network();
        network.addLayer(3, 3, ActivationFunction.LINEAR);
        network.addLayer(1, 3, ActivationFunction.LINEAR);

        BackpropTrainer trainer = new BackpropTrainer(config);
        BackpropTrainer.TrainingResult result = trainer.train(network, trainData, valData);

        assertThat(result.epochsTrained).isLessThan(100); // Should stop early
        assertThat(trainer.getEpochsTrained()).isEqualTo(result.epochsTrained);
    }

    @Test
    void shouldEvaluateCorrectly() {
        TrainingConfig config = createConfig();

        SlidingWindowDataset data = SlidingWindowDataset.builder()
                .inputWindow(2)
                .predictionHorizon(1)
                .build();
        data.build(java.util.List.of(1.0, 2.0, 3.0, 4.0, 5.0));

        Network network = new Network();
        network.addLayer(1, 2, ActivationFunction.LINEAR);
        network.setLearningRate(0.01);

        BackpropTrainer trainer = new BackpropTrainer(config);
        double loss = trainer.evaluate(network, data);

        assertThat(loss).isGreaterThanOrEqualTo(0.0);
    }

    @Test
    void shouldPredict() {
        TrainingConfig config = createConfig();

        Network network = new Network();
        network.addLayer(1, 3, ActivationFunction.LINEAR);
        Layer layer = network.getLayers().get(0);
        layer.setWeightsMatrix(new double[][]{{1.0, 0.0, 0.0}});
        layer.setBiases(new double[]{0.0});

        BackpropTrainer trainer = new BackpropTrainer(config);
        double[] prediction = trainer.predict(network, new double[]{5.0, 0.0, 0.0});

        assertThat(prediction[0]).isEqualTo(5.0);
    }
}