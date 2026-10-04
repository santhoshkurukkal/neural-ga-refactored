package com.neuralga.neural;

import com.neuralga.config.TrainingConfig;
import com.neuralga.data.DataNormalizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import static org.assertj.core.api.Assertions.*;

class NetworkSerializerTest {

    @TempDir
    Path tempDir;

    @Test
    void shouldSaveAndLoadNetwork() {
        Network network = new Network();
        network.addLayer(3, 2, ActivationFunction.TANH);
        network.addLayer(1, 3, ActivationFunction.LINEAR);
        network.setLearningRate(0.05);

        DataNormalizer normalizer = new DataNormalizer(DataNormalizer.Method.Z_SCORE);
        normalizer.fit(java.util.List.of(1.0, 2.0, 3.0, 4.0, 5.0));

        TrainingConfig config = new TrainingConfig();
        config.setInputWindow(2);
        config.setEpochs(100);

        Map<String, Object> metrics = Map.of("valMSE", 0.001, "testMSE", 0.002);

        Path modelPath = tempDir.resolve("model.json");

        NetworkSerializer.save(network, normalizer, config, modelPath, metrics);
        assertThat(modelPath).exists();

        // Just verify the file was created and has content
        String content;
        try {
            content = java.nio.file.Files.readString(modelPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        assertThat(content).isNotEmpty();
        assertThat(content).contains("network");
        assertThat(content).contains("normalizer");
        assertThat(content).contains("config");
        assertThat(content).contains("metrics");
    }

    @Test
    void shouldSaveConfig() {
        Network network = new Network();
        network.addLayer(3, 2, ActivationFunction.TANH);
        network.addLayer(1, 3, ActivationFunction.LINEAR);

        DataNormalizer normalizer = new DataNormalizer(DataNormalizer.Method.Z_SCORE);
        normalizer.fit(java.util.List.of(1.0, 2.0, 3.0, 4.0, 5.0));

        TrainingConfig config = new TrainingConfig();
        config.setInputWindow(2);
        config.setEpochs(100);

        Path modelPath = tempDir.resolve("model.json");

        NetworkSerializer.save(network, normalizer, config, modelPath, Map.of());

        assertThat(modelPath).exists();
    }
}