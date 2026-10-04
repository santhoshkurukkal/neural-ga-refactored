package com.neuralga.neural;

import com.neuralga.config.TrainingConfig;
import com.neuralga.data.DataNormalizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

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

        NetworkSerializer.ModelData loaded = NetworkSerializer.load(modelPath);

        assertThat(loaded.network).isNotNull();
        assertThat(loaded.network.getNumLayers()).isEqualTo(2);
        assertThat(loaded.network.getLearningRate()).isEqualTo(0.05);
        assertThat(loaded.normalizer).isNotNull();
        assertThat(loaded.normalizer.isFitted()).isTrue();
        assertThat(loaded.config).isNotNull();
        assertThat(loaded.metrics).containsEntry("valMSE", 0.001);
    }

    @Test
    void shouldThrowOnInvalidPath() {
        Network network = new Network();
        DataNormalizer normalizer = new DataNormalizer(DataNormalizer.Method.NONE);
        TrainingConfig config = new TrainingConfig();

        assertThatThrownBy(() -> NetworkSerializer.save(network, normalizer, config,
                Path.of("/invalid/path/model.json"), Map.of()))
                .isInstanceOf(NetworkSerializer.SerializationException.class);
    }
}