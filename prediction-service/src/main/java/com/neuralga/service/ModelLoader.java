package com.neuralga.service;

import com.neuralga.config.TrainingConfig;
import com.neuralga.data.DataNormalizer;
import com.neuralga.neural.CascadeCorrelationNetwork;
import com.neuralga.neural.Network;
import com.neuralga.neural.NetworkSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

public class ModelLoader {

    private static final Logger log = LoggerFactory.getLogger(ModelLoader.class);

    public static Model load(Path modelPath) throws IOException {
        if (!Files.exists(modelPath)) {
            throw new IOException("Model file not found: " + modelPath);
        }

        NetworkSerializer.ModelData modelData = NetworkSerializer.load(modelPath);

        Network network = modelData.network;
        DataNormalizer normalizer = modelData.normalizer;
        TrainingConfig config = modelData.config;

        log.info("Loaded model from: {}", modelPath);
        log.info("Network: input={}, output={}, layers={}",
                network.getInputSize(), network.getOutputSize(), network.getNumLayers());

        return new Model(network, normalizer, config, modelData.metrics);
    }

    public static class Model {
        private final Network network;
        private final DataNormalizer normalizer;
        private final TrainingConfig config;
        private final java.util.Map<String, Object> metrics;

        public Model(Network network, DataNormalizer normalizer,
                     TrainingConfig config, java.util.Map<String, Object> metrics) {
            this.network = Objects.requireNonNull(network);
            this.normalizer = normalizer;
            this.config = config;
            this.metrics = metrics;
        }

        public Network getNetwork() { return network; }
        public DataNormalizer getNormalizer() { return normalizer; }
        public TrainingConfig getConfig() { return config; }
        public java.util.Map<String, Object> getMetrics() { return metrics; }

        public int getInputWindow() { return config.getInputWindow(); }
        public int getPredictionHorizon() { return config.getPredictionHorizon(); }
    }
}