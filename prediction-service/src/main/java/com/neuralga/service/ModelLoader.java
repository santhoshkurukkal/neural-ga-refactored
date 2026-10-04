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

        if (!(modelData.network instanceof CascadeCorrelationNetwork)) {
            log.warn("Loaded network is not a CascadeCorrelationNetwork, wrapping");
        }

        CascadeCorrelationNetwork network = (CascadeCorrelationNetwork) modelData.network;
        DataNormalizer normalizer = modelData.normalizer;
        TrainingConfig config = modelData.config;

        log.info("Loaded model from: {}", modelPath);
        log.info("Network: {} hidden layers, input={}, output={}",
                network.getHiddenLayerCount(), network.getInputSize(), network.getOutputSize());

        return new Model(network, normalizer, config, modelData.metrics);
    }

    public static class Model {
        private final CascadeCorrelationNetwork network;
        private final DataNormalizer normalizer;
        private final TrainingConfig config;
        private final java.util.Map<String, Object> metrics;

        public Model(CascadeCorrelationNetwork network, DataNormalizer normalizer,
                     TrainingConfig config, java.util.Map<String, Object> metrics) {
            this.network = Objects.requireNonNull(network);
            this.normalizer = normalizer;
            this.config = config;
            this.metrics = metrics;
        }

        public CascadeCorrelationNetwork getNetwork() { return network; }
        public DataNormalizer getNormalizer() { return normalizer; }
        public TrainingConfig getConfig() { return config; }
        public java.util.Map<String, Object> getMetrics() { return metrics; }

        public int getInputWindow() { return config.getInputWindow(); }
        public int getPredictionHorizon() { return config.getPredictionHorizon(); }
    }
}