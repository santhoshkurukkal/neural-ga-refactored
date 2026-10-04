package com.neuralga.neural;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.neuralga.config.TrainingConfig;
import com.neuralga.data.DataNormalizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

public class NetworkSerializer {

    private static final Logger log = LoggerFactory.getLogger(NetworkSerializer.class);
    private static final ObjectMapper MAPPER = createMapper();

    private static ObjectMapper createMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        mapper.registerModule(new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule());
        return mapper;
    }

    public static void save(Network network, DataNormalizer normalizer, TrainingConfig config,
                            Path path, Map<String, Object> metrics) {
        ModelData modelData = new ModelData();
        modelData.timestamp = Instant.now().toString();
        modelData.network = network;
        modelData.normalizer = normalizer;
        modelData.config = config;
        modelData.metrics = metrics != null ? metrics : new HashMap<>();

        try {
            Files.createDirectories(path.getParent());
            MAPPER.writeValue(Files.newOutputStream(path), modelData);
            log.info("Saved model to: {}", path);
        } catch (IOException e) {
            log.error("Failed to save model to: {}", path, e);
            throw new SerializationException("Failed to save model", e);
        }
    }

    public static ModelData load(Path path) {
        try {
            ModelData data = MAPPER.readValue(Files.newInputStream(path), ModelData.class);
            log.info("Loaded model from: {}", path);
            return data;
        } catch (IOException e) {
            log.error("Failed to load model from: {}", path, e);
            throw new SerializationException("Failed to load model", e);
        }
    }

    public static class ModelData {
        public String timestamp;
        public Network network;
        public DataNormalizer normalizer;
        public TrainingConfig config;
        public Map<String, Object> metrics;
    }

    public static class SerializationException extends RuntimeException {
        public SerializationException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}