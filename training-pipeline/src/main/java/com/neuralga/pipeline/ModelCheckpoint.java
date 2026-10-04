package com.neuralga.pipeline;

import com.neuralga.data.DataNormalizer;
import com.neuralga.neural.CascadeCorrelationNetwork;
import com.neuralga.neural.NetworkSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

public class ModelCheckpoint {

    private static final Logger log = LoggerFactory.getLogger(ModelCheckpoint.class);

    private final Path checkpointDir;
    private final int saveInterval;
    private int counter = 0;

    private CascadeCorrelationNetwork bestNetwork;
    private double bestValLoss = Double.POSITIVE_INFINITY;

    public ModelCheckpoint(Path checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = saveInterval;
        try {
            Files.createDirectories(checkpointDir);
        } catch (IOException e) {
            log.error("Failed to create checkpoint directory", e);
        }
    }

    public void maybeSave(CascadeCorrelationNetwork network, DataNormalizer normalizer,
                          com.neuralga.config.TrainingConfig config, double valLoss) {
        counter++;

        if (valLoss < bestValLoss) {
            bestValLoss = valLoss;
            bestNetwork = network;
            saveCheckpoint(network, normalizer, config, valLoss, "best");
            log.info("New best model saved: valLoss={:.6f}", valLoss);
        }

        if (counter % saveInterval == 0) {
            saveCheckpoint(network, normalizer, config, valLoss, "iter-" + counter);
        }
    }

    private void saveCheckpoint(CascadeCorrelationNetwork network, DataNormalizer normalizer,
                                com.neuralga.config.TrainingConfig config, double valLoss, String suffix) {
        Path path = checkpointDir.resolve("checkpoint-" + suffix + ".json");
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("valLoss", valLoss);
        metrics.put("timestamp", Instant.now().toString());
        metrics.put("iteration", counter);

        try {
            NetworkSerializer.save(network, normalizer, config, path, metrics);
        } catch (Exception e) {
            log.error("Failed to save checkpoint: {}", path, e);
        }
    }

    public CascadeCorrelationNetwork getBestNetwork() { return bestNetwork; }
    public double getBestValLoss() { return bestValLoss; }
}