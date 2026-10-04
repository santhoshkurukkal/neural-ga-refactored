package com.neuralga.pipeline;

import com.neuralga.config.AppConfig;
import com.neuralga.config.CascadeConfig;
import com.neuralga.config.GAConfig;
import com.neuralga.config.TrainingConfig;
import com.neuralga.data.DataPipeline;
import com.neuralga.neural.CascadeCorrelationNetwork;
import com.neuralga.neural.DataNormalizer;
import com.neuralga.neural.NetworkSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

public class TrainingPipeline {

    private static final Logger log = LoggerFactory.getLogger(TrainingPipeline.class);

    private final TrainingConfig trainingConfig;
    private final CascadeConfig cascadeConfig;
    private final GAConfig gaConfig;
    private final AppConfig appConfig;

    private DataPipeline dataPipeline;
    private CascadeGATrainer cascadeTrainer;
    private CascadeCorrelationNetwork trainedNetwork;
    private DataNormalizer normalizer;

    public TrainingPipeline(TrainingConfig trainingConfig, CascadeConfig cascadeConfig,
                            GAConfig gaConfig, AppConfig appConfig) {
        this.trainingConfig = trainingConfig;
        this.cascadeConfig = cascadeConfig;
        this.gaConfig = gaConfig;
        this.appConfig = appConfig;
    }

    public TrainingPipeline loadData(String dataFile) {
        dataPipeline = new DataPipeline(trainingConfig, appConfig);
        dataPipeline.loadData(dataFile);
        return this;
    }

    public TrainingPipeline generateData(com.neuralga.data.SyntheticDataGenerator.DataType type, int length, double noise) {
        dataPipeline = new DataPipeline(trainingConfig, appConfig);
        dataPipeline.generateSyntheticData(type, length, noise);
        return this;
    }

    public TrainingPipeline prepareData() {
        if (dataPipeline == null) {
            throw new IllegalStateException("No data source. Call loadData() or generateData() first.");
        }
        dataPipeline.split().normalize();
        normalizer = dataPipeline.getNormalizer();
        return this;
    }

    public CascadeCorrelationNetwork train() {
        if (dataPipeline == null) {
            throw new IllegalStateException("Data not prepared. Call prepareData() first.");
        }

        cascadeTrainer = new CascadeGATrainer(trainingConfig, cascadeConfig, gaConfig);
        DataPipeline.PreparedData preparedData = dataPipeline.prepare();
        trainedNetwork = cascadeTrainer.train(preparedData);

        return trainedNetwork;
    }

    public void saveModel(String modelFile) {
        if (trainedNetwork == null || normalizer == null) {
            throw new IllegalStateException("No trained model to save");
        }

        Path modelPath = appConfig.getModelsDir().resolve(modelFile);
        Map<String, Object> metrics = cascadeTrainer.getTrainingMetrics(
                dataPipeline != null ? dataPipeline.getDataset() : null);

        metrics.put("timestamp", Instant.now().toString());
        metrics.put("totalCascadeIterations", cascadeTrainer.getTotalCascadeIterations());

        NetworkSerializer.save(trainedNetwork, normalizer, trainingConfig, modelPath, metrics);
    }

    public TrainingPipeline loadModel(String modelFile) throws IOException {
        Path modelPath = appConfig.getModelsDir().resolve(modelFile);
        NetworkSerializer.ModelData modelData = NetworkSerializer.load(modelPath);

        trainedNetwork = (CascadeCorrelationNetwork) modelData.network;
        normalizer = modelData.normalizer;

        // Configs are loaded from model, but we can use current ones
        log.info("Loaded model from: {}", modelPath);
        log.info("Model timestamp: {}", modelData.timestamp);
        log.info("Hidden layers: {}", trainedNetwork.getHiddenLayerCount());

        return this;
    }

    public CascadeCorrelationNetwork getTrainedNetwork() { return trainedNetwork; }
    public DataNormalizer getNormalizer() { return normalizer; }
    public CascadeGATrainer getCascadeTrainer() { return cascadeTrainer; }
}