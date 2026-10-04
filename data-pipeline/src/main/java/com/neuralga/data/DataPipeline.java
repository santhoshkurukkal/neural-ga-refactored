package com.neuralga.data;

import com.neuralga.config.AppConfig;
import com.neuralga.config.TrainingConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.List;

public class DataPipeline {

    private static final Logger log = LoggerFactory.getLogger(DataPipeline.class);

    private final TrainingConfig trainingConfig;
    private final AppConfig appConfig;

    private CsvTimeseriesReader reader;
    private SyntheticDataGenerator syntheticGenerator;
    private TrainValTestSplitter splitter;
    private DataNormalizer normalizer;
    private SlidingWindowDataset dataset;

    private List<Double> rawData;
    private TrainValTestSplitter.SplitResult splitResult;
    private boolean prepared = false;

    public DataPipeline(TrainingConfig trainingConfig, AppConfig appConfig) {
        this.trainingConfig = trainingConfig;
        this.appConfig = appConfig;
        this.reader = new CsvTimeseriesReader();
        this.splitter = new TrainValTestSplitter(
                1.0 - trainingConfig.getValidationSplit() - trainingConfig.getTestSplit(),
                trainingConfig.getValidationSplit(),
                trainingConfig.getTestSplit(),
                false, // chronological split for time-series
                appConfig.getRandomSeed()
        );
        this.normalizer = new DataNormalizer(DataNormalizer.Method.Z_SCORE);
        this.dataset = SlidingWindowDataset.builder()
                .inputWindow(trainingConfig.getInputWindow())
                .predictionHorizon(trainingConfig.getPredictionHorizon())
                .build();
    }

    public DataPipeline loadData(String filename) {
        Path dataPath = appConfig.getDataDir().resolve(filename);
        rawData = reader.loadFromPath(dataPath);
        log.info("Loaded raw data: {} samples from {}", rawData.size(), dataPath);
        return this;
    }

    public DataPipeline generateSyntheticData(SyntheticDataGenerator.DataType type, int length, double noise) {
        syntheticGenerator = SyntheticDataGenerator.builder()
                .type(type)
                .length(length)
                .noiseLevel(noise)
                .seed(appConfig.getRandomSeed())
                .build();
        rawData = syntheticGenerator.generate();
        log.info("Generated synthetic {} data: {} samples", type, rawData.size());
        return this;
    }

    public DataPipeline saveSyntheticData(String filename) {
        if (syntheticGenerator == null) {
            throw new IllegalStateException("No synthetic generator configured");
        }
        try {
            Path path = appConfig.getDataDir().resolve(filename);
            syntheticGenerator.saveToCsv(path);
        } catch (Exception e) {
            log.error("Failed to save synthetic data", e);
            throw new RuntimeException(e);
        }
        return this;
    }

    public DataPipeline split() {
        if (rawData == null) {
            throw new IllegalStateException("No data loaded. Call loadData() or generateSyntheticData() first.");
        }
        splitResult = splitter.split(rawData);
        return this;
    }

    public DataPipeline normalize() {
        if (splitResult == null) {
            throw new IllegalStateException("Data not split. Call split() first.");
        }
        normalizer.fit(splitResult.getTrain());
        log.info("Fitted normalizer on training data: mean={}, std={}", normalizer.getMean(), normalizer.getStd());
        return this;
    }

    public DataPipeline buildDataset() {
        if (splitResult == null || !normalizer.isFitted()) {
            throw new IllegalStateException("Data not prepared. Call split() and normalize() first.");
        }

        dataset.build(normalizer.normalize(splitResult.getTrain()));
        prepared = true;
        return this;
    }

    public PreparedData prepare() {
        if (!prepared) {
            buildDataset();
        }
        return new PreparedData(this);
    }

    public List<Double> getRawData() { return rawData; }
    public TrainValTestSplitter.SplitResult getSplitResult() { return splitResult; }
    public DataNormalizer getNormalizer() { return normalizer; }
    public SlidingWindowDataset getDataset() { return dataset; }

    public static class PreparedData {
        private final DataPipeline pipeline;

        public PreparedData(DataPipeline pipeline) {
            this.pipeline = pipeline;
        }

        public SlidingWindowDataset getTrainDataset() {
            return pipeline.dataset;
        }

        public SlidingWindowDataset getValDataset() {
            SlidingWindowDataset valDataset = SlidingWindowDataset.builder()
                    .inputWindow(pipeline.trainingConfig.getInputWindow())
                    .predictionHorizon(pipeline.trainingConfig.getPredictionHorizon())
                    .build();
            valDataset.build(pipeline.normalizer.normalize(pipeline.splitResult.getVal()));
            return valDataset;
        }

        public SlidingWindowDataset getTestDataset() {
            SlidingWindowDataset testDataset = SlidingWindowDataset.builder()
                    .inputWindow(pipeline.trainingConfig.getInputWindow())
                    .predictionHorizon(pipeline.trainingConfig.getPredictionHorizon())
                    .build();
            testDataset.build(pipeline.normalizer.normalize(pipeline.splitResult.getTest()));
            return testDataset;
        }

        public DataNormalizer getNormalizer() {
            return pipeline.normalizer;
        }

        public List<Double> getRawTrain() { return pipeline.splitResult.getTrain(); }
        public List<Double> getRawVal() { return pipeline.splitResult.getVal(); }
        public List<Double> getRawTest() { return pipeline.splitResult.getTest(); }

        public TrainingConfig getTrainingConfig() { return pipeline.trainingConfig; }
    }
}