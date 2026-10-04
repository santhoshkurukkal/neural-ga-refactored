package com.neuralga.cli.command;

import com.neuralga.config.*;
import com.neuralga.data.SyntheticDataGenerator;
import com.neuralga.pipeline.TrainingPipeline;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.Callable;

@Command(name = "train", description = "Train a new model")
public class TrainCommand implements Callable<Integer> {

    @Option(names = {"-d", "--data"}, description = "Path to training data file", required = true)
    String dataFile;

    @Option(names = {"-c", "--config"}, description = "Path to config YAML")
    String configPath = "config/training.yaml";

    @Option(names = {"-o", "--output"}, description = "Output model file")
    String outputFile = "models/best-model.json";

    @Option(names = {"--generate"}, description = "Generate synthetic data instead of loading")
    boolean generate = false;

    @Option(names = {"--type"}, description = "Synthetic data type")
    SyntheticDataGenerator.DataType type = SyntheticDataGenerator.DataType.SINE_WITH_NOISE;

    @Option(names = {"--length"}, description = "Synthetic data length")
    int length = 1000;

    @Option(names = {"--noise"}, description = "Noise level for synthetic data")
    double noise = 0.05;

    @Override
    public Integer call() {
        try {
            TrainingConfig trainingConfig = ConfigLoader.loadTrainingConfig(configPath);
            CascadeConfig cascadeConfig = ConfigLoader.loadCascadeConfig(configPath);
            GAConfig gaConfig = ConfigLoader.loadGAConfig(configPath);
            AppConfig appConfig = ConfigLoader.loadAppConfig(configPath);

            TrainingPipeline pipeline = new TrainingPipeline(trainingConfig, cascadeConfig, gaConfig, appConfig);

            if (generate) {
                pipeline.generateData(type, length, noise);
            } else {
                pipeline.loadData(dataFile);
            }

            pipeline.prepareData();
            pipeline.train();
            pipeline.saveModel(outputFile);

            System.out.println("Training complete. Model saved to: " + outputFile);
            return 0;
        } catch (Exception e) {
            System.err.println("Training failed: " + e.getMessage());
            e.printStackTrace();
            return 1;
        }
    }
}