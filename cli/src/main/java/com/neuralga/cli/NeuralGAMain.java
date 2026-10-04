package com.neuralga.cli;

import com.neuralga.config.*;
import com.neuralga.data.*;
import com.neuralga.pipeline.TrainingPipeline;
import com.neuralga.service.PredictionService;
import com.neuralga.service.ModelLoader;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.Callable;

@Command(name = "neural-ga", mixinStandardHelpOptions = true, version = "1.0.0",
        description = "Cascade Correlation Neural Network with Genetic Algorithm for Time-Series Prediction",
        subcommands = {
                TrainCommand.class,
                PredictCommand.class,
                EvaluateCommand.class,
                GenerateDataCommand.class
        })
public class NeuralGAMain implements Callable<Integer> {

    @Option(names = {"-c", "--config"}, description = "Path to config YAML file")
    String configPath = "config/training.yaml";

    @Option(names = {"-v", "--verbose"}, description = "Enable verbose logging")
    boolean verbose = false;

    public static void main(String[] args) {
        int exitCode = new CommandLine(new NeuralGAMain()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() {
        System.out.println("Neural-GA: Cascade Correlation + GA for Time-Series Prediction");
        System.out.println("Use --help to see available commands");
        return 0;
    }

    // Train command
    @Command(name = "train", description = "Train a new model")
    static class TrainCommand implements Callable<Integer> {
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

                pipeline.prepareData().train().saveModel(outputFile);

                System.out.println("Training complete. Model saved to: " + outputFile);
                return 0;
            } catch (Exception e) {
                System.err.println("Training failed: " + e.getMessage());
                e.printStackTrace();
                return 1;
            }
        }
    }

    // Predict command
    @Command(name = "predict", description = "Make predictions using a trained model")
    static class PredictCommand implements Callable<Integer> {
        @Option(names = {"-m", "--model"}, description = "Path to model file", required = true)
        String modelFile;

        @Option(names = {"-d", "--data"}, description = "Path to input data file")
        String dataFile;

        @Option(names = {"-o", "--output"}, description = "Output predictions file")
        String outputFile = "predictions.csv";

        @Option(names = {"--steps"}, description = "Number of steps to predict (if no data file)")
        int steps = 10;

        @Override
        public Integer call() {
            try {
                Path modelPath = Paths.get(modelFile);
                ModelLoader.Model model = ModelLoader.load(modelPath);
                PredictionService service = new PredictionService(model);

                if (dataFile != null && !dataFile.isEmpty()) {
                    // Load data and predict on each window
                    CsvTimeseriesReader reader = new CsvTimeseriesReader();
                    var data = reader.loadFromPath(Paths.get(dataFile));

                    SlidingWindowDataset dataset = SlidingWindowDataset.builder()
                            .inputWindow(model.getInputWindow())
                            .predictionHorizon(model.getPredictionHorizon())
                            .build();
                    dataset.build(data);

                    PredictionService.BatchPredictor predictor = new PredictionService.BatchPredictor(service);
                    var predictions = predictor.predictToList(dataset);

                    // Save predictions
                    StringBuilder sb = new StringBuilder();
                    sb.append("input,prediction\n");
                    for (int i = 0; i < predictions.size(); i++) {
                        double[] input = dataset.getInput(i);
                        double[] pred = predictions.get(i);
                        sb.append(String.join(",", java.util.Arrays.stream(input).mapToObj(String::valueOf).toArray(String[]::new)))
                                .append(",").append(pred[0]).append("\n");
                    }
                    Files.writeString(Paths.get(outputFile), sb.toString());
                    System.out.println("Predictions saved to: " + outputFile);
                } else {
                    // Interactive prediction from last known values
                    System.out.println("Enter " + model.getInputWindow() + " values for prediction (comma-separated):");
                    // This would need user input - simplified for now
                    System.out.println("Use --data for batch prediction or provide initial values");
                }

                return 0;
            } catch (Exception e) {
                System.err.println("Prediction failed: " + e.getMessage());
                e.printStackTrace();
                return 1;
            }
        }
    }

    // Evaluate command
    @Command(name = "evaluate", description = "Evaluate a trained model on test data")
    static class EvaluateCommand implements Callable<Integer> {
        @Option(names = {"-m", "--model"}, description = "Path to model file", required = true)
        String modelFile;

        @Option(names = {"-d", "--data"}, description = "Path to test data file", required = true)
        String dataFile;

        @Override
        public Integer call() {
            try {
                Path modelPath = Paths.get(modelFile);
                ModelLoader.Model model = ModelLoader.load(modelPath);
                PredictionService service = new PredictionService(model);

                CsvTimeseriesReader reader = new CsvTimeseriesReader();
                var data = reader.loadFromPath(Paths.get(dataFile));

                SlidingWindowDataset dataset = SlidingWindowDataset.builder()
                        .inputWindow(model.getInputWindow())
                        .predictionHorizon(model.getPredictionHorizon())
                        .build();
                dataset.build(data);

                PredictionService.BatchPredictor predictor = new PredictionService.BatchPredictor(service);
                var predictions = predictor.predictToList(dataset);

                // Calculate metrics
                double mse = 0, mae = 0;
                for (int i = 0; i < predictions.size(); i++) {
                    double target = dataset.getTarget(i)[0];
                    double pred = predictions.get(i)[0];
                    double diff = pred - target;
                    mse += diff * diff;
                    mae += Math.abs(diff);
                }
                mse /= predictions.size();
                mae /= predictions.size();

                System.out.printf("Test MSE: %.6f%n", mse);
                System.out.printf("Test MAE: %.6f%n", mae);

                return 0;
            } catch (Exception e) {
                System.err.println("Evaluation failed: " + e.getMessage());
                e.printStackTrace();
                return 1;
            }
        }
    }

    // Generate data command
    @Command(name = "generate-data", description = "Generate synthetic training data")
    static class GenerateDataCommand implements Callable<Integer> {
        @Option(names = {"-o", "--output"}, description = "Output file", required = true)
        String outputFile;

        @Option(names = {"--type"}, description = "Data type")
        SyntheticDataGenerator.DataType type = SyntheticDataGenerator.DataType.SINE_WITH_NOISE;

        @Option(names = {"--length"}, description = "Data length")
        int length = 1000;

        @Option(names = {"--noise"}, description = "Noise level")
        double noise = 0.05;

        @Option(names = {"--seed"}, description = "Random seed")
        long seed = System.currentTimeMillis();

        @Override
        public Integer call() {
            try {
                SyntheticDataGenerator gen = SyntheticDataGenerator.builder()
                        .type(type)
                        .length(length)
                        .noiseLevel(noise)
                        .seed(seed)
                        .build();

                gen.saveToCsv(Paths.get(outputFile));
                System.out.println("Generated " + length + " " + type + " samples to: " + outputFile);
                return 0;
            } catch (Exception e) {
                System.err.println("Generation failed: " + e.getMessage());
                e.printStackTrace();
                return 1;
            }
        }
    }
}