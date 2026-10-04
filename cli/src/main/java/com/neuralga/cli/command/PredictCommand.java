package com.neuralga.cli.command;

import com.neuralga.data.CsvTimeseriesReader;
import com.neuralga.data.SlidingWindowDataset;
import com.neuralga.service.ModelLoader;
import com.neuralga.service.PredictionService;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.Callable;

@Command(name = "predict", description = "Make predictions using a trained model")
public class PredictCommand implements Callable<Integer> {

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
                java.nio.file.Files.writeString(Paths.get(outputFile), sb.toString());
                System.out.println("Predictions saved to: " + outputFile);
            } else {
                // Interactive prediction from last known values
                System.out.println("Enter " + model.getInputWindow() + " values for prediction (comma-separated):");
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